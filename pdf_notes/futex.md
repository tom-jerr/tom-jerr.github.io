---
title: futex
date: 2024/4/18 19:29
update: 
comments: true
description: futex机制介绍
katex: true
tags: 
- linux
- 锁
categories: Knowledge
---

# Futex

- `Futex`(`Fast userspace mutex`，用户态快速互斥锁) ，是一种用户态与内核态共同作用的锁，其**用户态部分负责锁逻辑**，内核态部分负责锁调度。

- 当用户态线程请求锁时，先在用户态进行锁状态的判断维护
  - 若此时不产生锁的竞争，则直接在用户态进行上锁返回；
  - 反之，则需要进行线程的挂起操作，通过`Futex`系统调用**请求内核介入来挂起线程**，并维护阻塞队列。

- 当用户态线程释放锁时，先在用户态进行锁状态的判断维护
  - 若此时没有其他线程被该锁阻塞，则直接在用户态进行解锁返回；
  - 反之，则需要进行阻塞线程的唤醒操作，通过`Futex`系统调用请求内核介入来**唤醒阻塞队列中的线程**。

~~~c
/// 快速系统调用
static int futex(uint32_t *uaddr, int futex_op, uint32_t val,
                 const struct timespec *timeout, uint32_t *uaddr2,
                 uint32_t val3) {
  return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
}
/// 申请快锁
static void fwait(uint32_t *futexp) {
  long s;
  while (1) {
    const uint32_t one = 1;
    if (atomic_compare_exchange_strong(futexp, &one, 0))
      break; //申请快锁成功
    //申请快锁失败,需等待
    s = futex(futexp, FUTEX_WAIT, 0, NULL, NULL, 0);
    if (s == -1 && errno != EAGAIN)
      errExit("futex-FUTEX_WAIT");
  }
}
/// 释放快锁
static void fpost(uint32_t *futexp) {
  long s;
  const uint32_t zero = 0;
  if (atomic_compare_exchange_strong(futexp, &zero, 1)) { //释放快锁成功
    s = futex(futexp, FUTEX_WAKE, 1, NULL, NULL, 0); //唤醒等锁 进程/线程
    if (s == -1)
      errExit("futex-FUTEX_WAKE");
  }
}
~~~

## 内核结构

- 内核并没有**快锁**这个结构体，`key`就是快锁，它们的关系是 `1:N` 的关系 ，快锁分成了 **私有锁** 和 **共享锁** 两种类型。用`key`表示唯一性。共享锁用物理地址 , 私有锁用虚拟地址。

### Futex Node

~~~c
typedef struct {
    UINTPTR      key;           /* private:uvaddr | 私有锁，用虚拟地址         shared:paddr | 共享锁，用物理地址 */
    UINT32       index;         /* hash bucket index | 哈希桶索引 OsFutexKeyToIndex */
    UINT32       pid;           /* private:process id   shared:OS_INVALID(-1) | 私有锁:进程ID     ， 共享锁为 -1 */
    LOS_DL_LIST  pendList;      /* point to pendList in TCB struct | 指向 TCB 结构中的 pendList, 通过它找到任务(TaskCB中含有该属性)*/
    LOS_DL_LIST  queueList;     /* thread list blocked by this lock | 挂等待这把锁的任务，其实这里挂到是FutexNode.queueList , 通过 queueList 可以找到 pendList ,通过 pendList又可以找到真正的任务*/
    LOS_DL_LIST  futexList;     /* point to the next FutexNode | 下一把快锁节点*/
} FutexNode;
~~~

### 任务调度

#### Wait Task

- 找到对应的key，找到对应的hash bucket
- 拷贝值到内核空间
- 进行任务的调度

~~~c
/// 将当前任务挂入等待链表中
STATIC INT32 OsFutexWaitTask(const UINT32 *userVaddr, const UINT32 flags, const UINT32 val, const UINT32 timeOut)
{
    INT32 futexRet;
    UINT32 intSave, lockVal;
    LosTaskCB *taskCB = NULL;
    FutexNode *node = NULL;
    UINTPTR futexKey = OsFutexFlagsToKey(userVaddr, flags);//通过地址和flags 找到 key
    UINT32 index = OsFutexKeyToIndex(futexKey, flags);//通过key找到哈希桶
    FutexHash *hashNode = &g_futexHash[index];

    if (OsFutexLock(&hashNode->listLock)) {//操作快锁节点链表前先上互斥锁
        return LOS_EINVAL;
    }
    //userVaddr必须是用户空间虚拟地址
    if (LOS_ArchCopyFromUser(&lockVal, userVaddr, sizeof(UINT32))) {//将值拷贝到内核空间
        PRINT_ERR("Futex wait param check failed! copy from user failed!\n");
        futexRet = LOS_EINVAL;
        goto EXIT_ERR;
    }

    if (lockVal != val) {//对参数内部逻辑检查
        futexRet = LOS_EBADF;
        goto EXIT_ERR;
    }
    //注意第二个参数 FutexNode *node = NULL 
    if (OsFutexInsertTaskToHash(&taskCB, &node, futexKey, flags)) {// node = taskCB->futex
        futexRet = LOS_NOK;
        goto EXIT_ERR;
    }

    SCHEDULER_LOCK(intSave);
    OsTaskWaitSetPendMask(OS_TASK_WAIT_FUTEX, futexKey, timeOut);
    OsSchedTaskWait(&(node->pendList), timeOut, FALSE);
    OsSchedLock();
    LOS_SpinUnlock(&g_taskSpin);

    futexRet = OsFutexUnlock(&hashNode->listLock);
    if (futexRet) {
        OsSchedUnlock();
        LOS_IntRestore(intSave);
        goto EXIT_UNLOCK_ERR;
    }

    LOS_SpinLock(&g_taskSpin);
    OsSchedUnlock();

    /*
    * it will immediately do the scheduling, so there's no need to release the
    * task spinlock. when this task's been rescheduled, it will be holding the spinlock.
    */
    OsSchedResched();

    if (taskCB->taskStatus & OS_TASK_STATUS_TIMEOUT) {
        taskCB->taskStatus &= ~OS_TASK_STATUS_TIMEOUT;
        SCHEDULER_UNLOCK(intSave);
        return OsFutexDeleteTimeoutTaskNode(hashNode, node);
    }

    SCHEDULER_UNLOCK(intSave);
    return LOS_OK;

EXIT_ERR:
    (VOID)OsFutexUnlock(&hashNode->listLock);
EXIT_UNLOCK_ERR:
    return futexRet;
}
~~~

#### Wake Task

- 构建临时futex node，在链表中寻找该结点

~~~c
STATIC INT32 OsFutexWakeTask(UINTPTR futexKey, UINT32 flags, INT32 wakeNumber, FutexNode **newHeadNode, BOOL *wakeAny)
 {
     UINT32 intSave;
     FutexNode *node = NULL;
     FutexNode *headNode = NULL;
     UINT32 index = OsFutexKeyToIndex(futexKey, flags);
     FutexHash *hashNode = &g_futexHash[index];
     FutexNode tempNode = { // 先组成一个临时快锁节点,目的是为了找到哈希桶中是否有这个节点
         .key = futexKey,
         .index = index,
         .pid = (flags & FUTEX_PRIVATE) ? LOS_GetCurrProcessID() : OS_INVALID,
     };

     node = OsFindFutexNode(&tempNode);// 找快锁节点
     if (node == NULL) {
         return LOS_EBADF;
     }

     headNode = node;

     SCHEDULER_LOCK(intSave);
     OsFutexCheckAndWakePendTask(headNode, wakeNumber, hashNode, newHeadNode, wakeAny);// 再找到等这把锁的唤醒指向数量的任务
     if ((*newHeadNode) != NULL) {
         OsFutexReplaceQueueListHeadNode(headNode, *newHeadNode);
         OsFutexDeinitFutexNode(headNode);
     } else if (headNode->index < FUTEX_INDEX_MAX) {
         OsFutexDeleteKeyFromFutexList(headNode);
         OsFutexDeinitFutexNode(headNode);
     }
     SCHEDULER_UNLOCK(intSave);

     return LOS_OK;
 }
~~~