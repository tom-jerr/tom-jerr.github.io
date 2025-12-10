---

title: 8_Index Concurrency
created: 2024-10-31
tags:

- Database

---

# 8-Index Concurrency

## 8.0 Concurrency control

![](https://github.com/tom-jerr/MyblogImg/raw/15445/concurrency_control.png)

## 8.1 Latch

### 8.1.0 LOCKS VS. LATCHES

![](https://github.com/tom-jerr/MyblogImg/raw/15445/LOCKS_LATCHES.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/LOCKS_LATCHES2.png)

### 8.1.1 Latch Modes

![](https://github.com/tom-jerr/MyblogImg/raw/15445/latch_modes.png)

### 8.1.2 Latch implementations

Test-and-Set Spin Latch (TAS)；比如 std::atomic<T>

> 有效，不能支持大规模并发；对缓存不友好；对 OS 不友好

> CPU 空转，因为 NUMA 之间的通信，会增加硬件通信流量成本

```c++
std::atomic_flag latch;
while(latch.test_and_set(...){
  // Retry? Yield? Abort?
})
```

![](https://github.com/tom-jerr/MyblogImg/raw/15445/TAS.png)

Blocking OS Mutex：阻塞式实现；不能大规模并发；std::mutex

> OS 支持，竞争线程进入内核态类似 sleep

> 涉及到 OS 的系统调用，可能会拖慢整个系统的速度

![](img%5Cmutex.png)

Reader-Writer Latches

> 允许并发读

> 依据 spin latches 实现

![](https://github.com/tom-jerr/MyblogImg/raw/15445/read_write_latch.png)

### 8.2 Hash table latching

- 对一个哈希结构加锁
- 对一个哈希槽进行加锁

Compare-and-swap (CAS)

> 原子操作，对变量地址中的值进行改变

## 8.3 B+ Tree Concurrency Control

### latch crabbing/coupling

![](https://github.com/tom-jerr/MyblogImg/raw/15445/latch_crabbing.png)

> 释放 Latch 时，我们希望尽可能将阻塞更多线程的 latch 释放，所以我们**总是自顶向下释放 latch**

![](https://github.com/tom-jerr/MyblogImg/raw/15445/latch_crabbing2.png)

### Better Latching Algorithm

> 是乐观机制；在 split 和 merge 发生情况较少下效果更好。

> 先采用一路读锁到叶子节点的上一个节点，如果出现 split 或 merge，重新从头开始进行悲观锁加锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/better_latching.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/better_latching2.png)

### multithread 并发冲突

> 或者我们可以按照同一种顺序加锁，永远从左向右加锁

> T2 不知道 T1 发生了什么，确定 T1 的完成时间并等待是不现实的；实现线程间通信也是成本很高的；最佳选择是终止自己的事务并重新开始

![](https://github.com/tom-jerr/MyblogImg/raw/15445/leaf_node_scan.png)

> 这种竞态条件出现较为罕见；而且预防该种竞态条件成本高昂，一般方式采用双方事务都直接终止并重启

![](https://github.com/tom-jerr/MyblogImg/raw/15445/leaf_node_scan2.png)
