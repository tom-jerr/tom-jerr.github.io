---

title: The implement of Concurrent Component
created: 2025-06-11
update:
comments: true
description: 并发组件的内部实现浅析
katex: true
tags:

- C++

---

# Mutex

## pthread_mutex_t

一个 pthread_mutex_t 变量不仅仅是一个整数。它是一个结构体，包含了锁的状态、类型、持有者等信息。其具体结构会因架构和 glibc 版本而异，但其核心部分（在 x86_64 上）通常如下：

```c++
typedef union {
  struct {
    int __lock;             // 核心的 futex word
    unsigned int __count;   // 递归锁的计数器
    int __owner;            // 持有该锁的线程TID
    unsigned int __nusers;  // 等待者的数量
    int __kind;             // 锁的类型 (normal, recursive, error-checking)
    // ...
  } __data;
} pthread_mutex_t;
// __lock:
// 0: Unlocked
// 1: Locked, no waiters
// 2: Locked, with waiters
```

而 pthread_mutex_lock 和 pthread_mutex_unlock 的实现是基于原子指令和 futex 系统调用实现的。

### pthread_mutex_lock

1. 首先在用户态使用 CAS（Compare-And-Swap）尝试将 `__lock` 从 0（unlocked）变为 1（locked）。如果成功，直接返回。
1. 如果 CAS 失败，说明锁已经被其他线程持有，此时会进入自旋状态，尝试多次获取锁。
1. 如果自旋多次仍然失败，说明竞争激烈，将`__lock`变为 2，此时会调用 futex 系统调用，将当前线程挂起，等待锁被释放。
1. 其他线程释放锁后，该线程被唤醒，循环回到自旋状态，重新尝试获取锁。

```c++
if (atomic_compare_and_swap(&mutex->__lock, 0, 1) == 0) {
    mutex->__owner = self_tid;
    return 0;
}
for (int i = 0; i < SPIN_COUNT; ++i) {
    if (mutex->__lock == 0) {
        if (atomic_compare_and_swap(&mutex->__lock, 0, 1) == 0) {
            mutex->__owner = self_tid;
            return 0;
        }
    }
    // CPU "pause" 指令，减少功耗并提示CPU这是在自旋
    _mm_pause();
}
int old_lock = atomic_exchange(&mutex->__lock, 2);
if (old_lock != 0) {
    // 调用 futex 系统调用，让内核将当前线程睡眠。
    // 等待在 __lock 上，期望它的值是 2。
    syscall(SYS_futex, &mutex->__lock, FUTEX_WAIT, 2, ...);
}
```

### pthread_mutex_unlock

1. 首先原子地将 `__lock` 的值减 1，并返回之前的值。
1. 如果之前的值是 1，说明锁被成功释放，没有等待者，此时将 `__lock` 设为 0。
1. 如果之前的值是 2，说明有线程在等待，此时调用 futex_wake 系统调用，唤醒一个等待的线程。

```c++
int old_lock = atomic_fetch_sub(&mutex->__lock, 1);
if (old_lock != 1) {
    syscall(SYS_futex, &mutex->__lock, FUTEX_WAKE, 1, ...);
}
```

### futex

实际上主要逻辑分为 wait(do_futex_wait)以及 wake(do_futex_wake)

#### do_futex_wait

1. 锁定内核数据结构：获取管理该 uaddr 的内部锁。
1. 验证条件: 再次从用户空间读取 \*uaddr 的值，并与用户传入的 val 比较。这是至关重要的“原子性”保证。如果在用户态检查后、进入内核前，锁的状态改变了，这一步可以发现，并立即返回，避免不必要的休眠。
1. 加入等待队列: 如果条件仍然满足，将当前线程封装成一个等待节点，加入到 uaddr 对应的哈希等待队列中。
1. 休眠: 调用调度器 schedule()，放弃 CPU，进入睡眠状态。
1. 唤醒后清理: 被 FUTEX_WAKE 唤醒后，从等待队列中移除自己，然后返回用户空间。

#### do_futex_wake

1. 锁定内核数据结构：找到 uaddr 对应的等待队列。
1. 遍历队列: 遍历等待队列中的线程。
1. 唤醒线程: 对每个要唤醒的线程调用 wake_up_process()。这个函数是内核调度子系统的核心部分，它会改变线程的状态，使其有资格再次被 CPU 调度。
1. 返回: 返回被成功唤醒的线程数量。

# Atomic Variable

编译器在编译时，会将 C++ 源代码中的原子操作（如 load, store, fetch_add, compare_exchange_strong）映射到目标硬件平台提供的最优指令。我们可以用 `my_atomic.is_lock_free()` 方法来查询一个原子变量在当前平台上是否是无锁的。

- `std::atomic_flag` 是一个原子的布尔类型，也是唯一保证 lock-free 的原子类型，只能用`ATOMIC_FLAG_INIT` 初始化为 false

- 有原生原子指令的硬件 (主流): 在现代 CPU 架构（如 x86/x86_64, ARMv7+, AArch64）上，绝大多数原子操作都有对应的单条 CPU 指令。

- 没有原生原子指令的硬件 (非主流): 在一些较老或特殊的架构上，可能没有原生的原子指令。这时，C++ 标准库会使用自旋锁或其他机制来模拟原子操作。

  ```c++
  // 模拟 atomic<T>::fetch_add
  T fetch_add(T arg) {
      internal_global_lock.lock(); // 获取全局锁
      T old_value = this->value;
      this->value += arg;
      internal_global_lock.unlock(); // 释放锁
      return old_value;
  }
  ```

# Condition Variable

## pthread_cond_t

```c++
struct pthread_cond_t {
    // futex word 用于 condvar 的等待和唤醒
    int cond_futex;
    pthread_mutex_t *mutex;
    unsigned int total_seq;    // 总的 signal/broadcast 次数
    unsigned int wakeup_seq;   // 已被唤醒处理的次数
    // 其他字段，如等待者数量、时钟ID等...
};
```

## pthread_cond_wait

1. 读取当前的 `total_seq`。这是我们进入等待前的“版本快照”。我们将用这个值来检查在我们解锁后，是否有 signal 发生。
1. 解锁传入的互斥体。这是 `wait` 语义的关键部分。从这一刻起，其他线程可以获取锁并 signal。
1. 在我们解锁 mutex 和调用 futex_wait 之间，可能已经有 signal 发生。`wakeup_seq` 记录了被唤醒的线程应该“消费”到的序列号。如果 `wakeup_seq` 已经赶上或超过了我们记录的 `current_total_seq`，说明我们应该被唤醒的那个 signal 已经发生了。直接跳到 Fast Path
1. 进入内核等待 (Slow Path)。只有在确定没有错过信号的情况下，我们才去睡眠。我们告诉内核在 `cond->cond_futex` 上等待，并且只有当 `cond->cond_futex` 的值等于 `current_total_seq` 时才睡眠。
   - 这是一个简化的模型，核心是：如果在我们检查后，但在睡眠前，`signal` 发生并改变了状态，futex_wait 会立即返回。
1. 无论我们是从 `futex_wait` 被唤醒，还是因为错过了信号而直接跳到 Fast Path，我们都需要更新 `wakeup_seq`，表示我们已经“消费”了这个唤醒。
   - 我们尝试将 wakeup_seq 从我们之前看到的 `current_total_seq` 更新为 signal 发生后的 `current_total_seq + 1`。
   - 这通常通过一个 CAS 循环完成，以处理多个线程被 broadcast 唤醒的情况。只有一个线程会成功地将 wakeup_seq 加一。
1. 重新获取互斥锁。这是 `wait` 语义的另一半。函数返回前，必须重新持有锁。

```c++
int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex) {
    unsigned int current_total_seq = cond->total_seq.load(memory_order_relaxed);
    pthread_mutex_unlock(mutex);
    if (cond->wakeup_seq.load(memory_order_acquire) == current_total_seq) {
        syscall(SYS_futex, &cond->cond_futex, FUTEX_WAIT, some_expected_value, ...);
    }
    unsigned int old_wakeup_seq = cond->wakeup_seq.load(memory_order_relaxed);
    if (old_wakeup_seq == current_total_seq) {
        cond->wakeup_seq.compare_exchange_strong(old_wakeup_seq, current_total_seq + 1, ...);
    }
    pthread_mutex_lock(mutex);
    return 0; // Success
}
```

## pthread_cond_signal

1. 原子地增加总序列号计数器，这留下了 "signal" 发生过的痕迹。
1. 检查是否有潜在的等待者。如果 `wakeup_seq` 追上了 `total_seq` (增加前的值)，说明所有之前的 signal 都已经被 wait 线程“消费”了，很可能当前没有线程在等待。这是一个优化，避免不必要的系统调用。
1. 唤醒一个线程。我们告诉内核去 `cond->cond_futex` 这个地址上，唤醒最多 1 个线程。这个 `futex_wake` 调用是“发射后不管”的。即使没有线程在等待，它也只是一个空操作。

```c++
int pthread_cond_signal(pthread_cond_t* cond) {
    unsigned int old_total_seq = cond->total_seq.fetch_add(1, memory_order_acq_rel);
    if (cond->wakeup_seq.load(memory_order_relaxed) == old_total_seq) {
        syscall(SYS_futex, &cond->cond_futex, FUTEX_WAKE, 1, ...);
    }
    return 0; // Success
}
```
