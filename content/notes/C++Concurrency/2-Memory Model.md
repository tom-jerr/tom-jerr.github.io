---

title: C++ 内存序
created: 2025-06-10
update:
comments: true
description: C++ 内存模型和基于原子类型的操作
katex: true
tags:

- C++

---

# 内存模型

- `volatile`访问不会建立线程间的同步。
- 此外，`volatile`访问不是原子的（并发读写是一个数据竞争问题），并且不会对内存进行排序（非`volatile`的内存访问可以自由地在`volatile`访问周围重新排序）。

## 协议级别

- 原子操作的顺序一致语义被称为强内存模型，原子操作的自由语义被称为弱内存模型。

![](img/atomic_level.png)

## C++ 的内存序

原子操作默认的内存序是 std::memory_order_seq_cst，顺序一致性

```c++
enum memory_order{
  memory_order_relaxed,
  memory_order_consume,
  memory_order_acquire,
  memory_order_release,
  memory_order_acq_rel,
  memory_order_seq_cst
}
```

- 顺序一致: `memory_order_seq_cst`

- 获取-释放(Acquire-release)：`memory_order_consume` , `memory_order_acquire` ,`memory_order_release`和`memory_order_acq_rel`

- 自由序(Relaxed): `memory_order_relaxed`

## Sequentially-consistent ordering

顺序一致中，一个线程可以看到另一个线程的操作，因此也可以看到所有其他线程的操作。如果使用原子操作的获取-释放语义，那么顺序一致就不成立了。

- 标记为 memory_order_seq_cst 的原子操作不仅像释放/获取顺序那样对内存进行排序（一个线程中在存储操作之前发生的所有事情都成为另一个线程中加载操作的可见副作用），而且建立了一个所有这样标记的原子操作的单一总修改顺序。

## Release-Acquire ordering

- 如果线程 A 中的原子存储操作标记为`memory_order_release`，而线程 B 中对同一变量的原子加载操作标记为`memory_order_acquire`，并且线程 B 中的加载操作读取了线程 A 中存储操作写入的值，那么线程 A 中的存储操作与线程 B 中的加载操作之间就建立了同步关系（synchronizes-with）。
  所有在原子存储操作之前发生（从线程 A 的角度看）的内存写入操作（包括非原子操作和标记为`memory_order_relaxed`的原子操作）都将成为线程 B 中可见的副作用。也就是说，一旦原子加载操作完成，**线程 B 将能够看到线程 A 写入的所有内容**。这种保证仅在 B 实际返回线程 A 存储的值，或者返回释放序列中更晚的值时才成立。

- 这种同步仅在释放和获取同一原子变量的线程之间建立。其他线程可能会看到与同步线程之一或两者都不同的内存访问顺序。。

  > 互斥锁（如 std::mutex 或原子自旋锁）是释放-获取同步的一个例子：当线程 A 释放锁，线程 B 获取锁时，线程 A 在释放锁之前在临界区中发生的所有操作都必须对线程 B 可见（线程 B 在获取锁之后执行相同的临界区）。
  > 同样的原理也适用于线程的启动和汇入。这两种操作都是获取-释放操作。接下来是 wait 和 notify_one 对条件变量的调用；wait 是获取操作，notify_one 是释放操作。那 notify_all 呢？当然，也是一个释放操作。

- 在一个释放序列中，即使 RMW 操作使用了 memory_order_relaxed，它也不会破坏释放序列的同步效果。

  > RMW (Read-Modify-Write) 是一个包含“读取-修改-写入”三步骤的复合操作。

```c++
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

std::vector<int> data;
std::atomic<int> flag = {0};

void thread_1()
{
    data.push_back(42);
    flag.store(1, std::memory_order_release);
}

void thread_2()
{
    int expected = 1;
    // memory_order_relaxed is okay because this is an RMW,
    // and RMWs (with any ordering) following a release form a release sequence
    while (!flag.compare_exchange_strong(expected, 2, std::memory_order_relaxed))
    {
        expected = 1;
    }
}

void thread_3()
{
    while (flag.load(std::memory_order_acquire) < 2)
        ;
    // if we read the value 2 from the atomic flag, we see 42 in the vector
    assert(data.at(0) == 42); // will never fire
}

int main()
{
    std::thread a(thread_1);
    std::thread b(thread_2);
    std::thread c(thread_3);
    a.join(); b.join(); c.join();
}
```

- 错误案例：必须使用原子的 compare_and_exchange 来进行 flag 更改。
  > dataProduced.store(true, std::memory_order_release) 与 dataProduced.load(std::memory_order_acquire)同步。不过，并不意味着获取操作要对释放操作进行等待，而这正是下图中的内容。图中，dataProduced.load(std::memory_order_acquire) 在指令 dataProduced.store(true, std::memory_order_release)之前，所以这里没有同步关系。

![](img/wrong_example.png)

```c++
// acquireReleaseWithoutWaiting.cpp

#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

std::vector<int> mySharedWork;
std::atomic<bool> dataProduced(false);

void dataProducer(){
  mySharedWork = {1,0,3};
  dataProduced.store(true, std::memory_order_release);
}

void dataConsumer(){
     dataProduced.load(std::memory_order_acquire);
  myShraedWork[1] = 2;
}

int main(){

  std::cout << std::endl;

  std::thread t1(dataConsumer);
  std::thread t2(dataProducer);

  t1.join();
  t2.join();

  for (auto v : mySharedWork){
    std::cout << v << " ";
  }

  std::cout << "\n\n";

}
```

- 当 dataProduced.store(true, std::memory_order_release) 先行于 dataProduced.load(std::memory_order_acquire)，那么 dataProduced.store(true, std::memory_order_release) 之前和 dataProduced.load(std::memory_order_acquire) 之后执行的操作是所有线程可见的。

## Relax Ordering

Typical use for relaxed memory ordering is incrementing counters, such as the reference counters of std::shared_ptr, since this only requires atomicity, but not ordering or synchronization (note that decrementing the std::shared_ptr counters requires acquire-release synchronization with the destructor).

```c++
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

std::atomic<int> cnt = {0};

void f()
{
    for (int n = 0; n < 1000; ++n)
        cnt.fetch_add(1, std::memory_order_relaxed);
}

int main()
{
    std::vector<std::thread> v;
    for (int n = 0; n < 10; ++n)
        v.emplace_back(f);
    for (auto& t : v)
        t.join();
    std::cout << "Final counter value is " << cnt << '\n';
}
// always 10000
```

# Atomic

实现原子操作有两种实现方式：

1. lock-free: 这是 std::atomic 的理想实现方式。它不依赖于互斥锁（mutex）或其他操作系统提供的同步原语。相反，它直接利用 CPU 提供的原子指令。
1. lock-based: std::atomic<T> 对象内部会包含一个隐藏的互斥锁（mutex）。对这个 std::atomic 的每一次操作，实际上都会经历以下步骤：

- 加锁: 对内部的互斥锁进行加锁。
- 执行操作: 在锁的保护下，执行普通的数据操作（如读取、修改、写入）。
- 解锁: 释放互斥锁。

实际上 mutex 的实现内部也是使用 CPU 的原子指令，如 pthread_mutex_t，存在 fast-path 和 slow-path

- fast path:

  1. 原子地尝试获取锁: 代码会使用一个 CPU 原子指令（例如 CMPXCHG - Compare-and-Exchange）来尝试将 \_\_lock_status 从 0 (未锁定) 变为 1 (已锁定)
  1. 成功: 如果原子操作成功，意味着在这一瞬间，锁是未被占有的。当前线程现在成功获取了锁。
  1. 失败: 如果原子操作失败，意味着 \_\_lock_status 的值不是 0。这说明锁已经被其他线程持有。此时，我们不能再简单地重试，必须进入慢速路径。

  - 在真正进入内核前，线程会先进行一小段时间的用户态自旋。

- slow path

  1. 再次检查 (Double-Check): 在准备进行系统调用之前，代码可能会再次检查锁的状态。这是一种优化，防止在从快速路径转换到慢速路径的短暂间隙中，锁恰好被释放了。
  1. 调用系统调用进入内核: 如果锁仍然被占用，线程会执行一个系统调用（在 Linux 中，这是通过 futex 系统调用完成的）。pthread_mutex_lock 会调用 futex(FUTEX_WAIT, ...)。
  1. 在内核中等待: 内核会将该线程的状态从“运行中”变为“阻塞”，并将其从调度器的运行队列中移除。这个线程将不再消耗任何 CPU，静静地等待被唤醒。

```c++
// 这是一个非常简化的概念模型，真实结构更复杂
struct pthread_mutex_t {
    // 核心状态字，通常是一个整数。
    // 它的不同值/位代表不同的状态。
    // 0: 未锁定 (Unlocked)
    // 1: 已锁定，无等待者 (Locked, no waiters)
    // 2: 已锁定，有等待者 (Locked, with waiters)
    int __lock_status;

    // 其他字段，如：
    int __kind;      // 锁的类型 (normal, recursive, error-checking)
    int __owner_tid; // 当前持有锁的线程ID (用于递归锁和错误检查)
    int __recursion_count; // 递归计数值
    // 可能还有用于自旋的计数器等
};
```

## futex

实际上主要逻辑分为 wait(do_futex_wait)以及 wake(do_futex_wake)

### do_futex_wait

1. 锁定内核数据结构：获取管理该 uaddr 的内部锁。
1. 验证条件: 再次从用户空间读取 \*uaddr 的值，并与用户传入的 val 比较。这是至关重要的“原子性”保证。如果在用户态检查后、进入内核前，锁的状态改变了，这一步可以发现，并立即返回，避免不必要的休眠。
1. 加入等待队列: 如果条件仍然满足，将当前线程封装成一个等待节点，加入到 uaddr 对应的哈希等待队列中。
1. 休眠: 调用调度器 schedule()，放弃 CPU，进入睡眠状态。
1. 唤醒后清理: 被 FUTEX_WAKE 唤醒后，从等待队列中移除自己，然后返回用户空间。

### do_futex_wake

1. 锁定内核数据结构：找到 uaddr 对应的等待队列。
1. 遍历队列: 遍历等待队列中的线程。
1. 唤醒线程: 对每个要唤醒的线程调用 wake_up_process()。这个函数是内核调度子系统的核心部分，它会改变线程的状态，使其有资格再次被 CPU 调度。
1. 返回: 返回被成功唤醒的线程数量。

## std::atomic_flag

- `std::atomic_flag` 是一个原子的布尔类型，也是唯一保证 lock-free 的原子类型，只能用`ATOMIC_FLAG_INIT` 初始化为 false

### example 1

用 std::atomic_flag 实现自旋锁

```c++
class Spinlock {
 public:
  void lock() {
    while (flag_.test_and_set(std::memory_order_acquire)) {
    }
  }

  void unlock() { flag_.clear(std::memory_order_release); }

 private:
  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};
```

### example 2

用整型原子类型实现 Spinlock

```c++
class Spinlock {
 public:
  void lock() {
    int expected = 0;
    while (!flag_.compare_exchange_weak(expected, 1, std::memory_order_release,
                                        std::memory_order_relaxed)) {
      expected = 0;
    }
  }

  void unlock() { flag_.store(0, std::memory_order_release); }

 private:
  std::atomic<int> flag_ = 0;
};
```

## requirement

- 如果原子类型是自定义类型，该自定义类型必须可平凡复制（trivially copyable），也就意味着该类型不能有虚函数或虚基类。这可以用 is_trivially_copyable 检验。
- 自定义类型的原子类型不允许运算操作，只允许 is_lock_free、load、store、exchange、compare_exchange_weak、compare_exchange_strong，以及赋值操作和向自定义类型转换的操作。
