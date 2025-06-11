---
title: Synchronizing concurrent operation
date: 2025/6/10
update:
comments: true
description: C++ 中的同步并发操作
katex: true
tags:
  - C++
---

# Condition Variable

- 在并发编程中，一种常见的需求是，一个线程等待另一个线程完成某个事件后，再继续执行任务。对于这种情况，标准库提供了 std::condition_variable
- 但是 std::condition_variable 只能与 std::unique_lock 协作，std::condition_variable_any 可以和其他类型的锁来协作。
- 有多个能唤醒的任务时，notify_one() 会随机唤醒一个

  ```c++
  #include <condition_variable>
  #include <iostream>
  #include <mutex>
  #include <thread>

  class A {
  public:
    void step1() {
      {
        std::lock_guard<std::mutex> l(m_);
        step1_done_ = true;
      }
      std::cout << 1;
      cv_.notify_one();
    }

    void step2() {
      std::unique_lock<std::mutex> l(m_);
      cv_.wait(l, [this] { return step1_done_; });
      step2_done_ = true;
      std::cout << 2;
      cv_.notify_one();
    }

    void step3() {
      std::unique_lock<std::mutex> l(m_);
      cv_.wait(l, [this] { return step2_done_; });
      std::cout << 3;
    }

  private:
    std::mutex m_;
    std::condition_variable cv_;
    bool step1_done_ = false;
    bool step2_done_ = false;
  };

  int main() {
    A a;
    std::thread t1(&A::step1, &a);
    std::thread t2(&A::step2, &a);
    std::thread t3(&A::step3, &a);
    t1.join();
    t2.join();
    t3.join();
  }  // maybe: 123, 213, 231, 1-block
  ```

  ```c++
  // condition_variable_any
  #include <condition_variable>
  #include <iostream>
  #include <mutex>
  #include <thread>

  class Mutex {
  public:
    void lock() {}
    void unlock() {}
  };

  class A {
  public:
    void signal() {
      std::cout << 1;
      cv_.notify_one();
    }

    void wait() {
      Mutex m;
      cv_.wait(m);
      std::cout << 2;
    }

  private:
    std::condition_variable_any cv_;
  };

  int main() {
    A a;
    std::thread t1(&A::signal, &a);
    std::thread t2(&A::wait, &a);
    t1.join();
    t2.join();
  }  // 12
  ```

# Semaphore

即操作系统中的 PV 操作，P 操作对信号量减一，如果减到 0 则阻塞；V 操作对信号量加一。

- `std::counting_semaphore`(C++20)模拟信号量，函数传参设置信号量初始值
  - `acquire()`是 P 操作。
  - `release()`是 V 操作。
  - `std::binary_semaphore` 是最大值为 1 的信号量，即`std::counting_semaphore<1>`

# Barrier

阻塞所有的线程执行该指令后面的指令，一般是硬件指令，Linux 提供了系统调用接口。

- `std::barrier`(C++20)它用一个值作为要等待的线程的数量来构造.
  - `std::barrier::arrive_and_wait`：会阻塞至所有线程完成任务，当最后一个线程完成任务时，所有线程被释放，barrier 被重置。
  - 构造 std::barrier 时可以额外设置一个 noexcept 函数，当所有线程到达阻塞点时，由其中一个线程运行该函数。如果想从线程集中移除线程，则在该线程中对 barrier 调用 `std::barrier::arrive_and_drop`。
- `std::latch`(C++20)是一次性屏障，它用一个值作为计数器的初始值来构造。
  - `std::latch::count_down` 将计数器减 1。
  - `std::latch::wait` 将阻塞至计数器为 0，如果想让计数器减一并阻塞至为 0 则可以调用 `std::latch::arrive_and_wait`。

```c++
#include <barrier>
#include <cassert>
#include <iostream>
#include <thread>

class A {
 public:
  void f() {
    std::barrier sync_point{3, [&]() noexcept { ++i_; }};
    for (auto& x : tasks_) {
      x = std::thread([&] {
        std::cout << 1;
        sync_point.arrive_and_wait();
        assert(i_ == 1);
        std::cout << 2;
        sync_point.arrive_and_wait();
        assert(i_ == 2);
        std::cout << 3;
      });
    }
    for (auto& x : tasks_) {
      x.join();  // 析构 barrier 前 join 所有使用了 barrier 的线程
    }  // 析构 barrier 时，线程再调用 barrier 的成员函数是 undefined behavior
  }

 private:
  std::thread tasks_[3] = {};
  int i_ = 0;
};

int main() {
  A a;
  a.f();
}
```
