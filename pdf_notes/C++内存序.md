---
title: C++内存序
date: 2024/4/8 22：18
update:
comments: true
description: C++11并发操作的内存序
katex: true
tags: [C++,并发]
categories: Knowledge
---

# C++11并发操作的内存序

## 原子操作的关系

### Synchronized-with

> If thread A stores a value and thread B reads that value, there’s a synchronizes-with relationship between the store in thread A and the load in thread B.

### Happens-before

- 单线程：如果一个操作 A 排列在另一个操作 B 之前，那么这个操作 A happens-before B，一般单线程叫A sequenced-before B。
- 多线程：对于多线程而言, 如果一个线程中的操作A先于另一个线程中的操作B, 那么 A happens-before B(一般多线程间操作叫A inter-thread happens-before B)。

## 六种内存顺序

~~~c++
typedef enum memory_order {
  memory_order_relaxed,  // 无同步或顺序限制，只保证当前操作原子性
  memory_order_consume,  // 标记读操作，依赖于该值的读写不能重排到此操作前
  memory_order_acquire,  // 标记读操作，之后的读写不能重排到此操作前
  memory_order_release,  // 标记写操作，之前的读写不能重排到此操作后
  memory_order_acq_rel,  // 仅标记读改写操作，读操作相当于 acquire，写操作相当于 release
  memory_order_seq_cst   // sequential consistency：顺序一致性，不允许重排，所有原子操作的默认选项
} memory_order;
~~~

### Relaxed ordering

- 标记为 `memory_order_relaxed` 的原子操作不是同步操作，不强制要求并发内存的访问顺序，只保证原子性和修改顺序一致性

### Release-Consume ordering

- 对于标记为` memory_order_consume` 原子变量 x 的读操作 R，当前线程中依赖于 x 的读写不允许重排到 R 之前，其他线程中对依赖于 x 的变量写操作对当前线程可见

- consume语义是一种弱的acquire，**它只对关联变量进行约束**，这个实际编程中基本不用，且在某些情况下会自动进化成acquire语义(比如当用consume语义修饰的load操作在if条件表达式中时)。另外，C++17标准明确说明这个语义还未完善，建议直接使用acquire。

### Release-Acquire ordering

- 在release之前的所有store操作绝不会重排到(不管是编译器对代码的重排还是CPU指令重排)此release对应的操作之后，**也就是说如果release对应的store操作完成了，则C++标准能够保证此release之前的所有store操作肯定已经先完成了**，或者说可被感知了；
- 在acquire之后的所有load操作或者store操作绝对不会重排到此acquire对应的操作之前，也就是说只有**当执行完此acquire对应的load操作之后，才会执行后续的读操作或者写操作**。

~~~c++
#include <atomic>
#include <cassert>
#include <thread>

std::atomic<bool> x = false;
std::atomic<bool> y = false;
std::atomic<int> z = 0;

void write_x_then_y() {
  x.store(true, std::memory_order_relaxed);  // 1 happens-before 2，因为2使用了release
  y.store(true, std::memory_order_release);  // 2 happens-before 3（由于 3 的循环）
}

void read_y_then_x() {
  while (!y.load(std::memory_order_acquire)) {  // 3 happens-before 4，因为3使用了acquire
  }
  if (x.load(std::memory_order_relaxed)) {  // 4
    ++z;
  }
}

int main() {
  std::thread t1(write_x_then_y);
  std::thread t2(read_y_then_x);
  t1.join();
  t2.join();
  assert(z.load() != 0);  // 顺序一定为 1234，z一定不为 0
}
~~~

### Sequentially-consistent ordering

- memory_order_seq_cst 是所有原子操作的默认选项，可以省略不写
- 对于标记为 memory_order_seq_cst 的操作，大概行为就是对每一个变量都进行上面所说的Release-Acquire操作，读操作相当于 memory_order_acquire，写操作相当于 memory_order_release，读改写操作相当于 memory_order_acq_rel，此外还附加一个单独的 total ordering，即**所有线程对同一操作看到的顺序也是相同的。**

