---

title: Thread Shared Data Protection
created: 2025-06-10
update:
comments: true
description: C++ 中线程共享数据的保护方式
katex: true
tags:

- C++

---

# Mutex

- 使用 mutex 在访问共享数据前加锁，访问结束后解锁。一个线程用特定的 mutex 锁定后，其他线程必须等待该线程的 mutex 解锁才能访问共享数据
- `std::lock_guard` 可以自动处理 std::mutex 的加锁与解锁，它在构造时接受一个 mutex，并会调用 mutex.lock()，析构时会调用 mutex.unlock()
- `std::scoped_lock` 它可以接受任意数量的 mutex，并将这些 mutex 传给 std::lock 来同时上锁，它会对其中一个 mutex 调用 lock()，对其他调用 try_lock()，若 try_lock() 返回 false，则对已经上锁的 mutex 调用 unlock()，然后重新进行下一轮上锁，标准未规定下一轮的上锁顺序，可能不一致，重复此过程直到所有 mutex 上锁，从而达到同时上锁的效果。
- `std::unique_lock` 在构造时接受一个 mutex，并会调用 mutex.lock()，析构时会调用 mutex.unlock()，unique_lock 允许我们在中间过程去解锁并重新上锁。

# Reader-Writer Mutex

- 读锁：如果多个线程调用 shared_mutex.lock_shared()，多个线程可以同时读。
- 写锁：一个写线程调用 shared_mutex.lock()，则读线程均会等待该写线程释放锁后才上锁。

# Recursive Mutex

- 它可以在一个线程上多次获取锁，但在其他线程获取锁之前必须释放所有的锁。
- 多数情况下，如果需要递归锁，说明代码设计存在问题。比如一个类的每个成员函数都会上锁，一个成员函数调用另一个成员函数，就可能多次上锁，这种情况用递归锁就可以避免产生未定义行为。但显然这个设计本身是有问题的，更好的办法是提取其中一个函数作为 private 成员并且不上锁，其他成员先上锁再调用该函数

# Initialize Protect

- std::once_flag 和 std::call_once 来保证对某个操作只执行一次

```c++
#include <iostream>
#include <mutex>
#include <thread>

class A {
public:
  void f() {
    std::call_once(flag_, &A::print, this);
    std::cout << 2;
  }

private:
  void print() { std::cout << 1; }

private:
  std::once_flag flag_;
};

int main() {
  A a;
  std::thread t1{&A::f, &a};
  std::thread t2{&A::f, &a};
  t1.join();
  t2.join();
}  // 122
```

- static 局部变量在声明后就完成了初始化，这存在潜在的 race condition，如果多线程的控制流同时到达 static 局部变量的声明处，即使变量已在一个线程中初始化，其他线程并不知晓，仍会对其尝试初始化。为此，C++11 规定，如果 static 局部变量正在初始化，线程到达此处时，将等待其完成，从而避免了 race condition。只有一个全局实例时，可以直接用 static 而不需要 `std::call_once`

  ```c++
  template <typename T>
  class Singleton {
  public:
    static T& Instance();
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

  private:
    Singleton() = default;
    ~Singleton() = default;
  };

  template <typename T>
  T& Singleton<T>::Instance() {
    static T instance;
    return instance;
  }
  ```
