---

title: std::thread & std::jthread
created: 2025-06-10
update:
comments: true
description: C++11 的线程和 C++20 的RAII线程
katex: true
tags:

- C++

---

# std::thread

## thread 传递参数

- 有参数的函数也能传给 std::thread，参数的默认实参会被忽略

  ```c++
  #include <thread>

  void f(int i = 1) {}

  int main() {
    std::thread t{f, 42};  // std::thread t{f} 则会出错，因为默认实参会被忽略
    t.join();
  }
  ```

- 如果 thread 执行的函数参数是一个引用的话，需要`std::ref`

  ```c++
   void update_data_for_widget(widget_id w,widget_data& data); // 1
   void oops_again(widget_id w)  {
    widget_data data;  std::thread t(update_data_for_widget,w,std::ref(data)); // 2
    display_status();
    t.join();
    process_widget_data(data);
   }
  ```

- 这依赖于 std::thread 实例的可移动且不可复制性。不可复制性表示在某一时间点,一个 std::thread 实例只能关联一个执行线程。可移动性使得开发者可以自己决定,哪个实例拥有线程实际执行的所有权

- `std::thread::hardware_concurrency()` 会返回并发线程的数量。例如,多核系统中, 返回值可以是 CPU 核芯的数量。返回值也仅仅是一个标识,当无法获取时,函数返回 0。

## 并发 accumlate 的例子

```c++
// parallel accumulate
template<typename Iterator,typename T>
struct accumulate_block  {
  void operator()(Iterator first,Iterator last,T& result)  {  result=std::accumulate(first,last,result);
  }
};
template<typename Iterator,typename T>
T parallel_accumulate(Iterator first,Iterator last,T init)
{
  unsigned long const length=std::distance(first,last);
  if(!length) // 1
      return init;
  unsigned long const min_per_thread=25;
  unsigned long const max_threads=  (length+min_per_thread-1)/min_per_thread; // 2
  unsigned long const hardware_threads=  std::thread::hardware_concurrency();  unsigned long const num_threads= // 3
  std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);  unsigned long const block_size=length/num_threads; // 4
  std::vector<T> results(num_threads);
  std::vector<std::thread> threads(num_threads-1); // 5
  Iterator block_start=first;
  for(unsigned long i=0; i < (num_threads-1); ++i)
  {
      Iterator block_end=block_start;
      std::advance(block_end,block_size); // 6
      threads[i]=std::thread( // 7
          accumulate_block<Iterator,T>(),
          block_start,block_end,std::ref(results[i]));
      block_start=block_end; // 8
  }
  accumulate_block<Iterator,T>()(  block_start,last,results[num_threads-1]); // 9
  for (auto& entry : threads)  entry.join(); // 10
  return std::accumulate(results.begin(),results.end(),init); // 11
}
```

# jthread

- C++11 引入了 std::thread 类型,其与操作系统提供的线程对应,但该类型有一个严重的设计缺陷: 不是 RAII 类型。

## thread 存在的问题

std::thread 要求在其生命周期结束时,若表示正在运行的线程,则调用 join()(等待线程结束) 或 detach()(让线程在后台运行)。若两者都没有调用,析构函数会立即导致异常的程序终止(std::terminate)。

如果通过确保在离开作用域时调用 join() 来对异常作出反应,而不解决异常。不幸的是,这可能会导致阻塞 (永远)。然而,调用 detach() 也是一个问题,因为线程在程序的后台继续运行,使用 CPU 时间和资源,而这些时间和资源现在可能会销毁。若在更复杂的上下文中使用多个线程,问题会变得更糟,并且会产生非常糟糕的代码。

```c++
void foo(){
  std::thread t1{task1, name, val};
  std::thread t2;
  try {
    t2 = std::thread{task2, name, val};
    ...
  }
  catch(...){
    t1.join();
    if(t2.joinable()) {
      t2.join();
    }
    throw;
  }

  t1.join();
  t2.join();
}

```

## std::jthread

- std::jthread 解决了这些问题,它是 RAII 类型。若线程是可汇入的 (“j”代表“汇入”),析构函数会自动调用 join()。
- 内置停止机制：std::jthread 与 std::stop_token 集成，支持直接请求停止线程

```c++
void foo(){
  std::jthread t1{task1, name, val};
  std::jthread t2{task2, name, val};
  ...

  t1.join();
  t2.join();
}
```

## jthread 的停止令牌和停止回调

- 自动管理停止令牌：当使用 std::jthread 时，不需要手动创建 std::stop_source。std::jthread 自动包含一个内部的 std::stop_source，并在启动线程时将相关的 std::stop_token 传递给线程函数。
- 接收停止令牌：线程函数可以直接接受一个 std::stop_token 参数，该令牌由 std::jthread 提供，确保与线程的内部停止机制同步。
- 定期检查停止请求：在线程函数中，应定期调用 std::stop_token::stop_requested() 来检查是否接收到停止请求。这为安全且及时地停止执行提供了机制。
- 响应停止请求：一旦 std::stop_token 表明停止已被请求，线程函数应采取必要的步骤来安全地终止，这可能包括资源的清理和状态的保存。

```c++
#include <iostream>
#include <chrono>
#include <thread>

// 使用 std::jthread 运行的函数
void task(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        std::cout << "任务正在运行..." << std::endl;
        // 模拟一些工作
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "任务已收到停止请求，现在停止运行。" << std::endl;
}

int main() {
    // 创建 std::jthread，自动处理停止令牌
    std::jthread worker(task);

    // 模拟主线程运行一段时间后需要停止子线程
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "主线程请求停止子线程..." << std::endl;

    // 触发停止请求
    worker.request_stop();

    // std::jthread 在析构时自动加入
    return 0;
}
```

## std::stop_token 和 std::stop_callback 的其他使用案例;

另外, std::stop_token 和 std::stop_callback 并不局限于与线程（如 std::jthread）的使用,它们独立于线程的，用于程序中的任何地方，以提供一种灵活的停止信号处理机制。

```c++
#include <iostream>
#include <chrono>
#include <stop_token>

int main() {
    std::stop_source source;
    std::stop_token token = source.get_token();

    // 模拟一些可以被取消的工作
    auto startTime = std::chrono::steady_clock::now();
    auto endTime = startTime + std::chrono::seconds(10);  // 设定10秒后结束任务

    while (std::chrono::steady_clock::now() < endTime) {
        if (token.stop_requested()) {
            std::cout << "Task was canceled!" << std::endl;
            break;
        }
        std::cout << "Working..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // 模拟在某个条件下请求停止
        if (std::chrono::steady_clock::now() > startTime + std::chrono::seconds(5)) {
            source.request_stop();
        }
    }

    if (!token.stop_requested()) {
        std::cout << "Task completed normally." << std::endl;
    }

    return 0;
}

// result
Working...
Working...
Working...
Working...
Working...
Task was canceled!
```

- stop_token 与 std::thread 结合

```c++
#include <iostream>
#include <thread>
#include <stop_token>
#include <chrono>

void threadFunction(std::stop_token stoken) {
    std::stop_callback callback(stoken, []() {
        std::cout << "Stop request received.\n";
    });
   // 4. 定期检查停止请求
    while (!stoken.stop_requested()) {
        std::cout << "Running...\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
   // 5. 响应取消请求
    std::cout << "Thread finishing.\n";
}

int main() {
     // 1. 创建并发起取消请求的源
    std::stop_source stopSource;
     // 2. 生成停止令牌
    std::stop_token stoken = stopSource.get_token();
    // 3. 传递停止令牌
    std::thread t(threadFunction, stoken);

    std::this_thread::sleep_for(std::chrono::seconds(5));
     // 触发停止请求
    stopSource.request_stop();

    t.join();
    std::cout << "Thread stopped.\n";
    return 0;
}
```
