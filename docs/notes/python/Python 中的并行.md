# Python 中的并行

## 基于线程的并行

`threading` 模块提供了一种在单个进程内部并发地运行多个 [线程](<https://en.wikipedia.org/wiki/Thread_(computing)>) (从进程分出的更小单位) 的方式。 它允许创建和管理线程，以便能够平行地执行多个任务，并共享内存空间。 线程特别适用于 I/O 密集型的任务，如文件操作或发送网络请求，在此类任务中大部分时间都会消耗于等待外部资源。

### 伪并行

在 CPython 中，由于存在 [全局解释器锁(GIL)](https://docs.python.org/zh-cn/3/glossary.html#term-global-interpreter-lock)，同一时刻只有一个线程可以执行 Python 代码（虽然某些性能导向的库可能会去除此限制）。 如果你想让你的应用更好地利用多核心计算机的计算资源，推荐你使用 [`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") 或 [`concurrent.futures.ProcessPoolExecutor`](https://docs.python.org/zh-cn/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor "concurrent.futures.ProcessPoolExecutor")。 但是，如果你想要同时运行多个 I/O 密集型任务，则多线程仍然是一个合适的模型。

\>[`concurrent.futures.ThreadPoolExecutor`](https://docs.python.org/zh-cn/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor "concurrent.futures.ThreadPoolExecutor") 提供了一个高层级接口用来向后台线程推送任务而不会阻塞调用方线程的执行，同时仍然能够在需要时获取任务的结果。

> [`queue`](https://docs.python.org/zh-cn/3/library/queue.html#module-queue "queue: A synchronized queue class.") 提供了一个线程安全的接口用来在运行中的线程之间交换数据。

> [`asyncio`](https://docs.python.org/zh-cn/3/library/asyncio.html#module-asyncio "asyncio: Asynchronous I/O.") 提供了一个替代方式用来实现任务层级的并发而不要求使用多个操作系统线程。

### 优缺点

- **适用场景**：**I/O 密集型任务**。

  - 比如：爬虫、请求 Web API、读写磁盘文件。

- **优缺点**：

  - ✅ **开销小**：线程启动快，内存占用少。

  - ✅ **通信易**：共享内存，数据交换方便（但要小心线程安全）。

  - ❌ **伪并行**：在纯计算任务中，多线程反而可能因为切换上下文的开销变慢。

## 基于进程的并行

[`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") 包同时提供了本地和远程并发操作，通过使用子进程而非线程有效地绕过了 [全局解释器锁](https://docs.python.org/zh-cn/3/glossary.html#term-global-interpreter-lock)。 因此，[`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") 模块允许程序员充分利用给定机器上的多个处理器。

使用多进程时，**一般使用消息机制实现进程间通信，尽可能避免使用同步原语**，例如锁。

### 在进程之间交换对象

[`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") 支持进程之间的两种通信通道：

**队列**

队列是线程和进程安全的。 任何放入 [`multiprocessing`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#module-multiprocessing "multiprocessing: Process-based parallelism.") 队列的对象都将被序列化。

**管道**

[`Pipe()`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#multiprocessing.Pipe "multiprocessing.Pipe") 函数返回一个由管道连接的连接对象，默认情况下是双工（双向）。例如:

返回的两个连接对象 [`Pipe()`](https://docs.python.org/zh-cn/3/library/multiprocessing.html#multiprocessing.Pipe "multiprocessing.Pipe") 表示管道的两端。每个连接对象都有 `send()` 和 `recv()` 方法（相互之间的）。请注意，如果两个进程（或线程）同时尝试读取或写入管道的 _同一_ 端，则管道中的数据可能会损坏。当然，在不同进程中同时使用管道的不同端的情况下不存在损坏的风险。

`send()` 方法将序列化对象而 `recv()` 将重新创建对象。

### 优缺点

- **适用场景**：**CPU 密集型任务**。

  - 比如：繁重的数学计算（矩阵运算）、图像处理、视频编码、大规模数据清洗。

- **优缺点**：

  - ✅ **真并行**：利用多核优势，不受 GIL 限制。

  - ❌ **开销大**：启动进程慢，占用内存多（每个进程都要复制一份资源）。

  - ❌ **通信难**：进程间内存不共享，通信需要通过特殊的序列化（Pickle）机制，开销较大。
