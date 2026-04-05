# CUDA Basic Concept
## CUDA Async Copy & Pinned Memory
通常情况下，CPU 分配的内存是 Pageable（可分页） 的。操作系统为了节省物理内存，可能会把这些内存交换到磁盘上。

当你要把数据从 Pageable 内存拷贝到 GPU 时，CUDA 驱动其实做了一个幕后动作：

- 先在系统中申请一块临时存储区（即 Pinned Memory）。

- 将数据从 Pageable 内存拷贝到这块 Pinned Memory。

- 通过 DMA（直接内存访问） 将数据从 Pinned Memory 传给 GPU。

> 实际上多了一次 CPU 侧拷贝，所以性能会受到影响。

Pinned memory = page-locked memory，即页锁定内存。
- 正常的 CPU 内存（pageable memory）是可以被操作系统换页的。
- 通过 `cudaHostMalloc()` 分配的内存是 pinned memory，操作系统不会换页，可以直接被 GPU 访问。

```cpp
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```
它和 cudaMemcpy 的区别是：

- cudaMemcpy：通常是同步的，调用方会被阻塞，直到拷贝完成或至少到达某个同步点

- cudaMemcpyAsync：把拷贝操作提交到某个 stream，理论上 CPU 可以继续往下跑，GPU 也可以和别的工作重叠

- 对 CPU 异步
  - 调用 cudaMemcpyAsync 后，CPU 线程通常马上返回，不必一直等传输结束。

- 对 GPU 异步
  - 如果放在不同 stream，并且硬件支持 copy engine，那么数据拷贝和计算可以 overlap：
    - stream0: H2D 拷贝下一批
    - stream1: 执行当前批 kernel
# GPU 异步拷贝
## Pinned Memory
Pinned memory 就是 页锁定内存（page-locked memory），是主机内存里一块不会被操作系统随意换页的区域。

如果 host 端是普通 pageable memory：

- CUDA 很可能先偷偷拷到一块内部 staging pinned buffer
- 再 DMA 这样会多一次 copy
- 甚至可能退化成近似同步行为

而 pinned memory 的好处是：

- GPU/网卡可以直接 DMA 访问
- H2D / D2H 拷贝更快
- 更重要的是 支持真正的异步拷贝

## 异步拷贝
发起一次数据搬运后，CPU 不必堵塞等待，GPU/拷贝引擎可以在后台执行，和别的计算重叠。

比如 CUDA 里的：
```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
```
它表示：

- 把这个 copy 操作插入某个 stream，调用返回时，拷贝可能还没完成
- 后续可以继续发别的 kernel / copy
- 最后通过 event / stream synchronize 判断何时完成

## 注意事项
- 实际使用时会申请一大块内存作为 pinned memory pool 来避免频繁申请释放的开销
  - 避免频繁 cudaHostAlloc/cudaFreeHost 带来的高开销和抖动
- 使用时先从 pool 取一个 pinned buffer，再通过 cudaMemcpyAsync 发起异步 H2D/D2H 拷贝，并用 cudaEventRecord 标记完成时刻。CPU 侧等 event 完成后再消费数据，最后把 buffer 放回池中。
## Kernel Launch
- 一个 kernel launch 到 GPU 上发生了什么
  1. CPU 发起 kernel launch。
  2. GPU 收到 grid/block 配置。
  3. grid 中的 block 被调度到各个 SM。
  4. 每个 block 在某个 SM 上驻留并拆成多个 warp。
  5. SM 内的 warp scheduler 从 ready warp 中发指令。
  6. 某个 warp 因访存或依赖 stall 时，SM 切换到其他 ready warp。
- GPU 主要靠大量 ready warps 的快速切换来隐藏延迟，而不是像 CPU 那样主要靠大缓存、强分支预测和复杂乱序执行。