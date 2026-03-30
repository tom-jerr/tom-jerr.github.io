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