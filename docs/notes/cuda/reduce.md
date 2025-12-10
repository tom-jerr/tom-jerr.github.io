这种叫 Single Pass Reduction (单次归约)，但实现起来比较复杂，主要有两种思路：

原子加 (Atomic Add)：

所有 Block 算完后，直接 atomicAdd(global_final_sum, block_sum)。

缺点： 原子操作在数千个 Block 同时竞争写入同一个地址时，性能会急剧下降（串行化），通常不如两次 Kernel 启动快。

全局同步锁 (Global Sync / Last Block Optimization)：

利用原子计数器记录已经完成的 Block 数量。最后一个到达终点的 Block 负责把之前所有 Block 的结果加起来。

缺点： 代码非常复杂，容易死锁，且依赖 GPU 具体的调度行为，不如这种“两次启动”的方法稳健和通用。

所以，“两次启动内核”（Recursive Reduction）是工业界最标准、最稳妥的实现方式。