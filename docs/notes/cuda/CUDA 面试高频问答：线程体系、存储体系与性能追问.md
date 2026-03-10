---
title: CUDA 面试高频问答：线程体系、存储体系与性能追问
created: 2026-03-07
updated: 2026-03-07
tags:
  - CUDA
  - Interview
description: 基于仓库现有 CUDA 笔记、NVIDIA 官方文档、Nsight Compute 与 CUTLASS 资料整理的 CUDA 面试高频问答，覆盖线程体系、存储体系、occupancy、coalescing、bank conflict、spill 和 GEMM 映射。
---

# CUDA 面试高频问答：线程体系、存储体系与性能追问

## 使用说明

这篇文档不是 CUDA 教科书式笔记，而是按面试场景整理的问答题库。组织原则有两个：

- **答案以官方资料为准。** 技术结论优先采用 NVIDIA CUDA C++ Programming Guide、CUDA Best Practices、Nsight Compute 文档和 CUTLASS 文档。
- **题目频率参考公开面经。** 我只把牛客等公开面经当作“高频追问信号”，不把它们当作技术正确性的依据。

> [!NOTE]
> 结合公开面经和官方文档，CUDA 面试最容易被追问的点基本固定：`warp divergence`、`bank conflict`、`occupancy`、`coalescing`、`spill`、`block/SM 关系`、`GEMM 中线程与存储怎么映射`。

## 仓库内复习路径

| 主题 | 优先回看文件 | 用途 |
| --- | --- | --- |
| CUDA 线程体系、存储体系、SM、Warp、Occupancy | `docs/notes/cuda/cudaopt.md` | 线程层级、硬件结构、Warp 调度、基本优化概念 |
| 带宽利用、Latency Hiding、ILP/DLP、异步加载 | `docs/notes/cuda/GPU 内存系统演进：最大化带宽利用与延迟隐藏的技术路径.md` | 为什么 GPU 靠并发隐藏延迟，以及如何把访存打满 |
| GEMM 的 threadblock/warp/thread 分层、bank conflict、寄存器与 shared memory 权衡 | `docs/notes/cuda/从GEMM实践CUDA优化.md` | 面试里最常见的 GEMM 实战题 |
| 向量化访存、Roofline、memory-bound kernel 基础 | `docs/notes/cuda/Vector Add Optimization Example.md` | coalescing、向量化加载、带宽瓶颈 |

## 高频问题与标准答案

## 一、线程体系

### 1. CUDA 的线程体系是什么？Grid、Block、Thread、Warp 分别是什么？

**答案：**

CUDA 编程模型里，程序员显式组织的是 `grid -> block -> thread`，硬件真正调度执行的基本单位是 `warp`。在 NVIDIA GPU 上，一个 warp 通常是 32 个线程。block 是最关键的桥梁，因为：

- block 内线程可以共享 shared memory。
- block 内线程可以通过 `__syncthreads()` 同步。
- block 是被调度到某个 SM 上执行的基本合作单元。

所以面试里一句最稳妥的话是：

> CUDA 对外暴露的是 thread/block/grid 的分层编程模型，但底层真正执行和调度的基本粒度是 warp。

**常见追问：** `warp 是不是编程模型的一部分？`

答法：不是程序员显式 launch 的层级，但它决定了分支、访存、吞吐和调优方式。

**参考资料：**

- CUDA C++ Programming Guide, Thread Hierarchy: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 2. 一个 kernel launch 之后，GPU 侧大致发生了什么？

**答案：**

可以按这条链回答：

1. CPU 发起 kernel launch。
2. GPU 收到 grid/block 配置。
3. grid 中的 block 被调度到各个 SM。
4. 每个 block 在某个 SM 上驻留并拆成多个 warp。
5. SM 内的 warp scheduler 从 ready warp 中发指令。
6. 某个 warp 因访存或依赖 stall 时，SM 切换到其他 ready warp。

这个问题真正要答出的核心是：GPU 主要靠**大量 ready warps 的快速切换**来隐藏延迟，而不是像 CPU 那样主要靠大缓存、强分支预测和复杂乱序执行。

**参考资料：**

- CUDA C++ Programming Guide: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>

**对应本地文件：**

- `docs/notes/cuda/cudaopt.md`
- `docs/notes/cuda/GPU 内存系统演进：最大化带宽利用与延迟隐藏的技术路径.md`

### 3. 什么是 SM？一个 block 会不会跨多个 SM？

**答案：**

SM（Streaming Multiprocessor）可以理解为 GPU 的基本计算核心簇，内部包含 warp scheduler、CUDA cores、寄存器文件、shared memory/L1、load-store 单元等资源。

一个 block **不会跨多个 SM**。原因很直接：

- block 内线程要共享 shared memory。
- block 内线程可能调用 `__syncthreads()`。

因此 block 必须完整地驻留在同一个 SM 上执行，而且一旦开始运行就不会迁移到别的 SM。反过来，一个 SM 可以同时驻留多个 block，前提是寄存器、shared memory、线程数等资源够用。

**参考资料：**

- CUDA C++ Programming Guide, Thread Hierarchy / Thread Blocks: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy>
- CUDA C++ Programming Guide, Hardware Multithreading: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 4. 什么是 SIMT？它和 SIMD 有什么区别？

**答案：**

SIMD 更像“一条向量指令操作多个数据”；SIMT 则是“多个线程看起来像独立编程，但硬件按 warp 成组执行相同指令流”。

对程序员来说，CUDA 线程有自己的寄存器、局部变量、控制流；但在硬件上，同一个 warp 内线程通常是锁步推进的。因此 CUDA 的编程抽象更像线程，而底层执行效率又高度受 warp 级行为影响。

**参考资料：**

- CUDA C++ Programming Guide, SIMT Architecture: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 5. occupancy 是什么？为什么它不是越高越好？

**答案：**

occupancy 指的是：一个 SM 上实际活跃 warps 数，占该 SM 最大可支持活跃 warps 数的比例。

它高通常意味着更容易在某个 warp stall 时切到别的 warp，从而隐藏延迟；但它**绝不是越高越好**。为了强行追高 occupancy，你可能会：

- 降低每线程寄存器预算。
- 缩小 tile。
- 降低 shared memory 复用。
- 甚至引入 spill。

这些都可能让实际性能更差。真正该看的不是 occupancy 单一数值，而是：`当前 kernel 是不是 latency-bound`，以及 `active warps 是否足以覆盖 stall`。

**高分答法：**

> occupancy 是潜在并发度指标，不直接等于性能。它只说明“能挂多少 warp”，不说明“这些 warp 有没有高效工作”。

**参考资料：**

- CUDA C++ Best Practices Guide, Occupancy: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy>
- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>

**对应本地文件：**

- `docs/notes/cuda/cudaopt.md`
- `docs/notes/cuda/GPU 内存系统演进：最大化带宽利用与延迟隐藏的技术路径.md`

### 6. 什么决定 occupancy？

**答案：**

本质上看四类约束：

- 每个 block 的线程数。
- 每线程寄存器使用量。
- 每个 block 的 shared memory 使用量。
- 架构硬件上限：最大线程数、最大 warp 数、最大 block 数。

面试里别死背公式，讲清楚“先算每种资源各自允许驻留多少 block，再取最小值”就够了。

**参考资料：**

- CUDA Occupancy Calculator docs: <https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html>
- CUDA Runtime API Occupancy functions: <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 7. warp scheduler 是怎么做 latency hiding 的？

**答案：**

每个 SM 内通常有一个或多个 warp scheduler。它们从 ready warp 中挑选 warp 发射指令。如果当前 warp 正在等待 global memory、等待依赖结果、等待 barrier，那么 scheduler 就去执行别的 ready warp。

所以 GPU 隐藏延迟的主要机制不是“单线程做得更聪明”，而是“同一个 SM 上同时挂足够多的 ready warp”。

**参考资料：**

- CUDA C++ Programming Guide, Hardware Multithreading: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading>
- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 8. warp divergence 为什么会慢？

**答案：**

warp divergence 的本质是：同一个 warp 内线程走了不同控制流路径。由于 SIMT 模式下 warp 倾向于执行同一条指令流，所以硬件通常需要把不同分支路径串行跑一遍，再用 mask 控制哪些线程生效。这就让一部分线程在某条路径上空转，吞吐下降。

不只是 `if/else` 会触发 divergence，`switch`、循环退出条件不同、提前 return，都会导致同一 warp 内控制流分裂。

**优化思路：**

- 让相邻线程处理相似数据。
- 让 `threadIdx.x` 对应最规则的数据维度。
- 用 predication 或条件赋值替代很短的小分支。
- 对数据做分桶/重排，降低同 warp 内的不一致性。

**参考资料：**

- CUDA C++ Programming Guide, SIMT Architecture / Control Flow: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#simt-architecture>
- CUDA C++ Best Practices Guide, Branching and Divergence: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#branching-and-divergence>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 9. `__syncthreads()` 有什么作用？为什么 block 之间不能直接同步？

**答案：**

`__syncthreads()` 是 block 级 barrier。它保证：

- block 内所有线程都到达之后，才能继续执行。
- 对 shared memory 的写入在 barrier 之后对 block 内其他线程可见。

block 间默认不能直接同步，是因为 block 是独立调度的：

- 不保证同时运行。
- 不保证顺序。
- 有的 block 甚至可能在别的 block 结束后才启动。

如果允许任意 inter-block barrier，很容易死锁。默认的 grid 级同步通常依赖 kernel launch 边界；更特殊的场景可用 cooperative groups，但有硬件和 launch 约束。

**参考资料：**

- CUDA C++ Programming Guide, Synchronization Functions: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions>
- CUDA C++ Programming Guide, Cooperative Groups: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

## 二、存储体系

### 10. CUDA 的存储体系怎么分层？

**答案：**

最稳妥的面试回答是分成三层：

- **线程私有：** register、local memory
- **block 共享：** shared memory
- **全局可见：** global memory、constant memory、texture memory，以及硬件缓存 L1/L2

性能上通常是 `register > shared memory > global memory`，但要补一句：这只是经验顺序，真正性能还取决于访问模式、冲突和是否复用。

**参考资料：**

- CUDA C++ Programming Guide, Memory Hierarchy: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 11. register 和 local memory 有什么关系？为什么 local memory 不快？

**答案：**

register 是线程私有且最快的存储。普通局部变量如果能放进寄存器，编译器通常会优先放寄存器。

local memory 名字虽然叫 local，但它只是**逻辑上线程私有**，物理上通常走 global memory 路径，只是可以被缓存。因此它的访问代价接近 global memory，而不是 shared/register 这种片上高速存储。

典型触发 local memory 的场景：

- 局部数组太大。
- 编译器寄存器不够，发生 spill。
- 某些变量无法静态分配到寄存器。

**参考资料：**

- CUDA C++ Programming Guide, Local Memory: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#local-memory>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 12. shared memory 为什么快？什么时候它反而不划算？

**答案：**

shared memory 是 SM 片上的、block 内共享的高速存储。它快的根源是距离计算单元更近、延迟更低、带宽更高，而且访问模式由程序员显式控制。

它最适合做：

- tile 缓存
- block 内归约
- 转置
- producer-consumer 临时缓冲
- 减少对 global memory 的重复访问

但 shared memory 并不是永远值得用。如果数据只读一次就丢、没有复用，还要多做一次 global->shared 搬运，再加 `__syncthreads()` 和 bank conflict 风险，反而可能不划算。

**参考资料：**

- CUDA C++ Programming Guide, Shared Memory: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory>
- CUDA C++ Best Practices Guide, Shared Memory: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks>

**对应本地文件：**

- `docs/notes/cuda/cudaopt.md`
- `docs/notes/cuda/从GEMM实践CUDA优化.md`

### 13. shared memory bank conflict 是怎么产生的？

**答案：**

shared memory 被划分成多个 bank。以常见的 32-bank 组织为例，如果同一个 warp 内多个线程同时访问：

- **不同 bank：** 可以并行。
- **同一 bank 的不同地址：** 会冲突，访问被串行化。
- **同一 bank 的同一地址：** 常见情况可广播，不一定是严重冲突。

所以 bank conflict 的本质是：`同一 warp、同一时刻、同一 bank、不同地址`。

**常见优化：**

- padding
- 调整数据布局
- 转置存储
- swizzling

**经典例子：** `__shared__ float tile[32][33];` 里的 `+1`，就是为了打破跨行访问时的整齐步长，避免大量线程撞到同一 bank。

**参考资料：**

- CUDA C++ Best Practices Guide, Shared Memory and Memory Banks: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks>

**对应本地文件：** `docs/notes/cuda/从GEMM实践CUDA优化.md`

### 14. global memory 的关键优化点为什么是 coalescing？

**答案：**

global memory 容量大但延迟高，优化重点是让一个 warp 的访问尽量合并成少量内存事务。若 warp 内线程访问连续、对齐良好的地址，硬件就更容易把这些访问合并处理，减少 transaction 数量，提高带宽利用率。

面试里最实用的一句话是：

> 让 `threadIdx.x` 这个最快变化维度去读写内存中最快变化的连续维度。

如果访问步长很大、地址离散、对齐差，transaction 数会明显上升，cache line 利用率下降，吞吐掉得很快。

**参考资料：**

- CUDA C++ Best Practices Guide, Coalesced Access to Global Memory: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory>
- CUDA C++ Programming Guide, Device Memory Accesses: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses>

**对应本地文件：**

- `docs/notes/cuda/Vector Add Optimization Example.md`
- `docs/notes/cuda/GPU 内存系统演进：最大化带宽利用与延迟隐藏的技术路径.md`

### 15. constant memory 和 texture memory 分别适合什么场景？

**答案：**

- `constant memory` 适合小而只读、且 warp 内线程经常读取**相同地址**的数据。因为它有专门的 constant cache，广播效果好。
- `texture memory` 更适合具有二维/三维空间局部性、带采样或特殊寻址特征的读路径。现代纯数值 CUDA 内核里它不如 shared memory / L1/L2 常见，但概念上仍要知道。

面试时不需要把 texture 讲得很深，知道“它是只读访问路径，对空间局部性友好”通常就够了。

**参考资料：**

- CUDA C++ Programming Guide, Constant Memory: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constant-memory>
- CUDA C++ Programming Guide, Texture and Surface Memory: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 16. L1 和 L2 在 CUDA 优化里该怎么讲？

**答案：**

L1 更靠近 SM，容量小、服务本地访存；L2 是全 GPU 共享缓存，容量更大，对跨 SM 数据复用和 global memory 缓冲更重要。

面试里不要把重点放在“手动调 L1/L2”，而要强调：程序员通常更直接地控制 shared memory 和访问模式；L1/L2 更多依赖硬件缓存机制，但可以通过 tiling、coalescing、局部性设计来间接提高命中率。

**参考资料：**

- CUDA C++ Programming Guide, Memory Hierarchy: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

## 三、性能追问

### 17. 什么是 register spill？怎么判断一个 kernel 有没有 spill？

**答案：**

当线程需要的寄存器超出编译器/架构能承受的范围时，部分值会被溢出到 local memory，这就是 register spill。问题在于 local memory 通常走 global memory 路径，所以 spill 往往意味着更高延迟和更差吞吐。

最常见的检查方式：

- 编译时看 `-Xptxas -v` 输出里的 register 数、spill stores、spill loads。
- 用 Nsight Compute 看 local memory、寄存器和 stall 指标。

**常见优化：**

- 减少大局部数组。
- 缩短变量生命周期。
- 控制 unroll。
- 调整 block size 或 launch bounds。
- 拆 kernel 或拆复杂表达式。

**参考资料：**

- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>

**对应本地文件：**

- `docs/notes/cuda/cudaopt.md`
- `docs/notes/cuda/从GEMM实践CUDA优化.md`

### 18. block size 一般怎么选？为什么常说选 128/256/512？

**答案：**

经验上 block size 常选 32 的倍数，因为 warp 是 32 线程。128/256/512 常见，不是因为它们神奇，而是因为它们在大多数 kernel 里比较容易兼顾：

- warp 粒度对齐
- 足够并发
- 不至于一下把寄存器/shared memory 压爆

标准面试答法应该是：

> block size 要结合 warp 粒度、寄存器使用、shared memory 使用、访存模式和 kernel 特性一起选。先选 32 的倍数的合理起点，再用 profiling 验证。

**参考资料：**

- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- CUDA Occupancy Calculator: <https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 19. 为什么 GPU 适合高吞吐，不擅长强串行低延迟？

**答案：**

因为 GPU 的设计哲学是：

- 核心简单
- 并发很多
- 通过 warp 切换隐藏延迟
- 吞吐优先

而 CPU 更擅长：

- 单线程强
- 大缓存
- 强分支预测
- 复杂乱序执行

所以 GPU 很适合规则、数据并行、可批量化的问题；对分支复杂、串行依赖强、低延迟单任务场景往往不如 CPU。

**参考资料：**

- CUDA C++ Programming Guide: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>

**对应本地文件：** `docs/notes/cuda/cudaopt.md`

### 20. 一个高性能 GEMM kernel 里，线程和存储一般是怎么映射的？

**答案：**

最标准的分层回答是：

- `grid / threadblock` 负责结果矩阵 `C` 的大 tile。
- 一个 threadblock 内多个 `warp` 负责更小的 warp tile。
- 每个 `thread` 再负责更小的 thread tile，通常在寄存器里累加若干输出元素。
- `A/B` 的 tile 先从 global memory 搬到 shared memory，再从 shared memory 搬到寄存器，最后在寄存器里累计 `C` 的局部结果。

这其实就是：`global -> shared -> register` 的多级数据复用，以及 `threadblock tile -> warp tile -> thread tile` 的多级并行映射。

**高分补充：**

- 这样设计是为了同时兼顾 coalescing、shared memory 复用、寄存器复用、bank conflict 避免和 occupancy 权衡。
- CUTLASS 的 GEMM 组织就是典型模板。

**参考资料：**

- CUTLASS GEMM API: <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html>
- CUTLASS Efficient GEMM in CUDA: <https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/>

**对应本地文件：** `docs/notes/cuda/从GEMM实践CUDA优化.md`

### 21. 面试官问“GEMM 为什么快”，一段话怎么答？

**答案：**

高性能 GEMM 快，不是因为“每个线程算一个点”，而是因为它做了系统性的分层复用：

- 用 threadblock/warp/thread 多级 tile 划分计算。
- 让 A/B 子块在 shared memory 和寄存器里反复复用。
- 让 global memory 尽量 coalesced。
- 让 shared memory 尽量无 bank conflict。
- 让寄存器里尽量完成更多累加。

所以本质上，它是在同一套 kernel 里同时优化计算密度、访存模式和资源利用率。

**参考资料：**

- CUTLASS Efficient GEMM in CUDA: <https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/>
- CUTLASS GEMM API: <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html>

**对应本地文件：**

- `docs/notes/cuda/从GEMM实践CUDA优化.md`
- `docs/notes/cuda/cudaopt.md`

## 面试收口答案

### 3 分钟版

CUDA 编程模型从软件上看是 `grid / block / thread` 三层，但硬件真正调度执行的基本单位是 `warp`，通常一个 warp 是 32 个线程。block 是最关键的协作单位，因为 block 内线程能共享 shared memory，也能通过 `__syncthreads()` 做同步，所以一个 block 必须完整驻留在同一个 SM 上执行。SM 则通过维护多个 active warps 来隐藏访存延迟。

存储体系上，register 是线程私有且最快；如果寄存器不够会 spill 到 local memory，而 local memory 虽然逻辑上线程私有，物理上通常走 global memory 路径，所以很慢。shared memory 是 block 内共享的片上高速存储，适合做 tile、归约和数据复用，但要注意 bank conflict。global memory 容量最大，优化重点是 coalesced access 和减少重复访问。L1/L2 是硬件缓存，constant/texture 适合特定只读访问模式。

性能优化上，重点通常是：合理选 block size，平衡 occupancy 和资源使用，减少 warp divergence、bank conflict 和 spill，并通过 shared memory、寄存器复用与 coalesced global memory 访问把带宽和算力尽量喂满。GEMM 是最典型的综合题，因为它把线程映射、访存分层和资源权衡全部放到了一起。

### 1 分钟版

CUDA 从编程模型上是 `grid / block / thread`，但硬件按 warp 执行，一个 warp 通常是 32 个线程。block 是协作单位，因为 block 内线程能同步、能共享 shared memory，而且一个 block 只能跑在一个 SM 上。GPU 主要通过在 SM 上挂很多 ready warps 来隐藏访存延迟。

存储上，register 最快且线程私有，shared memory 是 block 内共享的片上高速存储，global memory 容量最大但延迟高，优化重点是 coalescing 和数据复用。local memory 名字叫 local，但通常落在 global memory 路径上；shared memory 要注意 bank conflict，寄存器太多会降低 occupancy 甚至 spill。CUDA 优化本质上就是平衡并发、访存和资源使用。

## 公开资料来源

### 官方文档 / 一线项目文档

- CUDA C++ Programming Guide: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- CUDA Occupancy Calculator: <https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html>
- CUDA Runtime API Occupancy functions: <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html>
- Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- CUTLASS GEMM API: <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html>
- CUTLASS blog: <https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/>

### 公开面经 / 讨论帖（只用于判断高频题，不作为技术结论依据）

- 牛客讨论：CUDA 高频追问相关帖子 1: <https://www.nowcoder.com/discuss/718716090212761600>
- 牛客讨论：CUDA 高频追问相关帖子 2: <https://www.nowcoder.com/discuss/738756774284533760>
- 牛客搜索结果：`site:nowcoder.com CUDA 面经 occupancy warp divergence bank conflict`: <https://www.google.com/search?q=site%3Anowcoder.com+CUDA+%E9%9D%A2%E7%BB%8F+occupancy+warp+divergence+bank+conflict>

## 建议放在哪个文件

这篇整理稿适合单独放在：

- `docs/notes/cuda/CUDA 面试高频问答：线程体系、存储体系与性能追问.md`

现有文件继续承担专题深入角色：

- `docs/notes/cuda/cudaopt.md`：总览型 CUDA 硬件与优化笔记。
- `docs/notes/cuda/GPU 内存系统演进：最大化带宽利用与延迟隐藏的技术路径.md`：访存与 latency hiding。
- `docs/notes/cuda/从GEMM实践CUDA优化.md`：GEMM、tiling、bank conflict、寄存器与 shared memory 权衡。
- `docs/notes/cuda/Vector Add Optimization Example.md`：memory-bound kernel、向量化、Roofline、coalescing。
