---
title: "CUDA 中的 ILP 与 MLP：从依赖链到延迟隐藏"
created: 2026-09-04
updated: 2026-09-04
tags:
  - CUDA
description: 从 warp scheduler、依赖链与 Little's Law 出发，区分 CUDA 中的指令级并行和内存级并行，并结合 GEMM、GEMV、FlashAttention、可对照的 Kernel 与 Nsight Compute 指标说明何时以及如何提升二者。
katex: true
---

# CUDA 中的 ILP 与 MLP：从依赖链到延迟隐藏

写 CUDA Kernel 时，我们经常听到两个建议：计算没有跑满就增加 ILP，访存没有跑满就增加 MLP。它们听起来像两套互不相关的技巧，实际却共享同一个目标：**当某条指令还没有完成时，给 SM 准备其他可以立即发射的工作。**

区别只在于“其他工作”是什么：

- **ILP（Instruction-Level Parallelism）**关注一条 warp 指令流中，有多少条互不依赖的指令可以继续推进。
- **MLP（Memory-Level Parallelism）**关注其中的访存部分，有多少个互不依赖的内存操作已经发出但尚未完成。

因此二者不是完全正交的概念。四条独立 FMA 只提升 ILP，不提升 MLP；四条独立 global load 同时提升了 ILP 和 MLP；而许多 warp 各自只发一条 load，也能依靠 TLP（Thread-Level Parallelism）在整个 SM 上形成足够的 MLP。

本文想回答五个实际问题：

1. ILP、MLP 在 CUDA 的执行模型中到底指什么？
2. 什么现象说明应该提升 ILP，什么现象说明应该提升 MLP？
3. 循环展开、每线程多元素、`float4`、Occupancy 和异步拷贝分别改变了什么？
4. 为什么“并行度更多”有时反而更慢？
5. 同样的 ILP/MLP 框架落到 GEMM、GEMV 与 FlashAttention 时，优化优先级为什么不同？

> [!NOTE]
> 文中的具体 stall 名称以 Nsight Compute 13.3 为参考。不同 GPU 和 Nsight Compute 版本提供的原始 metric 可能不同，因此诊断时优先看稳定的 section：Speed Of Light、Scheduler Statistics、Warp State Statistics、Memory Workload Analysis 和 Launch Statistics。

## 从 Warp Scheduler 看延迟隐藏

GPU 的执行不是“当前线程执行完一条指令，再执行下一条”。一个 SM 上同时驻留多个 warp；每个发射周期，warp scheduler 从 **eligible warp** 中选择一个 warp，发射它的下一条 ready instruction。这里的 ready 至少意味着操作数已经就绪，而且目标执行单元能够接收它。[^cuda-hardware]

假设一条指令产生结果需要等待 $L$ 个 cycle：

```text
只有一条依赖链：

FMA a0  ---- latency ---->  FMA a0  ---- latency ---->  FMA a0
          中间没有 ready instruction，当前 warp 只能等

四条独立依赖链：

FMA a0 -> FMA a1 -> FMA a2 -> FMA a3 -> FMA a0 -> ...
          用其他 accumulator 的工作覆盖 a0 的结果延迟
```

隐藏空隙有两条基本路径：

- 切换到另一个 ready warp，这是 TLP。
- 在同一个 warp 中发射另一条独立指令，这是 ILP。

CUDA 官方文档将 ILP 描述为单线程内流水化的独立指令；由于 SIMT 硬件以 warp 为发射单位，CUDA 源码中“每个线程有四条独立链”最终表现为“该 warp 有四条对应的独立指令链”。[^cuda-ilp] Vasily Volkov 的经典实验进一步说明，更多 ILP 可以用更少的线程隐藏算术延迟；所以 Occupancy 是提供 ready work 的一种手段，而不是性能本身。[^volkov]

内存延迟也遵循相同逻辑，只是等待通常更长。用 Little's Law 表示，稳态下有：

$$
N_{\text{in-flight}} = \text{throughput} \times \text{latency}
$$

如果 throughput 用 byte/s 表示，那么左侧就是 bytes in flight；如果用 request/s 表示，左侧就是未完成请求数。要达到目标带宽，软件必须暴露足够多的独立访问，让内存系统在等待旧请求返回时仍能接收新请求。**MLP 不会缩短单次 DRAM 访问的物理延迟，而是减少这段延迟暴露在 critical path 上的比例。**

## ILP 和 MLP 的准确边界

### ILP：同一指令流中可重叠推进的独立指令

如果指令 B 的输入依赖指令 A 的输出，那么在 A 的结果 ready 前，B 不能发射：

```cpp
x = __fmaf_rn(x, a, b);
x = __fmaf_rn(x, a, b);  // RAW dependency: 必须等上一条更新 x
```

如果两条指令更新不同的 accumulator，则它们之间没有 Read After Write（RAW）依赖：

```cpp
x0 = __fmaf_rn(x0, a, b);
x1 = __fmaf_rn(x1, a, b);  // 与上一条 FMA 独立
```

所以 ILP 不是“代码里连续写了多少条指令”，而是**在当前依赖图和硬件资源约束下，多少条指令有机会同时处于流水线的不同阶段**。分支、barrier、寄存器依赖、执行单元占满和编译器生成的实际 SASS 都会改变它。

### MLP：未完成的独立内存操作

NVIDIA 旧版 Profiler Guide 给出的软件视角定义很直接：MLP 是每个线程中 in flight 的独立内存操作数量。该文档也把循环展开、向量类型和每线程处理多个元素列为提高 MLP 的办法。[^mlp-profiler]

但在物理硬件上还要多走几步：线程中的 load 被编译成 warp 级内存指令，warp 中的地址经过 coalescing 形成一个或多个 memory transaction，请求又可能在 cache 层命中、合并或重放。因此：

$$
\text{source-level loads} \neq \text{warp instructions} \neq
\text{memory transactions} \neq \text{DRAM requests}
$$

工程上可以把 MLP 理解为两个相关层次：

- **单 warp / 单线程暴露的 MLP**：消费第一份数据前能连续发出多少条独立 load。
- **SM 或设备实际承载的 MLP**：所有 resident warp 合计在 cache、LSU 和内存分区中留下多少未完成请求。

这也解释了为什么 Nsight Compute 没有一个跨架构通用的“MLP = 7.3”指标。我们通常从 eligible warps、scoreboard stall、内存吞吐、请求效率和 throttle 反推并发请求是否不足。

### 二者和 TLP、向量化的关系

| 改动                             | ILP          | MLP                         | 主要改变                              |
| -------------------------------- | ------------ | --------------------------- | ------------------------------------- |
| 四个独立 accumulator             | 增加         | 不变                        | 独立计算依赖链                        |
| 每线程先发四条独立 load          | 增加         | 增加                        | 独立访存指令与在途请求                |
| 增加 resident warps              | 单 warp 不变 | SM 总 MLP 可能增加          | 用 TLP 提供更多 ready work            |
| `float` 改为对齐的 `float4` load | 不一定       | 请求数不一定增加            | 每条指令搬更多有效 byte，减少指令开销 |
| 改善 coalescing                  | 不一定       | 甚至可能减少 transaction 数 | 减少无效 transaction 和浪费的 byte    |
| 增加 `cp.async` stage            | 增加 overlap | 增加在途 copy               | 显式扩大 load 与 use 的距离           |

这里最容易混淆的是 `float4`。它可能让一条线程级 load 搬运 16 byte，增加 bytes in flight，并减少发射同样数据量所需的指令；但它没有自动创造四条互相独立的内存指令。**访问宽度、coalescing、MLP 是三个不同维度。**

## 什么时候应该提升 ILP

先不要从源码猜。Nsight Compute 官方建议只有在 scheduler 不能持续发射时，stall reason 才值得作为优化入口。[^ncu-scheduler] 一个较可靠的 ILP 信号组合是：

- Scheduler Statistics 中每个 scheduler 的 eligible warps 经常接近 0，并出现较多 skipped issue slots。
- 计算吞吐未接近目标 pipeline 上限，内存吞吐也不是主要上限。
- Warp State Statistics 或 Source Counters 将空隙定位到 fixed-latency execution dependency，例如较高的 `Wait`；或者源码/SASS 明确存在很长的 RAW 依赖链。
- 增加 block 数、提高 Occupancy 后收益很小或做不到，例如小问题、persistent kernel，或者寄存器/Shared Memory 让 TLP 有天然上限。

典型场景包括：

- reduction、dot product、卷积和 GEMM micro-kernel 中只有一个 accumulator，形成连续 FMA 依赖链；
- 一个线程负责多个输出，但源码按“完整算完输出 0，再算输出 1”的顺序组织；
- 地址计算、类型转换或其他独立标量工作没有与主计算交错；
- 低 Occupancy 是经过权衡后的结果，需要让少量 warp 自己提供足够的 ready instructions。

相反，如果 math pipeline 已经接近饱和并出现 `Math Pipe Throttle`，问题是执行单元供不应求，不是独立计算不够；继续增加 ILP 只会增加寄存器和指令数。如果 `Long Scoreboard` 才是主要阻塞，应该先看 MLP、局部性或访问效率。

## ILP Kernel：从一条累加链变成四条

下面的 micro-kernel 用 `CHAINS` 个独立 accumulator 处理相同总量的数据。`CHAINS=1` 时，每个线程只有一条循环携带依赖；`CHAINS=4` 时，同一个线程有四条可交错发射的 FMA 链：

```cpp
template <int CHAINS>
__global__ void fma_ilp_kernel(const float *__restrict__ input,
                               float *__restrict__ output,
                               int n, int iterations,
                               float mul, float add) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  float acc[CHAINS];

#pragma unroll
  for (int c = 0; c < CHAINS; ++c) {
    const int i = tid + c * stride;
    acc[c] = i < n ? input[i] : 0.0f;
  }

#pragma unroll 1
  for (int k = 0; k < iterations; ++k) {
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) {
      acc[c] = __fmaf_rn(acc[c], mul, add);
    }
  }

#pragma unroll
  for (int c = 0; c < CHAINS; ++c) {
    const int i = tid + c * stride;
    if (i < n) output[i] = acc[c];
  }
}
```

为了比较相同的总输出数量，launch 时让线程数约为 `ceil(n / CHAINS)`。内层循环足够长后，首尾的 global load/store 被摊薄，主要差异就是 accumulator 的数量：

```cpp
// Baseline: 一条 FMA 依赖链
fma_ilp_kernel<1><<<ceil_div(n, 256), 256>>>(
    input, output, n, iterations, mul, add);

// ILP=4: 四条彼此独立的 FMA 依赖链
fma_ilp_kernel<4><<<ceil_div(n, 256 * 4), 256>>>(
    input, output, n, iterations, mul, add);
```

`#pragma unroll` 本身不是收益来源。真正的变化是编译器能把 `acc[0]` 到 `acc[3]` 标量化进寄存器，并把不同链的 FMA 交错排列。如果只是把 `acc = f(acc)` 的同一条依赖链展开四次，RAW dependency 仍然存在，ILP 没有增加。

### 提升 ILP 的主要手段

#### 1. 多累加器与寄存器分块

这是 reduction、dot product 和 GEMM 中最常见的做法。把一个串行累加器拆成 2、4 或更多个局部累加器，最后再合并：

```cpp
float sum0 = 0.0f;
float sum1 = 0.0f;

for (int i = tid; i + stride < n; i += 2 * stride) {
  sum0 += x[i];
  sum1 += x[i + stride];
}
float sum = sum0 + sum1;
```

GEMM 的 thread tile 是同一思想的二维版本：一个线程持有 $T_M \times T_N$ 个 `C` accumulator，每次加载一小组 `A/B` fragment 后更新多个输出。收益来自输入复用和独立累加链，代价是 accumulator 寄存器数近似随 $T_M T_N$ 增长。

#### 2. 循环展开，但先确认跨迭代独立

编译器只有在迭代次数或展开因子可知、别名关系清晰，并且不同迭代没有强依赖时，才有空间重新排布指令。模板常量、固定 tile shape、`#pragma unroll N` 和手工展开都能暴露这个机会。

展开同时会：

- 减少 loop branch 和索引更新指令；
- 扩大编译器的调度窗口；
- 增加 live value 数量、寄存器压力和代码体积；
- 在大展开因子下增加 instruction-cache 压力。

所以应该扫描 `N=1/2/4/8`，而不是默认“完全展开最快”。

#### 3. 每线程处理多个输出

Thread coarsening 把原本由多个线程完成的独立输出交给一个线程。它可以带来三种收益：

- 暴露多个独立计算链；
- 复用已经加载到寄存器或 Shared Memory 的输入；
- 摊薄地址计算、边界判断等固定指令。

代价是 grid 中线程数减少、尾部处理变复杂、单线程寄存器增加。问题规模很小时，coarsening 还可能直接减少并行 block 数，让 GPU 无法铺满。

#### 4. 消除编译器看见的假依赖

`const`、`__restrict__`、固定索引和局部标量可以让编译器确认不同 load/store 或计算之间不存在 alias，从而提前发射和重排。但 `__restrict__` 是正确性契约：两个 restricted pointer 实际发生别名时，程序行为就不再可靠。

更激进的重排还可能改变浮点加法顺序。多累加器 reduction 与串行 reduction 通常只有数学意义上的等价，并不保证 bitwise identical；需要按业务误差预算验证。

#### 5. 软件流水化

把第 $k+1$ 次迭代的地址计算、load 或转换放到第 $k$ 次迭代的计算期间，本质上是拉大 producer 到 consumer 的距离：

```text
原始：load(k) -> wait -> compute(k) -> load(k+1)
流水：load(k+1) -> compute(k) -> wait(k+1) -> compute(k+1)
```

其中独立地址计算和计算指令增加 ILP；提前发出的 load 也增加 MLP。软件流水化正是两者最常见的交界处。

### 提升 ILP 的资源影响

| 影响                  | 为什么发生                               | 需要检查什么                                         |
| --------------------- | ---------------------------------------- | ---------------------------------------------------- |
| 寄存器增加            | 更多 accumulator 和预取值同时存活        | `registers/thread`、是否跨过 occupancy 分配台阶      |
| Occupancy 下降        | 每线程寄存器变多，每 SM 可驻留 warp 变少 | achieved/theoretical occupancy、active warps         |
| spill 到 local memory | live range 超过寄存器预算                | SASS 中 `LDL/STL`、local load/store、Long Scoreboard |
| 指令数与代码体积增加  | 展开复制循环体                           | instruction count、instruction fetch stall           |
| 数值结果变化          | reduction 次序或 fast math 改变          | 误差、NaN/Inf、边界输入                              |

这是一种资源交换：**用寄存器容量换取更少的 exposed dependency latency，并可能用 ILP 换掉一部分 TLP。** Volkov 的结果说明低 Occupancy 完全可能更快，但前提是减少的 warp 已经被每个 warp 更多的有效工作补偿，不能反过来把“低 Occupancy”当优化目标。

## 什么时候应该提升 MLP

适合提高 MLP 的信号组合是：

- Scheduler 的 eligible warps 不足，issue slot 经常空闲。
- `Long Scoreboard` 指向 global/local/texture load 的结果依赖，而不是 barrier 或计算依赖。当前 Nsight Compute 将其定义为等待 L1TEX 操作的 scoreboard dependency。[^ncu-stalls]
- DRAM/L2 吞吐显著低于同机、同访问模式可达到的带宽。
- global access 已经较好地 coalesced，sector 利用率合理，且 cache miss 是 workload 的固有部分。
- `LG Throttle`、`MIO Throttle` 或其他队列满指标并不高，否则请求管线可能已经过载。

典型场景包括 streaming copy、embedding lookup、gather/scatter、稀疏算子，以及 tiled GEMM/Attention 的 global-to-shared feeding 阶段。NVIDIA 的 cuEmbed 就通过每线程同时加载多个 embedding row、循环展开和 128-bit load 最大化 loads in flight；这个例子很重要，因为 embedding lookup 的地址不规则、计算又少，仅靠算术 ILP 无法填补内存等待。[^cuembed]

以下情况则不该先加 MLP：

- DRAM 或 L2 throughput 已经接近当前 workload 的可持续上限。此时应该减少 byte、提高复用或改变数据类型。
- load 不合并、对齐差或 sector 浪费严重。更多并发只会更快地产生无效流量。
- cache working set 本可复用但发生抖动。加深预取距离可能让数据在使用前就被逐出。
- `LG Throttle` 很高，说明 local/global memory instruction queue 已经难以接收更多请求；Nsight Compute 对这种情况建议减少冗余访问，并考虑用更少、更宽的指令。[^ncu-throttle]
- 核心是 pointer chasing：下一次地址依赖上一次 load 的结果。单条链在算法上只有 MLP=1，单纯 `unroll` 无法创造不存在的独立地址。

## MLP Kernel：先发四条 Load，再消费结果

下面的 copy kernel 与 Volkov GTC 示例采用相同思路。`ITEMS=1` 时，warp 发出 load 后很快就遇到依赖它的 store；`ITEMS=4` 时，编译器有机会在消费 `values[0]` 前发出四条独立 load：

```cpp
template <int ITEMS>
__global__ void copy_mlp_kernel(const float *__restrict__ src,
                                float *__restrict__ dst,
                                int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  float values[ITEMS];

  // Producer phase: 暴露 ITEMS 条独立 global load。
#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int i = tid + item * stride;
    values[item] = i < n ? src[i] : 0.0f;
  }

  // Consumer phase: 到这里才第一次使用 load 的结果。
#pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int i = tid + item * stride;
    if (i < n) dst[i] = values[item];
  }
}
```

同一个 `item` 下，warp 的连续线程访问连续地址，所以每条 load 仍然是 coalesced 的。比较时同样保持总元素数不变：

```cpp
copy_mlp_kernel<1><<<ceil_div(n, 256), 256>>>(src, dst, n);
copy_mlp_kernel<4><<<ceil_div(n, 256 * 4), 256>>>(src, dst, n);
```

`values` 必须被编译器标量化进寄存器；如果动态索引或过大的 `ITEMS` 使它落入 local memory，这个“优化”会制造额外 global traffic。还要查看 SASS 或 Nsight Compute Source View，确认生成顺序确实接近 `LDG, LDG, LDG, LDG, STG...`，而不是源码看起来并行、机器码仍然逐项 load/use。

这个例子展示的是**单 warp 暴露的 MLP**。即便 `ITEMS=1`，大量 resident warps 也可能让设备层面的请求数已经充足；因此 `ITEMS=4` 不保证更快。它通常在 Occupancy 受限、grid 偏小或每线程访问本就较多时更有价值。

### 提升 MLP 的主要手段

#### 1. 在 Use 前集中发出独立 Load

将：

```text
load0 -> use0 -> load1 -> use1
```

改写为：

```text
load0 -> load1 -> load2 -> load3 -> use0 -> use1 -> use2 -> use3
```

循环展开和每线程多元素只是实现这种调度的工具。前提是地址独立、编译器能证明没有 alias，而且硬件队列仍有容量。

对于 gather，可以先批量加载 index，再批量发起 table load；对于 embedding bag，可以同时处理多个 row；对于多条 linked list，可以让一个线程或 warp 同时推进多条独立 chain。若只有一条链：

```cpp
next = links[next];
next = links[next];  // 第二个地址依赖第一个 load，展开也仍然串行
```

必须通过批处理多条查询、改变数据布局或重新设计算法，才能得到真正的独立请求。

#### 2. 用更多 Warp 提供请求

提高 block 数、选择合适的 block size、减少不必要的寄存器/Shared Memory，都可能增加 resident warps。每个 warp 即使只有一条 outstanding load，很多 warp 合计也能形成较高的 SM-level MLP。

这条路径适合并行度充足、每线程工作简单的 kernel。它的局限是：

- 小 grid 或 persistent kernel 没有更多 block 可调度；
- barrier 可能让同一 CTA 的 warp 同时停下；
- 增加 warp 会分走寄存器、cache 容量和调度机会；
- 达到足够 latency hiding 后，继续提高 Occupancy 通常没有收益。

因此 MLP 可以由“少量 warp × 每 warp 多请求”提供，也可以由“更多 warp × 每 warp 少请求”提供。最佳点取决于寄存器、cache、LSU 队列和 workload shape。

#### 3. 增大单条指令的有效搬运量

在满足自然对齐和边界条件时使用 `float2`、`float4`、`int4` 等向量类型，可以减少 load/store 指令数，并让每次发射携带更多有效 byte。它提高的是 **bytes per instruction**，可能帮助满足 Little's Law 所需的 bytes in flight。

但要分别验证：

- 指针是否满足 8/16-byte 对齐；
- 尾部元素是否正确处理；
- SASS 是否真的生成 64/128-bit load；
- 更宽的 thread access 是否恶化 warp transaction 或寄存器压力。

向量化不是为了“让寄存器也变成 SIMD”。CUDA Best Practices Guide 明确指出，没有必要因为寄存器访问而把数据打包成 `float4`；它的主要价值在内存指令和数据搬运。[^registers]

#### 4. 先修复 Coalescing、对齐与局部性

这些手段通常不增加 MLP，却会让同样数量的在途请求更有价值：

- coalescing 减少一个 warp 指令产生的无效 transaction；
- 对齐减少跨 sector/cache line 的额外访问；
- Shared Memory tiling 和 cache reuse 减少重复 global load；
- 合理的 block 遍历顺序减小瞬时 working set。

这就是为什么看到 `Long Scoreboard` 时不能立刻加 unroll。等待可能来自“请求太少”，也可能来自“每个请求都在做无效工作或频繁 miss”。前者需要 MLP，后者需要优化访问路径。

#### 5. 异步拷贝与多级流水线

对规则的 global-to-shared tile，Ampere 及以后架构可以使用硬件加速的异步拷贝；Hopper 还提供 TMA。CUDA 官方文档说明，async copy 可以让 global-to-shared 搬运与计算重叠，并避免传统同步路径中的中间寄存器。[^async-copy]

两级 pipeline 的核心不是 API 名称，而是 buffer ownership：

```text
prologue:  发起 tile 0 的异步拷贝

steady state:
  等待 tile k ready
  发起 tile k+1 的异步拷贝
  计算 tile k
  释放 tile k 的 buffer

epilogue: 消费最后一个已提交 tile
```

使用 `cuda::pipeline` 时，producer acquire/commit 和 consumer wait/release 明确描述了这个生命周期。pipeline stage 是有限资源；stage 全部占用后，producer 仍会阻塞。[^cuda-pipeline]

增加 stage count 会扩大 copy 到 use 的距离，提高在途 tile 数，但也线性增加 Shared Memory 容量，并可能降低每 SM 的 resident CTA 数。理想 stage 数满足：计算 tile $k$ 的时间足以覆盖 tile $k+1$ 的传输；再多的 stage 只会占资源和增加 prologue/epilogue 成本。

### 提升 MLP 的资源影响

| 影响               | 为什么发生                              | 需要检查什么                       |
| ------------------ | --------------------------------------- | ---------------------------------- |
| 寄存器增加         | 多个 load 结果在 use 前同时存活         | registers/thread、spill、Occupancy |
| Shared Memory 增加 | 多 stage 要同时保存多个 tile            | bytes/block、active CTAs/SM        |
| cache 污染         | 预取距离过远或工作集扩大                | L1/L2 hit rate、重复 DRAM bytes    |
| 请求队列拥塞       | 发出的 load 超过 LSU/L1TEX/MIO 可承载量 | `LG/MIO/TEX Throttle`              |
| 带宽饱和           | MLP 已足够后，瓶颈转为物理带宽          | DRAM/L2 throughput、memory SOL     |
| 尾部与同步开销     | pipeline 需要 fill、drain、barrier      | 小 shape 曲线、barrier stall       |

这同样是资源交换：**用寄存器、Shared Memory 和 queue capacity 换取更多 bytes in flight。**

## 三个算子里的 ILP 与 MLP：GEMM、GEMV 与 FlashAttention

只说“计算密集型提高 ILP，访存密集型提高 MLP”还不够，因为一个真实 Kernel 往往同时包含 global load、片上搬运、矩阵乘加、归约和写回。更可靠的办法是先问两个问题：

1. 数据从 HBM 搬进来后能被复用多少次，也就是算术强度能否随问题规模增长？
2. 当前 critical path 是计算依赖、内存依赖，还是设备上根本没有足够的 CTA？

GEMM、GEMV 和 FlashAttention 恰好给出了三种不同答案。下面的公式只建立理想下界：假设输入各从 HBM 读取一次、输出写回一次，不计 cache 重读、对齐浪费、epilogue 和 $\beta C$ 或 $\beta y$ 的额外读取。令 $s_A,s_B,s_C$ 表示对应元素的 byte 数。

对于 $C_{M\times N}=A_{M\times K}B_{K\times N}$：

$$
\mathrm{AI}_{\mathrm{GEMM}}
\approx
\frac{2MNK}{s_A MK+s_B KN+s_C MN}
$$

当 $M=N=K=n$ 且数据宽度相同时，算术强度约为 $2n/(3s)$，会随 $n$ 线性增长。大矩阵只要分块复用做得好，就有机会从带宽瓶颈转向计算瓶颈。

对于 cuBLAS 定义的 $y=\alpha\operatorname{op}(A)x+\beta y$，若先看 $\beta=0$ 的最低流量：[^cublas-gemv]

$$
\mathrm{AI}_{\mathrm{GEMV}}
\approx
\frac{2MN}{s_A MN+s_xN+s_yM}
\xrightarrow[M,N\to\infty]{}
\frac{2}{s_A}
$$

因此 FP32 GEMV 的理想上界也只有约 $0.5$ FLOP/byte，FP16/BF16 约为 $1$ FLOP/byte。它不会像 GEMM 那样通过增大矩阵维度获得不断增长的算术强度，因为矩阵 $A$ 的每个元素只参与一次乘加。这是后面优化优先级不同的根因。

### GEMM：用寄存器扩大 ILP，用流水线保证供料

先看最朴素的 GEMM：一个线程计算一个 $C_{mn}$，循环内只有一个 `sum`。它同时有两个问题：

- `sum = fma(a, b, sum)` 形成长度为 $K$ 的单 accumulator 依赖链；
- 相邻输出反复从 global memory 读取相同的 $A/B$ 元素，没有利用片上复用。

第一项是 ILP 问题，第二项却不是 MLP 不足，而是**搬了太多本可复用的 byte**。若直接让每个线程预取更多 global load，可能只是更快地制造冗余流量。高性能 GEMM 会先做 CTA、warp 和 thread 三级分块：CTA tile 在 Shared Memory 中复用，warp/thread tile 在寄存器中复用。CUTLASS 的线程级 GEMM 正是让每个线程计算二维小块，并发出一组独立的累加指令。[^cutlass-gemm]

下面是简化后的 mainloop 调度，不对应某个特定架构的可编译 API，但保留了依赖关系：

```cpp
float acc[TM][TN] = {};        // 多个独立 C accumulator

prefetch_to_smem(/* k_tile = */ 0, /* stage = */ 0);

for (int kt = 0; kt < K_TILES; ++kt) {
  const int read_stage = kt % STAGES;
  wait_stage(read_stage);

  // 当前 tile 计算时，提前搬运下一 tile。
  if (kt + 1 < K_TILES) {
    prefetch_to_smem(kt + 1, (kt + 1) % STAGES);
  }

#pragma unroll
  for (int kk = 0; kk < BK; ++kk) {
    float a_frag[TM];
    float b_frag[TN];
    load_fragments_from_smem(a_frag, b_frag, kk, read_stage);

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        acc[i][j] = __fmaf_rn(a_frag[i], b_frag[j], acc[i][j]);
      }
    }
  }
  release_stage(read_stage);
}
```

这里有三种不能混为一谈的收益：

- `acc[TM][TN]` 把一条长依赖链拆成多条独立 FMA 链，这是**计算 ILP**；同一 `a_frag[i]` 或 `b_frag[j]` 又被多个输出复用。
- `prefetch_to_smem(kt + 1)` 在消费当前 tile 时发起下一 tile 的 global-to-shared copy，这是**MLP 与 load/compute overlap**。
- CTA tiling 减少了 HBM 请求总量，这是**数据复用**，不是提高 MLP。

CUTLASS 使用 threadblock 级 Shared Memory 双缓冲，以及 warp 级寄存器 fragment 双缓冲来实现这种软件流水；它也明确指出 accumulator 往往占据线程寄存器预算的很大一部分，所以高性能 GEMM 可以在较低 Occupancy 下依靠 ILP 与流水线维持吞吐。[^cutlass-gemm]

GEMM 中该提高哪一个，可以按阶段判断：

- 已经有良好 tiling，但 `Wait` 高、目标 math pipe 未满：增大 thread/warp tile，增加独立 accumulator 或 MMA fragment，优先提高 ILP。
- `Long Scoreboard` 指向 A/B 的 global-to-shared 路径，DRAM 吞吐又未满：增加 `cp.async`/TMA stage、提前地址计算，优先提高 MLP 和流水重叠。
- DRAM 已满：继续加 stage 没有意义，应扩大复用、调整 CTA tile 或改善 L2 locality。
- $M$ 或 $N$ 很小而 $K$ 很大：即使单 CTA 内 ILP 很高，整个 GPU 仍可能没有足够 block。此时应换小 tile、做 split-K 或 grouped/batched GEMM；CUTLASS 也将 split-K 用于小问题缺少 threadblock 并行度的情形。[^cutlass-gemm]

GEMM 的核心交换因此是：更大的输出 tile 提高复用和 ILP，却消耗更多 accumulator 寄存器；更多 pipeline stage 提高 MLP，却消耗更多 Shared Memory。两者都可能压低 Occupancy，必须联合扫描，而不是分别拉到最大。

### GEMV：MLP 只能帮助打满带宽，不能突破带宽

典型 GEMV 可以让一个 warp 负责一行：lane 分别读取该行的连续列，计算局部点积，最后用 warp shuffle 做归约。下面的核心循环一次处理四个 warp-sized chunk；为了突出依赖关系，省略了不能整除时的尾部：

```cpp
template <int UNROLL = 4>
__global__ void gemv_warp_rows(const float *__restrict__ A,
                               const float *__restrict__ x,
                               float *__restrict__ y,
                               int M, int N) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= M) return;

  float sum[UNROLL] = {};

  for (int base = lane; base + (UNROLL - 1) * 32 < N;
       base += UNROLL * 32) {
    float a[UNROLL];
    float xv[UNROLL];

    // Producer：先暴露多组彼此独立且仍然合并的 load。
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int col = base + u * 32;
      a[u] = A[warp * N + col];
      xv[u] = x[col];
    }

    // Consumer：多 accumulator 同时缩短单链依赖。
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      sum[u] = __fmaf_rn(a[u], xv[u], sum[u]);
    }
  }

  float total = 0.0f;
#pragma unroll
  for (int u = 0; u < UNROLL; ++u) total += sum[u];
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    total += __shfl_down_sync(0xffffffff, total, offset);
  }
  if (lane == 0) y[warp] = total;
}
```

这段改写同时改变了 ILP 和 MLP：

- 四组 `A/x` load 在第一次消费前被发出，提高单 warp 的 MLP；每个 `u` 在整个 warp 上仍访问 32 个连续元素，所以没有牺牲 coalescing。
- `sum[0..3]` 是四条独立 FMA 链，提高计算 ILP，并将归约依赖推迟到循环之后。
- 同一 block 若计算多行，可将 `x` 的 tile 放进 Shared Memory，让多个 warp 复用；这减少的是 x 的流量，但占比最大的 $A$ 仍要被流式读取。

这种展开在 `Long Scoreboard` 高、带宽没有打满、矩阵行数太少导致 resident warp 不足时最有价值。代价是 `a/xv/sum` 的 live range 增长，寄存器用量上升；过度展开会降低 active warps，甚至 spill，最终把增加的 MLP 又抵消掉。

一旦 HBM 带宽已经饱和，继续从 `UNROLL=4` 加到 8 通常不能突破 Roofline。此时有效方向是减少 byte：使用更窄的数据类型、量化权重、融合 bias/activation，或改善 $x$ 的复用。还有一个更根本的办法：若同时有多个向量 $X_{N\times B}$，把它们合并成 $Y=AX$。当 $B>1$ 后，每个 $A$ 元素可以服务多个输出，算子从 GEMV 逐渐变成 GEMM，算术强度也随 $B$ 增长。**Batching 不是“给 GEMV 增加一点 MLP”，而是改变了数据复用和算子类别。**

小 GEMV 还可能短到连一个完整 GPU wave 都铺不满。此时优先考虑批处理、persistent/grouped kernel 或与相邻逐元素操作融合；单 CTA 内再多 ILP，也不能补出不存在的设备级并行度。

### FlashAttention：先减少 IO，再重叠搬运、矩阵乘与 Softmax

标准 Attention 将 $S=QK^T$ 和 $P=\operatorname{softmax}(S)$ 物化到 HBM。FlashAttention 的第一性改动不是“发出更多 load”，而是通过 tiling 和 online softmax 不再写回这两个 $N\times N$ 中间矩阵。原论文给出的 IO 分析表明：标准 Attention 的 HBM 访问量为 $\Theta(Nd+N^2)$，FlashAttention 在 SRAM 容量为 $M_{\mathrm{SRAM}}$ 时为 $\Theta(N^2d^2/M_{\mathrm{SRAM}})$。[^flashattention]

所以它展示了一个比增加 MLP 更优先的原则：**如果能从算法上删除 HBM traffic，就先删 byte，再隐藏剩余访问的延迟。**

固定一个 $Q_i$ tile 后，简化的前向 mainloop 可以写成：

```text
prefetch K[0], V[0]

for j in KV tiles:
    wait K[j], V[j]
    prefetch K[j+1], V[j+1]       # 与本轮计算独立，增加 MLP

    S[j] = Q[i] @ K[j]^T          # GEMM 0
    (m, l, P[j]) = online_softmax(S[j], m, l)
    O = rescale(O, m_old, m) + P[j] @ V[j]  # GEMM 1
```

其中状态存在真正的循环携带依赖：

$$
(m_j,\ell_j,O_j)=F(S_j,V_j,m_{j-1},\ell_{j-1},O_{j-1})
$$

下一块的 online softmax 不能无条件越过上一块状态更新。这和 GEMM 中多个互不相关的 $C$ accumulator 不同。可以自由提前的是下一块 $K/V$ 的搬运和地址计算；可以增加的是单块内部 MMA accumulator、row max/row sum 归约的 ILP；不能凭 `#pragma unroll` 消掉的是跨 KV tile 的数值递推。

FlashAttention 的演进也能用这套边界解释：

- **FlashAttention-2** 在 batch/head 之外沿 query sequence 增加 threadblock 并行，并重新划分 warp 工作以减少 Shared Memory 通信。原论文将收益归因于更高 Occupancy、更少非 matmul FLOPs 和更少 warp 间通信。[^flashattention2] 这主要是 TLP、工作划分和 traffic 优化，不能简单记成“增加 ILP”。
- **FlashAttention-3** 在 Hopper 上用 producer/consumer warp specialization、循环 Shared Memory buffer、TMA 和异步 WGMMA。producer 连续发起 $K/V$ tile 搬运，consumer 执行矩阵乘；同时让一个 warpgroup 的 Softmax 与另一个 warpgroup 的 GEMM 交错。[^flashattention3] TMA pipeline 增加的是 MLP 与跨引擎 overlap；多个 MMA/Softmax 的交错扩大了可发射工作，但跨 warpgroup 调度严格说已超出经典“单线程 ILP”，更准确的名字是 CTA 内任务级流水。

这里的资源交换比 GEMM 更紧：增加 KV stage 会线性增加 Shared Memory；增大 $Q/KV$ tile 会提高 Tensor Core 利用率，却增加 accumulator、Softmax 状态和尾部浪费；更多 consumer warpgroup 能提供计算并行，又会压缩每个线程的寄存器预算。还要同时检查 causal mask、变长序列和 head dimension，因为它们会改变有效 tile 比例与负载均衡。

最后必须区分 **prefill/training attention** 与 **decode attention**。前者的 query length 较大，两个块矩阵乘有 GEMM 特征，优化重点通常是 tile reuse、Tensor Core ILP 和 load/compute/Softmax 流水。自回归 decode 每步通常只有一个或少量 query，却要扫描不断增长的 KV cache，更接近 GEMV：HBM byte、请求并发和可供调度的 CTA 才是主线。此时 split-KV/Flash-Decoding 是沿序列拆出更多独立 block，最后合并 online-softmax 状态；它主要补的是 TLP 和设备级 MLP，而不是让单条 softmax 递推链突然具有更多 ILP。[^flashdecoding]

### 三个算子的统一判断

| 算子/阶段                         | 数据复用与常见上限                         | 优先提升 ILP 的信号                         | 优先提升 MLP 的信号                           | 更优先的其他动作                         |
| --------------------------------- | ------------------------------------------ | -------------------------------------------- | ---------------------------------------------- | ---------------------------------------- |
| 大而规则的 GEMM mainloop          | tile 复用高，常走向 math pipe 上限         | `Wait` 高、math pipe 未满、供料已稳定        | A/B load 的 `Long Scoreboard` 高且带宽未满     | 先做好 CTA/warp/thread tiling            |
| 小 $M$ 或 $N$、大 $K$ GEMM        | 单 CTA 工作多，但设备 block 数不足          | 单 CTA 内确有计算依赖时适度增加               | 通常不是第一矛盾                               | 小 tile、split-K、grouped/batched GEMM   |
| GEMV                              | $A$ 近似只读一次，通常受 HBM 带宽限制       | 点积单 accumulator 依赖明显且 math pipe 未满  | 带宽低、`Long Scoreboard` 高、访问已合并       | 饱和后减 byte、融合，或 batching 成 GEMM |
| FlashAttention prefill/training   | tile 内 GEMM 复用高，且已删除 $S/P$ HBM IO | MMA/Softmax 依赖让 Tensor Core 或 SFU 空转    | KV pipeline 断流、TMA/load 等待且带宽未满      | 先减少 IO、改善 sequence/warp 工作划分   |
| Decode attention / Flash-Decoding | KV cache 流式读取，更接近 GEMV              | 局部 dot/reduction 单链明显时才考虑           | KV 请求不足、长延迟暴露                        | split-KV、量化 KV、融合、改善 page layout |

这三类算子给出了一条很实用的优先级：

1. 先减少无效或可复用的 byte，GEMM tiling 与 FlashAttention 融合都属于这一层。
2. 再让剩余 load 足够并发，用 MLP 把 memory pipeline 填起来。
3. 再拆计算依赖链，用 ILP 保持 CUDA Core、Tensor Core 或其他目标 pipeline 有 ready work。
4. 若 grid 本身太小，就增加 TLP 或改变任务组织；ILP/MLP 不能替代缺失的 CTA。

实际实现并不总按这个顺序逐项发生，但诊断必须区分这四层。否则很容易把 GEMM 的“寄存器分块”、GEMV 的“多 load 在途”和 FlashAttention 的“删除中间矩阵”都叫作同一种并行优化，进而在错误的资源上继续加码。

## 用 Nsight Compute 决定提升哪一个

可以先收集完整 section，再从高层到低层下钻：

```bash
ncu --set full --kernel-name regex:target_kernel ./benchmark
```

建议按下面的顺序判断，而不是先盯住某个 stall 百分比：

### 第一步：Scheduler 真的断粮了吗

查看 Scheduler Statistics：

- Active Warps 高，只表示 warp 已驻留。
- Eligible Warps 表示当下能够发射下一条指令的 warp。
- Issued Warps 和 skipped issue slots 才能说明 scheduler 是否持续得到 ready work。

如果 issue 已经接近硬件允许的水平，某类 stall 占比高也未必限制性能，因为 scheduler 同时可以从其他 warp 发射。Nsight Compute 官方文档也明确提醒：只有 scheduler 未能每 cycle 发射时，才应聚焦 stall reason。[^ncu-scheduler]

### 第二步：哪个资源先到上限

在 Speed Of Light 和 Memory Workload Analysis 中区分：

- Compute pipeline 已高：先看指令类型、Tensor Core shape、精度或算法，ILP 往往不是答案。
- DRAM/L2 已高：先减少 byte 或提高复用，MLP 往往已经足够。
- Compute 和 Memory 都低：更像 latency、依赖、同步、小 grid 或指令供给问题，再进入 stall 分析。

### 第三步：把 Stall 定位回 Producer

| 观测组合                                              | 更可能的原因                                              | 首选实验                                             |
| ----------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------- |
| Eligible 低，`Wait` 高，计算 pipe 未满                | fixed-latency 计算依赖                                    | 2/4 个 accumulator、重排独立计算、提高 TLP           |
| Eligible 低，`Long Scoreboard` 高，带宽低，访问效率好 | global/local load-to-use 距离太短                         | 每线程 2/4 个独立 load、更多 warps、预取             |
| `Long Scoreboard` 高，sector 利用率差或 miss 异常     | 访问模式和 locality 问题                                  | coalescing、对齐、tiling、cache reuse                |
| `Short Scoreboard` 高                                 | 常见于 Shared Memory/MIO 依赖，也可能是特殊数学或动态分支 | 查 bank conflict、MIO source，不要直接当成寄存器依赖 |
| `LG Throttle` 高                                      | local/global 指令队列压力                                 | 减少冗余 load、处理 spill、合并成更宽指令            |
| `MIO Throttle` 高                                     | Shared Memory、特殊数学或分支等 MIO pipeline 压力         | 减少冲突和指令数、使用更宽 Shared load               |
| `Not Selected` 高且 issue 正常                        | eligible warp 已足够                                      | 可以尝试减少 TLP、把资源换给 ILP/reuse               |
| register 或 Shared Memory 限制 Occupancy              | 资源预算限制                                              | 同时扫描 unroll/tile/stage 与 active warps           |

`Long Scoreboard` 只说明 consumer 正在等某个 L1TEX producer，并不自动证明 DRAM，也不自动证明 MLP 不足。一次 L1 miss、local-memory spill、随机 gather 和错误的 load-to-use 调度都可能表现为 Long Scoreboard。需要在 Source View 找到 producer instruction，再结合 cache、transaction 和 local-memory 指标判断。

### 第四步：做参数扫描，而不是相信单点结果

对 ILP/MLP 优化至少扫描：

- `ITEMS/CHAINS = 1, 2, 4, 8`；
- 线程块大小，例如 128、256、512；
- 问题规模，覆盖小 grid、一个完整 wave 和大规模稳态；
- pipeline stage 数，例如 1、2、3、4；
- 对齐、stride、cache-hot/cache-cold 输入。

每组记录 kernel time、有效吞吐、registers/thread、active warps、eligible warps、主要 stall、DRAM/L2 throughput 和 local-memory traffic。性能曲线通常会先上升、平台，再因 spill、Occupancy 台阶或 queue throttle 回落；平台或拐点比某个“最佳展开因子”更能说明机制。

## 一张决策表：ILP、MLP 还是别的优化

| Kernel 现象                           | 应该优先做什么                          | 为什么                                                   |
| ------------------------------------- | --------------------------------------- | -------------------------------------------------------- |
| 串行 FMA/reduction 链，计算资源未满   | 提升 ILP                                | 独立 accumulator 能覆盖执行依赖延迟                      |
| 访存等待高、带宽低、访问已合并        | 提升 MLP                                | 缺少 in-flight requests，延迟直接暴露                    |
| 访存等待高、访问离散且浪费 sector     | 先修 coalescing/layout                  | 增加 MLP 会放大无效流量                                  |
| DRAM/L2 已饱和                        | 减少 bytes、提高 reuse、fusion/量化     | 并行度不能突破吞吐上限                                   |
| Math Pipe Throttle 高                 | 改 instruction mix/算法或接受上限       | 不是 ready instruction 不够，而是目标 pipe 已满          |
| LG/MIO Throttle 高                    | 减少或加宽内存指令，处理 spill/conflict | 请求入口已拥塞，不该继续灌请求                           |
| Occupancy 低但 issue 与目标 pipe 都高 | 不必追求更高 Occupancy                  | 当前 ILP/MLP 已足以隐藏延迟                              |
| Occupancy 低且 eligible warp 不足     | 在 ILP、MLP、TLP 间重分资源             | 需要增加 ready work，但要看等待类型                      |
| 单条 pointer chain                    | 并行多条 chain 或改算法/布局            | 数据依赖决定单链 MLP 无法超过 1                          |
| 小 shape、不到一个 wave               | batching、persistent/grouped kernel     | ILP/MLP 不能补齐完全缺失的设备级并行度与 launch 固定成本 |

## 几个常见误区

### `#pragma unroll` 等于提升 ILP

不等于。展开只扩大编译器看见的基本块；如果不同迭代仍依赖同一个 accumulator、pointer 或状态，依赖图没有变。必须查看独立变量、producer-consumer 距离以及生成的 SASS。

### `float4` 等于四倍 MLP

不等于。它通常是一条更宽的 load，不是四条独立 load。它可能增加 bytes per instruction、减少指令数并改善带宽利用，但 MLP 的 request count 未必增加。

### Coalescing 会提升 MLP

不一定。Coalescing 往往让一个 warp 的访问用更少 transaction 完成，物理请求数可能下降；它的价值是减少无效 byte 和队列压力。MLP 追求“足够多的有效请求”，不是请求越多越好。

### Occupancy 越高越好

Occupancy 只描述 resident warp 相对硬件上限的比例，不描述 eligible warp、issue rate、cache locality 或执行单元利用率。提高 ILP/MLP 常常增加寄存器或 Shared Memory，导致 Occupancy 下降；只要 ready work 和目标资源吞吐上升，这个交换就是合理的。

### Long Scoreboard 高就应该增加 MLP

Long Scoreboard 是症状，不是病因。先问四个问题：scheduler 是否真的漏发射、带宽是否已满、访问是否有效、producer 是 global load 还是 spill/local load。只有“请求有效、带宽未满、并发不足”这一支才直接指向增加 MLP。

## 总结

ILP 和 MLP 最好不要记成两份优化技巧列表，而要放回同一个依赖图中理解：

- **ILP**让一个 warp 在等待某条计算或访存指令时，还有其他独立指令可以发射。
- **MLP**专门描述未完成的独立内存操作；它可以来自单 warp 的多个 load，也可以来自更多 resident warps。
- **TLP、ILP、MLP 共同提供 latency hiding**，但它们争用寄存器、Shared Memory、cache 和队列容量。
- 提升 ILP 的核心是拆依赖链，常用多累加器、寄存器分块、thread coarsening、展开和软件流水。
- 提升 MLP 的核心是扩大 load-to-use 距离并增加有效在途请求，常用多元素预取、更多 warp、宽访问、批处理独立查询和异步多级流水。
- 当计算或带宽已经饱和时，并行度不再是答案；当请求队列已经 throttle 时，更多 MLP 甚至会更差。

最终的判断标准不是“用了几级 pipeline”或“Occupancy 有多高”，而是：**scheduler 是否持续拿到 ready instruction，critical resource 是否更接近有效上限，以及为此付出的寄存器、Shared Memory 和额外流量是否值得。**

## Reference

[^cuda-hardware]: [CUDA Programming Guide: Hardware Multithreading](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-multithreading)

[^cuda-ilp]: [CUDA Programming Guide: Hardware Implementation](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#hardware-implementation)

[^volkov]: Vasily Volkov, [Better Performance at Lower Occupancy](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf), GTC 2010. 文中的具体 GPU 数字属于当时的 G80/GF100/GF104；本文只迁移其“ILP 与 TLP 都能隐藏延迟”的方法论。

[^mlp-profiler]: [CUDA Profiler User's Guide 12.9: Warp State](https://docs.nvidia.com/cuda/archive/12.9.0/profiler-users-guide/index.html#warp-state). 这是 legacy profiler 的定义；当前 kernel profiling 应使用 Nsight Compute。

[^ncu-scheduler]: [Nsight Compute Profiling Guide: Scheduler Statistics](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sets-and-sections)

[^ncu-stalls]: [Nsight Compute Profiling Guide: Warp Stall Reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons)

[^ncu-throttle]: [Nsight Compute Profiling Guide: LG/MIO Throttle](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#warp-stall-reasons)

[^cuembed]: NVIDIA, [Accelerating Embedding Lookups with cuEmbed](https://developer.nvidia.com/blog/accelerating-embedding-lookups-with-cuembed/)；对应实现见 [NVIDIA/cuEmbed](https://github.com/NVIDIA/cuEmbed)。

[^registers]: [CUDA C++ Best Practices Guide: Registers](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#registers)

[^async-copy]: [CUDA C++ Best Practices Guide: Asynchronous Copy from Global Memory to Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#asynchronous-copy-from-global-memory-to-shared-memory)

[^cuda-pipeline]: [CUDA Programming Guide: Pipelines](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html)

[^cutlass-gemm]: NVIDIA CUTLASS, [Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html). 文档给出了 CTA/warp/thread 的分层 tiling、寄存器外积，以及 Shared Memory tile 与 register fragment 的双缓冲。

[^cublas-gemv]: NVIDIA, [cuBLAS: `cublas<t>gemv()`](https://docs.nvidia.com/cuda/cublas/#cublas-t-gemv). 算术强度上界是依据该运算定义，在理想最小 HBM 流量假设下推导得到。

[^flashattention]: Tri Dao et al., [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), NeurIPS 2022.

[^flashattention2]: Tri Dao, [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691), ICLR 2024.

[^flashattention3]: Jay Shah et al., [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608), NeurIPS 2024.

[^flashdecoding]: Tri Dao et al., [Flash-Decoding for Long-Context Inference](https://princeton-nlp.github.io/flash-decoding/), 2023. 该方案沿 KV sequence 拆分并行工作，再通过额外归约合并各 split 的输出与 log-sum-exp。
