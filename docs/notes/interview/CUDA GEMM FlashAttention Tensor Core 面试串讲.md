---
title: CUDA GEMM FlashAttention Tensor Core 面试串讲
created: 2026-03-07
updated: 2026-03-07
tags:
  - CUDA
  - LLMInference
  - Interview
description: 面向 CUDA、GEMM、Tensor Core、FlashAttention、Flash Decoding 的面试串讲稿，重点整理 decode 场景下的 split-k 与 split-q 方法。
---

# CUDA GEMM FlashAttention Tensor Core 面试串讲

## 这篇怎么用

这篇文档不是论文笔记，也不是只列八股名词，而是按面试官常见追问顺序，把这条链一次讲顺：

`CUDA -> GEMM -> Tensor Core -> FlashAttention -> Flash Decoding -> split-q / split-k`

如果你只能记住一条主线，那就是：

> FlashAttention 本质上是在做“像 GEMM 一样分块、像 online softmax 一样递推、像高性能 CUDA kernel 一样管理 shared memory / register / warp / Tensor Core”的 attention；而 Flash Decoding 则是在 decode 场景下，把原来几乎没有 query 并行度的问题，改写成对 KV 维度做 split-k 并在最后 merge attention states 的问题。

## 一、总起：为什么 CUDA / GEMM / FlashAttention / Tensor Core 经常被一起问

因为这四件事本来就是一条技术链：

- CUDA 负责给你线程、warp、shared memory、寄存器、同步这些基础抽象。
- GEMM 是 GPU 上最成熟的高性能模板，几乎所有 kernel 设计都在借鉴它的分层分块方法。
- Tensor Core 是现代 NVIDIA GPU 上最强的矩阵乘硬件单元，高性能 GEMM 基本都围绕它设计。
- FlashAttention 则是在 attention 上尽量逼近 GEMM 的效率，把 attention 里多余的 HBM 读写去掉，再把真正贵的部分变成更接近 GEMM 的块级计算。

所以当面试官从 CUDA 一路问到 FlashAttention，本质上是在确认你是否真的理解：

1. GPU 为什么喜欢规整、分块、可复用的计算。
2. attention 为什么比 GEMM 更难打满。
3. decode 阶段为什么又比 prefill 更难。

## 二、面试开场版：3 分钟串讲

可以直接按下面这版答。

### 2.1 开场回答

CUDA 提供的是 grid、block、thread 的分层并行模型，硬件真正按 warp 调度执行。高性能 kernel 的核心目标，是把 global memory 的访问次数降下来，把热点数据留在 shared memory 和寄存器里，并让 warp 有足够多的 ready work 来隐藏延迟。

GEMM 是这个思路最标准的模板。高性能 GEMM 会把计算拆成 threadblock tile、warp tile、thread tile，让 A/B 子块先搬到 shared memory，再搬到寄存器，最后在寄存器里累计 C 的局部结果。Tensor Core 进一步把 warp 级矩阵乘做成专用 MMA 指令，因此只要你的数据布局、tile 设计和访存路径够好，就能逼近硬件峰值。

FlashAttention 本质上是在 attention 上复用 GEMM 的分块思想。传统 attention 慢，不只是算得多，更是因为会把中间的 score / probability 矩阵反复写回 HBM。FlashAttention 用块级 Q/K/V 计算和 online softmax，把完整 attention matrix 的 materialization 去掉，于是显著减少 HBM 往返。FlashAttention-2 又继续优化了两个点：一是减少非 matmul FLOPs，二是改进 threadblock 和 warp 内的 work partition，使 attention 更接近 GEMM 的执行效率。

但 decode 场景和 prefill 不一样。decode 时 query 长度通常是 1，这意味着沿 Q 维度几乎没有并行度，FlashAttention 原本依赖的 split-q 路径不再能把 SM 填满，所以会退化。Flash Decoding 的核心就是在 decode 场景下改成 split-k：把 KV cache 沿 sequence dimension 切块，让多个 CTA 并行处理不同的 KV chunk，各自产生 partial attention state，再通过 log-sum-exp / attention state merge 做最终规约。这样即使 batch 很小、q_len=1，也能把更多 SM 利用起来。

一句话总结：

> prefill 更像“attention 版 GEMM”，主优化是 split-q 和块内重用；decode 更像“memory-bound 的长 K 规约问题”，主优化是 split-k 和最后的 state merge。

## 三、主线一：从 CUDA 到 GEMM

### 3.1 为什么 GEMM 是所有 CUDA 面试的母题

因为 GEMM 把 CUDA 优化里最重要的几个点都一次性包含了：

- global memory coalescing
- shared memory staging
- register blocking
- warp-level work partition
- occupancy 与寄存器/SMEM 的平衡
- Tensor Core / WMMA / MMA 指令
- bank conflict 规避
- double buffer / async copy

所以面试官问你 FlashAttention，常常不是想听论文摘要，而是想听你能不能把它还原成一个“特殊形态的 GEMM-like kernel”。

### 3.2 GEMM 的面试标准答法

高性能 GEMM 的常见分层是：

- **threadblock tile**：一个 CTA 负责 C 的一个大块。
- **warp tile**：一个 warp 负责 CTA 内更小的一块。
- **thread tile**：单线程在寄存器里累计更小的输出微块。
- **K-tile**：沿 K 维度循环分块，重复加载 A/B 子块。

这背后的资源映射是：

- global memory 负责存大矩阵。
- shared memory 负责 block 内复用 A/B 子块。
- register 负责线程私有的 accumulator 与 fragment。
- warp 负责发起 Tensor Core MMA 或标量/向量 FMA。

**高频追问：GEMM 为什么快？**

标准答案：

> 因为它通过分层分块把 A/B 的复用做到了 shared memory 和 register 两层，并把最贵的 global memory 访问压缩到了 tile 级，同时让计算部分尽量映射到 Tensor Core 的 MMA 指令，所以可以接近硬件峰值。

### 3.3 Tensor Core 在这里扮演什么角色

Tensor Core 是 warp 级矩阵乘加专用单元。你可以通过：

- WMMA API
- 更底层的 `mma.sync` / `ldmatrix`
- CUTLASS / cuBLAS / cuDNN 等库

去用它。

面试里最稳妥的说法是：

> Tensor Core 不是“自动加速所有矩阵运算”的黑盒，它要求你把数据布局、tile 形状、对齐和 warp 分工组织成适合 MMA 指令的形式。高性能 GEMM 和高性能 attention kernel 的很多复杂性，本质都来自“如何把问题改写成 Tensor Core 喜欢的形状”。

## 四、主线二：为什么 FlashAttention 接近 GEMM，但又不等于 GEMM

### 4.1 传统 attention 为什么慢

传统 attention 的计算图是：

`S = QK^T -> P = softmax(S) -> O = PV`

问题在于：

- `S` 很大，通常是 `Lq x Lk`。
- `P` 也很大。
- 如果你把 `S` 和 `P` 都显式写到 HBM，再读回来做 softmax 和乘 V，就会有大量不必要的 global memory 读写。

FlashAttention 的核心不是改变数学结果，而是改变 **I/O 过程**。它把 attention 写成块级递推，不显式 materialize 全量 `S` 和 `P`，而是只维护每个 query block 的 online softmax 统计量和 partial output。

### 4.2 为什么说 FlashAttention 是 IO-aware 的

因为它优化的第一目标不是减少 FLOPs，而是减少 HBM traffic。它抓住的关键事实是：

- attention 常常是 memory-bound，不是纯 compute-bound。
- HBM 往返代价远高于在 shared memory / register 里多做一点算术。

所以 FlashAttention 用更多块内计算，换更少全局存取。

### 4.3 FlashAttention 和 GEMM 的相同点

相同点：

- 都是 tile-based。
- 都高度依赖 shared memory 和 register 复用。
- 都围绕 warp / threadblock 层级做 work partition。
- 都尽量把主要算力映射到 Tensor Core。

### 4.4 FlashAttention 和 GEMM 的不同点

不同点：

- GEMM 的 reduction 更规则，直接沿 K 维度累加即可。
- attention 中间夹着 softmax，这是一个带归一化和数值稳定要求的非线性步骤。
- 所以 attention 不能简单地当成一个纯 GEMM；它必须额外维护 `row max`、`row sum`、`lse` 这类统计量。

一句话概括：

> FlashAttention 的难点不在“矩阵乘”本身，而在“怎么在分块的情况下把 softmax 正确地拼回来”。

## 五、主线三：FlashAttention-2 为什么更像“工程化的高性能 CUDA kernel”

FlashAttention-2 的论文和实现强调三件事：

1. 减少 non-matmul FLOPs。
2. 跨 threadblock 增加并行度。
3. 在 block 内重新分配 warp 工作，减少 warp 间 shared memory 通信。

面试里可以直接说：

> FlashAttention-1 已经证明了 online softmax + tile attention 的正确性，FlashAttention-2 主要是在 GPU work partition 上继续逼近 GEMM：让更多 threadblock 同时工作，让 block 内 warp 少做同步和 shared memory 往返，让 attention 的执行形态更接近 Tensor Core 友好的矩阵乘。

这也是为什么论文会强调它已经能到 A100 上理论峰值 FLOPs 的 50%~73%，开始接近 GEMM 的效率区间。

## 六、关键转折：为什么 decode 场景让 FlashAttention 失效一半

### 6.1 prefill 和 decode 的根本差异

- **prefill**：`q_len` 大，`kv_len` 也大。Q 维度和 sequence 维度都有可观并行度。
- **decode**：通常 `q_len = 1`，但 `kv_len` 很长。每一步只算一行 attention。

这会带来一个直接后果：

> 你沿 Q 维度几乎没有 tile 可以切了。

而经典 FlashAttention 的 threadblock 并行，很大程度依赖 query blocks。如果 `q_len=1` 且 batch 也不大，那么 GPU 上大量 SM 根本吃不到活。

### 6.2 为什么 decode attention 常常是 IO-bound

因为 decode 时每个新 token 都要读完整的 KV cache，但计算量并没有按同样比例增加。于是：

- operational intensity 很低
- kernel 常被内存带宽卡住
- 如果 batch 小，还会出现 SM 利用率很低的问题

所以 decode attention 的优化目标通常不是“多做一点算术”，而是：

1. 把更多 SM 调动起来。
2. 尽量让读 KV 的过程高效、连续、并发。
3. 把 partial result 的合并代价控制住。

## 七、flash decoding：核心思想到底是什么

### 7.1 面试版定义

Flash Decoding 的本质是：

> 在 decode 场景下，由于 `q_len` 太小，无法像 prefill 那样靠 split-q 填满 GPU，于是改为把长 KV cache 沿 sequence dimension 切成多个 chunk，多个 CTA 并行计算各自 chunk 的 partial attention，再在最后通过 log-sum-exp 相关统计量把这些 partial result 正确合并。

### 7.2 为什么它需要“两阶段”

因为每个 KV chunk 上都只能得到局部 attention 结果。局部 softmax 不能直接拼成全局 softmax，所以需要额外保存：

- partial output
- partial log-sum-exp / lse

最后再做一次 merge / reduce，把所有 chunk 的结果按正确缩放系数组合起来。

Together AI 的 Flash-Decoding 文章把它拆成三步：

1. split KV into chunks
2. 对每个 chunk 并行做局部 attention，并输出 lse
3. 用 lse 做最终规约，得到完整输出

### 7.3 这和 FlashInfer 的 recursive attention 是什么关系

FlashInfer 的 attention states / recursive attention，本质上是在给 split-k / split-kv 这类方法一个更通用的数学抽象：

- 一个 attention state 可以写成 `(s(I), v(I))`
- 其中 `s(I)` 是某个子集上的 log-sum-exp，`v(I)` 是对应 attention 输出
- 不同子集的 state 可以通过一个可交换、可结合的 merge operator 合并

这件事的工程含义很重要：

> 你可以把不同 KV chunk 的 attention 分到不同 CTA、不同 kernel，甚至不同设备上算，只要最后按正确的 state merge 规则合并，结果就是正确的。

## 八、split-k 和 split-q 到底分别是什么

这是你这次最需要讲透的部分。

### 8.1 split-k 的来源

split-k 最早是 GEMM 里常见的技巧：

- 当 M/N 维度并行度不够时，把 reduction 维 K 切开。
- 多个 CTA 分别计算不同 K chunk 的 partial sum。
- 最后再做规约。

decode attention 借用了同样的想法，只不过这里的 “K” 不再是 GEMM 中矩阵乘的 K，而更像 **attention 中沿 KV sequence 的 reduction 维**。

所以在 LLM serving 里经常会把它叫：

- split-k
- split-kv
- split over sequence dimension

这三个说法在 decode attention 语境里基本是在讲同一类事情。

### 8.2 split-k 在 decode 里为什么有效

因为 decode 时问题通常是：

- `q_len=1`
- batch 小
- `kv_len` 很大

也就是说：

- 沿 Q 维度没什么可切的。
- 沿 batch/head 维度也可能不够。
- 唯一够长的维度是 KV sequence。

于是把 KV cache 切成多个 chunk，让多个 CTA 并行读取不同 chunk，就能显著提高 SM 利用率。

### 8.3 split-k 的代价是什么

代价是最后必须 merge partial results。你不能把每个 chunk 内 softmax 出来的结果直接相加，因为各 chunk 的归一化分母不同。

因此 split-k 一定会引入额外的：

- partial lse 写出
- partial output 写出
- final reduction / merge kernel

所以面试里不能把它讲成“白送的并行度”，正确说法是：

> split-k 用额外的 reduction 成本换取更多 CTA 并行度，是否划算取决于 q_len、kv_len、batch size、SM 数量和带宽/算力平衡点。

### 8.4 split-q 是什么

split-q 就是把 query 维度切开，不同 CTA 或不同 warp 去处理不同 query rows / query blocks。

它在下面这些场景最自然：

- prefill
- training attention
- append attention（当 q_len 不小）
- speculative decoding 中 target 一次验证多个 token

因为这些场景里 `q_len` 不再是 1，而是几十、几百，甚至更大，Q 维度本身就能提供足够并行度。

### 8.5 为什么 prefill 更像 split-q 问题

prefill 时你通常有很多 query rows，要和同一段 KV 做 attention。最自然的并行方式就是：

- 不同 CTA / block 负责不同 query blocks
- 每个 block 内再沿 head_dim / tile 维度配合 Tensor Core 做块乘

所以 FlashAttention / FlashAttention-2 的主并行方向，本质上是更偏 split-q 的。

### 8.6 为什么 decode 里 split-q 常常不成立

decode 的标准场景是 `q_len=1`。这意味着：

- 你几乎没有 query blocks 可切。
- 就算切，也只会产生很少 CTA。
- GPU 大部分 SM 还是会空着。

所以在经典单 token decode 中，split-q 不是主解法，split-k 才是。

### 8.7 什么时候 decode 里 split-q 又重新有意义

有两类典型情况：

1. **append attention / chunked append**
   这时 q_len 可能是 64、128、256，而 kv_len 更长。此时 query 维度已经足够大，split-q 会重新变得有效。

2. **speculative decoding / verification**
   target model 一次验证多个 draft tokens，相当于 q_seq_len > 1。这个时候 verify kernel 更像 append/prefill，而不像单 token decode，因此又可以用 split-q 或 Q 维度并行。

这也是为什么很多 serving kernel 会按阶段区分：

- `decode kernel`: 偏 split-k
- `append/prefill kernel`: 偏 split-q

### 8.8 一句话比较 split-q 和 split-k

可以直接背下面这句：

> split-q 适合 query 维度足够大、计算更接近 prefill 的场景；split-k 适合 query 很短但 KV 很长、需要把 reduction 维拆开以榨干 GPU 的 decode 场景。

## 九、面试高频追问

### 9.1 为什么说 FlashAttention 在 prefill 更有效，而 Flash-Decoding 在 decode 更有效？

**答案：**

因为两阶段的瓶颈不同：

- prefill 有较大的 q_len，本身就有 query 维度并行度，FlashAttention 的 split-q / tile attention 能很好工作。
- decode 的 q_len 通常是 1，FlashAttention 原有的 query 并行路径几乎没有空间，导致 GPU 利用率很低。这时必须转向 split-k，把并行度从 KV sequence 维度里找出来。

### 9.2 split-k 的数学正确性怎么保证？

**答案：**

靠的不是“局部 softmax 直接相加”，而是维护每个 chunk 的 attention state，通常是局部输出和局部 lse，然后通过可交换、可结合的 merge operator 合并。Flash-Decoding 和 FlashInfer recursive attention 本质上都在做这件事。

### 9.3 split-k 会不会破坏数值稳定性？

**答案：**

如果只是生硬地把 chunk 输出相加，当然会错；但如果用 log-sum-exp / online softmax 的方式保存局部统计量，再用正确缩放系数 merge，数学上是等价的。真正的问题更多是工程上的：

- merge kernel 开销
- 非确定性规约顺序
- CUDA Graph 下 chunk 数变化带来的 shape 不稳定

FlashInfer 文档甚至专门给了 `disable_split_kv` 选项，用于在一些图捕获/确定性场景下禁用 split-kv。

### 9.4 split-k 一定更快吗？

**答案：**

不是。只有在“原始并行度不够、KV 足够长、最终 reduction 成本相对可接受”的时候才更快。对某些带宽较小但 CUDA Core 较强的卡，split-k 收益可能没那么明显。FlashInfer 的公开 benchmark 也明确提到过这点。

### 9.5 Tensor Core 在 FlashAttention / Flash-Decoding 里什么时候最有价值？

**答案：**

当 query block 足够大、head_dim 和数据类型适合 MMA 指令、tile 形状够规整时最有价值。prefill 和 append 更容易把大块 matmul 映射到 Tensor Core；decode 单 token 时，由于 q_len 太小，光靠 Tensor Core 不一定能解决并行度不足的问题，所以还要靠 split-k 把更多 CTA 调起来。

### 9.6 GQA / MQA 会如何改变 split-k 与 split-q 的讨论？

**答案：**

GQA/MQA 降低了 KV heads 数量，减少了 KV cache 读取流量，会改变 attention 的 operational intensity。某些实现里，decode attention 在 GQA 下反而更接近 compute-bound，此时 Tensor Core 利用会变得更重要，FlashInfer 也明确提到过在 GQA 场景下会用更偏 prefill-style 的 Tensor Core kernel 来做 decode attention。

## 十、你可以直接背的收口答案

### 10.1 面试 2 分钟版

CUDA 优化的核心是合理组织 block、warp、shared memory 和寄存器，让 global memory 访问更少、复用更高。GEMM 是最标准的模板：threadblock tile、warp tile、thread tile，再配合 Tensor Core 把矩阵乘做成 warp 级 MMA。FlashAttention 本质上是在 attention 上借鉴 GEMM 的分块思想，用 online softmax 避免把完整 score/probability 矩阵写回 HBM，所以它是 IO-aware 的 exact attention。FlashAttention-2 又进一步改进 threadblock 和 warp 的 work partition，让 attention 更接近 GEMM 的效率。

但 decode 场景下 q_len 通常是 1，这时沿 query 维度几乎没有并行度，原本偏 split-q 的 FlashAttention 路径没法把 GPU 填满，所以 attention 会退化成一个低并行、强 IO-bound 的 kernel。Flash Decoding 的核心就是改做 split-k：把 KV cache 沿 sequence 维度切块，让多个 CTA 并行算不同 chunk 的 partial attention，再通过 log-sum-exp / attention states merge 得到最终结果。简单说，prefill 更像 split-q 问题，decode 更像 split-k 问题；append 和 speculative verify 则介于两者之间。

### 10.2 面试 30 秒版

GEMM 是 CUDA 高性能 kernel 的模板，Tensor Core 是它的核心算力单元。FlashAttention 本质上是在 attention 上做 GEMM 式分块和 online softmax，避免中间矩阵写回 HBM，所以能显著降 I/O。FlashAttention-2 继续优化了 threadblock 和 warp 的 work partition。decode 时由于 q_len=1，split-q 不再够用，GPU 利用率会掉下来，所以 Flash Decoding 改成 split-k：把长 KV cache 切块并行算，再用 lse / attention state 做最终规约。prefill 主看 split-q，decode 主看 split-k。

## 十一、仓库内的对应文件

| 主题 | 文件 | 为什么看它 |
| --- | --- | --- |
| CUDA 线程体系、存储体系、SM、Warp、Occupancy | `docs/notes/cuda/cudaopt.md` | 串讲的底座 |
| GEMM 分层分块、bank conflict、寄存器/SMEM 权衡、WMMA | `docs/notes/cuda/从GEMM实践CUDA优化.md` | GEMM 和 Tensor Core 面试主线 |
| 带宽利用、latency hiding、ILP/DLP、async copy | `docs/notes/cuda/GPU 内存系统演进：最大化带宽利用与延迟隐藏的技术路径.md` | 为什么 attention/decode 会 memory-bound |
| 向量化访存、Roofline、memory-bound kernel 基础 | `docs/notes/cuda/Vector Add Optimization Example.md` | 理解 decode 的 bandwidth ceiling |
| FlashAttention v1/v2 原理 | `docs/blogs/posts/FlashAttention 原理 v1-v2.md` | FlashAttention 主体逻辑 |
| attention 形态、decode 图示 | `docs/notes/llm_inference/attention&transformer.md` | 补 attention 背景 |
| speculative / append / verify 场景 | `docs/feishu/Eagle2 in SGLang.md` | 为什么 q_seq_len>1 时 split-q 又回来 |

## 参考资料

### 官方文档 / 官方项目 / 一线开源文档

- CUDA C++ Programming Guide: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- CUDA C++ Best Practices Guide: <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- CUTLASS Overview: <https://docs.nvidia.com/cutlass/latest/overview.html>
- NVIDIA CUTLASS blog: <https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/>
- NVIDIA Tensor Core / WMMA blog: <https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/>
- NVIDIA FlashAttention-3 blog: <https://developer.nvidia.com/blog/next-generation-of-flashattention/>
- FlashInfer docs: <https://docs.flashinfer.ai/api/attention.html>
- FlashInfer recursive attention tutorial: <https://docs.flashinfer.ai/tutorials/recursive_attention.html>
- FlashInfer blog: <https://flashinfer.ai/2024/02/02/introduce-flashinfer.html>
- FlashAttention official repo: <https://github.com/Dao-AILab/flash-attention>

### 论文 / 技术文章

- FlashAttention paper: <https://arxiv.org/abs/2205.14135>
- FlashAttention-2 paper: <https://arxiv.org/abs/2307.08691>
- Flash-Decoding blog: <https://www.together.ai/blog/flash-decoding-for-long-context-inference>
- FlashInfer paper: <https://arxiv.org/abs/2501.01005>

## 建议放在哪个文件

这篇串讲稿适合放在：

- `docs/notes/cuda/CUDA GEMM FlashAttention Tensor Core 面试串讲.md`

它和已有文件的分工建议如下：

- `docs/notes/cuda/cudaopt.md`：CUDA 基础与硬件/资源模型。
- `docs/notes/cuda/从GEMM实践CUDA优化.md`：GEMM 代码级实践。
- `docs/blogs/posts/FlashAttention 原理 v1-v2.md`：FlashAttention 算法细节。
- `docs/notes/cuda/CUDA GEMM FlashAttention Tensor Core 面试串讲.md`：面试口述主线，负责把前三者串起来。
