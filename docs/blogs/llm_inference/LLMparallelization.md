---
title: Parallelization in LLM Inference
date: 2025/9/28
update:
comments: true
katex: true
tags:
  - LLM Inference
---

# Parallelization in LLM Inference

之前我们通过 cse234 课程的视角介绍了 MlSys 领域的并行化的基本概念，本文档将结合 LLM 推理的特点，介绍现在业界常用的并行化技术和策略。现在 LLM 基本是 Transformer-based 的自回归预训练模型，本文档中我们将主要介绍这种模型推理中的并行化技术。

> 我们这次通过 Transformer 中使用的各个模块的并行化来介绍 LLM 推理中的并行化技术。

## MatMul Parallelism

在 A x B x C 矩阵乘法场景，B 矩阵列切， C 矩阵行切。

- 计算量减半，多了一次集群通信（allReduce）中间值的存储大小减半，Input, Weight 减半

![](img/llm_dp.jpg)

## 线性层并行

线性层的并行运算，包括层内切分、层间切分以及参数冗余消除，并行方式包括了数据并行（DP）、序列并行(SP)、张量并行（TP）、层并行（PP）、参数冗余消除（Zero）

### 层内并行

- DP: batch size 切分
- SP: seq len 切分
- TP: hidden size 切分
  $$ activation size:[batch\_size, seq\_len, hidden\_size]$$

SP 和 TP 通常一起使用，因为 **TP 仅仅切分了 weights**，而 input 没有切分，GPU 上仍然保存完整的 X 的 activation，而 **TP 则切分了 activation**

**Example**
megatron-v3 中使用 SP+TP 来减小 activation 的大小，TP 的计算过程类似 MatMul Parallelism 的描述

- 因为 GeLU 非线性，必须使用 allGather 来收集 activation
  ![](img/sq.png)

### 层间并行

Pipeline Parallelism，Expert Parallelism，Attention-FFN Decoupling

现在的普遍做法是 AFD(Attention-FFN Decoupling)，这是一种**层间异构并行策略**，把 Transformer 里的 Attention 和 MoE-FFN 拆开到不同设备上执行，以适配它们算力/显存需求的差异，从而提高推理吞吐和硬件利用率，MoE 的不同专家也分配到不同的 GPU 上

- Attention 是 Matmul+softmax+activation，通常是访存瓶颈（memory access bound）
- FFN 是占用显存大，算力要求高，通常表现为计算瓶颈（compute bound）
  ![](img/afd.png)

### 冗余参数消除

一般使用 ZeRO，将参数分散存储
模型参数较小时，一般选择参数广播

![](img/pbroadcast.jpg)
参数较大时，将每一层的权重 n 等分，每个 GPU 设备上面存一份，当需计算时将其 allgather 回来。
![](img/pallgather.png)

## Attention Parallelism

- DP: batch size
- TP: heads & $d_k$
- SP/CP(context parallelism): seq_len
  $$attention \space size: [batch\_size, heads, seq\_len, d_k]$$
  Attention 的层间并行、冗余参数消除方式与线性层的方式一致，**层内并行的主要差异是 TP 和 SP**

### Sequence Parallelsim
