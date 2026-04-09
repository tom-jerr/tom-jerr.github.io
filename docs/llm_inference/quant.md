---
title: Quantization in LLM Inference
created: 2026-03-12
updated: 
tags:
  - LLMInference
description: 
# cover: /img/parallelism_in_training.png
---

# Quantization in LLM Inference
- 对称量化：认为数值分布大致以 0 为中心
  - 量化：$q = round(\frac{x}{s})$
  - 反量化：$\hat{x} = q \cdot s$
  - 其中 $s$ 是 scale，表示量化单位的大小，通常根据数值范围和量化位数 $b$ 计算得到：
    - $s = \frac{\alpha}{2^b - 1}$
    - $\alpha$ 是数值范围的绝对值上界，例如 $\alpha = \max(|x_{min}|, |x_{max}|)$

- 非对称量化：允许数值分布不以 0 为中心，因此需要额外的偏移量（zero-point）来表示 0 的位置
  - 量化：$q = round(\frac{x}{s}) + z$
  - 反量化：$\hat{x} = (q - z) \cdot s$
  - 其中 $z$ 是 zero-point，表示量化后数值为 0 的位置，通常根据数值范围和量化位数 $b$ 计算得到：
    - 若浮点范围是 $[x_{\min}, x_{\max}]$，整数范围是 $[q_{\min}, q_{\max}]$，例如 uint8 的 [0,255]，常见做法：
      - $s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$
      - $z = round(\frac{-x_{\min}}{s} + q_{\min})$
## AWQ
不进行回归或重建，仅通过统计校准集中的激活值来选择关键权重并进行缩放。这种基于激活分布的策略不会直接修改模型权重，因此可以更好地保留模型的原始特性和知识。

AWQ 具体方法：

- 通过激活分布（按大小）选择 1% 关键权重
- 通过缩放关键权重降低量化误差
- 量化感知训练（QAT）：需要重新训练模型，计算成本极高，不适用于大规模LLMs
- 后训练量化（PTQ）：无需重新训练，但在低比特（如4-bit）情况下会显著降低模型性能。

## GPTQ
GPTQ 做 group-wise quantization 时，量化误差和每个输入通道的重要性有关，所以它会先把更重要的通道排到前面，再分组量化
```cpp
C = A[M, K] @ W[K, N]
```
- 一旦这样做了，W 的 K 维顺序就不再是原顺序了。为了让数学结果不变，推理时 A 的列也必须按同样顺序重排。这就是我说的 “A 的列重排”。

## Marlin 格式

Marlin 是 kernel 期望的一套预排布布局，核心是把量化后的 tile、bit-pack 顺序和 scale/zp 顺序改成 kernel 访问友好的形式
- 权重按固定 tile 组织，tile_k=16，tile_n=64。
- 在每个 16x64 tile 内，再做一层针对 Tensor Core / warp 访存模式的置换和 bit-pack。
- scales 和 zero-points 也要按同样的 tile 访问顺序重排，否则反量化时会对错组、错列

AWQ checkpoint 进来先要进行 repack 重排
- 先把 AWQ 的列打包/interleave 解释回来。
- 再按 16x64 tile 重排。
- 再按 Marlin kernel 需要的顺序重新 pack 成 int32

### 计算
- 每次通过异步指令将 GEMM 需要的 tile  的权重从 HBM 加载进来反量化后做 MMA（Tensor Core Matmul）
> [!IMPORTANT]
>  这里不会生成完整的反量化权重矩阵，而是每次只反量化当前 tile 需要的部分，节省显存带宽和中间存储。
权重已经预排成 Marlin 的 16x64 tile 布局，kernel 可以顺着 Tensor Core 访存模式直接读，不需要运行时转置/散读。marlin.cuh (line 30)
用 cp.async 做全局内存到 shared memory 的流水线预取，加载和计算重叠。marlin_template.h (line 712)
只在 tile 进入计算时做局部反量化，没有生成完整 fp16 权重矩阵，省显存带宽和中间存储。
scale 和 zero-point 只在“进入新 group”时加载，不重复搬运。