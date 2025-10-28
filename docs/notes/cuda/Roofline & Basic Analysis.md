---
title: Roofline & Basic Analysis
date: 2025/10/20 23:15
tags:
  - LLMInference
---

# Roofline & Basic Analysis

## AI

## Roofline

Roofline 模型（屋顶线模型）是一种用来**分析程序性能瓶颈**（计算受限还是带宽受限）的方法。  
它把**计算性能**（FLOPs/s）和**访存性能**（Bytes/s）联系在一起，以可视化的方式展示性能上限。

$$
Achievable FLOPs=min(AI×Memory BW,Peak FLOPs)
$$

### 以 vector add 为例

- 最简单的累加方式，每个 thread 负责一个线程的计算

```cpp
// FP32
// ElementWise Add grid(N/256),
// block(256) a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void vector_add_kernel(const float *a, const float *b, float *c,
                                  int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}
```

| 指标           | 数值             |
| -------------- | ---------------- |
| 每元素 FLOPs   | 1                |
| 每元素 Bytes   | 12               |
| AI             | 0.083 FLOPs/Byte |
| Peak Bandwidth | 1008 GB/s        |
| Peak Compute   | 82.6 TFLOPs/s    |

```shell
性能 (GFLOPs/s)
↑
|                   ────────────────  ← 82.6 TFLOPs/s (平顶线)
|                  /
|                 /
|    o           /
|   (VectorAdd) /
|______________/__________________→ AI (FLOPs/Byte)
                 0.083

```

- 左下角：访存主导（memory-bound）
- 右上角：计算主导（compute-bound）
- 中间交点：分界点（称为 **ridge point**）
