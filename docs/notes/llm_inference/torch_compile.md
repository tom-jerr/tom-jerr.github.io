---
title: torch.compile 原理和实践
tags:
  - LLMInference
created: 2025-12-12
---

# torch.compile 原理和实践

## 原理
`torch.compile` 是 PyTorch 提供的一个用于加速模型推理和训练的工具。它通过将 PyTorch 代码转换为更高效的中间表示（IR），并利用各种优化技术来提升性能。其核心原理包括以下几个方面：
- Dynamo: 拦截 Python 执行代码，提取 tensor op 构成静态 IR 图，跳过 Python 解释器
- Inductor: 算子融合 + kernel 合并 + 内存优化
- CUDAGraph: 减少 CPU 调度开销，一次录制多次执行
