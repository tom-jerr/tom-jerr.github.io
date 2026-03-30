---
title: torch.compile 原理和实践
tags:
  - LLMInference
created: 2025-12-12
description: 介绍 torch.compile 的原理和在大模型推理中的实践应用
cover: /img/torch_compile.png
---

# torch.compile 原理和实践
```shell
Python eager 代码
→ TorchDynamo 截获 Python frame / 字节码
→ 抽取 FX Graph + 记录 guards(这份编译结果成立的前提条件)
→ AOTAutograd（训练场景）做前反向图分解
→ TorchInductor 做图级优化、fusion、调度、代码生成
→ 后端落到 Triton / CPP / ATen 等
→ Triton 再把 kernel 编译成设备可执行代码
→ 运行时按 guards 检查是否可复用旧编译结果
```
- Dynamo 更像一个 Python 层图抽取器 + 特化入口，不是最后生成 GPU kernel 的地方。官方文档也明确说：每个 frame 都会尝试编译，并把**编译结果缓存在 code object 上；如果后续调用不满足之前的 guards，就会发生 guard failure，然后重新编译**

## compile cache
### 第一层：Dynamo 的 frame / guard 级缓存

缓存的是：
- 某段 Python frame 对应的已编译结果
- 它配套的 guards

只要这次调用还满足旧 guards，就直接复用，不重新抓图/编译。官方文档明确说它会把编译结果缓存在 code object 上，并在 guard failure 时重新编译。

### 第二层：Inductor 的图级缓存

缓存的是：

- FX Graph / 规范化后的图表示
- 对应生成过的后端代码
- 一些中间 IR 或编译结果索引

PyTorch 官方 compile caching recipe 里明确提到，默认有本地磁盘缓存，里面包括 FXGraphCache 等模块化缓存。

### 第三层：Triton kernel 编译缓存

缓存的是：

- Triton kernel 源/IR 对应的设备代码
- 和 launch 相关的元数据
- autotune 结果或编译 key 相关信息

这一层更接近“传统 JIT kernel cache”。
## 原理
`torch.compile` 是 PyTorch 提供的一个用于加速模型推理和训练的工具。它通过将 PyTorch 代码转换为更高效的中间表示（IR），并利用各种优化技术来提升性能。其核心原理包括以下几个方面：
- Dynamo: 拦截 Python 执行代码，提取 tensor op 构成静态 IR 图，跳过 Python 解释器
- Inductor: 算子融合 + kernel 合并 + 内存优化
- CUDAGraph: 减少 CPU 调度开销，一次录制多次执行
