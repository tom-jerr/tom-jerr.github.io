**From 11-23 to 12-23**

## 📅 第 1 周：建立“Roofline”思维与内存这一生之敌

**核心目标：** 学会 Roofline Model，学会用 NCU 看带宽利用率。

### 📚 理论学习 (Theory)

- [ ] **阅读 PMPP/官方文档：** 深入理解 Memory Hierarchy (Global -> **L2 Cache** -> Shared Memory)。

- [ ] **理解 KV Cache：** 搞清楚 L2 Cache 在 LLM 推理中对 KV Cache 的作用。

- [ ] **死磕 Coalesced Access：** 彻底理解合并访问机制。

- [ ] **掌握 Roofline Model：** 学会计算公式 `理论带宽 * 实测算力 / 理论算力`。

### 💻 实战编码 (Coding & Profiling)

#### 1. Vector Add (带宽测试版)

- [ ] 编写基础 Kernel。

- [ ] **NCU 分析：** 运行 Nsight Compute，查看 "Memory Throughput"。

- [ ] **KPI 检查：** 确认 Kernel 是否达到 GPU 理论带宽的 **80%** 以上 (例如 RTX 3090 > 750GB/s)。

  - [ ] _如果是，通过。_

  - [ ] _如果否，分析原因并优化。_

#### 2. Naive SGEMM (最简版)

- [ ] 编写一个没有任何优化的矩阵乘法 Kernel。

- [ ] **NCU 源码分析：** 打开 NCU 的 **Source View**。

- [ ] **定位瓶颈：** 找出具体哪一行代码导致了 Memory Stall (内存停顿)。

### 🗣️ 面试准备 (Interview Prep)

- [ ] **绘图练习：** 能手绘 SM 简图 (包含 Tensor Core, L1/Shared Mem)。

- [ ] **口述练习：** 解释“为什么 Uncoalesced Memory Access 会导致带宽浪费？” (必须提到 Transaction 的 32B/128B 粒度)。

______________________________________________________________________

## 📅 第 2 周：跨越到 Tensor Core (WMMA & Async Copy)

**核心目标：** 放弃纯 CUDA Core 执念，掌握现代 GPU 加速核心 (实习面试分水岭)。

### 📚 理论学习 (Theory)

- [ ] **学习 WMMA API：** 阅读 `nvcuda::wmma` 文档，理解 Fragment (片段) 概念。

- [ ] **学习 Async Copy：** 理解 `cp.async` 如何绕过寄存器直接从 Global 搬运到 Shared Memory。

### 💻 实战编码 (Coding & Profiling)

#### SGEMM 进化之路

- [ ] **V1: Shared Memory Tiling (FP32)**

  - [ ] 实现分块算法，解决 Bank Conflicts。

- [ ] **V2: Tensor Core GEMM (FP16) [核心]**

  - [ ] 将数据类型转换为 `half` (FP16)。

  - [ ] 使用 `wmma::load`, `wmma::mma`, `wmma::store` 重写 Kernel。

  - [ ] **KPI 检查：** 性能必须大幅碾压 V1 版本。

- [ ] **V3: Double Buffering + Async Copy [高阶选做]**

  - [ ] 在 V2 基础上引入 `cp.async`。

  - [ ] 实现计算与数据搬运的流水线重叠 (Overlapping)。

### 🗣️ 面试准备 (Interview Prep)

- [ ] **口述练习：** 回答“Tensor Core 计算时，SM 其他单元在干什么？” (关键词：INT32 单元并行计算地址/指针)。

______________________________________________________________________

## 📅 第 3 周：Memory-Bound 算子与 Warp 魔法

**核心目标：** 针对 LLM 核心组件 (Softmax/RMSNorm) 进行专项训练。

### 📚 理论学习 (Theory)

- [ ] **掌握 Warp Primitives：** 彻底搞懂 `__shfl_down_sync` (归约) 和 `__shfl_xor_sync` (蝴蝶交换)。

- [ ] **学习 Online Softmax：** 理解 FlashAttention 的数学基础 (如何在未知全局 Max 的情况下计算)。

### 💻 实战编码 (Coding & Profiling)

#### 1. Warp Reduction

- [ ] 编写 Kernel：一个 Warp 内 32 个线程求和。

- [ ] **限制条件：** 不使用 Shared Memory，仅使用 Shuffle 指令。

#### 2. RMSNorm (Llama 同款)

- [ ] 实现公式：$x / \\sqrt{\\text{mean}(x^2) + \\epsilon} * \\gamma$。

- [ ] **优化技巧：** 使用 `float4` 向量化加载数据。

- [ ] **流程实现：** 平方和 -> Warp Reduce -> 广播结果 -> 计算输出。

#### 3. Online Softmax

- [ ] 实现单 Pass 计算 Softmax。

- [ ] **数值稳定性：** 确保处理了减去 Max 值的逻辑。

______________________________________________________________________

## 📅 第 4 周：LLM 综合实战与简历包装

**核心目标：** 理论结合实践，构建“微型推理引擎”概念。

### 📚 理论深度 (Theory Deep Dive)

- [ ] **FlashAttention V1：** 阅读论文或技术博客。

  - [ ] **重点：** 手写/默写伪代码，理清 Tiling 循环顺序 (Outer Loop: K, V; Inner Loop: Q)。

- [ ] **Quantization (量化)：**

  - [ ] 理解 **W8A16** (权重 INT8，激活 FP16) 模式。

  - [ ] 理解 De-quantization (反量化) 在 Kernel 中的计算流。

### 📝 总结与复盘 (Review)

- [ ] 整理所有代码到 GitHub 仓库。

- [ ] 编写 README，贴上 NCU 分析截图和 Roofline 性能对比图。

- [ ] 模拟面试：串联 4 周的知识点，准备讲述一个“从 Naive 到 Tensor Core 优化”的故事。
