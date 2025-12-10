## 准备

### 第一部分："SGLang Diffusion Warm-up Optimization" Issue

**结论：这是一个“金矿”级别的任务。**

这个 Issue 对两个团队都有极高的杀伤力：

1. **对 Seed 团队（多模态）：** SGLang 正在从纯 LLM 扩展到 Diffusion（生图/视频），这正是 Seed 团队的主战场。你做这个，证明你懂 **多模态推理流程**。

1. **对 Data-AML 团队（系统优化）：** Warm-up（预热）的核心是 **CUDA Graph Capture（CUDA 图捕获）** 和 **Memory Allocation（显存预分配）**。这是为了消除 First Token Latency（首字延迟）和避免运行时抖动的关键技术。

**你应该在这个 Issue 中学到（并能在面试中讲出）的东西：**

- **CUDA Graphs 原理：** 为什么 Diffusion 模型特别需要 CUDA Graph？（因为 Diffusion 是多步迭代去噪，循环次数多，Kernel Launch overhead 极大，CUDA Graph 可以把几百个 Kernel 打包成一个，一次发射）。

- **Static vs Dynamic Shape：** Warm-up 实际上是在固定 Input Shape 的情况下预先跑一遍。你需要理解 SGLang 如何处理动态 Shape 和 CUDA Graph 对固定 Shape 要求的冲突。

- **显存池化 (Mempool)：** Warm-up 过程中是否涉及 PyTorch 的显存分配？SGLang 是如何接管这些显存的？

**行动建议：**

- 全力以赴解决这个 Issue。

- **记录过程：** 记录下优化前后的 Latency 对比（比如首图生成时间从 3s 降到 2.5s），截取 Nsight Systems 的图，展示 Kernel Launch 之间的 Gap 被消除了。**这直接就是面试时的 PPT 素材。**

### 第二部分：知识储备清单（针对 Data-AML & Seed）

要拿下这两个团队，你的知识不能只停留在“会写代码”，要上升到“体系结构”。

#### 1. CUDA 进阶（目标：超越基础 Kernel）

- **CUDA Graphs (必考)：** 既然你在做 Warm-up，必须彻底搞懂 `cudaStreamBeginCapture` 和 `cudaGraphLaunch`。面试题：_“CUDA Graph 遇到 CPU 判断逻辑（if-else）会发生什么？如何处理动态控制流？”_

- **Stream & Event 异步并发：** 理解如何利用多流（Multi-stream）实现 Compute-Copy Overlap（计算与通信重叠）。

- **Tensor Core (WMMA/MMA)：** 不要求手写极其复杂的 GEMM，但要懂 Layout（`nwc` vs `ncw`）以及 Fragment 的概念。

#### 2. SGLang 架构深度（目标：懂调度）

- **RadixAttention vs BlockManager：** 彻底理清 vLLM 的 PagedAttention 和 SGLang 的 RadixAttention 在数据结构上的区别（Radix Tree 也就是基数树，是如何做前缀匹配的）。

- **Compiler Backend：** 了解 SGLang 底层是如何调用 Triton 生成代码的。SGLang 不仅仅是一个 Runtime，它有一套类似编译器的前端语言。

#### 3. Triton 认知（目标：字节非常看重 Compiler）

- **为什么用 Triton：** 它是 Python DSL 到 PTX 的桥梁。

- **关键概念：** `tl.block`（分块处理），`tl.load/store`（向量化读写）。

- **练手：** 试着用 Triton 写一个简单的 **RMSNorm** 或 **Softmax**。面试时说：“我发现手写 CUDA 开发周期长，所以我也在尝试用 Triton 快速验证算子逻辑。”——这句话非常这就很符合字节“务实/效率”的价值观。

______________________________________________________________________

### 第三部分：MiniInfer 升级指南（Project Portfolio）

为了配合这个 Issue，你的 MiniInfer 项目需要添加 1-2 个“高大上”的功能，让它看起来像一个微缩版的工业级引擎。

#### 功能 1：集成 CUDA Graph 支持 (优先级：最高)

- **理由：** 完美配合你正在修的 SGLang Issue。

- **怎么做：** 在 MiniInfer 的 Decoding 阶段（因为 Decoding 阶段 Shape 固定），引入 `cudaGraph` 机制。

- **面试话术：** _“我在修复 SGLang Diffusion Warm-up 的过程中深刻体会到了 Kernel Launch Overhead 的影响，所以我顺手把 CUDA Graph 复刻到了我的 MiniInfer 中，实测 decoding 速度提升了 20%。”_ —— **绝杀。**

#### 功能 2：引入简单的 PyTorch/Triton 算子混合调用

- **理由：** 字节 JD 里提到了“异构”和“Triton”。

- **怎么做：** 在 C++ 工程里，尝试通过 `libtorch` 或者 Python C API，调用一个用 Triton 写的算子（比如 LayerNorm）。

- **目的：** 证明你懂 **Hybrid Backend（混合后端）** 架构。

#### 功能 3：KV Cache 的量化 (Int8/FP8)

- **理由：** Data-AML 极其看重显存效率。

- **怎么做：** 实现一个简单的 Int8 KV Cache 存储。即在保存 `K` 和 `V` 时将其 cast 成 `int8`，读取时 cast 回 `fp16`。

- **目的：** 证明你懂**量化对精度的影响**以及**显存带宽的节省**。
