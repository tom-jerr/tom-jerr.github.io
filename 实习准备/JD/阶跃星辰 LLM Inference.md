## 职位描述

**岗位描述**

1. 参与分布式大模型推理框架的开发与优化，提升推理性能与吞吐量；
1. 针对不同场景的 LLM 请求特点，优化 GPU 计算流程，打造业内领先的高效 LLM 推理引擎；
1. 调研并引入前沿机器学习系统技术，推动系统架构的持续优化升级；
1. 与算法团队深度合作，探索算法-系统协同优化方案，提升整体推理效率。

**任职要求**

1. 计算机、电子、自动化、软件等相关专业，本科及以上在读，能实习 6 个月以上优先；
1. 具备操作系统、计算机体系结构等基础知识，对底层性能优化有浓厚兴趣；
1. 熟悉SGLang、vLLM、Megatron 等框架，有开源项目贡献或相关经验者优先；
1. 熟悉CUDA编程和GPU上性能优化，有Triton、CUTLASS等开发经验者优先；
1. 具备良好的沟通能力和团队协作精神。
   这是一个非常硬核且极具前景的岗位，方向是**大模型推理系统的底层优化**。

## 准备

基于你目前的背景——**CS研究生、正在开发MiniInfer（C++自研推理引擎）、深度研究SGLang/vLLM源码、且有NebulaGraph的C++系统开发经验**——你几乎完美契合这个岗位的核心要求。特别是JD中明确点名了**SGLang**，这正是你近期的主攻方向。

为了拿下这个Offer，你需要将你的经历向“**分布式**”和“**算子级优化**”这两个方向进一步靠拢。以下是针对该JD的定制化准备建议：

### 1. 核心知识储备：针对性查漏补缺

这个JD相比之前的OceanBase岗位，更强调**分布式（Megatron）**和**底层算子（CUDA/Triton/CUTLASS）**。

- **分布式推理原理（重中之重）：**

  - **并行策略：** 必须彻底理解 Tensor Parallelism (TP) 和 Pipeline Parallelism (PP)。

  - **通信原语：** 理解 `all-reduce`, `all-gather` 在多卡推理中的作用。

  - **Megatron-LM：** 不需要精通代码，但要懂它切分模型权重的逻辑（比如MLP层和Attention层是如何切分的）。

  - **准备话术：** “我在开发MiniInfer时虽然目前是单卡，但我研究过vLLM如何利用Ray进行分布式调度，以及Megatron如何通过TP降低单卡显存压力。”

- **GPU 编程与算子优化（Triton/CUTLASS）：**

  - **Triton 是加分项：** JD明确提到了Triton。建议你花2-3天时间，用 OpenAI Triton 写一个简单的算子（比如 Softmax 或 LayerNorm），对比一下 PyTorch 原生的性能。这能证明你具备“引入前沿技术”的能力。

  - **CUDA 进阶：** 结合你写 RoPE 算子的经验，深入复习一下 Shared Memory Bank Conflict、Coalesced Access（合并访问）等概念。

  - **CUTLASS：** 这是一个高性能矩阵乘法库。作为实习生，你不必精通，但需要知道它主要用于解决什么问题（如 GEMM 优化、Epilogue 融合）。

- **框架源码深度理解 (SGLang/vLLM)：**

  - **SGLang 特性：** 你需要能讲清楚 **RadixAttention** 是如何实现 KV Cache 复用的（这是 SGLang 的灵魂），以及它相比 vLLM 的 PagedAttention 在多轮对话或 System Prompt 场景下的优势。

  - **调度器设计：** 对比 SGLang 和 vLLM 的调度策略（Continuous Batching 的具体实现差异）。

### 2. 项目经历包装：直击JD痛点

将你手头的项目换个角度描述，以匹配JD中的关键词。

#### 项目一：自研高性能 LLM 推理引擎 MiniInfer

- **对标JD：** “熟悉CUDA编程和GPU上性能优化”、“打造业内领先的高效LLM推理引擎”。

- **简历描述建议：**

  - **Title：** 核心开发者 | C++ / CUDA

  - **内容：** 从零构建基于 C++ 的轻量级 LLM 推理系统。

  - **关键点：**

    1. **手写 CUDA Kernel：** 实现了 RoPE 旋转位置编码和 RMSNorm，深入理解 GPU 线程束（Warp）调度机制。

    1. **显存管理：** 实现了类似 PagedAttention 的显存分页机制，解决了显存碎片化问题。

    1. **调度优化：** 实现了 Continuous Batching（行内批处理），显著提升了 decode 阶段的 GPU 利用率。

  - **目的：** 证明你不止会调包，具备“造轮子”的底层编码能力。

#### 项目二：SGLang/vLLM 源码剖析与特性研究

- **对标JD：** “熟悉SGLang、vLLM”、“调研并引入前沿机器学习系统技术”。

- **简历描述建议：**

  - **Title：** 源码研究与性能分析

  - **内容：** 深度分析 SGLang 框架架构，专注于 RadixAttention 对长文本前缀缓存（Prefix Caching）的优化机制。

  - **关键点：**

    1. **技术调研：** 对比了 vLLM 的 BlockManager 与 SGLang 的 RadixTree 在显存管理上的差异。

    1. **性能分析：** 研究了不同并发请求下，SGLang 在处理结构化输出（JSON/Regex）时的性能优势（这是SGLang的一大卖点）。

  - **目的：** 直接回应JD对SGLang的要求，展示你的技术敏锐度。

#### 项目三：NebulaGraph 分布式存储引擎开发

- **对标JD：** “参与分布式大模型推理框架”、“具备良好的团队协作精神”。

- **简历描述建议：**

  - **目的：** 虽然不是AI项目，但它是**分布式系统**。

  - **话术：** 强调你在 C++ 复杂工程下的开发经验，对 Raft 协议的理解（有助于理解分布式推理中的状态同步），以及在大型开源社区（NebulaGraph）的代码规范和协作流程。这能给面试官极大的安全感，证明你的工程落地能力很强。

### 3. 面试必问“硬核”问题预测

准备好如何回答这些问题，能让你在面试中脱颖而出：

1. **场景题：** “如果 SGLang 的 RadixAttention 显存命中率很低，会导致什么问题？你会如何优化对应的 Eviction Policy（淘汰策略）？”

1. **CUDA题：** “写 RoPE 算子时，你是如何处理高维 Tensor 的内存布局的？有没有遇到 Shared Memory 不够用的情况？”

1. **架构题：** “vLLM 的 Continuous Batching 在处理极长 Prompt（Prefill阶段）时会阻塞 Decode 阶段的请求，也就是 Head-of-Line Blocking 问题，你知道现在的业界解决方案是什么吗？”（答案关键词：Chunked Prefill / Piggybacking）。

1. **分布式题：** “如果要将你的 MiniInfer 扩展成支持两张卡跑 Llama-70B，你需要做哪些改动？通信同步点在哪里？”

### 总结

这个岗位非常适合你。**SGLang** 是目前推理界最火的框架之一，而你正好在研究它。

**建议行动：**

1. 简历上要把 **SGLang** 和 **MiniInfer** 放得非常显眼。

1. 这两天快速过一遍 **Triton** 的官方 Tutorials（只需要看懂 vector-add 和 softmax 即可），面试时提到“我正在学习 Triton 并尝试用它重写部分算子”，会是非常大的加分项。
