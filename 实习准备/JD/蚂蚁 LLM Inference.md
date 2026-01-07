## JD

**职位描述**

你将与OB的资深工程师紧密合作，参与自研LLM推理框架的研发，通过学习开源社区优秀实践，结合工程创新，打造具有极致性价比的下一代LLM推理系统。 作为实习生，你可以直面工业级AI系统的真实挑战，在师兄的指导下，你可以完整经历从技术调研、方案设计、代码实现到效果验证的全研发流程，积累顶尖企业级项目经验，同时系统掌握LLM推理优化的方法论与技术体系。 你的优秀成果将直接整合进OceanBase智能计算引擎，服务亿级用户场景。

**职位要求**

1. 推理框架全链路优化: 包括但不限于常规的量化，speculative decoding，DP，EP等开源社区已采纳的最优实践，也要探索通信优化，kv cache offload，计算/通信overlap等新的优化机会。
1. 异构计算探索: 在PC/一体机等真实场景中，探索CPU+GPU混合计算的最佳实践。通过智能调度与并行策略设计，让每一份计算资源都发挥最大价值。
1. 长上下文优化: 研究KV Cache压缩/过滤/持久化等前沿技术，破解长上下文推理成本难题。在这里，你将亲手优化处理超长文本的工业级解决方案。
1. 只要你熟悉PyTorch框架或CUDA编程基础，对底层优化有强烈好奇心，接触过Transformer架构优化即可，我们会提供系统化指导。当然如果你研究过vLLM/SGLang/ktransformer等开源项目能更快上手。

## 准备

这份职位描述（JD）含金量极高，来自于 **OceanBase（蚂蚁集团旗下的企业级分布式数据库）** 的AI团队。从描述来看，他们正在自研或深度魔改一套LLM推理引擎，目的是为了在“端侧”（一体机/PC）或“私有化部署”场景下，以极低成本运行大模型。

根据你之前的对话记录（特别是你正在开发的 **MiniInfer** 项目、对 **SGLang** 的钻研，以及你作为计算机研究生的背景），**这个职位几乎是为你量身定做的**。你目前的技能栈与他们的需求重合度极高。

为了确保万无一失，拿下这个Offer，你需要从以下几个维度整理你的知识储备和项目经历：

### 1. 核心知识储备（针对JD逐条拆解）

你需要展示出“不仅会用，而且懂底层原理”的能力。

- **基础架构与数学原理：**

  - **Transformer 深度剖析：** 必须能徒手推导 Self-Attention 公式，清楚 Multi-head Attention 与 Multi-query Attention (MQA/GQA) 的区别（这一点在显存带宽优化中极其重要）。

  - **位置编码：** 彻底理解 RoPE（旋转位置编码）的数学原理及其代码实现（既然你MiniInfer里写过，面试时要能讲清楚它是如何处理长文本外推的）。

- **推理优化核心技术 (JD重点)：**

  - **显存管理 (KV Cache)：** 你必须精通 **PagedAttention** (vLLM的核心) 的原理。理解 Block table 如何解决显存碎片化问题。

  - **调度策略：** 深刻理解 **Continuous Batching (Orca)**。要能对比 Static Batching 和 Continuous Batching 的吞吐量差异。

  - **量化 (Quantization)：** JD提到了“常规量化”。你需要了解 **W4A16**（权重4bit，激活16bit）、**AWQ**、**GPTQ** 或 **SmoothQuant** 的基本原理。如果还没看过，重点补一下 AWQ 的论文。

  - **投机采样 (Speculative Decoding)：** 理解 Draft model 机制，知道它是如何用“小模型预测+大模型验证”来换取时间的。

- **异构计算与底层编程 (JD特色)：**

  - **CUDA 编程：** 不需要你是CUDA专家，但要懂 Kernel fusion（算子融合），理解 GPU Memory Hierarchy (HBM vs SRAM)。

  - **CPU+GPU 混合推理 (关键点)：** JD 提到了 `ktransformers`，这是一个利用 CPU 内存承载 KV Cache 或部分权重的项目。你需要了解 **KV Cache Offloading** 技术（何时把 KV 搬到 CPU，何时搬回 GPU，如何掩盖 PCIe 传输延迟）。

### 2. 针对性项目经历（如何包装你现有的项目）

这部分是你的杀手锏。不要只罗列项目，要**针对他们的痛点**来描述。

#### 项目一：从零构建轻量级推理引擎 (基于你的 MiniInfer)

- **对标JD要求：** “对底层优化有强烈好奇心”、“接触过 Transformer 架构优化”。

- **简历/面试话术：**

  - 强调你是**从零 (Scratch)** 手写的推理引擎，而不是调包。

  - 重点描述你如何用 C++/CUDA 实现了 **RoPE** 算子。

  - 描述你实现的 **Continuous Batching** 调度逻辑，以及它是如何提升 GPU 利用率的。

  - **加分项：** 提到你在开发过程中遇到的具体困难（例如：Mask处理、Tensor形状对齐），以及你是如何 Debug 的。

#### 项目二：开源社区贡献与源码分析 (基于你的 SGLang/vLLM 计划)

- **对标JD要求：** “熟悉 vLLM/SGLang 等开源项目”、“学习开源社区优秀实践”。

- **简历/面试话术：**

  - 不要只说“看过源码”，要说“**分析过 SGLang 的 RadixAttention 实现机制**”。

  - 如果你已经按照之前的计划开始阅读 SGLang 源码，可以具体谈谈它是如何利用 Prefix Caching 来优化多轮对话或 System Prompt 的。这直接对应 JD 里的“长上下文优化”和“KV Cache过滤”。

  - 提到你对 **Chunked Prefill** 的理解（解决首字延迟与吞吐量的权衡），这是目前非常前沿的优化点。

#### 项目三：分布式系统与存储经验 (基于你的 NebulaGraph 经历)

- **对标JD要求：** OceanBase 是一家数据库公司。

- **隐藏优势 (Hidden Gem)：**

  - 虽然这是一个AI岗位，但面试官极大概率是 OB 的资深工程师，他们非常看重**系统稳定性**和**C++工程能力**。

  - 利用你在 NebulaGraph 的经历，强调你对 **C++ 内存管理**、**高并发系统设计** 以及 **Raft 协议** 的理解。这会给面试官一种“即使不懂AI，你的工程底子也足够好”的安全感。

### 3. 查漏补缺（现在的行动指南）

为了完美匹配这个 JD，建议你接下来一周重点突击以下两个盲区：

1. **研究 `ktransformers` 或 `llama.cpp`：**

   - JD 特别提到了异构计算。去看一下 `llama.cpp` 是如何把层切分到 CPU 和 GPU 上的（`--n-gpu-layers` 参数背后的逻辑）。

   - 思考：如果 GPU 显存不够，如何把 KV Cache 放在 CPU 主存里，并在计算 Attention 时通过 PCIe 快速读取？

1. **了解长文本优化：**

   - 搜索关键词：**StreamingLLM** (Sink token), **Ring Attention** (虽然实习生未必需要实现，但要知道概念)。

   - JD 提到的“KV Cache 压缩/持久化”，本质上是在解决 RAG（检索增强生成）场景下的上下文重用问题。

### 4. 模拟面试问题（自测）

在面试前，试着回答这几个问题：

1. _“在你的 MiniInfer 中，如果 Batch Size 变大，哪个算子会通过 Roofline Model 分析成为瓶颈？是 Compute-bound 还是 Memory-bound？”_

1. _“SGLang 相比 vLLM，最大的改进点在哪里？RadixAttention 是如何复用 KV Cache 的？”_

1. _“如果我们要在显存只有 24GB 的机器上跑 70B 的模型（权重约 140GB），你有什么技术方案？”_ （答案方向：4-bit量化 + 逐层加载/CPU Offload）。
