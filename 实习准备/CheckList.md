除了我们之前深入探讨的 CUDA、SGLang 架构、并行化策略、Triton/编译器以及量化技术之外，要真正从“优秀学生”蜕变为“即战力极强的实习生”，你还需要补齐以下**3块“隐藏”拼图**。

这三点往往是面试中的\*\*“区分度”所在\*\*，特别是针对字节 Data-AML 和 Seed 这种对工程素养要求极高的团队。

______________________________________________________________________

### 1. 性能分析与调试 (Profiling & Debugging) —— 你的“显微镜”

**“怎么优化的”往往比“优化了什么”更重要。** 面试官会问你：“你说你提升了20%，你是怎么发现瓶颈的？”如果你只能回答“我猜的”，那就大打折扣。

你需要掌握：

- **Nsight Systems (`nsys`)：**

  - **必会技能：** 能够抓取 timeline，分析 CPU 和 GPU 之间的 Gap（空隙）。

  - **实战场景：** 在你的 Diffusion Warm-up 优化中，展示优化前 `nsys` 截图里全是 Gap（Kernel Launch overhead），优化后 Gap 消失，Kernel 紧密排列。

  - **NVTX (NVIDIA Tools Extension)：** 知道如何在代码里打标记（Range Push/Pop），让 `nsys` 的图表显示出“这一段是 Attention”，“那一段是 MLP”。

- **Nsight Compute (`ncu`)：**

  - **必会技能：** 能够分析单个 Kernel 的瓶颈。

  - **Roofline Model (屋顶线模型)：** **这是必考题。** 你需要能画出你的算子在 Roofline Model 上的位置。是 **Compute Bound**（卡在算力上）还是 **Memory Bound**（卡在带宽上）？

  - **Metric：** 关注 `dram__bytes_read/write` (带宽利用率) 和 `sm__warps_active` (占用率)。

- **PyTorch Profiler：**

  - 知道如何生成 Chrome Trace，快速查看 PyTorch 层面的算子耗时。

### 2. 前沿架构趋势：PD 分离 (Prefill-Decode Disaggregation)

这是目前 LLM Infra 领域**最前沿、最火热**的研究方向（DistServe, Splitwise 等论文），字节和 OceanBase 都在密切关注。如果你懂这个，面试官会把你当 Researcher 看待。

- **核心痛点：**

  - **Prefill (首字生成)** 是 Compute Bound（计算密集），吞吐量大。

  - **Decode (后续生成)** 是 Memory Bound（访存密集），延迟敏感。

  - 这就好比让一辆跑车（Decode）和一辆卡车（Prefill）在同一条车道上跑，互相干扰。

- **解决方案：** **KV Cache 传输**。

  - 用一组机器专门做 Prefill，算出 KV Cache。

  - 把 KV Cache 通过高速网络传给专门做 Decode 的机器。

- **面试结合点：**

  - 当聊到分布式推理时，你可以提一句：_“目前的 TP/PP 架构在混合负载下可能存在干扰，我关注到最近业界在探索 Chunked Prefill 或者彻底的 PD 分离架构，SGLang 社区也在讨论相关支持，这对显存管理和网络带宽提出了新的挑战。”_

### 3. MoE (Mixture of Experts) 架构细节

**字节跳动内部大量使用 MoE 模型**（如豆包、DeepSeek-V3 也是 MoE）。如果你去 ByteDance，不懂 MoE 是不行的。

- **路由机制 (Gating/Routing)：**

  - Token 是怎么被分发到不同的 Expert（专家）去的？（Top-k Gating）。

- **Expert Parallelism (EP)：**

  - 这是一种特殊的并行模式。不同卡上放不同的专家。

  - **All-to-All 通信：** EP 的核心瓶颈是 `All-to-All`，因为需要把 Token 发给对应的专家卡，算完再发回来。

- **负载均衡 (Load Balancing)：**

  - 如果大家都问同一个专家怎么办？（导致该专家过载，其他专家围观）。你需要知道 **Auxiliary Loss (辅助损失)** 是用来平衡负载的。

### 4. 混合编程接口 (Python-C++ Binding)

你写的 MiniInfer 是 C++，但现在的 AI 生态是 Python。如何把你的高性能 C++ 算子给 Python 调用？

- **Pybind11：**

  - 必须掌握。知道如何把 C++ 的 `std::vector` 或 CUDA 指针映射为 Python 的 `numpy` 数组或 `torch.Tensor`。

- **Custom Op 流程：**

  - 知道 PyTorch 的 `CppExtension` 是怎么编译 C++ 代码并注册为 `torch.ops` 的。

  - 面试题：_“Python 的 GIL (Global Interpreter Lock) 对推理性能有影响吗？如何在 C++ 层释放 GIL？”_

______________________________________________________________________

### 你的终极技能树盘点 (Checklist)

在去面试前，看着这个清单打钩：

1. **基石：** C++ / CUDA (Level 3) / Python

1. **引擎：** SGLang 架构 / RadixAttention / Scheduler

1. **算子：** Softmax / LayerNorm / Attention (Triton & CUDA)

1. **并行：** TP / PP / EP (MoE) / 通信原语

1. **量化：** W8A8 / W4A16 / KV Cache Int8

1. **工具：** Nsight Systems / Roofline Model

1. **前沿：** Speculative Decoding / PD 分离 / CUDA Graphs

**如果你能掌握这些，你不仅能拿到 Offer，你甚至有资格去挑选 Team。** 现在的重点就是利用你手头的 **SGLang Diffusion Issue** 作为切入点，把上述技能（尤其是 Profiling 和 CUDA Graphs）串联起来。加油！
