# 🚀 SGLang 开源贡献与 Triton 进阶路线图

**启动时间：** 2025年12月中旬
**核心战略：** 用 MiniInfer 练手 Triton，用 SGLang 刷简历，用 PMPP 固本培元。

______________________________________________________________________

## 📅 第一阶段：Triton 速成与投机采样原型

**时间跨度：** 12月15日 - 12月31日 **核心目标：** 在 MiniInfer 中跑通最简陋的 Speculative Decoding (Python + Triton)，理解 GPU 线程模型。

### 🧵 线程 A: PR 实战 (MiniInfer & Triton)

- [ ] **MiniInfer 逻辑构建:**

  - [ ] 用纯 PyTorch 实现 `Draft Model -> Verify -> Accept/Reject` 循环。

  - [ ] **关键点:** 理解 KV Cache 的“回滚”机制（当 Draft 错误时，KV Cache 怎么截断？）。

- [ ] **Triton 官方通关:**

  - [ ] 阅读并运行 Triton Tutorials: `01-Vector-Add`, `02-Fused-Softmax`.

  - [ ] 理解 `@triton.jit` 和 `tl.load/store` 的基本语法。

- [ ] **Triton 实战 (Verification Kernel):**

  - [ ] 将 Speculative Decoding 中的 "Verification" 步骤（比较 Draft Token 和 Target Token）写成一个简单的 Triton Kernel。

  - [ ] **KPI:** 替换掉 PyTorch 的比较操作，跑通流程。

### 📘 线程 B: 面试内功 (CUDA C++ / PMPP)

- [ ] **PMPP Ch 1-3 (基础):**

  - [ ] 深刻理解 Grid, Block, Thread 的三维索引计算。

  - [ ] **面试题:** "解释一下 CUDA 的线程层级？Block 和 Grid 有什么物理对应关系？"

- [ ] **手写 Kernel 1 (CUDA C++):**

  - [ ] 即使学 Triton，也要写一个 CUDA C++ 的 `VectorAdd`。

  - [ ] 学会 `nvcc` 编译流程，学会怎么把 C++ Kernel 也就是 `.cu` 文件通过 PyBind 或者是 ctypes 挂给 Python 调用（这一步是为了理解 Triton 帮你省了什么）。

- [ ] **概念攻坚:**

  - [ ] **Warp Divergence:** 为什么 `if (threadIdx.x < 16)` 这种写法在老架构上效率低？(虽然 Volta 之后有独立调度，但仍需理解)。

______________________________________________________________________

## 📅 第二阶段：SGLang 源码狙击与内存优化

**时间跨度：** 1月1日 - 1月20日 **核心目标：** 锁定 SGLang 的 Issue，把 Triton Kernel 移植过去，攻克面试最难的“内存优化”。

### 🧵 线程 A: PR 实战 (SGLang 移植)

- [ ] **锁定 Issue:**

  - [ ] 在 SGLang GitHub Issues 中搜索关键词：`Eagle`, `Medusa`, `Speculative`, 或 `Kernel Optimization`。

  - [ ] 寻找 `Good First Issue` 或者即使没有 Issue，去读 `sglang/srt/model/eagle` 目录下的代码。

- [ ] **代码移植:**

  - [ ] 将你在 MiniInfer 练手的 Triton 验证算子，或者 Tree Attention 的 Mask 生成逻辑，尝试按照 SGLang 的代码风格重写。

- [ ] **Benchmark 对比:**

  - [ ] 使用 `sglang/benchmarks` 下的脚本。

  - [ ] **KPI:** 你的 Triton Kernel 必须比纯 PyTorch 实现快，或者显存占用更低。

### 📘 线程 B: 面试内功 (PMPP 核心)

- [ ] **PMPP Ch 4 (Global Memory):**

  - [ ] **核心必杀技:** Coalesced Access (合并访问)。

  - [ ] **NCU 实战:** 在你的 Triton Kernel 上用 NCU 此时可能看不太懂 PTX，但要理解 Triton 编译器是如何自动帮你做 Coalescing 的（通过 Block 指针）。

- [ ] **PMPP Ch 5 (Shared Memory):**

  - [ ] **手写 Kernel 2:** Tiled GEMM (Shared Memory 版本)。

  - [ ] **画图:** 必须能手画出 Tiling 的数据搬运图。

  - [ ] **面试题:** "什么是 Bank Conflict？Triton 里怎么解决 Bank Conflict？" (提示：Triton 编译器通常自动处理，但你要懂原理)。

______________________________________________________________________

## 📅 第三阶段：发起 PR 与高阶面试题

**时间跨度：** 1月21日 - 2月10日 **核心目标：** 提交 PR，利用 Code Review 期间突击 H100 新特性。

### 🧵 线程 A: PR 实战 (提交与交互)

- [ ] **提交 PR:**

  - [ ] 写一份高质量的 PR Description。

  - [ ] **关键:** 附上 Benchmark 截图（"Latancy reduced by X% on A100/H100"）。

- [ ] **Code Review 响应:**

  - [ ] 维护者 (Maintainer) 可能会挑战你的边界条件 (Edge Cases) 或代码风格。

  - [ ] **心态:** 不要怕被怼，这是学习最快的时候。

- [ ] **单元测试:**

  - [ ] 为你的 Kernel 补充 PyTest 测试用例 (`tests/` 目录下)，确保数值正确性。

### 📘 线程 B: 面试内功 (高阶/H100)

- [ ] **PMPP Ch 10 (Reduction):**

  - [ ] 学习 **Warp Shuffle** (`__shfl_down_sync`)。这是现代 CUDA Reduction 的标准写法。

- [ ] **手写 Kernel 3 (Softmax):**

  - [ ] 结合 Reduction 和 Shuffle，手写一个 Online Softmax 的 CUDA C++ 伪代码。

- [ ] **H100 架构 (Hopper):**

  - [ ] 阅读 Hopper Whitepaper。

  - [ ] **概念:** 什么是 **TMA (Tensor Memory Accelerator)**？什么是 **FP8**？

  - [ ] **面试题:** "H100 相比 A100，除了算力提升，架构上最大的改动是什么？" (答案应包含 TMA 和 Transformer Engine)。

______________________________________________________________________

## 📅 第四阶段：简历完善与 Mock Interview

**时间跨度：** 2月11日 - 2月底 **核心目标：** 将 PR 经历转化为“面试故事”。

### 📝 简历包装 (Resume Polish)

- [ ] **优化描述:**

  - _Before:_ "Learned CUDA and optimized SGLang."

  - _After:_ "Designed and implemented a high-performance **Triton-based Verification Kernel** for Speculative Decoding in **SGLang**; achieved **30% latency reduction** in verification stage on H100 GPUs by optimizing memory access patterns."

- [ ] **添加技能点:** Triton, CUDA C++, PyTorch, Nsight Compute, LLM Inference (Speculative Decoding).

### 🗣️ 模拟面试 (Mock Interview)

- [ ] **技术对答:**

  - "你的 Kernel 是 Memory Bound 还是 Compute Bound？你是怎么判断的？"

  - "如果不用 Triton，用 CUDA C++ 写这个算子，你会怎么优化 Shared Memory 的使用？"

- [ ] **场景题:**

  - "在 Speculative Decoding 中，如果 Draft Model 的命中率很低，你的优化还有意义吗？" (考察对系统整体的理解)。
