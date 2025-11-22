## CUDA
### 第 1 周：建立心智模型与内存优化（PMPP 基础篇）

**目标：** 能够写出正确的 CUDA 代码，并深刻理解“为什么慢”。

- **PMPP 重点章节：**
    
    - **Ch 1-3 (Introduction & Data Parallelism):** 快速过。搞懂 Grid, Block, Thread 的映射关系，以及 `idx = blockIdx.x * blockDim.x + threadIdx.x` 这种坐标变换。
        
    - **Ch 4 (Memory Architecture):** **精读。** 这是面试的重灾区。必须搞懂 Global Memory 的 Coalesced Access（合并访问）。
        
    - **Ch 5 (Performance Considerations):** **精读。** 理解 Bandwidth 瓶颈和 Compute 瓶颈的区别。
        
- **实战任务（必须手写）：**
    
    1. **Vector Add:** 写一个最简单的向量加法，分别尝试调整 Block Size (32, 128, 256, 512, 1024)，观察耗时变化。
        
    2. **Naive SGEMM (矩阵乘法):** 写一个最朴素的 `C = A * B` (Global Memory 直接读写)。
        
    3. **关键练习：** 故意写一个**非合并访问**（Memory Uncoalesced）的 Kernel，然后写一个**合并访问**的 Kernel，用 `nvprof` 或 Nsight Compute (NCU) 截图对比两者的吞吐量。**（面试时这是绝佳的谈资）**
        
- **面试必杀技储备：**
    
    - 能画出 SM (Streaming Multiprocessor) 的简图。
        
    - 能解释为什么 Warp Divergence（线程束分歧）会降低性能。
        

---

### 第 2 周：攻克计算密集型算子 —— SGEMM 优化（PMPP 进阶篇）

**目标：** 通过优化矩阵乘法（GEMM），掌握 Shared Memory 和寄存器分块。这是推理加速的基石。

- **PMPP 重点章节：**
    
    - **Ch 6 (Matrix Multiplication):** **圣经级章节，反复读。** 理解 Tiling（分块）技术如何减少 Global Memory 访问。
        
- **实战任务（SGEMM 进化之路）：**
    
    - **V1: Shared Memory Tiling:** 将矩阵块加载到 Shared Memory 中再计算。解决 Bank Conflicts（PMPP 中有讲 Padding）。
        
    - **V2: Register Tiling (2D Register Blocking):** 每个线程不再只算 1 个点，而是算 4x4 或 8x8 的小块，利用寄存器极高的带宽。
        
    - **V3: Vectorized Load:** 使用 `float4` 指令从 Global Memory 读取数据（一条指令读128位），极大提升带宽利用率。
        
- **关键工具学习：**
    
    - 学习使用 **Nsight Compute (NCU)**。不要只看运行时间，要学会看 **"Memory Throughput"** 和 **"Compute Throughput"**。
        
    - **面试题预演：** “你的 Kernel 现在的瓶颈在哪里？是访存还是计算？Occupancy 是多少？”
        

---

### 第 3 周：攻克访存密集型算子 —— Softmax/LayerNorm（LLM 核心）

**目标：** 掌握 Reduction（规约）算法和 Warp Shuffle 原语。LLM 推理中，除了矩阵乘，剩下的瓶颈全在这里。

- **PMPP 重点章节：**
    
    - **Ch 10 (Reduction):** 学习并行求和、求最大值的各种优化策略（避免 Warp Divergence，利用 Shared Memory）。
        
- **补充知识点（PMPP 可能讲得不够深）：**
    
    - **Warp Shuffle Instructions (`__shfl_down_sync`):** 现在的 GPU 优化 Reduction 不再完全依赖 Shared Memory，而是用寄存器在 Warp 内直接通信。去搜 Nvidia 官方博客或文档学习这个 API。
        
- **实战任务：**
    
    1. **Parallel Reduction:** 实现一个求数组 Sum 或 Max 的 Kernel。
        
    2. **Softmax Kernel:** 结合 Reduction (求Max) -> Exp -> Reduction (求Sum) -> Div。
        
        - **进阶要求：** 实现 **Online Softmax**（FlashAttention 的基础），防止数值溢出，且只遍历一次数据。
            
    3. **RMSNorm / LayerNorm:** 尝试写一个简单的 LayerNorm Kernel。
        

---

### 第 4 周：LLM 特性与模拟面试（综合提升）

**目标：** 将知识串联，了解前沿技术，准备简历项目。

- **PMPP 选读：** 此时书已经是参考书了，不需要通读。
    
- **必须补充的 LLM 知识（PMPP 里没有）：**
    
    - **FlashAttention 原理：** 找一篇好的博客（如知乎上的解析），理解它如何通过 Tiling Q, K, V 来利用 SRAM 计算，从而减少 HBM 访问。你需要能口述其逻辑。
        
    - **量化（Quantization）：** 理解 Int8/FP16 的区别。了解 Weight-Only Quantization (W8A16) 是什么。
        
    - **Tensor Core:** 只需要知道它存在，以及如何用 `wmma` API 调用。不需要精通，但要知道它比 CUDA Core 快在哪（一次指令算一个 16x16 矩阵块）。
        
- **最终项目（简历亮点）：**
    
    - 整理你前三周的代码，建一个 GitHub 仓库，起名如 `Tiny-CUDA-LLM-Ops`。
        
    - 包含：优化过的 SGEMM、Softmax、LayerNorm。
        
    - **关键：** 写一个 `README.md`，放上你的**性能对比图**（Baseline vs Optimized vs PyTorch/CuBLAS）。
        
    - _例如：“通过使用 Shared Memory 和 float4 向量化，我的 SGEMM 性能达到了 CuBLAS 的 80%。”_ —— 这句话在面试中非常有分量。

## SGLang

### 📅 第一阶段：Triton 速成与投机采样原型 (11/22 - 12/10)

**目标：** 在 MiniInfer 中跑通最简陋的 Speculative Decoding (Python + Triton)，同时理解 GPU 基础。

|**线程 A: PR 实战 (Triton/Python)**|**线程 B: 面试内功 (CUDA C++ / PMPP)**|**对应面试题**|
|---|---|---|
|**[ ] MiniInfer 逻辑:** 用纯 PyTorch 写一个 `Draft Model -> Verify -> Accept/Reject` 的 Python 循环。理解 KV Cache 回滚机制。|**[ ] PMPP Ch 1-3:** 理解 Grid/Block/Thread 和基本的内存模型。|"解释一下 CUDA 的线程层级？"|
|**[ ] Triton 入门:** 阅读 Triton 官方 Tutorials (01-Vector-Add, 02-Fused-Softmax)。|**[ ] 手写 Kernel 1:** 用 CUDA C++ 写一个 `VectorAdd`，学会怎么编译、运行、看结果。|"如何写一个基本的 Kernel？"|
|**[ ] Triton 实战:** 将 Speculative Decoding 中的 "Verification" 步骤（比较 Token ID）用 Triton 写一个简单的 Kernel。|**[ ] 概念理解:** 什么是 Warp Divergence？为什么 if-else 在 GPU 上很慢？|"什么是 Warp Divergence？"|

- **里程碑:** 你的 MiniInfer 能用 "Draft-Verify" 模式跑通，哪怕很慢。
    

---

### 📅 第二阶段：SGLang 源码狙击与内存优化 (12/11 - 12/31)

**目标：** 锁定 SGLang 的 Issue，把你的 Triton Kernel 移植过去，同时攻克面试中最难的“内存优化”。

|**线程 A: PR 实战 (SGLang)**|**线程 B: 面试内功 (PMPP 核心)**|**对应面试题**|
|---|---|---|
|**[ ] 锁定 Issue:** 在 SGLang 寻找 Eagle/Medusa 相关 Issue，或者 "Kernel Optimization" 标签。|**[ ] PMPP Ch 4 (Global Mem):** 彻底搞懂 **Coalesced Access (合并访问)**。这是面试必杀技。|"为什么不连续访问显存性能会差？"|
|**[ ] 代码移植:** 将你在 MiniInfer 写的验证算子，或者 Tree Attention 的 Mask 生成逻辑，尝试按照 SGLang 的格式重写。|**[ ] 手写 Kernel 2:** **Tiled GEMM (Shared Memory)**。不用写得很完美，但要能画出 Tiling 的图。|"如何优化矩阵乘法？Shared Memory 怎么用？"|
|**[ ] Benchmark:** 跑 SGLang 自带的 benchmark 脚本，对比你的 Kernel 和 PyTorch 原生实现的性能差异。|**[ ] PMPP Ch 5 (Shared Mem):** 理解 **Bank Conflict**。|"什么是 Bank Conflict？如何解决？"|

- **里程碑:** 你在本地的 SGLang 仓库里跑通了你的修改，并且看到了性能提升（Latency 降低）。
    

---

### 📅 第三阶段：发起 PR 与高阶面试题 (1/1 - 1/21)

**目标：** 提交 PR，并在 Code Review 等待期间，突击 H100 和高阶算法知识。

|**线程 A: PR 实战 (提交与修改)**|**线程 B: 面试内功 (高阶/H100)**|**对应面试题**|
|---|---|---|
|**[ ] 提交 PR:** 写一份漂亮的 Description，附上 Benchmark 截图（"X% Speedup on H100"）。|**[ ] PMPP Ch 10 (Reduction):** 学习 **Warp Shuffle** 指令。这是现代 CUDA 优化的标配。|"如何实现高效的 Sum Reduction？"|
|**[ ] 响应 Review:** Maintainer 会挑战你的代码风格、边界情况（Edge Cases）。这是学习最快的时候。|**[ ] 手写 Kernel 3:** **Softmax**。结合 Reduction 和 Shuffle 实现一个安全版 Softmax。|"手写一下 Softmax 的 CUDA 实现思路？"|
|**[ ] 单元测试:** 补充 SGLang 的 PyTest 测试用例，确保你的 Kernel 算得对。|**[ ] H100 概念:** 阅读 Hopper 架构白皮书，理解 **TMA** (异步拷贝) 和 **FP8** 的基本原理。|"H100 相比 A100 有什么架构改进？"|

- **里程碑:** PR 被 Merge，或者进入 Final Review 阶段。简历上可以写 "Contributor to SGLang"。
    

---

### 📅 第四阶段：简历完善与 Mock Interview (1/22 - 2月中)

**目标：** 将 PR 经历转化为面试故事，查漏补缺。

- **[ ] 简历包装:**
    
    - _Bad:_ "Learned CUDA and optimized SGLang."
        
    - _Good:_ "Designed and implemented a Triton-based Tree Verification kernel for Speculative Decoding in SGLang, reducing verification latency by 30% on H100 GPUs."
        
- **[ ] 模拟面试 (Mock):**
    
    - 对着镜子讲清楚：你的 Kernel 是怎么处理 **Memory Bound** 的？
        
    - 准备好回答：**"如果不用 Triton，用 CUDA C++ 写这个算子，你会怎么优化？"** (此时你可以祭出 Stage 2 学的 Shared Memory 和 Coalescing 知识)。