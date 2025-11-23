## CUDA
### 第 1 周：建立“Roofline”思维与内存这一生之敌

**目标变化：** 不仅要会写，要学会用 **Nsight Compute (NCU)** 算账。一切优化始于“带宽利用率”。

- **理论升级 (PMPP + 官方文档)：**
    
    - **Memory Hierarchy:** 重点看 Global Memory -> **L2 Cache** -> Shared Memory 的路径。理解 L2 Cache 在 LLM 推理（特别是 KV Cache）中的巨大作用。
        
    - **Coalesced Access (合并访问):** 依然是核心，必须死磕。
        
    - **Roofline Model:** 必须学会算：`理论带宽 * 实测算力 / 理论算力`。
        
- **实战任务 (现代化改造)：**
    
    1. **Vector Add (带宽测试版):**
        
        - 写完后，用 NCU 跑一下，查看 **"Memory Throughput"**。
            
        - **KPI:** 你的 Kernel 能跑到 GPU 理论带宽（比如 3090 是 936GB/s）的 80% 以上吗？如果不到，为什么？
            
    2. **Naive SGEMM:**
        
        - 写个最烂的版本。
            
        - **关键练习升级:** 不仅对比吞吐量，还要打开 NCU 的 **Source View**，看看到底是哪一行代码导致了 Memory Stall（内存停顿）。
            
- **面试必杀技储备：**
    
    - 能画出 SM 简图（包含 Tensor Core 和 L1/Shared Mem）。
        
    - **新增：** 能口述“为什么 Uncoalesced Memory Access 会导致带宽浪费？”（答案要涉及 Transaction 的 32 字节/128 字节粒度）。
        

---

### 第 2 周：从 CUDA Core 跨越到 Tensor Core (最痛苦但也最重要的一周)

**目标变化：** 放弃纯 CUDA Core 优化的执念，拥抱 **WMMA** 和 **Async Copy**。这是实习面试的**分水岭**。

- **理论升级 (跳出 PMPP)：**
    
    - **Tensor Core API:** 学习 C++ 接口 `nvcuda::wmma`。理解 Fragment（片段）的概念：数据必须先加载到 Fragment 才能进 Tensor Core。
        
    - **Async Copy (`cp.async`):** 学习如何绕过寄存器，直接从 Global Memory 搬运到 Shared Memory。这是现代 GPU 掩盖延迟的神器。
        
- **实战任务 (SGEMM 现代化进化之路)：**
    
    - **V1: Shared Memory Tiling (FP32):** 这一步保留，为了理解分块原理。解决 Bank Conflicts。
        
    - **V2: Tensor Core GEMM (FP16):** **(核心新增)**
        
        - 将数据类型改为 `half` (FP16)。
            
        - 使用 `wmma::load_matrix_sync`, `wmma::mma_sync`, `wmma::store_matrix_sync` 实现矩阵乘。
            
        - **KPI:** 此时你的性能应该轻松碾压 V1 版本。
            
    - **V3: Double Buffering + Async Copy:** **(高阶选做)**
        
        - 在 V2 的基础上，尝试用 `cp.async` 预取下一块数据。即使写不出来，也要看懂相关代码逻辑。
            
- **面试题预演：**
    
    - “Tensor Core 计算的时候，SM 在干什么？”（答案：SM 的 INT32 单元可以并行做地址计算/指针移动，这就是并行的魅力）。
        

---

### 第 3 周：Memory-Bound 算子与 Warp 魔法

**目标变化：** 针对 LLM 的 **Softmax** 和 **RMSNorm** 进行针对性训练。

- **理论升级：**
    
    - **Warp Primitives:** 彻底搞懂 `__shfl_down_sync` (归约用) 和 `__shfl_xor_sync` (蝴蝶交换用)。
        
    - **Online Softmax:** 理解如何在**不知道全局最大值**的情况下计算 Softmax（FlashAttention 的数学基础）。
        
- **实战任务：**
    
    1. **Warp Reduction:** 写一个 Kernel，一个 Warp 内 32 个线程求和，**不使用 Shared Memory**，只用 Shuffle 指令。
        
    2. **RMSNorm (Llama 同款):**
        
        - 实现公式：$x / \sqrt{\text{mean}(x^2) + \epsilon} * \gamma$
            
        - 技巧：输入数据通常是 `float4` 加载，先算出平方和，做一次 Warp Reduce，然后广播结果，最后计算输出。
            
    3. **Online Softmax:**
        
        - 尝试在一个 Kernel Pass 内完成 Softmax 计算。
            
        - **关键点:** 数值稳定性处理（减去 Max 值）。
            

---

### 第 4 周：LLM 综合实战与简历包装

**目标变化：** 将前三周的积木拼成一个“微型推理引擎”。

- **理论知识 (LLM 特供)：**
    
    - **FlashAttention:** 不要只看博客，要看 **FlashAttention V1 的伪代码**。理解 Tiling Q/K/V 的循环顺序：`Outer Loop: K, V blocks; Inner Loop: Q blocks` (或者反过来，取决于具体实现)。
        
    - **Quantization (量化):** 了解 **W8A16** (权重 INT8，激活 FP16)。知道 Kernel 里需要把 INT8 权重转换回 FP16 再计算（De-quantization）。
        
- **最终项目 (GitHub: Tiny-LLM-CUDA):**
    
    - **结构建议：**
        
        Plaintext
        
        ```
        Tiny-LLM-CUDA/
        ├── 01_basics/
        │   └── vec_add_profiled.cu  (附带 NCU 截图)
        ├── 02_gemm_tensorcore/
        │   └── wmma_gemm_fp16.cu    (展示你懂 Tensor Core)
        ├── 03_ops/
        │   ├── rms_norm_shfl.cu     (展示你懂 Warp Shuffle)
        │   └── online_softmax.cu    (展示你懂 LLM 算子特性)
        └── README.md
        ```
        
    - **README 写法：**
        
        - 不要只贴代码。要贴**图表**。
            
        - _Case Study:_ "在 RMSNorm 中，通过将 Shared Memory Reduction 替换为 Warp Shuffle，Kernel Latency 从 10us 降低到了 2us，带宽利用率提升至 90%。"
            

---

### 每日时间分配建议 (高强度版)

- **前 3 天 (周一至周三):** 痛苦地死磕代码。比如这就写一个 `wmma_gemm`，写不出来就去抄 NVIDIA 官方 Sample，然后一行行注释读懂。
    
- **周四:** **Profile Day。** 打开 NCU，对着你的 Kernel 找红线（瓶颈）。
    
- **周五:** **Refactor Day。** 根据 NCU 的建议优化代码（比如加 `#pragma unroll`，或者改一下 Block Size）。
    
- **周末:** 整理 GitHub 和 博客/笔记。面试官很喜欢看“优化日志”。
    

### 总结：为什么这个 V2 计划更强？

原计划让你成为一个“扎实的 CUDA 工程师”。

V2 计划让你成为一个**“可以直接上手 vLLM/TensorRT-LLM 开发的实习生”**。

在面试中，当你随口说出：“我之前在写 GEMM 时，发现从 Global Memory 读数据如果不由 cp.async 预取，Tensor Core 就会因为数据饥饿而 Stall...”

面试官就知道，你是自己人。

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