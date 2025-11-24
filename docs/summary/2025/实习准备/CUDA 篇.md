以下我为你划分了三个层级（Level 1-3），以及针对 LLM Infra 岗位的**必知必会技术点**。

---

### Level 1: 基础扎实（必须烂熟于心）

_这是门槛。你需要能流利地解释这些概念，并在 MiniInfer 中有对应的代码体现。_

1. **线程模型 (Thread Hierarchy)：**
    
    - 清楚 Grid, Block, Warp, Thread 的映射关系。
        
    - **Warp (32 threads):** 必须理解 Warp 是 GPU 调度的最小单位。理解 **Warp Divergence (分支发散)** 的代价（即 `if-else` 导致同一 Warp 内线程空转）。
        
    - **Occupancy (占用率)：** 理解寄存器使用量（Register Pressure）和 Shared Memory 大小是如何限制每个 SM 上能跑多少个 Block 的。
        
2. **内存层次 (Memory Hierarchy) —— 重中之重：**
    
    - **Global Memory:** 必须懂 **Coalesced Access (合并访问)**。为什么读取 float16 时，按照 128-bit（float4）对齐读取比逐个读取快得多？（MiniInfer 的 RoPE 算子一定要体现这一点）。
        
    - **Shared Memory:** 它是用户可控的 L1 Cache。你需要懂如何用它来做 Tiling（分块），减少 Global Memory 的带宽压力。
        
    - **Bank Conflict:** 知道 Shared Memory 被划分为 32 个 Bank。当同一个 Warp 内的多个线程访问同一个 Bank 的不同地址时会发生冲突，导致串行化。
        
3. **基本同步机制：**
    
    - `__syncthreads()` 的作用范围（Block 内）。
        
    - 知道为什么 Grid 间同步很难（通常需要结束 Kernel 或使用原子操作）。
        

---

### Level 2: 进阶优化（字节/OB 面试的核心考点）

_这部分决定了你是否能过二面/三面。你需要结合 MiniInfer 或 SGLang 的场景来谈。_

1. **Warp-Level Primitives (Warp 原语)：**
    
    - **Shuffle Instruction (`__shfl_sync`, `__shfl_down_sync`):**
        
        - **面试必杀技：** 在写 **Softmax** 或 **Reduce Sum** 算子时，不要只用 Shared Memory 做归约。要能说出：“为了极致性能，我在 Warp 内部使用了 Shuffle 指令进行寄存器级别的数据交换，不需要经过 Shared Memory，速度更快。”
            
    - 这是高性能算子（如 FlashAttention）的基础。
        
2. **异步并发与掩盖延迟 (Latency Hiding)：**
    
    - **CUDA Streams:** 理解 Default Stream 和 Non-default Stream 的区别。
        
    - **Overlap (重叠):** 如何让数据拷贝（H2D/D2H）和 Kernel 计算同时进行？
        
    - **SGLang 关联：** 你正在修的 Warm-up Issue，本质上就是在解决 Kernel Launch 的 CPU 开销导致 GPU 出现空闲（Gap）的问题。**CUDA Graphs** 是解决这个问题的终极方案（一次 Launch，GPU 自己调度依赖）。
        
3. **向量化内存访问 (Vectorized Load/Store)：**
    
    - 在写 Kernel 时，尽量使用 `float4` (或者 `int4` for quant) 类型进行读写。这能把指令吞吐量提升 4 倍（相比 float1）。
        
    - **MiniInfer 实战：** 检查你的 RoPE 或 Elemwise 算子，是否用了 `reinterpret_cast<float4*>`？如果没有，加上去，这就是优化点。
        

---

### Level 3: 领域专家（针对 LLM/Transformer 场景）

_这部分是针对 Seed/Data-AML 的加分项，证明你懂 Transformer 的硬件特性。_

1. **Tensor Core (混合精度编程)：**
    
    - 不需要手写复杂的 GEMM，但要懂 **WMMA (Warp Matrix Multiply Accumulate)** API。
        
    - **概念：** 知道 `fragment`，知道 `load_matrix_sync` -> `mma_sync` -> `store_matrix_sync` 的流程。
        
    - **Layout：** 理解矩阵在内存中的布局（Row-major vs Col-major）对 Tensor Core 性能的巨大影响。
        
2. **FlashAttention 的核心思想 (Tiling + Recomputation)：**
    
    - 不需要默写代码，但要能画图解释：为什么传统 Attention 需要 $O(N^2)$ 的 HBM 读写，而 FlashAttention 通过分块（Tiling）把计算限制在 SRAM（Shared Memory）里，从而将 HBM 读写降到了 $O(N)$。
        
3. **量化相关的位操作：**
    
    - 如何高效处理 **int4** 数据？
        
    - 你需要懂位移操作：`(val >> 4) & 0xF`。了解如何利用 `BFE` (Bit Field Extract) 等指令加速解包。
### 总结：你需要准备的“必背”代码片段/概念

为了通过字节和 OB 的面试，请确保你能够手写或详细描述以下 3 个 Kernel 的实现逻辑：

1. **高效的 Softmax Kernel：**
    - 不要写 Naive 版本。
    - **思路：** Load Global -> Block Reduce (求 Max) -> Block Reduce (求 Sum) -> Update -> Store。
    - **关键点：** 显式使用 Warp Shuffle 优化 Reduce 过程。
        
2. **LayerNorm / RMSNorm Kernel：**
    - **思路：** 典型的两趟扫描（Two-pass）算法（求均值/方差 -> 归一化）。或者一趟扫描（Welford 算法，虽然面试不一定考这么深）。
    - **关键点：** 向量化读取 (`float4`)。
        
3. **矩阵转置 (Matrix Transpose)：**
    - 这是考察 **Coalesced Access** 和 **Shared Memory Bank Conflict** 的经典题目。
    - **思路：** Global (Coalesced) -> Shared Memory (解决转置后的非合并写入) -> Global (Coalesced)。
        

### 针对你当前 SGLang Issue 的 CUDA 知识

由于你在做 SGLang 的 Warm-up，你**必须**掌握：

1. **CUDA Graphs API:** `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch`.
    
2. **Mempool (内存池):** 为什么 Graph Capture 要求显存地址是固定的？（因为 Graph 记录了指针地址，如果下一次运行地址变了，Graph 就失效了）。

## 算子

面试考察的是**原理掌握程度**和**手写代码的能力**，而不是让你现场写一个库。

#### 1. Softmax / RMSNorm / LayerNorm

- **你掌握的技巧：** `float4` 向量化读写 + Warp Shuffle Reduce + Shared Memory 缓存。
    
- **结果：** 这一套组合拳打下去，你的 Kernel 性能通常能达到 PyTorch 原生算子的 80%-120%（取决于具体 Shape）。
    
- **面试评价：** 这证明你懂显存带宽瓶颈（Memory Bound），懂如何利用寄存器通信避免 Shared Memory 开销。这是满分答案。
    

#### 2. GEMM (矩阵乘法)

- **你掌握的技巧：** Tiling (分块) + Shared Memory 避免重复读取 + Tensor Core (WMMA)。
    
- **现状：** 即使你用了这些技巧，手写 GEMM 也很难跑过 cuBLAS 或 CUTLASS。因为极致的 GEMM 需要汇编级的流水线优化。
    
- **面试策略：** **不要试图手写一个通用的 GEMM。**
    
    - 面试官更看重你是否懂 **Tiling 的逻辑** 和 **Tensor Core 的 API (WMMA)**。
        
    - 如果你能写出一个简单的 _Block Tiling + WMMA_ 的 Demo，证明你懂混合精度计算的流程，这就足够了。
        

#### 3. FlashAttention

- **你掌握的技巧：** Tiling + Recomputation (重计算) + Softmax 的数学技巧。
    
- **结果：** 这是一个极其复杂的算子。如果你能用 CUDA C++ 把 FlashAttention V1 的逻辑跑通（哪怕性能只有官方的 50%），在校招或社招初中级岗位中已经是**Top 5%** 的水平。
    
- **关键点：** FlashAttention 的难点不仅仅是 CUDA 技巧，更是**算法逻辑**（Online Softmax 的数值稳定性公式）。
    

---

### 二、 如果要在 MiniInfer 实战中跑得更快，你还缺什么？ (Level 4: 隐藏关卡)

如果你想让你的 FlashAttention 或 GEMM 真正接近 SGLang/vLLM 的性能，上述 Level 1-3 的技巧是基础，你还需要引入以下 **3 个高级机制**。这在面试中属于“加分项”，但在高性能库开发中是“必选项”。

#### 1. 异步数据拷贝 (Async Copy / `cp.async`)

- **问题：** 你的代码逻辑可能是：`Load Global -> Wait -> Compute -> Store`。GPU 计算单元在等待数据加载时是空闲的。
    
- **Level 4 技巧：** Ampere (A100) 架构引入了 `cp.async` 指令。它允许计算单元发出“搬运数据”的指令后，立刻去干别的事（比如计算上一步的数据），不需要线程阻塞等待。
    
- **应用场景：** FlashAttention 的 Q、K、V 加载。
    

#### 2. 双缓冲 (Double Buffering / Pipelining)

- **问题：** 即使有了异步拷贝，如果只有一个缓冲区，你还得等。
    
- **Level 4 技巧：** 在 Shared Memory 中开辟两块空间（Buffer A 和 Buffer B）。
    
    - 当 Tensor Core 在计算 Buffer A 时。
        
    - Load Unit 利用 `cp.async` 把下一批数据预取到 Buffer B。
        
    - 这就是所谓的 **掩盖延迟 (Latency Hiding)** 的终极形态。
        
- **面试话术：** “在优化 FlashAttention 时，为了掩盖 HBM 的访问延迟，我设计了一个主循环流水线，利用 Shared Memory 做 Double Buffering，并配合 `cp.async` 指令实现计算与访存的完美重叠。”
    

#### 3. Shared Memory Swizzling (解决 Bank Conflict 的终极方案)

- **问题：** 在矩阵乘法或 FlashAttention 中，当 Thread Block 把数据从 Shared Memory 加载到寄存器喂给 Tensor Core 时，非常容易发生 Bank Conflict。
    
- **Level 4 技巧：** 简单的 Padding（填充）可能不够。你需要用 **XOR Swizzling**（异或重排）这种地址映射技巧，让连续的内存地址映射到不同的 Bank 上。
    
- **CUTLASS/SGLang 现状：** 它们内部大量使用了这种技巧来保证 Tensor Core 吃数据的速度不被 Shared Memory 卡住。
    

---

### 三、 针对你 MiniInfer 项目的建议路径

既然你要做 MiniInfer，我建议你采取以下**分治策略**，既能学到东西，又不会陷入造轮子的泥潭：

1. **Elemwise & Reduction (RoPE, Softmax, RMSNorm, Silu):**
    
    - **全部手写！**
        
    - 使用 Level 1 & 2 的技巧（`float4`, `shfl`, `grid stride loop`）。
        
    - 这是你练习 CUDA 基本功最好的地方，而且这部分很容易做到比 PyTorch 快。
        
2. **Attention (FlashAttention):**
    
    - **核心挑战项目。**
        
    - 先尝试写一个 Naive 版本（不带 Double Buffering），跑通 Online Softmax 的逻辑。
        
    - 重点展示你对 **SRAM 管理** 的理解。
        
3. **Linear / GEMM:**
    
    - **不要手写 Kernel。**
        
    - 直接调用 **cuBLAS** 或者集成 **CUTLASS**。
        
    - **原因：** 除非你是去 NVIDIA 写库，否则手写 GEMM 投入产出比太低。你应该把精力花在“如何把 GEMM 和后面的 Activation 融合 (Fusion)”上，而不是写 GEMM 本身。
        

### 总结

**你列出的技巧是“九阳神功”的内功心法，完全足够支撑你通过面试。**

只要你在写代码时，脑子里有：

1. **Coalesced Access** (Global Mem 不浪费带宽)
    
2. **Shared Memory** (作为显式缓存)
    
3. **Vectorized Access** (`float4` 提升吞吐)
    
4. **Warp Primitive** (寄存器级通信)
    

你就已经超越了绝大多数竞争者。至于 Double Buffering 和 Swizzling，那是等你先把基础版跑通后，用来刷榜（SOTA）用的。

**下一步建议：** 不用再犹豫是否“足够”，立刻开始写一个 **Vectorized RMSNorm**。写完了发给我，我帮你 Code Review，看看有没有达到 Level 2 的标准。