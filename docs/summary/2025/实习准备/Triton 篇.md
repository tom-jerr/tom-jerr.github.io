以下是具体的掌握标准和技术点拆解：

---

### 第一部分：OpenAI Triton (必须掌握，达到能写、能调优的程度)

Triton 是目前的“当红炸子鸡”，SGLang、vLLM、TorchInductor 底层全都在用。面试官默认你如果不写 CUDA，就得会写 Triton。

#### 1. 核心思维转变：Block-based Programming

- **CUDA vs. Triton：**
    
    - CUDA 是 **SIMT (Single Instruction Multiple Threads)**，你控制的是一个个线程（Thread），需要自己处理线程到 Block 的映射。
        
    - Triton 是 **Block-based**，你操作的是 **块（Tile/Block）**。你要习惯写 `tl.load(ptr + offsets)` 一次加载一个块的数据。
        
- **考核点：** 能解释清楚为什么 Triton 编写起来比 CUDA 简单（**自动处理了 Shared Memory 的管理和部分同步**）。
    

#### 2. 必须掌握的 API 与概念

- **指针运算 (Pointer Arithmetic)：** 理解 `tl.program_id`，以及如何生成 `offsets` 来切分大矩阵。
    
- **Masking (掩码)：** 处理边界条件（当 Tensor 大小不是 Block 大小的整数倍时）。`tl.load(..., mask=...)` 是必考点。
    
- **高级优化指令：**
    
    - `tl.dot`: 矩阵乘法指令，Triton 编译器会自动把它映射到 Tensor Core (MMA) 上。
        
    - **Block Pointers (新特性)：** 如果你知道 `tl.make_block_ptr`，这证明你跟进了最新版本，知道这能简化多维数据的加载。
        

#### 3. 内存层级控制

- 即使是 Triton，你也要懂 **Coalescing**。Triton 编译器虽然聪明，但如果你写的 access pattern 很乱，它也救不了你。
    
- **L2 Cache Optimization：** 了解 Swizzle（重新排序 Block 的执行顺序）以增加 L2 Cache 命中率。这在 Triton 的教程里有专门的一章（Matrix Multiplication），**必看**。
    

**📌 检验标准：** 你能用 Triton 在 20 行代码内写出一个性能尚可的 **Softmax** 或 **LayerNorm**，并能解释代码里每一行的作用。

---

### 第二部分：AI 编译器原理 (MLIR / TVM / TorchCompile)

对于实习生，**不需要**你会写 C++ Pass，但必须懂 **“编译器的 pipeline 是怎么工作的”** 以及 **“它到底优化了什么”**。

#### 1. 核心概念：Lowering (降级) & IR (中间表示)

- **流程：** High-level Graph (PyTorch) -> Logical IR (计算图) -> Physical IR (Triton/CUDA) -> Machine Code (PTX/SASS)。
    
- **MLIR 的作用：** 它提供了一套通用的基础设施。你需要知道 **Dialect (方言)** 的概念。比如 `Linalg` Dialect 描述线性代数运算，`Affine` Dialect 描述循环。
    
- **面试话术：** _“MLIR 解决了编译器碎片化的问题，通过多层 Dialect 的逐级 Lowering，让算子融合和代码生成变得更模块化。”_
    

#### 2. 最重要的优化手段：Operator Fusion (算子融合)

- 这是 AI 编译器存在的最大意义，也是字节面试 **必问** 的点。
    
- **垂直融合 (Vertical Fusion)：** 比如 `Conv -> BatchNorm -> ReLU` 融合成一个 Kernel，减少显存读写。
    
- **水平融合 (Horizontal Fusion)：** 把两个不相关的、形状相同的矩阵乘法拼在一起做。
    
- **Loop Tiling & Reordering：** 编译器如何自动把大循环切块（Tile）以适应 CPU/GPU 的 Cache。
    

#### 3. 重点关注：TorchDynamo & TorchInductor

- 这是 PyTorch 2.0 的核心，也是目前工业界的主流。
    
- **工作流：**
    1. **Dynamo:** 捕获 PyTorch 的计算图（解决动态图捕获难题）。
    2. **Inductor:** 也就是后端编译器，它会生成 **Triton Kernel**。
- **字节 Data-AML 考点：** 他们非常关注如何处理 **Dynamic Shape**（动态形状）。你需要了解编译器在遇到动态 Shape 时，是会重新编译（Recompile）还是生成通用的 Kernel（Symbolic Shape）。
    

---

### 针对你背景的“投机取巧”策略

你时间有限，不要去啃《编译原理》龙书。利用你现有的项目来“伪装”成专家：

#### 策略 1：利用 SGLang 谈编译

SGLang 本身就有一层类似编译器的设计。

- **观察：** SGLang 的 `sgl.function` 装饰器其实就是在做 Tracing（追踪），生成了一个内部的 IR（中间表示），然后由解释器去执行。
    
- **话术：** _“在研究 SGLang 源码时，我发现它处理 Prompt 的方式其实类似于编译器前端的 Tracing。它构建了一个计算图来管理 KV Cache 的分配，这让我对计算图优化有了具体的理解。”_
    

#### 策略 2：用 Triton 优化 MiniInfer

- **动作：** 在你的 MiniInfer 里，用 C++ 调用一个 Python 生成的 Triton Kernel（或者只是在简历里说你对比过）。
    
- **展示：** 甚至你可以用 Triton 重写 MiniInfer 里的 RoPE。
    
- **目的：** 证明你懂 Triton 的语法，以及它生成的 PTX 代码质量如何。
    

### 总结：你需要掌握到什么程度？

1. **Triton:** **Level 4 (熟练应用)。** 必须能看懂 SGLang 里的 Triton Kernel 代码。建议手写一遍 **Matrix Multiplication** 和 **FlashAttention (简化版)** 的 Triton 教程。
    
2. **TorchCompile/Inductor:** **Level 3 (理解架构)。** 知道它怎么把 PyTorch 变成 Triton 的。
    
3. **MLIR/TVM:** **Level 2 (概念认知)。** 知道它们是做什么的，知道 **Lowering**、**Fusion**、**Tiling** 这三个词的含义即可。除非你去专门做编译器的组，否则不需要写 MLIR 代码。

**一句话总结：** 字节的面试官会问你：“为什么 PyTorch 2.0 `torch.compile` 会变快？” **满分回答要包含：** “因为它利用 Dynamo 捕获了计算图，消除了 Python 开销；通过 Inductor 进行了激进的算子融合（Operator Fusion），减少了 HBM 访问；最后生成了针对当前硬件特化的 Triton Kernel。”