以下是具体的掌握标准拆解：

______________________________________________________________________

### Level 1: API 与生命周期 (必须熟练使用)

这是基础，你的 MiniInfer 和 SGLang 任务都必须用到。

1. **核心流程 (The Workflow)：**

   - **Capture (捕获):** `cudaStreamBeginCapture` / `cudaStreamEndCapture`。知道这一步是在“录像”，GPU 不会立刻执行。

   - **Instantiate (实例化):** `cudaGraphInstantiate`。这是最耗时的一步（Validation + Optimization），它把 Graph 变成了可执行的 `GraphExec` 对象。**这就是为什么你需要 Warm-up 的核心原因——为了提前把这一步做完。**

   - **Launch (发射):** `cudaGraphLaunch`。极快，几乎没有 CPU 开销。

1. **Stream 的关系：**

   - 知道 Capture 是针对特定 Stream 的。在 Capture 期间，向该 Stream 提交的所有 Kernel 都会被录入 Graph。

   - 知道向 _其他_ Stream 提交任务会发生什么（通常会导致 Capture 失败或变成依赖关系）。

______________________________________________________________________

### Level 2: 限制与陷阱 (面试核心考点)

这是**字节面试官最喜欢问的**，因为这体现了你是否有实际工程经验。

1. **静态图 vs. 动态逻辑 (The Static Nature)：**

   - **CPU 逻辑失效：** 在 Capture 期间，任何 CPU 端的 `if-else` 或 `for` 循环只有在“录制”那一刻生效。Graph 录制下来的是一条固定的执行路径。

   - **面试题：** _“如果在 Capture 期间有一个 `if (random() > 0.5)`，Graph 运行的时候会保留这个随机性吗？”_

   - **答案：** 不会。它会永远执行录制时走的那条路。

1. **指针地址固定 (Pointer Baking)：**

   - **这是最大的坑：** Graph 录制时，Kernel 参数里的\*\*显存指针地址（Pointer Address）\*\*是被“写死”在 Graph 里的。

   - **后果：** 如果你下一轮推理，KV Cache 换了个地方存（SGLang 的 Memory Pool 重新分配了地址），但你还在用旧 Graph，程序会崩溃或算错。

   - **SGLang 的解法：** SGLang 使用 **Mempool (内存池)** 或 **Private Buffer**。Warm-up 时确定的地址，在后续推理时必须保证不变（或者保证复用同一块 Buffer）。**你在修 Issue 时必须确认这一点：Warm-up 用的 Buffer 和实际推理用的 Buffer 必须在物理地址上兼容。**

1. **动态 Shape 问题：**

   - Graph 对 Input Tensor 的 Shape 也是敏感的（通常）。

   - **解法：** 针对常见的 Batch Size (1, 2, 4, 8...) 分别录制不同的 Graph 并缓存起来（Graph Caching）。

______________________________________________________________________

### Level 3: 高级优化与调试 (Data-AML/Seed 进阶要求)

1. **Graph Update (图更新)：**

   - 如果只是 input tensor 的**内容**变了（地址没变），不需要更新 Graph。

   - 如果 input tensor 的**地址**变了，可以使用 `cudaGraphExecUpdate` 来更新参数，而不需要重新 Capture 和 Instantiate（前者太慢）。

   - **实战：** 了解 SGLang 是否使用了 `ExecUpdate`，还是简单粗暴地维护了多个 Graph 实例？（通常为了稳定性，维护多实例更常见）。

1. **Mempool 管理：**

   - CUDA Graph 能够捕获 `cudaMalloc` 吗？

   - **答案：** 现代 CUDA (11.x+) 支持 `cudaGraphAddMemAllocNode`，支持在 Graph 内部做内存分配。但这通常很难用。

   - **工业界做法：** **Static Allocation**。在 Capture 之前就把最大显存申请好，Graph 内部只做计算，不碰内存分配。**SGLang 就是这么做的，它预先分配了巨大的 KV Cache 池。**

1. **调试 (Debugging)：**

   - 当使用了 CUDA Graph 后，`nsys` (Nsight Systems) 看到的不再是一个个碎的 Kernel，而是一个巨大的块（Graph Launch）。

   - **技巧：** 知道如何在 `nsys` 中展开 Graph 内部的 Kernel 详情（通常工具会自动支持，但有时需要特定版本）。

______________________________________________________________________

### 针对你的 SGLang Diffusion Warm-up 任务

你现在手头的这个 Issue，完美对应了上述知识点。你需要搞清楚：

1. **Warm-up 到底在 Warm-up 什么？**

   - 它在跑一遍模型，触发 PyTorch 的 Lazy Initialization，分配显存，**并且（关键点）录制 CUDA Graph**。

1. **Diffusion 的特殊性：**

   - Diffusion 是一个 UNet 循环运行几十次。

   - 如果没有 Graph，每一次循环都要 CPU 发射几百个 Kernel，开销巨大。

   - **你的目标：** 把这几十次循环（或者单次去噪步）录制成 Graph。

1. **你需要解决的冲突：**

   - Warm-up 时用的 Input Tensor 是假的（Dummy Input）。

   - 实际推理时用的是真的。

   - **问题：** 怎么保证 Graph 录制的 Dummy Input 地址和实际 Input 地址的兼容性？（通过 `torch.cuda.make_graphed_callables` 或者 SGLang 自己的 Graph Runner 封装）。

### 总结：你需要掌握到什么程度？

1. **API：** 熟练。

1. **原理：** 深刻理解 **“指针固化”** 和 **“CPU 逻辑消失”** 这两个特性。

1. **工程：** 知道如何结合 **Mempool** 来规避地址变化的问题。

**面试必问模拟题：** _“我们在 SGLang 中使用了 CUDA Graph，但是发现显存池（Mempool）碎片化整理后，推理结果全错了，可能是什么原因？”_ **(答案：碎片整理移动了物理内存地址，但 Graph 内部还持有旧的指针地址。必须 Re-capture 或 Update Graph。)**
