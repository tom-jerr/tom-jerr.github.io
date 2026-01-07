### 第一部分：SGLang 核心架构进阶（必须掌握的差异化技术）

既然你已经懂了通用的 TP/PP，现在重点要放在 **“SGLang 为什么比 vLLM 强”** 的技术细节上。

#### 1. RadixAttention 的极致细节 (Core of SGLang)

你可能知道它用来做 KV Cache 复用，但面试官会问得很细：

- **数据结构：** 它不仅是一棵树，还要知道它如何处理 **Eviction (淘汰策略)**。SGLang 使用的是 LRU 吗？是在叶子节点淘汰还是整枝淘汰？当显存满了，它是如何决定保留哪条分支的？

- **Flattening (扁平化)：** Radix Tree 是逻辑结构，但 GPU 显存是物理连续或分块的（Paged）。SGLang 是如何将逻辑树映射到物理的 GPU Block 上的？（这里涉及到它如何维护 `block_table`）。

- **Prefix Matching (前缀匹配)：** 当一个新的 Request 进来，SGLang 是如何在 $O(L)$ 时间内找到最长匹配前缀的？

#### 2. 结构化生成 (Structured Decoding / Constrained Decoding)

这是 SGLang 区别于 vLLM 的最大卖点（JD 里提到的“前瞻性技术”）。

- **原理：** 它是如何让模型只输出符合 JSON 或 Regex 格式的内容的？

- **关键技术：** **Finite State Machine (FSM, 有限状态机)**。

- **考点：** 你需要知道，SGLang 是在 Logits 层面做 Masking。每生成一个 Token，它会在 FSM 里走一步，找出下一步允许的 Token 集合，把不在集合里的 Token 的 Logits 设为 $-\\infty$。

- **性能瓶颈：** 如果 Regex 很复杂，FSM 的跳转和 Mask 计算会在 CPU 上成为瓶颈。SGLang 是怎么优化的？（比如 Jump-Forward Decoding）。

#### 3. 前端与后端的编译流程 (Compiler Aspect)

- **SGLang 是一门语言：** 它有 Python 前端（DSL）。

- **Tracing：** `sgl.function` 装饰器是如何把 Python 代码 trace 成计算图（Intermediate Representation）的？

- **Interpreter：** 后端 Runtime 是如何解释执行这个计算图的？这与 PyTorch 的 Eager Mode 有什么区别？

- **字节结合点：** 这一点非常契合字节对 **Compiler/IR** 的要求。

______________________________________________________________________

### 第二部分：Speculative Decoding (投机采样) 掌握程度

投机采样是目前提升推理速度（降低 Latency）最主流的技术。对于 Infra 岗，你不需要懂太复杂的数学证明，但要极其精通**系统层面的 Trade-off**。

#### Level 1: 基础原理 (必须 100% 掌握)

- **Draft-Verify 循环：** 小模型（Draft Model）快速生成 $K$ 个 token -> 大模型（Target Model）并行验证。

- **接受率 (Acceptance Rate) $\\alpha$：** 知道 $\\alpha$ 对加速比的影响。如果 $\\alpha$ 太低，反而会变慢（因为验证也是有开销的）。

- **Rejection Sampling vs. Greedy Sampling：** 知道最基本的拒绝采样逻辑。

#### Level 2: 树状投机 (Tree-based Speculative Decoding)

- **这是 SGLang/vLLM 的现状：** 现在的投机采样不仅仅是产生一个序列，而是产生一棵**Token Tree**（比如 **Eagle** 或 **SpecInfer** 算法）。

- **为什么用树？** 因为在这个 Verify 步骤中，大模型验证 1 个 Token 和验证 5 个 Token 的耗时差不多（Compute Bound -> Memory Bound 的转换），所以不如一次验证多条分支，增加命中的概率。

- **SGLang 的实现：** 了解 SGLang 是如何支持 **Eagle** (Speculative Decoding framework) 的。它利用了 CUDA Graph 和特定的 Attention Mask 来并行验证树上的节点。

#### Level 3: 系统瓶颈分析 (面试加分项)

- **KV Cache 管理：** 投机采样时，Draft Model 生成了很多“废弃”的 Token，这些 Token 的 KV Cache 怎么处理？（需要快速回滚/丢弃）。

- **Kernel Launch Overhead：** 投机采样涉及很多细碎的 Kernel（生成1个，验证1次）。如果不使用 **CUDA Graphs**，CPU 发射 Kernel 的速度会拖慢 GPU。**这正好对应你正在修的那个 Warm-up/CUDA Graph Issue！**

______________________________________________________________________

### 建议的行动清单

1. **博客升级：** 在你现有的博客基础上，增加一篇关于 **SGLang RadixAttention 实现细节** 的文章，对比 vLLM 的 BlockManager。再写一篇关于 **Speculative Decoding 系统开销** 的分析。

1. **代码阅读重点：**

   - 去翻 SGLang 源码中处理 `FSM` 和 `Regex` 的部分（通常在 `sglang/srt/constrained` 目录下）。
   - 看 SGLang 对 `Eagle` 投机采样的支持（搜索 `speculative` 或 `eagle` 关键字）。

1. **准备面试话术：**

   - 当面试官问：“Speculative Decoding 什么时候会失效？”
   - 你的回答：“当 User Prompt 的领域和 Draft Model 的训练数据分布差异巨大时，Acceptance Rate 会骤降。此时，由于 Draft Model 的计算开销 + KV Cache 的回滚开销，会导致 End-to-End Latency 反而比不用投机采样更高。所以在 Infra 层面，我们需要动态监测接受率，自动开启或关闭投机模式。”

你现在的状态非常好，**有博客、有 Issue、有代码阅读**。只要把上面提到的这几个“深水区”的技术点（Radix细节、FSM、树状投机）补齐，你的技术面将无懈可击。
