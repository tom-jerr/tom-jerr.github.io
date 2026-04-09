
# CUDA Graph
## 原理
CUDA Graph 将一段 GPU kernel 序列录制为静态 DAG，之后只需一次 CPU launch 即可重放整个计算流，以此消除逐个 kernel launch 的 CPU 开销。在此基础上，我们更进一步理解 CUDA Graph 的一些核心机制。
## 构造过程
1. Capture：捕获或者是录制 CUDA Graph。
   - 调用 cudaStreamBeginCapture() 后，CUDA runtime 进入录制模式——后续所有提交到该 stream 的操作（kernel launch、memcpy、memset 等）都不会真正执行，而是被记录为 DAG 中的节点。
   - 每个节点保存的信息包括：要调用哪个 kernel、grid/block 维度、以及所有参数的值（对 tensor 而言就是其 GPU 虚拟地址）。
   - 节点之间的边由 stream 上的提交顺序和跨 stream 的 event 同步自动推断。
   - 录制结束时，调用 cudaStreamEndCapture() **返回一个 cudaGraph_t 作为纯粹的拓扑关系的描述**，不能直接执行。
2. Instantiate：实例化 CUDA Graph。
      - 得到 cudaGraph_t 后，进一步调用 cudaGraphInstantiate() 将其编译为 cudaGraphExec_t。capture 类似于录制脚本，**instantiate 则是编译脚本为可执行二进制**——前者是声明式的描述，后者是命令式的执行计划：
     - 依赖分析与调度：遍历 DAG 拓扑，确定哪些 kernel 之间有真正的数据依赖、哪些可以并发执行，生成一份最优的执行计划。
     - 参数绑定与固化：将 capture 阶段录制的所有 kernel 参数，比如 tensor 的 GPU 指针等等，烘焙（bake）进可执行对象中。从中我们也能看出，之后每次利用 CUDA Graph replay 时 tensor 的地址必须保持不变，因为虚拟地址已经被焊死在 cudaGraphExec_t 里了。
     - 合法性校验：检查图中是否存在不支持的操作（如 host-device sync），不合法则 instantiate 返回失败。
3. Replay：CUDA Graph 重放。
   - 调用 cudaGraphLaunch(exec, stream) 将**整个 cudaGraphExec_t 一次性提交到指定 stream 上**
   - CPU 只发出一次 launch 指令，GPU 端的调度器按 instantiate 阶段生成的执行计划依次（或并发地）执行所有 kernel，消除了逐个 kernel launch 的 CPU 开销。
   - 由于 replay 不经过 Python/PyTorch 的 dispatcher，也没有 CPU 端的逐 operation 调度，CPU overhead 几乎降为零。
  
对于推理这种高度重复的固定计算流（每个 decode step 执行相同的 kernel 序列），capture 一次、instantiate 一次、replay 无数次，节省下大量 CPU launch 开销，这就是 CUDA Graph 的核心价值。

## 约束条件
启动阶段执行的操作能够进一步推导出 CUDA Graph 的约束条件，也即在什么情况下 CUDA Graph 会被破坏：
| 约束                     | 含义                                                       | 为什么                                                            |
| ------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------- |
| 指针稳定性               | replay 时必须保证 GPU 虚拟地址不变                         | capture 时录制的是内存地址，地址变了，kernel 就会读写错误的内存   |
| 不能有动态内存分配       | capture 期间所有 tensor 必须预分配                         | 动态创建如 `torch.zeros(...)` 会触发 allocator，导致地址不可预测  |
| 不能有 host-device sync  | 不能调用 `.item()`、`torch.multinomial` 等会导致同步的操作 | sync 会中断 stream capture，导致 capture 失败                     |
| 静态控制流               | 循环次数、分支条件在 capture 时需要固定                    | graph 是静态 DAG，不支持运行时再决定条件分支                      |
| graph 录制后不会自动更新 | 改了代码路径后必须重新 capture                             | 录制完成后，kernel 序列已经固化，不会随着 Python 代码变化自动改变 |

一份 graph 只能服务一种 batch size。具体来说，考虑到 capture 阶段录制 kernel 参数（grid 维度、tensor shape/地址）会被固化进 cudaGraphExec_t，而 batch size 改变，则这些参数会全部失效。


## CUDA Graph 显存开销与共享机制
每个 graph 在 capture 阶段执行的所有中间计算都会产生中间 tensor（attention score、FFN 中间激活、残差加法的输出等）。这些中间 tensor 的地址会被写入 cudaGraphExec_t，replay 时 kernel 直接读写这些固定地址。
因此，capture 期间 memory pool 分配出的这块显存区域会被 CUDA Graph 整体锁定，不会还给 PyTorch 的通用 allocator。
- 我们可以将 CUDA Graph 的 memory pool 视作一块封闭的独立显存区域：graph 内部的 tensor 只能使用 pool 内的显存，外部 tensor 不能占用 pool 里的空间，内外隔离。
- 但在这个围起来的区域内部，PyTorch 的 caching allocator 仍然正常工作：先产生的中间 tensor 如果后续不再被引用，其地址可以被后面的 tensor 复用。

> [!IMPORTANT]
> **单个 graph 持有的显存约等于 capture 期间的显存峰值（high-water mark），而非所有曾经出现过的 tensor 的简单加总。**

举个例子，一个 32 层 Transformer 的 forward，每层的中间激活在下一层开始后就可以被复用，high-water mark 可能只相当于几层的中间 tensor，而远非 32 层的总和。


尽管如此，如果有 12 个不同 batch size 的 graph 且各自独立持有一份 high-water mark 的显存，总占用仍然是 12 倍——这在大模型推理中依然不可接受。这里进一步引出**不同 Graph 之间的显存共享机制**。
- 前文提及的内外隔离，是指 graph pool 与非 graph 的普通 PyTorch 代码之间的隔离，但多个 graph 之间并不需要隔离。
- PyTorch 通过 torch.cuda.graph(pool=...) 允许多个 graph 共享同一个 pool，它们的中间 tensor 都从同一块显存区域中分配。
> [!IMPORTANT]
> 这之所以安全，是因为 decode 阶段每个 step 只会选择一个 batch size 对应的 graph 来 replay，不同 graph 的中间 tensor 不会同时存活，可以轮流复用。
> 
> 这样，不管有多少个 graph，显存开销只相当于最大那个 graph 的一份 high-water mark。
## Piecewise Graph
部分算子（主要是 Attention）与 CUDA Graph capture 不兼容，但又希望模型的其他部分能被 CUDA Graph 加速。解决方案是将整个模型的 FX 图在这些算子处"切断"，形成若干子图——包含 Attention 的"splitting subgraph"在 CUDA Graph 外执行，周围的子图则被 CUDA Graph 捕获。
- eager 操作比如 attention 已经在 custom_op 中注册。
- 
### Graph Break
重要区分：这里的 graph break 不是 Dynamo 的 graph break（那是 bug，vLLM 要求 Dynamo 必须全图捕获）。这是一个发生在 Dynamo 之后、Inductor 之前的有意识的 FX 图划分。

**策略 1：FX 级别 split（默认）**

在 VllmBackend.__call__ 中，Dynamo 传来的完整 FX 图被 split_graph() 切割：
- split_graph() 根据 splitting_ops 列表（默认包含 Attention）将 FX 图切成多个子图
- 每个子图被标记为 splitting（包含 Attention）或 non-splitting

**策略 2：Inductor 级别 partition**
The full FX graph is passed to Inductor; Inductor's own scheduler performs partitioning after all fusion passes have run
- 当它找到操作名出现在 torch._inductor.config.custom_should_partition_ops 的 FallbackKernel 节点时，返回 True，迫使图在该边界被拆分
- eagler 算子比如 attention 使用已经注册的 custom_op 实现，其他算子可以正常被 Inductor 处理和融合

### 流程
- 对每个非 splitting 子图，创建 PiecewiseBackend 并编译所有 ranges（包括 symbolic range 和 static sizes）
- PiecewiseBackend 在初始化时调用 compile_all_ranges()，对每个 compile_range 提前编译好对应的 runnable。运行时根据 runtime_shape 查找对应 range 的 runnable 并执行
- 非 splitting 子图的 PiecewiseBackend 最终被 wrap_with_cudagraph_if_needed() 包裹成 CUDA Graph wrapper
# torch.compile
```shell
Python eager 代码
→ TorchDynamo 截获 Python frame / 字节码
→ 抽取 FX Graph + 记录 guards(这份编译结果成立的前提条件)
→ AOTAutograd（训练场景）做前反向图分解
→ TorchInductor 做图级优化、fusion、调度、代码生成
→ 后端落到 Triton / CPP / ATen 等
→ Triton 再把 kernel 编译成设备可执行代码
→ 运行时按 guards 检查是否可复用旧编译结果
```
- Dynamo 更像一个 Python 层图抽取器 + 特化入口，不是最后生成 GPU kernel 的地方。官方文档也明确说：每个 frame 都会尝试编译，并把**编译结果缓存在 code object 上；如果后续调用不满足之前的 guards，就会发生 guard failure，然后重新编译**

## compile cache
### 第一层：Dynamo 的 frame / guard 级缓存

缓存的是：
- 某段 Python frame 对应的已编译结果
- 它配套的 guards

只要这次调用还满足旧 guards，就直接复用，不重新抓图/编译。官方文档明确说它会把编译结果缓存在 code object 上，并在 guard failure 时重新编译。

### 第二层：Inductor 的图级缓存

缓存的是：

- FX Graph / 规范化后的图表示
- 对应生成过的后端代码
- 一些中间 IR 或编译结果索引

PyTorch 官方 compile caching recipe 里明确提到，默认有本地磁盘缓存，里面包括 FXGraphCache 等模块化缓存。

### 第三层：Triton kernel 编译缓存

缓存的是：

- Triton kernel 源/IR 对应的设备代码
- 和 launch 相关的元数据
- autotune 结果或编译 key 相关信息

这一层更接近“传统 JIT kernel cache”。
## 原理
`torch.compile` 是 PyTorch 提供的一个用于加速模型推理和训练的工具。它通过将 PyTorch 代码转换为更高效的中间表示（IR），并利用各种优化技术来提升性能。其核心原理包括以下几个方面：
- Dynamo: 拦截 Python 执行代码，提取 tensor op 构成静态 IR 图，跳过 Python 解释器
- Inductor: 算子融合 + kernel 合并 + 内存优化
- CUDAGraph: 减少 CPU 调度开销，一次录制多次执行
