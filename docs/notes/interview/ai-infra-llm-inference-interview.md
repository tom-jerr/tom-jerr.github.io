# AI Infra / LLM Inference 面试：问答题库 + 知识点总结（基于本仓库笔记 + 面经/JD）

> 适用岗位：大模型推理系统 / 推理优化 / AI Infra（Serving + CUDA + 分布式 + 云原生）  
> 目标：用一份文档把“你本地笔记的知识框架”对齐到“面经/JD的考察点”，便于刷题复习与系统设计表达。

---

## 0. 本仓库的复习路径（把题目落到你的笔记上）

**Attention / Transformer 推理基础**
- `docs/notes/llm_inference/attention&transformer.md`
- `docs/notes/llm_inference/transformerbasedllm.md`

**FlashAttention / Kernel**
- `docs/blogs/posts/FlashAttention 原理 v1-v2.md`
- `docs/blogs/posts/PageAttention.md`

**Serving 调度 / Batching / KV Cache / Prefix Cache（SGLang 视角）**
- `docs/notes/sglang/大模型推理服务中的 Batching.md`
- `docs/notes/sglang/SGLang Scheduler 技术变迁.md`
- `docs/notes/sglang/从代码看 SGLang 的 KV Cache.md`
- `docs/notes/sglang/RadixAttention 你需要知道的全部细节.md`

**PD 分离（Prefill/Decode Disaggregation）**
- `docs/feishu/PD Disaggregation in SGLang.md`

**并行与分布式（推理/训练交界）**
- `docs/notes/llm_inference/LLMparallelization.md`

---

## 1. 岗位画像（从 JD 里抽象出来的“他们到底想要什么”）

把 JD 关键词统一成 4 条主线（面试也基本沿这 4 条追问）：

1) **性能目标会算**：吞吐/时延/显存/成本的 tradeoff；TTFT/TPOT、prefill vs decode、compute-bound vs memory-bound。  
2) **系统能落地**：路由/队列/调度/隔离/限流/弹性；多租户、稳定性、可观测。  
3) **GPU/Kernel 真懂**：内存层次结构、访存与并行、算子融合、FlashAttention/PagedAttention 类 kernel 的边界条件与工程细节。  
4) **分布式与网络不虚**：TP/PP/SP、AllReduce/AllGather、RDMA、P/D 分离下的 KV 传输与一致性/容错/背压。

（典型 JD 示例可见：小红书 MaaS 推理系统研发、快手 AI Infra 研发等，文末有链接）

---

## 2. 一句话性能模型（面试最吃香的表达）

- **Prefill**：更像“高算力大矩阵计算”，常见瓶颈是 **FLOPs / TensorCore 利用率 / 并行策略**。  
- **Decode**：每步只生成 1 token（或很少 token），更像“搬运 KV + 小算子”，常见瓶颈是 **HBM 带宽 / KV 访问模式 / cache miss**。  
- **Serving 优化的本质**：把 GPU 从“单请求串行”改成“多请求并行”，并且让 **KV Cache 的存储与访问** 变得可控、可回收、可迁移。

---

## 3. 知识点总结（按面经高频排序）

### 3.1 指标体系与压测方法
- 指标：`throughput (tok/s)`, `latency`, `TTFT`, `TPOT`, `P95/P99`, `goodput`（满足 SLO 的吞吐）。  
- 场景拆分：短 prompt / 长 prompt；长输出 / 短输出；突发 / 平稳；多租户混部。  
- 归因常用：prefill 占比、decode 占比、KV 占比、排队占比、网络占比。

### 3.2 KV Cache：容量、碎片、回收与迁移
- KV Cache 规模估算（面试常问你能不能心算）：大致与 `layers * seq_len * hidden * bytes` 同阶（精确要乘 head 与 K/V 两份以及实现细节）。  
- 核心矛盾：**为每个请求分连续大块显存** 会导致严重碎片；而 **非连续块** 则要求 kernel 端支持 gather。  
- 工程问题：OOM 时策略（recompute vs swap-out）、抢占/回退（retract）、跨实例迁移（PD 分离）。

### 3.3 Batching 与调度演化
- Static batching → Continuous batching（迭代级调度） → Paged/PageAttention（异构 batch 的 kernel 支撑） → Chunked prefill（降低 TTFT） → Overlap / zero-overhead schedule（减少 CPU launch / stream 同步开销）。  
- 常见策略维度：prefill-first、decode-first、混合 batch、chunk size、fairness（防止长请求饿死短请求）。

### 3.4 Prefix Cache：Radix / Trie / 命中率驱动调度
- 目标：复用“相同前缀”的 KV，避免重复 prefill。  
- 关键点：prefix 的组织（radix tree + path compression）、命中判断、驱逐策略（LRU/成本模型）、尾部不足页如何处理、并发安全。  

### 3.5 FlashAttention / Attention Kernel（你要能讲到“为什么快”）
- 传统 attention 的瓶颈：`QK^T` 与 softmax 中间结果导致大量 HBM 读写。  
- FlashAttention 核心：**tiling + kernel fusion + online (safe) softmax**，把中间状态留在 SRAM/寄存器，减少 HBM 流量。  
- v2 重点：更好的 warp 分工、并行与非 matmul 的优化；解码侧常见还有 “Flash Decoding / split-k + reduce” 思路。

### 3.6 CUDA / GPU 基础（面经里最容易被细问）
- 访存：coalescing、对齐、vectorized load/store、bank conflict 与 swizzle。  
- 并行：warp、occupancy、register pressure、shared memory、异步拷贝（架构相关）、stream 并发与同步。  
- 常见手撕：reduce / softmax / transpose / GEMM 变体；以及 LRU/缓存结构（后端岗位也爱问）。

### 3.7 分布式推理：TP/PP/SP 与通信
- TP：AllReduce/ReduceScatter 常在 MLP/attention 的投影层；decode 阶段通信更频繁且更敏感于延迟。  
- SP/CP：长上下文下切 seq，常要讨论 ring vs tree attention。  
- 组合策略：prefill 与 decode 的并行策略可能不同（这也是 PD 分离常见动机）。

### 3.8 PD 分离（Prefill/Decode Disaggregation）
- 直觉：prefill compute-bound，decode bandwidth-bound；拆开后可以 **分别选最合适的并行与资源形态**。  
- 难点：KV 传输（RDMA/高速网络）、控制面/数据面、状态机、背压与容错、跨集群调度。

### 3.9 LLM 推理系统瓶颈分析（怎么定位“到底慢在哪”）
- 先拆指标：TTFT（排队 + prefill） vs TPOT（decode step + 调度开销）；同时盯 P50/P95/P99 与 goodput。  
- 再拆链路：网关/鉴权/限流 → tokenizer → queue/scheduler → GPU kernel → 通信（TP/PP/SP/RDMA）→ 输出回传。  
- 判定 compute-bound vs memory-bound：prefill 常偏 compute-bound；decode 常偏 memory-bound（KV 读写）+ launch/sync；用 profiler 看 `achieved FLOPs`、`HBM throughput`、`SM occupancy`、以及 kernel 时间占比。  
- 最后落优化：调度（continuous batching / chunked prefill / overlap）+ 内存（paged allocator / eviction / prefix cache）+ kernel（FlashAttention/FlashDecoding/PagedAttention/fusion）+ 分布式（通信/并行策略）+ 系统（隔离/背压/限流/弹性）。

### 3.10 CUDA illegal memory access（定位与修复）
- 现象：报错通常是异步的（真正越界发生在某个 kernel，但在后续 `cudaDeviceSynchronize()`/下一次 CUDA API 才抛）。  
- 定位套路：先最小复现 + 固定随机种子；打开 `CUDA_LAUNCH_BLOCKING=1` 定位到具体 op；用 `compute-sanitizer --tool memcheck/racecheck` 做精确检查；必要时启用 device-side assert（如 PyTorch 的 `TORCH_USE_CUDA_DSA=1`）。  
- 高频根因：越界 index（block table / page table / stride 计算）；vectorized load/store 跨界（`float4/half2` 对齐与尾部处理）；shared memory 越界；并发写导致 race；使用了已经释放/复用的指针（memory pool + 异步 stream）。  
- 修复思路：边界检查（尾部 mask load/store）+ 对齐与 padding + 明确 stream 同步关系 + 对关键 index 做 assert/单测；对“可能负数/溢出”的索引先在 host 侧做校验（尤其是 retract/preempt 路径）。

---

## 4. 面试问答题库（按“从浅到深”排列）

下面每题给出一个“面试可用的答题骨架”（不是论文式长文；重点是结构与关键点）。

### A. LLM 推理基础（prefill/decode/指标）

**Q1：Prefill 和 Decode 的区别？分别容易被什么瓶颈卡住？**  
A：prefill 计算量大（大矩阵），更受 FLOPs/TensorCore/并行影响；decode 每步 token 少，主要搬 KV 和做小算子，更受 HBM 带宽、KV 访问模式、launch/sync 开销影响。落到指标上：prefill 影响 TTFT，decode 影响 TPOT。

**Q2：TTFT、TPOT、吞吐三者如何权衡？**  
A：吞吐最大化倾向大 batch / mixed batch / chunked prefill；TTFT 需要减少排队与大 prefill 的阻塞（prefill-first、chunked prefill、PD 分离）；TPOT 需要优化 decode 的 KV 访问与 kernel（Paged/PageAttention、Flash Decoding、减少同步）。

**Q3：为什么 decode 不能简单“把 batch 做大就好”？**  
A：decode 的 per-step token 少，batch 过大可能导致：排队延迟增大、尾部 request 依赖阻塞（所有 request 同步前进）、KV 压力上升引发驱逐/回退，反而恶化 P99。

### B. KV Cache / PagedAttention / PageAttention

**Q4：KV Cache 存的是什么？为什么能加速？**  
A：每层 attention 的 K/V（以及实现相关的布局），加速来自避免在每次生成新 token 时重复计算历史 token 的 K/V。

**Q5：为什么 KV Cache 会把显存“吃爆”？**  
A：KV 随序列长度线性增长；长上下文 + 并发请求会把“每个请求一条长 KV”叠加起来，显存上升很快；而碎片化会进一步降低可用容量。

**Q6：PagedAttention 的核心解决了什么？代价是什么？**  
A：用“分页/块”管理 KV，解决连续分配导致的内部/外部碎片；代价是 kernel 需要支持非连续块的 gather，block table/索引维护也有开销；以及某些访问模式可能更难做到极致连续带宽。

**Q7：PageAttention kernel（你博客里的那个实现）一般按什么结构组织？**  
A：分层并行（grid/block/warp/thread-group），把 Q/K/V 切成 tile；局部 qk_max、全局规约得到稳定 softmax；再计算 LV 并规约写回。优化点主要在：访存向量化、减少 bank conflict、规约策略、warp 分工。

### C. Batching / 调度 / Overlap

**Q8：Static batching 为什么不适合在线服务？**  
A：必须等够 batch，TTFT 不稳定；请求长度异构会产生浪费；无法利用“请求随时到达”的并发。

**Q9：Continuous batching 的关键是什么？**  
A：迭代级调度：每个 decode step 都允许把新请求插入 batch；需要维护 request state、KV cache、以及对不同长度的处理（异构 batch）。

**Q10：Chunked prefill 为什么能降低 TTFT？**  
A：把超长 prefill 拆成多段，每段之间插入 decode/新请求，减少一个超长 prompt 独占 GPU 的时间；代价是调度复杂与可能的 kernel/同步开销。

**Q11：你会如何设计一个“prefill-first but 不饿死 decode”的策略？**  
A：用预算/配额：每轮给 prefill 固定 token budget，剩余给 decode；或者用 SLO 驱动：优先满足 TTFT 的请求，但对 decode 设最大等待阈值；再引入长请求的 chunk 化与重排队机制。

**Q12：SGLang 的 overlap/zero-overhead schedule（从你笔记角度）解决什么？**  
A：减少 CPU launch 开销与 stream 同步；把上一 batch 的输出作为下一 batch 的输入依赖，通过 future/事件机制在 CPU 侧编排，达到更高的流水化与更低的调度开销。

### D. Prefix Cache / RadixAttention

**Q13：RadixAttention 与“普通 prefix cache”本质差异？**  
A：核心是 prefix 的组织结构（radix tree + 路径压缩）与高效匹配/更新；调度层可以利用命中价值排序；并且与 block/page KV 管理结合更紧密。

**Q14：尾部不足一个 page 的 token 为什么往往不共享？真的不能共享吗？**  
A：实现上为了简化 block 对齐与并发安全，常把 tail 作为独占块；理论上可以通过更细粒度的 sub-page 管理或 copy-on-write 共享，但会引入更复杂的引用计数/同步与潜在的碎片/写放大。

**Q15：高并发下 Radix tree 并发安全怎么做？GIL 会不会成为瓶颈？**  
A：工程上常用读多写少的锁策略（RWLock）、分段锁、或把写操作集中在调度线程；GIL 影响取决于热点路径是否在 Python 层以及是否释放 GIL/用原生扩展；关键是把“匹配/插入/驱逐”的代价变成可控且可度量。

### E. Speculative Decoding（推测解码）

**Q16：推测解码为什么能加速？收益上限在哪里？**  
A：用小模型一次预测多个 token（draft），大模型验证并一次吞下多 token；收益受 draft 命中率、验证开销、以及系统是否能把验证做成高效 batch 的影响；上限接近“每次验证平均接受的 token 数”。

**Q17：工程上最容易踩的坑？**  
A：命中率波动导致 tail latency 不稳定；验证阶段的 kernel/通信变复杂；KV cache 的写入/回滚；以及多租户下对公平性与成本控制更难。

### F. FlashAttention / Kernel 优化（必问）

**Q18：Online safe softmax 的“online”指什么？**  
A：softmax 需要 max 与 sum；online 是指按 tile 流式遍历时持续更新 running max 与 running sum，并保持数值稳定（safe），避免一次性 materialize 全量矩阵。

**Q19：FlashAttention 为什么能减少 HBM 读写？**  
A：tile 化后 Q/K/V 子块进 SRAM/寄存器，在片上完成 `QK^T + softmax + (softmax*V)` 融合，避免存 `QK^T` 与 softmax 矩阵；写回只写最终输出（以及必要的中间标量）。

**Q20：FlashAttention v2 你会强调哪些工程点？**  
A：warp 分工与并行策略、减少非 matmul 的开销（softmax/归约/转置类）、更好地利用 shared memory / register、以及更少的同步与更高 occupancy。

**Q21：decode 侧的 Flash Decoding 常见思路？**  
A：split-K / split-seq：每个 split 计算部分注意力并做局部 reduce，再做跨 split combine reduce；本质是让并行度足够高，同时把 softmax 的数值稳定合并做对。

### G. CUDA / GPU 基础（面经高频细问）

**Q22：CUDA 内存层次（从慢到快）以及你会怎么用？**  
A：HBM(global) → L2 → shared memory → registers；策略：尽量把重用数据放到 shared/register；保证 global 访问合并；减少共享内存 bank conflict；控制寄存器数量避免 occupancy 掉得太狠。

**Q23：什么是 memory coalescing？如何保证？**  
A：同一 warp 的线程访问连续对齐地址可合并为少量事务；保证访问模式按 threadIdx 线性映射到连续地址、对齐到 16/32/128B 边界、使用向量化 load/store（如 float4）等。

**Q24：shared memory bank conflict 怎么来的？常见解法？**  
A：同一 bank 被多个线程同周期访问产生冲突；解法：padding、改变布局（transpose + padding）、swizzle、调整访问步长与向量化方式。

**Q25：reduce 类算子怎么优化（面经原题）？**  
A：分层归约：warp-level primitives（shuffle）→ block-level shared reduction → grid-level（多 block 合并）；选择合适的并行维度（N 大 vs M 大的不同策略）；减少原子操作；对齐与向量化；必要时两阶段 kernel。

**Q26：CUDA cache 可配置吗？你会怎么用这个点？**  
A：部分架构/设置允许调整 L1/shared 的分配或 cache policy；面试表达重点是：理解“shared vs cache 的 tradeoff”，并能结合 kernel 的重用与访存模式解释为什么要调。

### H. 分布式推理与网络（AI Infra 必问）

**Q27：TP/PP/SP（或 CP）各自解决什么问题？代价是什么？**  
A：TP 切权重减单卡计算但引入通信；PP 切层降低单卡显存/计算但有 bubble；SP/CP 切序列支撑长上下文但 attention 通信更复杂。代价总体是：通信/同步/调度复杂度上升。

**Q28：ring vs tree attention 你怎么选？**  
A：ring 通信更均匀、实现相对简单；tree 可能更低延迟/更少步数但实现复杂；选型要结合拓扑、消息大小、并发与容错需求（以及框架实现）。

**Q29：PD 分离下 KV 传输的关键难点？**  
A：数据面：高吞吐低延迟（RDMA）、内存注册与 pin、zero-copy、分块传输与顺序；控制面：连接管理、元数据一致性、状态机、重试/超时；系统面：背压、限流、容错与资源隔离。

### I. 系统设计题（最后一轮常见）

**Q30：设计一个多租户 LLM Serving：你会有哪些核心组件？**  
A：入口网关（鉴权/限流/路由）→ 调度器（队列+策略）→ worker pool（prefill/decode，可能分离）→ KV/状态管理（cache/evict/迁移）→ 观测（metrics/log/trace）→ 弹性（autoscale）与故障恢复（重试/隔离/熔断）。

**Q31：如何把“成本”写进调度目标？**  
A：引入 cost model（KV 占用、prefill 代价、命中价值、网络代价），把策略从“纯吞吐”变成“goodput/成本”；例如命中率驱动调度、KV 使用率与排队负载感知调度、P/D 分离下的流量切分。

**Q32：如何排查 P99 延迟偶发尖刺？**  
A：先切分：排队 vs GPU kernel vs 网络；看是否与长 prompt、OOM 回退、cache eviction、GC/内存碎片、CPU launch/sync 抖动相关；用 trace 把一个请求的生命周期打点（enqueue → schedule → prefill chunks → decode steps → egress）。

### J. 瓶颈分析 & 线上 Debug（AI Infra 加分区）

**Q33：给你一个 LLM Serving，“吞吐还行但 P99 很差”，你如何系统定位瓶颈？**  
A：按请求生命周期拆解：排队（queueing）/tokenizer/调度/通信/GPU kernel；先用全链路 trace 拿到 P99 的时间分布，再用 profiler（Nsight Systems/Compute 或框架自带 trace）锁定热点；重点区分 TTFT vs TPOT，并检查是否有长 prompt、OOM 回退（retract/recompute/swap）、prefix cache 命中波动、以及 CPU launch/sync 抖动导致的尖刺。

**Q34：如何判断当前瓶颈是 compute-bound 还是 memory-bound？**  
A：prefill/decode 分开看：prefill 看 GEMM/attention 的 FLOPs 利用率与 TensorCore；decode 看 HBM 带宽、L2 hit、以及 KV gather 相关 kernel；用 roofline 思维：如果 HBM 已接近饱和而 SM 利用不高，多半是 memory-bound；反之是 compute-bound；然后用“减少 HBM 流量（FlashAttention/fusion）”或“增加并行/减少同步（batch/overlap/CUDA graph）”分别对症。

**Q35：为什么做了 chunked prefill 后 TTFT 下降了，但吞吐反而掉了？怎么解释与改？**  
A：chunked prefill 会引入更多调度与 kernel 边界（launch/sync）开销，且可能破坏部分算子的大块连续计算；改法是：调大 chunk 让 launch 次数下降；把 decode 用 CUDA graph/overlap 隐藏 CPU 开销；对 prefill 做预打包/融合；或在低负载时退回大 chunk/非 chunked 以提升吞吐。

**Q36：遇到 “CUDA error: an illegal memory access was encountered”，你怎么一步步定位？**  
A：先把异步错误同步出来：加 `CUDA_LAUNCH_BLOCKING=1` 或在关键点插 `cudaDeviceSynchronize()`/`torch.cuda.synchronize()`；再用 `compute-sanitizer --tool memcheck` 找到越界线程与访问地址；如果是自写 kernel/extension，加 `-lineinfo` 产出行号（必要时 debug 编译 `-G`）；最后回到三类根因：index 越界、对齐/向量化跨界、并发 race/生命周期（stream + memory pool）错误。

**Q37：如果你在实现 PagedAttention/PageAttention 这类 kernel 时，怎么“设计上”降低 illegal memory 风险？**  
A：把索引体系做成可验证：block/page table 的边界与单调性约束（host 侧校验）；尾部统一走 mask 路径；vectorized load/store 前先保证对齐，不对齐就退回标量；关键路径加 cheap assert（debug 开关）；并把 compute-sanitizer/racecheck 作为 PR/CI 的最小回归（小 batch、小 seq、多随机 shape）。

---

## 5. 30 分钟速记（面试前快速过一遍）

- 先开场：用一句话把 prefill/decode + TTFT/TPOT 讲清楚。  
- 能算：KV cache 容量/成本的数量级，和长上下文/并发如何放大。  
- 能画：Continuous batching 的 event loop；PagedAttention 的 block table；PD 分离的数据面/控制面。  
- 能落地：说出“策略 + 指标 + 观测 + 兜底（OOM/回退/限流）”。  
- 能深入：FlashAttention 的 online softmax 与 tile；reduce/softmax 优化套路；RDMA/背压/状态机。

---

## 6. 项目深挖（基于 `e:\CV\CV_chinese\zh-cv.pdf`）

这部分是“把简历项目变成 AI Infra 面试可追问的故事线”：每个项目给 1 句定位 + 可量化结果 + 可深挖点 + 下一步优化点。

### 6.1 SGLang（开源贡献 / 推理优化）
- 一句话：围绕 **overlap scheduling / speculative（EAGLE/EAGLE2）/ PD 分离** 做推理端优化，关注 TPOT 与系统开销。
- 可讲结果（来自简历抽取）：CPU-GPU overlap 带来 **约 21% TPOT 改善**（示例：`4.07ms → 3.22ms`）；在 decode-heavy benchmark 上有 **约 27.8%** 的提升；涉及 PR `#14410`、`#18906` 等。
- 深挖点（面试爱问）：为什么 overlap 有效（隐藏哪段开销/依赖怎么管理）；logprobs/padding/accept len 等细节如何影响收益；如何保证 correctness（数值差异 `max_diff`、随机性、回归）。
- 下一步优化：把 overlap 的收益做成“稳定可控”（对不同 prompt/output/并发的 P95/P99）；加入更细的 cost model（KV/通信/验证开销）驱动调度；在多租户下做隔离与限流。

### 6.2 MiniInfer（教学/实验性推理框架）
- 一句话：实现 Tokenizer→Scheduler→ModelRunner→Detokenizer 的推理链路，覆盖 **continuous batching / chunked prefill / KV cache / paged allocator / CUDA graph / Triton kernel**。
- 可讲结果（来自简历抽取）：在 `RTX 3080 Ti + Qwen2-0.5B-Instruct` 上吞吐从 `1807.69 tok/s` 提升到 `6600+ tok/s`（约 **+265.2%**）；并展示了 prefill 侧 `33ms → 28ms`（约 **15%**）的优化案例。
- 深挖点（面试爱问）：prefill-only/decode-only/mixed batch 的 tradeoff；`new_token_ratio` 与 retract 的意义；`req_to_token → block_table` 的索引体系；CUDA graph 解决的到底是 CPU launch 还是同步；Triton 与 CUDA kernel 各自适用边界。
- 下一步优化：把“吞吐提升”转化成 “SLO 提升”（TTFT/TPOT/P99）；做 prefix cache（Radix）命中率模型与驱逐；接入 FlashDecoding/更强的 kernel fusion；对 OOM 回退策略做可观测与可回归测试。

### 6.3 OceanBase（向量检索 / HNSW 优化）
- 一句话：围绕 HNSW 做向量检索性能优化（SIMD、Int8 等），关注 QPS 与 I/O 瓶颈。
- 可讲结果（来自简历抽取）：QPS `592 → 1095`；并标注了 I/O 占比（如 `I/O 70%`）与整体提升（如 `QPS 201%`）的结果呈现。
- 深挖点（面试爱问）：HNSW 参数（`M/efConstruction/efSearch`）如何影响精度/延迟；SIMD/量化如何落地；I/O 为什么会成为瓶颈（索引结构是否驻留内存、邻居列表布局、cache 命中）。
- 下一步优化：邻接表布局与预取（cache-aware）；分层缓存（hot graph in RAM）；异步 I/O 与并发控制；更细粒度的 recall@k/latency/QPS 联合评估。

### 6.4 Nebula Graph ANN Search（图数据库内的向量索引与检索）
- 一句话：在图数据库内实现/集成 ANN Search（HNSWlib/FAISS），打通 **DDL/DML → Parser/Planner/Executor → Storage（RocksDB Column Family）** 的链路。
- 可讲结构（来自简历抽取）：AnnIndex / AnnIndexScan，Graphd/Storaged 两侧配合；向量数据与索引存储落在 RocksDB。
- 深挖点（面试爱问）：索引构建/更新与一致性（WAL/raft/事务边界）；查询计划如何把 ANN scan 下推到存储；并发与资源隔离；冷热数据与 cache。
- 下一步优化：增量构建与后台 compaction 的配合；索引与数据的共置/分置策略；批量查询与向量算子融合；更好的回放/恢复与观测。

---

## 7. 参考资料（面经 + JD + 学习资料）

### 7.1 牛客面经（AI Infra / 推理 / CUDA）
- 百融云创 AI Infra 面经（含 reduce/flashattention/vllm/pagedattention 等提问）：https://www.nowcoder.com/discuss/724396208503947264
- 小鹏 AI Infra 后端一面记录（偏后端/系统）：https://www.nowcoder.com/discuss/727263110091784192
- ppio AI Infra 社招面经（偏网络/容器/iptables）：https://www.nowcoder.com/discuss/684773148192952320
- CUDA 算子手撕与面试（含常见 kernel 优化套路与项目链接）：https://www.nowcoder.com/discuss/697901950464954368

### 7.2 职位描述（JD：小红书 / 快手 / 推理优化）
- 小红书｜大模型推理系统 MaaS 研发工程师（实习，来源：小红书官网转发）：https://campus.niuqizp.com/job-vyy5Lt5aL.html
- 小红书｜大模型推理服务(MaaS)研发工程师/专家（来源：小红书官网转发）：https://jobs.niuqizp.com/job-vw85LLzMt.html
- 小红书｜大模型推理服务(MaaS)研发工程师/专家（面试马转发）：https://www.mianshima.com/job/15/16080
- 快手｜AI Infra 研发工程师（校招 JD，含 GPU/RDMA/K8s/vLLM/SGLang）：https://www.nowcoder.com/jobs/detail/405622
- 大模型推理加速工程师 JD（含 vLLM/FlashAttention/PageAttention/Continuous Batching/Speculative Decoding）：https://www.boshijob.com/job/201297.html

### 7.3 学习资料（Serving/框架/综述）
- 大模型推理系统全景（框架对比：vLLM/SGLang/TensorRT-LLM/MLC）：https://jimmysong.io/book/ai-handbook/llm/inferences-overview/

### 7.4 论文/官方文档（建议按需补充）
- FlashAttention (v1)：https://arxiv.org/abs/2205.14135
- FlashAttention-2：https://arxiv.org/abs/2307.08691
- vLLM（PagedAttention / Serving）：https://arxiv.org/abs/2309.06180
- SGLang 官方文档：https://docs.sglang.ai/
- vLLM 官方文档：https://docs.vllm.ai/
- NVIDIA Compute Sanitizer（memcheck/racecheck）：https://docs.nvidia.com/compute-sanitizer/
- NVIDIA Nsight Systems（全链路 timeline）：https://docs.nvidia.com/nsight-systems/
- NVIDIA Nsight Compute（kernel 级分析）：https://developer.nvidia.com/nsight-compute
