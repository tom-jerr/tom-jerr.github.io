---
title: LLM 推理优化岗位深研：调度侧与推测解码侧
created: 2026-03-07
updated: 2026-03-07
tags:
  - LLMInference
  - Interview
description: 面向腾讯、字节跳动、阿里巴巴的 LLM 推理优化岗位复习提纲，聚焦调度侧与推测解码侧，整理高频问题、标准答案、公开参考资料与仓库内对应笔记。
---

# LLM 推理优化岗位深研：调度侧与推测解码侧

## 使用方式

这篇文档不是再讲一遍大模型推理基础，而是把当前仓库里已经写过的内容，和 2023-03-06 到 2026-03-07 可检索到的公开资料做一次重新编排，输出四类信息：

- 这类岗位到底在考什么。
- 不同公司更偏哪条线。
- 高频面试题的标准回答应该怎么组织。
- 每道题在当前仓库里该回看哪个文件。

> [!NOTE]
> 字节跳动官方职位页很多是动态渲染，公开可稳定检索到的完整 JD 少于腾讯云文档、阿里云文档、vLLM/SGLang/TensorRT-LLM/Llumnix 等公开资料。本文对字节部分会明确标注“样本有限”，更多结合你已有的 JD 摘要和可检索到的公开岗位标题来归纳。

## 结论先看

### 岗位画像

当前中国大厂的 LLM 推理优化岗位，尤其是“推理系统 / 推理优化 / 推理调度 / Serving Infra”方向，已经明显不是只考 kernel 或 CUDA 小优化，而是考下面这条主线：

`请求生命周期 -> batching -> token budget -> KV cache -> 显存碎片 -> 调度策略 -> 多实例/多节点负载均衡 -> 指标闭环`

如果面的是调度侧与推测解码侧，面试官通常会默认你已经知道 FlashAttention、KV cache、continuous batching、paged attention 这些名词；真正拉开差距的是你能不能回答下面两件事：

1. 某个优化为什么会改变 scheduler 的决策边界。
2. 某个优化为什么会改变 KV cache 的生命周期和显存预算。

### 三家公司侧重点

#### 腾讯

腾讯公开资料最完整的是 TACO-LLM / TACO Kit 文档。它对外直接列出了 Continuous Batching、Paged Attention、投机采样、Auto Prefix Caching、CPU 辅助加速、长序列优化等能力。因此腾讯语境里的面试，通常不是问“你知不知道”，而是问“这些技术分别改进了哪个指标，代价是什么，什么时候反而没收益”。

#### 阿里巴巴 / 阿里云 / PAI

阿里云公开资料非常适合复习。ACK/PAI 文档、AlibabaPAI 的 Llumnix、以及和 vLLM/SGLang 的集成说明，直接把分布式推理、异构资源、推理框架选型、PD 分离、动态重调度、观测体系串起来了。阿里这条线的典型考法是：

- 云上多实例/多节点怎么调度。
- vLLM 和 SGLang 怎么选。
- KV-cache-aware rescheduling / migration 如何降低 TTFT、P99、preemption stall。

#### 字节跳动

字节的公开官方长文本 JD 样本有限，但从你已有的 JD 摘要和可检索岗位标题看，字节更像“平台调度 + 推理系统”混合画像：

- 海量异构资源调度编排。
- 多租户配额、隔离、优先级。
- 多阶段推理调度，尤其是 PD 分离、KV-cache-centric routing 这类问题。
- 对网络与资源治理的要求通常比单机引擎岗位更高。

所以准备字节时，不能只会讲 vLLM/PagedAttention，还要能把它提升成“资源模型 + 调度目标函数 + 多租户治理 + 拓扑约束”的语言。

## 当前仓库内的复习路径

| 主题 | 优先回看文件 | 用途 |
| --- | --- | --- |
| Continuous batching / chunked prefill / token budget | `docs/notes/sglang/大模型推理服务中的 Batching.md` | 调度主线、静态批处理到连续批处理的演进 |
| SGLang scheduler 工作流 | `docs/notes/sglang/SGLang Scheduler 技术变迁.md` | batch 状态机、waiting/running/new batch、zero-overhead scheduler |
| KV cache 生命周期 | `docs/notes/sglang/从代码看 SGLang 的 KV Cache.md` | req_to_token_pool、token_to_kv_pool、tree cache |
| Prefix caching / RadixAttention | `docs/notes/sglang/RadixAttention 你需要知道的全部细节.md` | 前缀路由、匹配粒度、缓存复用 |
| PD 分离 | `docs/feishu/PD Disaggregation in SGLang.md` | prefill/decode 分离、KV 传输、调度与状态机 |
| 推测解码 / EAGLE2 | `docs/feishu/Eagle2 in SGLang.md` | speculative decoding、EAGLE 路线、SGLang 集成 |
| 并行策略 | `docs/notes/sglang/SGLang 中的 TP + PP.md` | TP/PP 对系统行为的影响 |
| 更泛化的推理并行知识 | `docs/notes/llm_inference/LLMparallelization.md` | DP/TP/PP/EP、通信与吞吐/延迟权衡 |

## 高频问题与标准答案

## 一、调度侧

### 1. 为什么 LLM 推理要做 PD 分离？

**答案：**

LLM 推理天然分成两个阶段：prefill 是 compute-bound，decode 是 memory-bound。把两者放在同一个统一调度器里，会出现两个常见问题：

- 新的 prefill batch 会频繁打断 decode，恶化 TPOT / ITL。
- 同一套并行策略无法同时适配 prefill 和 decode 的资源特征。

PD 分离的本质不是“把流程拆开”这么简单，而是让 prefill 池和 decode 池分别用更合适的 GPU、并行策略和 batching 策略。代价是引入 KV cache transfer、路由、超时与故障恢复问题，所以面试里必须继续讲到网络和状态管理。

**推荐答法关键词：** `compute-bound vs memory-bound`、`TTFT vs TPOT`、`KV cache transfer`、`router`、`disaggregation`

**参考资料：**

- SGLang PD Disaggregation: <https://docs.sglang.ai/advanced_features/pd_disaggregation.html>
- SGLang PD backend doc: <https://docs.sglang.ai/backend/pd_disaggregation.html>
- DistServe paper: <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>

**对应本地文件：** `docs/feishu/PD Disaggregation in SGLang.md`

### 2. Continuous batching 相比 static batching 到底解决了什么？

**答案：**

static batching 的核心问题是“请求必须一起进、一起出”，导致两种浪费：

- 短请求必须等长请求结束，GPU 有空泡。
- 新请求必须等整个 batch 结束，排队延迟变大。

continuous batching 把调度粒度从 request-level 变成 iteration-level / token-level。每轮 decode 结束后，已经完成的请求立刻退出，新的请求可以立刻补进来。于是 GPU 利用率更高，排队更短，系统开始围绕 token budget 而不是固定 batch size 运转。

它真正难的部分不是“动态加减请求”，而是“不同长度、不同阶段、不同 KV 长度的请求如何共享一次 engine step”。这就需要和 paged attention、chunked prefill、prefix cache 一起看。

**参考资料：**

- vLLM PagedAttention paper: <https://arxiv.org/abs/2309.06180>
- 腾讯云 TACO Kit 特性文档: <https://cloud.tencent.com/document/product/1573/115169>
- vLLM Optimization: <https://docs.vllm.ai/en/latest/configuration/optimization.html>

**对应本地文件：** `docs/notes/sglang/大模型推理服务中的 Batching.md`

### 3. token budget 应该怎么理解？为什么它比 batch size 更关键？

**答案：**

在线推理里真正稀缺的资源不是“有多少条请求”，而是“这一轮最多还能塞多少 token 而不把显存和 kernel 形状搞爆”。所以很多现代 scheduler 更关心 `max_num_batched_tokens` 或者类似的 token budget，而不是固定 batch size。

它至少控制三类东西：

- 当前轮 prefill + decode 一共能进多少 token。
- KV cache 还能分配多少块页。
- speculative decoding 开启时，draft token 会不会把预算提前吃掉。

面试里如果只说“为了防 OOM”是不够的，最好补一句：token budget 是 scheduler、KV allocator、prefill chunking、spec decode 共享的统一约束面。

**参考资料：**

- vLLM Optimization: <https://docs.vllm.ai/en/latest/configuration/optimization.html>
- TensorRT-LLM speculative decoding guide: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html>

**对应本地文件：** `docs/notes/sglang/大模型推理服务中的 Batching.md`

### 4. 为什么 KV cache 是在线推理系统里的“一等公民”？

**答案：**

没有 KV cache，每次生成一个新 token 都要把历史 token 全部重新做 attention，代价太大；有了 KV cache 之后，计算量降了，但新的瓶颈变成了显存容量、显存带宽和缓存管理。

所以系统设计的重点会从“怎么少算”变成“怎么少占、少搬、少碎片化、少失效”。很多 serving 优化其实都可以重写成一句话：

> 想办法让更多 request 在有限 KV cache 下并发地活着，并尽量延长可复用前缀的生命周期。

**参考资料：**

- vLLM paper: <https://arxiv.org/abs/2309.06180>
- vLLM metrics: <https://docs.vllm.ai/en/latest/usage/metrics.html>

**对应本地文件：** `docs/notes/sglang/从代码看 SGLang 的 KV Cache.md`

### 5. PagedAttention 为什么像操作系统的虚拟内存？

**答案：**

传统做法要求每个请求的 KV cache 在显存里连续分配，这会带来严重的内部碎片和外部碎片。PagedAttention 借鉴 OS 的分页思想，把 KV cache 切成固定大小的 block/page，再用 block table 把逻辑 token 位置映射到物理块。

收益有三点：

- 最后一页之外几乎没有内部碎片。
- 不要求连续物理内存，外部碎片大幅下降。
- prefix sharing / parallel sampling 时可以共享物理块。

真正高级的回答要再补一层：PagedAttention 不是纯内存优化，而是 memory-compute co-design。因为 kernel 也必须知道如何在非连续物理块上正确 gather K/V 并完成 attention。

**参考资料：**

- vLLM paper: <https://arxiv.org/abs/2309.06180>
- 腾讯云 TACO-LLM 文档: <https://www.tencentcloud.com/document/product/457/69509>

**对应本地文件：** `docs/notes/sglang/大模型推理服务中的 Batching.md`

### 6. Prefix caching / RadixAttention 的命中条件和调度意义是什么？

**答案：**

命中条件本质上是“新请求前缀与历史请求前缀在 token 序列上足够一致”，然后系统可以直接复用那段 prefix 对应的 KV cache，而不必重新 prefill。

调度上的意义比 cache 本身更大：

- router 会优先把相同前缀/同会话请求发往缓存命中概率更高的实例。
- scheduler 会优先处理高命中价值请求，因为它们能直接省掉 prefill 开销。
- prefix 粒度如果按 page 而不是逐 token，会影响命中率和 lookup 成本。

SGLang 的 RadixAttention 强调的是前缀组织结构和高效匹配；vLLM 生态里常见的是 automatic prefix caching。面试时可以把两者统一成“前缀感知调度”。

**参考资料：**

- SGLang docs: <https://docs.sglang.ai/>
- vLLM Automatic Prefix Caching: <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
- 腾讯云 TACO Kit 特性: <https://cloud.tencent.com/document/product/1573/115169>

**对应本地文件：**

- `docs/notes/sglang/RadixAttention 你需要知道的全部细节.md`
- `docs/notes/sglang/从代码看 SGLang 的 KV Cache.md`

### 7. vLLM 的 preemption 为什么 V1 更偏向 RECOMPUTE 而不是 SWAP？

**答案：**

当 KV cache 不够时，系统要么把请求状态换出到 CPU 再换回，要么直接丢掉中间状态，等有资源时重算。vLLM V1 默认更偏向 `RECOMPUTE`，官方原因很直接：在 V1 架构下，recompute 的整体开销更低。

什么时候 `SWAP` 仍然有价值？典型是 beam search 或多序列组场景，此时简单地重算可能破坏已有的多分支状态管理，swap 反而更稳。

答这题不能停在定义层，最好再补一句：preemption 本质是显存压力向计算或 PCIe/CPU 内存压力的转移。

**参考资料：**

- vLLM Optimization: <https://docs.vllm.ai/en/latest/configuration/optimization.html>

**对应本地文件：** `docs/notes/sglang/SGLang Scheduler 技术变迁.md`

### 8. 为什么多实例一次性 dispatch 不够，需要 continuous rescheduling？

**答案：**

因为请求长度、输出长度、prefix 命中率、生成速度都高度异质，一次性把请求分发到某个实例后，负载很快就会失衡：

- 某些实例排队很长。
- 某些实例显存碎片严重。
- 某些实例频繁 preemption。
- 某些实例明明 prefix 命中更高但没流量。

Llumnix 的核心贡献就是把“请求运行中不能动”这个假设打掉，通过 KV-cache-aware scheduling + live migration 持续重调度，请求和内存态都能迁移，于是能同时改善 TTFT、P99 和 preemption stall。

**参考资料：**

- Llumnix GitHub: <https://github.com/AlibabaPAI/llumnix>
- Llumnix OSDI page: <https://www.usenix.org/conference/osdi24/presentation/sun-biao>

**对应本地文件：** `docs/feishu/PD Disaggregation in SGLang.md`

### 9. KV cache 迁移 / 传输的系统难点是什么？

**答案：**

难点不止是“带宽不够”，而是三件事一起出现：

- 迁移本身会吃网络带宽，挤压正常推理通信。
- 迁移期间请求不能长时间停机，否则 TTFT / TPOT 抖动很大。
- scheduler 必须知道迁移成本，否则会做出“迁移比不迁移更差”的错误决定。

成熟答案通常要包含这些关键词：`near-zero-overhead migration`、`pipelining`、`backpressure`、`topology-aware routing`、`timeout / failure recovery`。

**参考资料：**

- Llumnix GitHub: <https://github.com/AlibabaPAI/llumnix>
- SGLang PD backend doc: <https://docs.sglang.ai/backend/pd_disaggregation.html>
- vLLM metrics 里的 NIXL KV connector 指标: <https://docs.vllm.ai/en/latest/usage/metrics.html>

**对应本地文件：** `docs/feishu/PD Disaggregation in SGLang.md`

### 10. TTFT 抖动时，你会先看哪些指标？

**答案：**

先把问题拆成三段：排队、prefill、传输/调度。

优先看：

- request-level: `time_to_first_token_seconds`、`e2e_request_latency_seconds`
- server-level: `num_requests_waiting`、`num_requests_running`、`kv_cache_usage_perc`
- cache 相关: `prefix_cache_queries`、`prefix_cache_hits`
- spec decode 相关: accepted draft tokens / draft tokens
- 如果是 PD / 多实例: KV transfer bytes、failed notifications、迁移次数、迁移耗时

面试里加分点在于你会说：单看 TTFT 没意义，要和 waiting queue、KV 占用、prefix hit、preemption 次数一起看，才能知道是 admission 太激进、cache 不够，还是路由失衡。

**参考资料：**

- vLLM metrics overview: <https://docs.vllm.ai/en/latest/usage/metrics.html>
- vLLM metrics design: <https://docs.vllm.ai/en/latest/design/v1/metrics.html>
- 阿里云 ACK LLM inference monitoring: <https://www.alibabacloud.com/help/en/ack/cloud-native-ai-suite/user-guide/configure-monitoring-for-llm-inference-services>

**对应本地文件：** `docs/notes/sglang/SGLang Scheduler 技术变迁.md`

## 二、推测解码侧

### 11. 推测解码为什么能加速？

**答案：**

核心原因是 decode 阶段 memory-bound，单步只生成一个 token 时 GPU 算力经常没吃满。如果先用 draft model 或 draft head 提前提出多个候选 token，再让 target model 一次性 verify 多个位置，就可能用接近一次大模型前向的代价，接受多个 token。

加速上限主要由三件事决定：

- draft 生成开销有多大。
- verify 是否能高效并行。
- draft token 的接受率有多高。

如果接受率低、draft 太贵、batch 很大导致 verify 本来就吃满了，那 speculative decoding 可能没收益甚至变慢。

**参考资料：**

- Speculative Decoding paper: <https://arxiv.org/abs/2211.17192>
- TensorRT-LLM speculative decoding: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html>

**对应本地文件：** `docs/feishu/Eagle2 in SGLang.md`

### 12. 推测解码为什么必须改 scheduler，而不是只改 sampler？

**答案：**

因为 draft token 不是“逻辑上存在、资源上不存在”的东西。它们会真实占用：

- token budget
- KV cache pages
- CUDA graph / kernel dispatch 形状

TensorRT-LLM 的文档把这件事说得很清楚：在真正调度前，就要先让 scheduler 和 KV cache manager 知道这一轮会 speculative decoding，否则调度器会低估资源需求，导致预算错配甚至 OOM。

这题最标准的一句话是：

> speculative decoding 改变的不只是采样语义，还改变了请求在一次 engine step 中的资源画像。

**参考资料：**

- TensorRT-LLM speculative decoding developer guide: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html>

**对应本地文件：** `docs/feishu/Eagle2 in SGLang.md`

### 13. draft token 对 KV cache 生命周期有什么影响？

**答案：**

draft token 先被分配 KV 页，但最后不一定都被接受。于是 KV cache 生命周期多了一步：`allocate -> verify -> partially accept -> rewind/free rejected pages`。

这会带来两个调度后果：

- scheduler 要为“可能被拒绝的 token”先预留空间。
- KV allocator 要支持低成本 rewind，而不能真的做高成本内存释放。

TensorRT-LLM 里就明确提到 rejected draft token 的 page 会在下一轮前被 rewind/free，这本质上是为了避免 KV cache manager 维护过于复杂的复用逻辑。

**参考资料：**

- TensorRT-LLM speculative decoding guide: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html>

**对应本地文件：** `docs/feishu/Eagle2 in SGLang.md`

### 14. Medusa、EAGLE、draft-target 三种路线怎么快速区分？

**答案：**

可以按“draft 从哪里来”来区分：

- `draft-target`: 单独的小模型先起草，再由大模型验证。优点是概念简单；缺点是多一套模型和显存。
- `Medusa`: 不额外上小模型，而是在大模型顶部加多个预测头。优点是工程更集中；缺点是后续位置预测信息不足，接受率容易掉。
- `EAGLE`: 不直接猜 token，而是更关注 feature-level extrapolation，用更丰富的中间表示去做 draft，因此通常比 Medusa 更稳。

如果面试官继续追问工程落地，记得回答：树状验证、接受率、draft 长度、和 scheduler/token budget 的耦合，比“论文名字”更重要。

**参考资料：**

- Medusa paper: <https://arxiv.org/abs/2401.10774>
- EAGLE paper: <https://arxiv.org/abs/2401.15077>
- TensorRT-LLM speculative decoding docs: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html>

**对应本地文件：** `docs/feishu/Eagle2 in SGLang.md`

### 15. speculative decoding 在什么情况下可能没收益？

**答案：**

典型有五种：

- 接受率低。
- draft model 太大或 draft head 太贵。
- batch 已经很大，target verify 额外收益变小。
- KV cache 很紧张，draft token 反而挤占正常并发。
- 请求分布偏长上下文/高并发，scheduler 已经更受显存和队列约束而不是单请求 decode 约束。

所以面试里不要把 speculative decoding 讲成“必定加速”。更准确的表述是：它是在低到中等 batch、可接受率较高、且 scheduler 能正确感知额外预算时，比较容易拿到收益。

**参考资料：**

- TensorRT-LLM speculative decoding guide: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html>
- NVIDIA blog on lookahead decoding: <https://developer.nvidia.com/blog/optimizing-qwen2-5-coder-throughput-with-nvidia-tensorrt-llm-lookahead-decoding/>

**对应本地文件：** `docs/feishu/Eagle2 in SGLang.md`

## 三、系统设计 / 行为题

### 16. 设计一个支持多租户的推理集群配额与隔离系统，你会怎么做？

**答案：**

建议按四层回答：

- 资源模型：GPU、CPU、内存、网络带宽、KV cache budget 都要进入配额模型。
- admission control：按 tenant / queue 做准入、优先级和抢占策略。
- runtime isolation：限制单租户的 `max_num_batched_tokens`、实例数、prefix cache 占比、迁移带宽。
- observability：按租户维度暴露 TTFT、TPOT、拒绝率、抢占率、prefix hit、cost per token。

如果是平台调度岗位，最好顺带提 `queue quota`、`hierarchical queue`、`binpack vs spread`、`noisy neighbor`。

**参考资料：**

- Volcano overview: <https://volcano.sh/en/>
- Volcano queue management: <https://volcano.sh/en/docs/queue_resource_management/>
- Kubernetes device plugins: <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>

**对应本地文件：**

- `docs/notes/sglang/大模型推理服务中的 Batching.md`
- `docs/notes/llm_inference/LLMparallelization.md`

### 17. TP、PP、DP、EP 在推理场景里最容易被问什么？

**答案：**

最容易被问的不是定义，而是系统后果：

- TP：单层内切分，decode 每步同步频繁，网络很敏感。
- PP：层间切分，能省单卡显存，但会引入流水线 bubble 和 stage 间延迟。
- DP：最容易做横向扩展，但 prefix 路由和负载均衡决定上限。
- EP：MoE 模型常见，专家路由和 all-to-all 通信开销显著。

面试里最好回答成一句 tradeoff：`TP 更像拿网络换单请求 latency，PP 更像拿 pipeline 复杂度换可部署性和吞吐，DP 更依赖全局调度，EP 更依赖通信拓扑和热点控制。`

**参考资料：**

- 阿里云 ACK distributed inference doc: <https://www.alibabacloud.com/help/en/ack/cloud-native-ai-suite/user-guide/deploy-multi-machine-distributed-inference-services>
- NCCL Developer Guide: <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2212/nccl-developer-guide/index.html>

**对应本地文件：**

- `docs/notes/sglang/SGLang 中的 TP + PP.md`
- `docs/notes/llm_inference/LLMparallelization.md`

### 18. NCCL / 网络通信为什么在推理岗位里也越来越重要？

**答案：**

因为现代推理已经不是单卡单实例问题了。只要涉及 TP、EP、PD 分离、KV cache migration、cross-instance prefix sharing，网络就会直接影响 TTFT、TPOT 和吞吐稳定性。

面试里至少要会三件事：

- 解释 all-reduce / all-gather 的语义。
- 知道带宽和时延对不同并行策略影响不同。
- 知道拓扑感知部署为什么重要，比如同 rack、同 NVLink domain、同 replica 组尽量靠近。

**参考资料：**

- NCCL Developer Guide: <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2212/nccl-developer-guide/index.html>
- LWS overview: <https://lws.sigs.k8s.io/docs/overview/>
- vLLM LWS deployment: <https://docs.vllm.ai/en/latest/deployment/frameworks/lws/>

**对应本地文件：** `docs/notes/llm_inference/LLMparallelization.md`

### 19. 为什么 LWS / Volcano / topology-aware scheduling 会成为加分项？

**答案：**

因为多节点推理的调度对象不再是“单个 Pod”，而是“一个必须一起存活、一起通信、一起调度的 Pod 组”。

- LWS 解决 leader-worker 组的部署与生命周期问题。
- Volcano 解决 gang scheduling、队列配额、binpack、异构设备等更强的 AI 调度问题。
- topology-aware scheduling 解决跨节点通信距离和带宽瓶颈。

这类问题特别像字节 / 平台调度岗会问的题，因为它已经超出了单引擎优化，进入云原生资源治理层。

**参考资料：**

- LWS GitHub: <https://github.com/kubernetes-sigs/lws>
- LWS overview: <https://lws.sigs.k8s.io/docs/overview/>
- Volcano overview: <https://volcano.sh/en/>
- Volcano gang scheduling tutorial: <https://volcano.sh/en/docs/tutorials/>

**对应本地文件：** `docs/notes/llm_inference/LLMparallelization.md`

### 20. vLLM 和 SGLang 该怎么选？

**答案：**

比较稳妥的回答方式是按“生态 vs 调度/缓存特性”来讲：

- vLLM 的优势是 PagedAttention 起家、生态成熟、部署和兼容性强，很多云平台和监控体系默认先接它。
- SGLang 的优势是 RadixAttention、zero-overhead CPU scheduler、PD 分离、speculative decoding、structured outputs 这些能力比较完整，适合继续向调度和复杂 serving 场景深入。

如果业务更重多实例协同、PD 分离、prefix-aware routing、spec decode 深度调优，SGLang 往往更有吸引力；如果业务更看重生态兼容、快速部署、成熟监控，vLLM 更常见。

**参考资料：**

- SGLang docs: <https://docs.sglang.ai/>
- vLLM docs: <https://docs.vllm.ai/en/latest/>
- 阿里云 inference framework support: <https://www.alibabacloud.com/help/doc-detail/2950163.html>

**对应本地文件：**

- `docs/notes/sglang/SGLang Scheduler 技术变迁.md`
- `docs/notes/sglang/从代码看 SGLang 的 KV Cache.md`
- `docs/notes/sglang/大模型推理服务中的 Batching.md`

## 公司定向复习建议

### 腾讯

优先顺序建议：

1. `Continuous Batching -> Paged Attention -> Prefix Caching -> Speculative Decoding`
2. `这些优化分别改进 TTFT / TPOT / 吞吐 / 显存 的哪一项`
3. `它们的副作用和失效场景`

优先回看：

- `docs/notes/sglang/大模型推理服务中的 Batching.md`
- `docs/feishu/Eagle2 in SGLang.md`

### 阿里

优先顺序建议：

1. `vLLM vs SGLang`
2. `PD 分离`
3. `Llumnix 动态重调度与 KV 迁移`
4. `ACK/PAI 场景下的多实例、多节点、观测`

优先回看：

- `docs/feishu/PD Disaggregation in SGLang.md`
- `docs/notes/sglang/SGLang Scheduler 技术变迁.md`
- `docs/notes/sglang/从代码看 SGLang 的 KV Cache.md`

### 字节

优先顺序建议：

1. `资源模型：GPU/CPU/内存/网络`
2. `多租户 quota / 隔离 / 优先级`
3. `PD 分离、多阶段调度、KV-cache-centric routing`
4. `拓扑感知调度、LWS/Volcano、网络与带宽`

优先回看：

- `docs/feishu/PD Disaggregation in SGLang.md`
- `docs/notes/llm_inference/LLMparallelization.md`
- `docs/notes/sglang/大模型推理服务中的 Batching.md`

## 参考资料

### 官方文档 / 官方项目

- Tencent Cloud, TACO-LLM: <https://www.tencentcloud.com/document/product/457/69509>
- 腾讯云 TACO Kit 主要特性: <https://cloud.tencent.com/document/product/1573/115169>
- SGLang Documentation: <https://docs.sglang.ai/>
- SGLang PD Disaggregation: <https://docs.sglang.ai/advanced_features/pd_disaggregation.html>
- vLLM Documentation: <https://docs.vllm.ai/en/latest/>
- vLLM Optimization and Tuning: <https://docs.vllm.ai/en/latest/configuration/optimization.html>
- vLLM Production Metrics: <https://docs.vllm.ai/en/latest/usage/metrics.html>
- vLLM Metrics Design: <https://docs.vllm.ai/en/latest/design/v1/metrics.html>
- TensorRT-LLM Speculative Decoding: <https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/features/speculative-decoding.html>
- AlibabaPAI Llumnix: <https://github.com/AlibabaPAI/llumnix>
- Alibaba Cloud distributed inference docs: <https://www.alibabacloud.com/help/en/ack/cloud-native-ai-suite/user-guide/deploy-multi-machine-distributed-inference-services>
- Alibaba Cloud inference framework support: <https://www.alibabacloud.com/help/doc-detail/2950163.html>
- Alibaba Cloud monitoring for inference service: <https://www.alibabacloud.com/help/en/ack/cloud-native-ai-suite/user-guide/configure-monitoring-for-llm-inference-services>
- NCCL Developer Guide: <https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2212/nccl-developer-guide/index.html>
- Kubernetes Device Plugins: <https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/>
- LeaderWorkerSet overview: <https://lws.sigs.k8s.io/docs/overview/>
- LWS GitHub: <https://github.com/kubernetes-sigs/lws>
- Volcano: <https://volcano.sh/en/>
- Volcano Queue Resource Management: <https://volcano.sh/en/docs/queue_resource_management/>

### 论文 / 开源论文页面

- vLLM: <https://arxiv.org/abs/2309.06180>
- Speculative Decoding: <https://arxiv.org/abs/2211.17192>
- Medusa: <https://arxiv.org/abs/2401.10774>
- EAGLE: <https://arxiv.org/abs/2401.15077>
- Llumnix OSDI: <https://www.usenix.org/conference/osdi24/presentation/sun-biao>
- DistServe OSDI: <https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>

## 建议放在仓库中的位置

这篇总览文档适合放在：

- `docs/feishu/LLM 推理优化岗位深研：调度侧与推测解码侧.md`

现有文件继续承担“专题深入”的角色，不建议把这些内容再拆碎到别的地方，否则会和已有笔记重复：

- `docs/feishu/Eagle2 in SGLang.md`: 继续放推测解码专题。
- `docs/feishu/PD Disaggregation in SGLang.md`: 继续放 PD 分离专题。
- `docs/notes/sglang/*.md`: 继续放代码级和机制级细节。
