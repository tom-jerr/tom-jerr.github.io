---
title: Agent 长上下文边缘推理优化研究报告
created: 2026-04-07
updated: 2026-04-07
tags:
  - LLMInference
description: 面向 Agent 场景的长上下文边缘大模型推理优化研究报告，整理问题定义、系统主线、工程 trade-off 公式、实验假设与回改后的论文提纲。
katex: true
---

# Agent 长上下文边缘推理优化研究报告

## 结论先看

这篇报告的核心观点只有一句话：

> 在边缘设备上的 Agent 推理里，真正稀缺的资源不再只是模型权重，而是能够被长期保存、快速复用、低代价迁移的上下文状态，尤其是 KV cache。

这意味着研究对象不能再被建模为“一次性长 prompt 推理”，而应被建模为“跨轮、长会话、状态持续增长的在线推理系统”。

与普通长 prompt 相比，Agent 长会话至少有三点本质不同：

| 维度 | 静态长 prompt | Agent 长会话 |
| --- | --- | --- |
| 输入结构 | 一次性给定 | 多轮持续追加 |
| 重复模式 | 单请求内部重复较少 | 跨轮稳定前缀重复很多 |
| 核心难点 | prefill 计算量大 | `prefill + decode + KV 生命周期管理` 联合优化 |
| 主要状态 | prompt token | `system/tool/doc/history/trace` 组成的会话状态 |
| 关键决策 | 如何加速 attention | 如何在 `保留/压缩/换出/重算/丢弃` 之间做 runtime 决策 |

因此，本研究最适合定位为一篇**系统优化型硕士论文**：不改基础模型参数，不自研复杂 attention kernel，而是围绕 Agent 长会话的状态管理、KV 管理和调度管理，设计一个边缘友好的推理运行时。

## 1. 问题重新定义

### 1.1 从长 prompt 到长会话

传统长上下文工作通常默认请求形态是：

$$
Prompt = [Instruction, Document, Query]
$$

其上下文长度虽然长，但生命周期很短，通常在一次推理结束后就可以被整体释放。

Agent 场景中的请求结构更接近：

$$
Context_t = [S, U, D_t, H_t, O_t]
$$

其中：

- $S$：system prompt，与 Agent 角色和规则相关，通常跨轮稳定。
- $U$：tool schema 或工具描述，通常稳定或低频变化。
- $D_t$：当前轮检索到的文档片段，可能在多轮中重复出现。
- $H_t$：历史对话，随轮次不断累积。
- $O_t$：历史 observation / action / tool trace，长度增长快，但价值衰减也快。

于是 Agent 会话在第 $t$ 轮的总上下文长度可写为：

$$
T_t = T_S + T_U + T_{D_t} + T_{H_t} + T_{O_t}
$$

这里最重要的不是 $T_t$ 本身大，而是不同部分具有**不同的价值、复用率和生命周期**。

### 1.2 边缘场景的特殊约束

本文默认研究边界如下：

- 单边缘设备闭环，不依赖多机集群。
- 模型规模为 1.5B-7B，允许采用量化模型。
- 平台默认是 Jetson Orin 级边缘 GPU。
- 工作负载只聚焦两类：
  - 多轮文档问答 Agent。
  - 工具调用 Agent。

在这个边界下，边缘推理的矛盾可以概括为：

> 有限显存、有限带宽、有限能耗预算，需要支撑会话状态持续增长、前缀持续复用、请求持续到达。

这也是本文选择把重点放在 runtime 与状态管理，而不是新模型结构或训练改造上的原因。

## 2. 统一符号表

为了避免后续公式冲突，本文统一使用下列符号：

| 符号 | 含义 |
| --- | --- |
| $L_{layer}$ | Transformer 层数 |
| $T$ | 当前上下文 token 长度 |
| $T_{reuse}$ | 本轮可直接复用的前缀 token 数 |
| $T_{new}$ | 本轮必须新计算的 token 数 |
| $T_{active}$ | decode 时仍需要参与注意力读取的有效上下文长度 |
| $n_{kv}$ | KV head 数，MHA 对应 head 数，GQA 对应 KV 组数 |
| $d_{head}$ | 每个 head 的维度 |
| $b$ | KV cache 单元素字节数 |
| $M_{kv}$ | 某段上下文对应的 KV 容量 |
| $BW_{mem}$ | GPU 显存有效带宽 |
| $BW_{gpu\to cpu}$ | GPU 到 CPU/pinned memory 的有效迁移带宽 |
| $BW_{cpu\to gpu}$ | CPU/pinned memory 到 GPU 的有效迁移带宽 |
| $C_{prefill}(x)$ | 长度为 $x$ 的 prefill 计算代价 |
| $\eta_{gpu}$ | GPU 计算效率折算因子 |
| $C_{offload}$ | KV 换出代价 |
| $C_{restore}$ | KV 恢复代价 |
| $C_{recompute}$ | 重算代价 |
| $\Delta M$ | 某策略带来的显存节省 |
| $\Delta T$ | 某策略带来的额外时延 |
| $\Delta Q$ | 某策略带来的质量损失 |
| $B_{ctx}$ | 当前会话允许保留的上下文预算 |
| $v_i$ | 第 $i$ 个上下文块的保留价值 |
| $m_i$ | 第 $i$ 个上下文块占用的预算 |
| $H_{prefix}$ | 请求的前缀命中率或命中收益估计 |
| $MemPressure(r)$ | 请求 $r$ 带来的边缘显存压力 |

## 3. Agent 长会话的瓶颈拆解

### 3.1 Prefill 代价高，但不再是唯一问题

静态长 prompt 的主要矛盾通常是：

- prompt 越长，prefill 越慢；
- attention 中间张量和计算量快速增长。

Agent 场景下这个问题依然存在，但它不再是全部。因为多轮会话里，很多 token 并不是“第一次见”，而是可复用的前缀。如果系统不能识别和命中这些前缀，就会把“可复用状态”错误地当成“新输入”重复计算。

### 3.2 Decode 更容易变成带宽瓶颈

随着会话变长，decode 阶段每生成一个 token 都要读取越来越长的历史 KV。此时瓶颈通常从计算转向显存带宽和访问局部性，这一点和已有 `prefill compute-bound / decode memory-bound` 的分析一致，但在 Agent 场景下更突出，因为历史更长、生命周期更久、冷热分布更明显。

### 3.3 真正稀缺的是 KV 生命周期预算

边缘设备常常无法把所有历史 KV 都留在 GPU 上。因此系统必须回答三个问题：

1. 哪些状态值得长期保留。
2. 哪些状态应该迁移到 CPU/pinned memory 或压缩层。
3. 哪些状态不值得保存，直接重算更划算。

一旦把问题提到这一层，研究重点就从“有没有一个更快的 attention kernel”转向“有没有一套更合理的状态管理策略”。

### 3.4 调度与缓存不能分开看

在 Agent 场景里，一个请求值不值得优先跑，不只取决于它有多急，还取决于：

- 它是否命中长前缀。
- 它会不会占用大量新的 KV。
- 它是否会触发大规模 restore 或重算。

因此，调度问题本质上已经变成**缓存感知调度**而不是单纯的 FCFS 或 shortest-job-first。

## 4. 一阶工程成本模型

本节不追求微架构级精确预测，而是给出能指导 runtime 决策的一阶模型。

### 4.1 KV 容量模型

对一段长度为 $T$ 的上下文，其 KV cache 容量近似为：

$$
M_{kv}=2 \cdot L_{layer}\cdot T \cdot n_{kv}\cdot d_{head}\cdot b
$$

其中第一个 $2$ 表示 K 和 V 两部分。

这个式子直接说明三件事：

- 长上下文下，$M_{kv}$ 与 $T$ 线性增长。
- GQA/MQA 通过降低 $n_{kv}$ 直接降低 KV 容量和带宽压力。
- KV 低比特化通过降低 $b$ 直接降低容量压力。

### 4.2 前缀复用带来的 TTFT 改善

若本轮请求中有 $T_{reuse}$ 个 token 可以直接命中前缀缓存，则被跳过的 prefill 代价近似为：

$$
\Delta TTFT \approx \frac{C_{prefill}(T_{reuse})}{\eta_{gpu}}
$$

若进一步把 prefill 代价近似看作与 token 数成正相关，可写为：

$$
\Delta TTFT \propto T_{reuse}
$$

这个公式的意义不是精确预测毫秒数，而是说明：**Agent 场景的收益上限与可复用前缀长度成正相关**，因此系统能否识别稳定前缀，比单纯追求一次性 attention 加速更关键。

### 4.3 Decode 带宽压力模型

decode 阶段的平均单 token 代价可近似写成：

$$
TPOT \propto \frac{M_{read}^{kv}(T_{active})}{BW_{mem}}
$$

其中 $M_{read}^{kv}(T_{active})$ 表示为了生成下一个 token 需要读取的历史 KV 数据量。

该式说明：

- 当 $T_{active}$ 持续增长时，TPOT 往往恶化。
- 即使 prefix 已经命中，decode 仍可能受历史 KV 读取影响。
- 如果上下文预算控制能有效缩小 $T_{active}$，它不仅省显存，也能改善 TPOT。

### 4.4 Swap 与 Recompute 的判据

对一段冷 KV，系统可以选择换出后再恢复，也可以不存、需要时重算。一个简单而实用的判据是：

$$
\text{choose swap if } C_{offload}+C_{restore} < C_{recompute}
$$

其中：

$$
C_{offload}\approx \frac{M_{kv}}{BW_{gpu\to cpu}}
$$

$$
C_{restore}\approx \frac{M_{kv}}{BW_{cpu\to gpu}}
$$

$$
C_{recompute}\approx \alpha \cdot T_{prefix}
$$

这里 $\alpha$ 是单位 token 的重算成本。这个判据非常适合写成边缘 runtime 里的门限策略：

- 对长而稳定的前缀，更倾向于保留或换出。
- 对短且低频的冷前缀，更倾向于直接重算。

### 4.5 KV 压缩判据

是否对 warm/cold KV 进行低比特压缩，可写成一个简单效用式：

$$
U_{compress}= \lambda_m \Delta M - \lambda_q \Delta Q - \lambda_t \Delta T
$$

若 $U_{compress} > 0$，则压缩是值得的。

这个式子把三个常常混在一起讨论的问题拆开了：

- $\Delta M$：节省了多少显存或 CPU 内存。
- $\Delta Q$：对任务质量造成多少影响。
- $\Delta T$：额外引入多少压缩、解压或恢复时延。

因此，KV 压缩不应被写成“总是更好”，而应被写成“在边缘内存紧张、且质量容忍度允许时更好”。

### 4.6 上下文预算分配模型

把上下文按块划分后，可将预算分配抽象成一个带价值的受限选择问题：

$$
\max \sum_i v_i x_i
$$

subject to

$$
\sum_i m_i x_i \le B_{ctx}
$$

其中：

- $x_i=1$ 表示保留该块；
- $x_i=0$ 表示不保留该块；
- $v_i$ 表示价值；
- $m_i$ 表示预算开销。

若扩展到多动作版本，则每个块不再只有保留或丢弃两种状态，而是可以从 `retain / summarize / drop / recompute` 中选一个动作。

### 4.7 调度优先级模型

对请求 $r$，可以定义前缀感知调度分数：

$$
Score(r)=\beta_1 H_{prefix}+\beta_2 Urgency(r)-\beta_3 Cost_{prefill}(r)-\beta_4 MemPressure(r)
$$

这个式子比“命中缓存就优先”更完整，因为它同时考虑：

- 命中收益；
- 时延紧迫性；
- prefill 新增代价；
- 显存压力。

在边缘设备上，这类综合分数比单一的 FCFS 或短作业优先更符合实际。

## 5. 可优化点总表

下表把核心优化点放到同一视角下比较：

| 优化方向 | 优化对象 | 作用阶段 | 主要收益 | 主要代价 | 与 Agent 的关系 | 更适合的场景 | 不适合的场景 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 前缀复用 | 稳定 prompt 与共享文档前缀 | Prefill | 降低 TTFT，减少重复计算 | 需要索引、匹配和缓存保留 | Agent 多轮稳定前缀多，天然高价值 | 系统提示、工具 schema、共享检索片段反复出现 | 会话间几乎无共享、prompt 高度个性化 |
| 分层 KV 管理 | GPU/CPU/pinned/压缩层状态 | Decode + 生命周期管理 | 缓解显存压力，提升会话连续性 | 迁移与恢复开销 | Agent 状态寿命长、冷热分化明显 | 长会话、显存紧张、状态可复用 | 极短会话，状态保留价值低 |
| KV 压缩 | warm/cold KV | 存储层 | 节省容量，增加并发或连续性 | 质量损失和恢复开销 | 旧 observation/tool output 常适合压缩 | 边缘显存吃紧、对旧信息精度容忍度较高 | 高精度要求、上下文本就不长 |
| 上下文预算控制 | 历史消息与工具轨迹 | Prefill + Decode | 降低 $T_{active}$，同时省内存和带宽 | 可能损失长期依赖信息 | Agent 历史增长快，必须预算化 | 工具输出冗长、历史可分块管理 | 每个历史片段都高相关的任务 |
| Chunked Prefill | 超长 prompt 的进入方式 | Prefill | 减少单次阻塞，便于与调度协同 | 调度更复杂，chunk 之间有管理成本 | Agent 检索结果和历史可分段进入 | 超长检索文档、长会话续跑 | 很短 prompt 或极低并发 |
| Prefix-aware Scheduling | 请求排序与批处理 | 系统级 | 提高前缀命中和总体 goodput | 可能牺牲局部公平性 | Agent 请求间共享模式更稳定 | 多会话并发、共享前缀明显 | 请求低并发、共享极少 |

这张表可以直接转化为论文里“问题空间与优化空间”的总览图。

## 6. 统一 runtime 决策：保留、压缩、换出、重算、丢弃

为了避免系统设计变成“很多孤立技巧”，更好的写法是把所有优化统一成针对上下文状态的五类动作：

- `retain`：保留在 GPU 上，追求最快复用。
- `compress`：保留但压缩，牺牲一部分质量或恢复开销来节省容量。
- `swap`：迁移到 CPU/pinned memory，保留可恢复性。
- `recompute`：不长期保存，需要时重算。
- `drop`：彻底丢弃，只依靠被压缩后的文本摘要或直接放弃。

### 6.1 五类动作的适用逻辑

| 动作 | 适用对象 | 主要目标 | 代价 |
| --- | --- | --- | --- |
| retain | 热前缀、当前 decode window、近期高价值历史 | 最低恢复时延 | 占用宝贵 GPU 显存 |
| compress | 冷但仍可能被引用的状态 | 减容量 | 质量损失、恢复处理开销 |
| swap | 长而稳定、重算代价高的前缀 | 节省 GPU 显存同时保留可恢复性 | 迁移带宽和恢复时延 |
| recompute | 冷、小、低频使用的状态 | 避免长期占用容量 | 重新 prefill 成本 |
| drop | 低价值、过期、冗长中间状态 | 直接回收预算 | 可能丢失长期依赖信息 |

### 6.2 三条最重要的判据

#### 判据一：swap vs recompute

若

$$
C_{offload}+C_{restore} < C_{recompute}
$$

则优先换出；否则优先重算。

工程含义是：

- 长前缀、高复用前缀更适合换出。
- 短前缀、低复用前缀更适合重算。

#### 判据二：compress vs keep

若

$$
U_{compress}= \lambda_m \Delta M - \lambda_q \Delta Q - \lambda_t \Delta T > 0
$$

则优先压缩；否则保持原状。

工程含义是：

- 当显存压力高时，$\lambda_m$ 上升，压缩更容易被选中。
- 当任务质量要求高时，$\lambda_q$ 上升，压缩更难被选中。

#### 判据三：retain vs drop

可把每个上下文块的价值密度写成：

$$
\rho_i = \frac{v_i}{m_i}
$$

优先保留 $\rho_i$ 高的块，淘汰 $\rho_i$ 低的块。

在 Agent 场景中，通常可按下面规律初始化 $v_i$：

- system prompt、tool schema：高价值。
- 最近若干轮历史：中高价值。
- 冗长且过时的 observation：低价值。
- 重复出现的检索前缀：高价值。

这比“按时间先后直接截断历史”更符合 Agent 工作负载。

## 7. EdgeAgentCache：收敛后的系统主线

论文系统不宜写成很多分散模块，更适合收敛为四个核心机制。

### 7.1 Session Analyzer

输入：

- tokenized prompt
- session metadata
- retrieval metadata
- tool metadata

输出：

- `prefix_span`
- `reuse_candidate_id`
- `cache_priority`
- `context block tags`

它的目标不是做复杂语义理解，而是把请求切分成**稳定前缀、可复用前缀、易失上下文**三类。

### 7.2 Hierarchical KV Manager

它管理三层状态：

- GPU KV：热前缀和活跃 decode window
- CPU/pinned memory KV：warm KV
- 压缩层 KV：cold KV

它提供的核心操作是：

- `pin`
- `evict`
- `compress`
- `restore`
- `recompute`

### 7.3 Agent Context Budgeter

它不改模型参数，只负责决定上下文块采取什么动作：

- `retain`
- `summarize`
- `drop`
- `recompute`

它的价值在于把“会话管理”从 prompt 构造逻辑中独立出来，变成 runtime 可调策略。

### 7.4 Prefix-Aware Scheduler

它基于前缀命中、请求紧迫性和显存压力进行调度，并与 chunked prefill 协同：

- 高命中前缀请求优先。
- 超长 prompt 按 chunk 进入。
- 当显存吃紧时，根据成本模型在 swap 与 recompute 间切换。

### 7.5 数据流

建议把系统流程收敛成下面这条主线：

1. 请求进入后，Session Analyzer 对上下文分块并识别前缀。
2. Agent Context Budgeter 根据预算与价值模型输出保留策略。
3. Hierarchical KV Manager 查询已有 KV，并决定保留、迁移、压缩还是重算。
4. Prefix-Aware Scheduler 根据命中收益和资源压力组织批次与 prefill 方式。
5. 推理完成后更新会话元数据、前缀索引和各层 KV 状态。

这条主线已经足够支撑一篇系统优化型硕士论文，不需要再把复杂 CUDA kernel 作为主创新点。

## 8. 可测指标与实验假设

### 8.1 指标映射

本文中的公式应映射到可测指标，而不是停留在抽象层。

| 模型/判据 | 对应可测指标 | 说明 |
| --- | --- | --- |
| $M_{kv}$ | `peak GPU memory`、`CPU memory usage` | 验证分层 KV 与压缩层的容量收益 |
| $\Delta TTFT$ | `TTFT` | 验证前缀复用与 chunked prefill 的收益 |
| $TPOT \propto M_{read}^{kv}/BW_{mem}$ | `TPOT`、`memory bandwidth utilization` | 验证 decode 是否转为带宽瓶颈 |
| $C_{offload}+C_{restore}$ vs $C_{recompute}$ | `restore latency`、`prefill recompute latency` | 验证 swap/recompute 门限是否合理 |
| $U_{compress}$ | `任务质量`、`恢复时延`、`显存节省` | 验证压缩收益是否覆盖代价 |
| $Score(r)$ | `end-to-end latency`、`prefix hit rate`、`KV hit rate` | 验证调度收益是否来自缓存感知 |

此外，建议将能耗指标纳入：

$$
E_{resp} \approx P_{gpu}\cdot t_{gpu} + P_{cpu}\cdot t_{cpu}
$$

虽然这也是一阶近似，但足够支持“边缘能耗收益”这一论证方向。

### 8.2 三条核心假设

建议实验围绕三条主假设组织，而不是围绕功能模块逐个罗列。

#### H1：Agent 场景中的主要收益来自重复前缀复用，而不仅仅来自一般长上下文优化

验证方式：

- 对比普通长 prompt 与多轮 Agent 会话。
- 观察 prefix hit rate 与 TTFT 改善是否同步上升。

#### H2：随着会话增长，系统瓶颈会在 `prefill -> decode -> migration` 间迁移

验证方式：

- 固定模型与平台，逐步增加会话长度。
- 观察 TTFT、TPOT、restore latency、带宽利用率的变化。

#### H3：不同会话类型存在不同的最优状态策略

验证方式：

- 文档问答 Agent 更偏向保留共享文档前缀。
- 工具调用 Agent 更偏向保留系统提示和工具 schema，压缩旧 observation。

### 8.3 建议 baseline

- 原始本地 runtime
- 仅 prefix cache
- 仅 swap/recompute
- 仅 KV quantization
- 完整 EdgeAgentCache

### 8.4 建议消融

- 去掉 prefix-aware scheduler
- 去掉 hierarchical KV
- 去掉 context budgeter
- 去掉 cold KV compression
- 固定所有请求使用 recompute
- 固定所有冷状态使用 swap

## 9. 回改后的论文提纲

下面给出一版更收敛的提纲。与原提纲相比，重点是让第 3、4、6 章形成闭环，而不是平铺技术模块。

### 第 1 章 绪论

- 介绍边缘大模型与 Agent 应用的发展背景。
- 指出核心矛盾：有限显存、有限带宽、有限能耗预算，与持续增长的会话状态之间的矛盾。
- 说明本文定位为系统优化型研究，重点是运行时状态管理与调度管理。
- 概述研究内容、技术路线和主要贡献。

### 第 2 章 相关技术与研究现状

- Transformer 推理、prefill/decode 分离及 KV cache 基础。
- 长上下文推理与注意力优化。
- Prefix caching、PagedAttention、chunked prefill、缓存感知调度。
- 边缘端 LLM 推理与长会话管理的现状与不足。
- 小结：已有工作多针对单点优化，缺少面向 Agent 长会话边缘推理的统一系统视角。

### 第 3 章 问题建模与统一成本模型

- 定义 Agent 长会话的上下文结构：
  - system prompt
  - tool schema
  - retrieved documents
  - dialogue history
  - tool traces
- 给出边缘推理目标：
  - 最小化 TTFT
  - 最小化 TPOT
  - 最小化峰值显存
  - 最小化单响应能耗
  - 最大化 session continuity
- 建立统一成本模型：
  - KV 容量模型
  - 前缀复用收益模型
  - decode 带宽压力模型
  - swap/recompute 判据
  - compress/keep 判据
  - retain/drop 价值密度模型
- 明确决策变量：
  - 哪段状态保留在 GPU
  - 哪段状态迁移到 CPU/pinned memory
  - 哪段状态压缩
  - 哪段状态直接重算或丢弃

### 第 4 章 面向 Agent 长会话的边缘推理系统设计

- 给出系统目标：在不修改基础模型参数的前提下，提高长会话连续性并降低时延、显存和能耗。
- 以四个核心机制组织系统：
  - Session Analyzer
  - Hierarchical KV Manager
  - Agent Context Budgeter
  - Prefix-Aware Scheduler
- 说明四者之间的数据流与协同关系。
- 说明成本模型如何被映射为 runtime 策略。
- 分析系统开销：
  - 索引开销
  - 迁移开销
  - 压缩开销
  - 调度开销

### 第 5 章 原型实现

- 说明实现底座，如 `llama.cpp` 或 `MLC-LLM`。
- 描述关键数据结构：
  - session metadata
  - prefix index
  - KV handle
  - tier metadata
  - budget planner
- 说明工程取舍：
  - 不自研复杂 attention kernel
  - 不做分布式多机
  - 不做训练阶段改模

### 第 6 章 实验设计与结果分析

- 实验平台：Jetson Orin 级边缘 GPU。
- 模型：Qwen/Gemma/Llama 的 1.5B-7B 量化版本。
- 工作负载：
  - 多轮文档问答 Agent
  - 工具调用 Agent
  - 长历史连续对话
- 指标：
  - TTFT
  - TPOT
  - end-to-end latency
  - peak GPU memory
  - CPU memory usage
  - energy per response
  - 任务质量
  - prefix hit rate
  - KV hit rate
- 三条主假设：
  1. 收益是否主要来自前缀复用。
  2. 瓶颈如何在 `prefill / decode / migration` 间迁移。
  3. 不同会话类型的最优状态策略是否不同。
- 消融：
  - 去掉 prefix-aware scheduler
  - 去掉 hierarchical KV
  - 去掉 context budgeter
  - 调整 swap 阈值
  - 调整 KV 压缩比特数

### 第 7 章 总结与展望

- 总结本文提出的统一问题定义、系统设计与实验结论。
- 展望：
  - 云边协同 KV 迁移
  - speculative decoding 与状态管理协同
  - MLA/GQA 与边缘 runtime 的联合设计

## 10. 这篇报告如何映射回论文正文

如果后续开始正式写论文，可以按下面方式复用本报告内容：

- 本文第 1-3 节可直接拆入论文第 3 章的问题定义部分。
- 本文第 4-7 节可直接拆入论文第 3、4 章的成本模型与系统设计部分。
- 本文第 8 节可直接迁移到论文第 6 章，作为实验问题与指标设计的基础。
- 本文第 9 节已经是一版可直接继续细化的章节骨架。

不建议直接原样搬入论文的部分主要有两类：

- 过于工程化的门限描述，正式成稿时应改写为更学术的“策略设计与分析”语言。
- 带有明显综述口吻的段落，正式成稿时应进一步收束为问题驱动的论证。

## 11. 写作建议

这条研究线最容易写散的地方有两个：

第一，容易把论文写成“很多优化点的堆叠”。更好的写法是始终围绕一句主线展开：

> Agent 长会话让边缘推理的瓶颈从单次长 prompt 计算，转向会话状态的长期管理与复用。

第二，容易把系统设计写成“很多工程模块”。更好的写法是只抓住一个统一问题：

> 在有限边缘预算下，系统如何对不同上下文状态做保留、压缩、换出、重算与丢弃决策。

只要论文始终围绕这两个中心句展开，整条叙事就会稳定很多。

## 参考与衔接材料

- [Transformer-based LLM](./transformerbasedllm.md)
- [Long Context Attention](./longcontext.md)
- [Model Forward 瓶颈分析以及优化方法](./inferenceperf.md)
- [LLM 推理优化岗位深研：调度侧与推测解码侧](../interview/LLM%20推理优化岗位深研：调度侧与推测解码侧.md)
- [从代码看 SGLang 的 KV Cache](../sglang/从代码看%20SGLang%20的%20KV%20Cache.md)
- [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](../../paperreadings/llm/distserve.md)
