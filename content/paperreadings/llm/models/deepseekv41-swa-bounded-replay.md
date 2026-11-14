---
title: DeepSeek-V4.1-Flash 的 SWA Bounded Replay：保存前缀与精确重算之间
created: 2026-10-07
tags:
  - Paper Notes
  - LLMInference
description: 从 DeepSeek-V4 的磁盘前缀缓存策略出发，解释为什么长期保存 SWA 与精确恢复都昂贵，以及 V4.1-Flash 如何用近似的窗口回放换取更小的持久 KV 缓存。
katex: true
---

# DeepSeek-V4.1-Flash 的 SWA Bounded Replay：保存前缀与精确重算之间

在[上一篇 DeepSeek-V4 架构与 SGLang 实现](deepseekv4.md)中，`UnifiedRadixCache` 的 Full/SWA components 给出了一个很直观的设备缓存现象：一段前缀的长期压缩 KV 还在，旧 SWA 窗口却可能已经被回收。树上的 SWA tombstone 让系统知道“token 前缀相同”和“此刻能直接继续计算”不是一回事。

把问题放大到 SSD 上，取舍更尖锐：**长期保存每个可能复用点的 SWA 状态很贵；丢掉它以后，按原模型精确恢复又可能要回放多个窗口。** DeepSeek-V4 技术报告给出全量保存、周期 checkpoint、零保存三种磁盘 SWA 策略；DeepSeek-V4.1-Flash 则进一步用 **SWA Bounded Replay**：只回放最近一个窗口，接受与完整前缀计算不完全相同的状态。本文关注这条从精确复用到近似恢复的成本链。[^v4-disk] [^v41-cache] [^v41-replay]

## 先区分三种状态

DeepSeek-V4 的 CSA/C4 与 HCA/C128 把长历史压缩为全局 KV；SWA 在每层保存最近窗口的未压缩 KV。压缩历史的价值随前缀长度增长，某个 SWA 状态却只服务于相应的续算位置。CSA/HCA 还有尚未凑齐压缩块的尾部状态，命中前缀时，完整压缩块可从磁盘读取，未完成块的尾部仍要重新计算。它与 SWA 窗口恢复是两笔不同的账。[^v4-layout] [^v4-disk]

| 状态 | 命中时的作用 | 生命周期与恢复问题 |
| --- | --- | --- |
| 全局压缩 KV | 提供完整块的长程历史 | 前缀越长，复用价值越高；V4 论文选择持久保存 |
| SWA KV | 让新 token 读取当前位置附近的细粒度历史 | 只需最近窗口，却必须对应**命中末端**的状态 |
| 未完成压缩块 | 保留生成下一个压缩条目所需的尾部输入 | 完整块的 KV 命中不能自动恢复这部分状态 |

这也解释了为什么“保留完整前缀”不是简单保存一棵 radix tree。树上的 key、Full slot 或磁盘索引只是定位信息；真正占容量的是每层 KV 字节。V4 报告指出，如果把所有 token 的 SWA KV 都写盘，它的体量约为压缩 CSA/HCA KV 的 **8 倍**，而一次命中只会读末端的一个窗口，形成高写入、低读取比例。[^v4-disk]

## V4：精确语义下的三个存储点

V4 报告把磁盘 SWA 策略摆成一条连续的存算曲线：[^v4-disk]

| 策略 | 持久保存什么 | 命中后做什么 | 代价落点 |
| --- | --- | --- | --- |
| Full SWA Caching | 所有 token 的 SWA KV | 读取命中末端的窗口 | 写入量、SSD 容量与长期保留 |
| Periodic Checkpointing | 每隔 $p$ 个 token 保存一次末端窗口 | 读取最近 checkpoint，再重算后续 token | 用 $p$ 调节保存量与尾部重算 |
| Zero SWA Caching | 不持久保存 SWA KV | 依靠已缓存的全局 KV 精确恢复 SWA | 深层窗口依赖的重算 |

V4.1 论文补充了实际部署的口径：V4 的持久缓存并非按 Full 策略保存所有位置，而是在 **prompt 末端和 output 末端**保存 SWA 窗口，以支持续写与多轮对话；全局 KV 与 SWA KV 独立按 LRU 管理。即便如此，SWA 仍占持久缓存容量的**近一半**。多轮短 turn 会产生大量续算点，每个点都可能需要一份未压缩的窗口状态。这里“全量 SWA 约为压缩 KV 的 8 倍”是 V4 报告讨论的全 token 保存方案；“近一半”是 V4.1 回顾的实际持久缓存，两者的保存粒度不同，不能拿来直接互相否定。[^v41-cache]

为什么不干脆采用 Zero SWA Caching？因为 **SWA 的计算范围是一个窗口，跨层依赖范围却会扩张**。第 $l$ 层末端窗口的 KV 取自第 $l$ 层的 hidden states；这些 hidden states 又受到第 $l-1$ 层更早一个窗口的影响。逐层向下追溯，精确恢复 $L$ 层的末端 SWA KV，需要触及约 $L\,n_{\mathrm{win}}$ 个历史 token 的跨度。已缓存的全局 KV 能免掉长程压缩分支的重算，但不能补出这些缺失的局部依赖。V4.1 论文称这样的完整 forward 在生产部署中成本过高。这里的 $L\,n_{\mathrm{win}}$ 是**回放跨度**，不是一个可直接换算成延迟或 FLOPs 的加速比。[^v4-disk] [^v41-cache]

以 $n_{\mathrm{win}}=128$、40 层作量级示意，跨层精确恢复涉及约 5120 token 的历史跨度；只回放 128 token 时，最前端的局部依赖必然缺失。这个示意说明依赖范围为何比单层窗口大，不能当作 V4.1 生产路径的实测开销：该模型采用 20 层 encoder 加 20 层 decoder，两段恢复路径不同。[^v41-arch] [^v41-replay]

## V4.1：把回放限定在一个窗口

SWA Bounded Replay 把回放起点设为 $s$，仅处理命中前缀末端的 $n_{\mathrm{win}}$ 个 token。对回放中的位置 $i$，原本会读取的局部 key 被截到当前回放片段内：

$$
\text{SWA keys}(i)=\left[\max\left(s, i-n_{\mathrm{win}}+1\right),\ i\right].
$$

这条截断就是近似发生的位置。回放片段开头的 token 看不到 $s$ 之前本应属于其窗口的 SWA KV；误差可随层传播。由于**全局 KV 仍从命中的前缀缓存读取**，回放的工作主要用于补局部 SWA 状态，成本不再随整个前缀长度增长。模型没有“无损恢复”这个保证；同一段后缀在不同命中边界之后产生的全局 KV 与 SWA KV，也可能不再数学等价。[^v41-replay]

论文的实现分成两条路径：

1. **Encoder SWA Bounded Replay**：当全局 KV 命中但 encoder SWA 已过期，回放缓存前缀末端一个窗口，并与未命中的 suffix 一起处理。回放的旧 token 只重新生成 SWA KV，不重算或覆盖已缓存的全局 KV；新 suffix 则同时产生两类 KV。因此，后续全局 KV 虽被保存，其值可能受这个近似边界影响。[^v41-replay]
2. **Decoder SWA Bounded Replay**：V4.1 的 Causal Encoder–Decoder（CED）从 encoder 最终 hidden states 投影 decoder 的全局 KV，prompt 无须完整穿过 decoder；但 decoder 各层的 SWA KV 仍要供第一步 decode 使用。论文让末端一个窗口的 encoder 输出经过 decoder，并在窗口内截断局部注意力；生成的 decoder SWA 只用于解码，不进入前缀缓存。作者还在 post-training 中模拟这一回放，以适配近似状态。[^v41-arch] [^v41-replay]

第二条路径是容易遗漏的成本：即使持久缓存里不存在 SWA miss，CED 仍要为 decode 准备 decoder 的局部状态。论文用有界回放把长 prompt 的 prefill 主体留在 encoder；其复杂度写为 $\mathcal{O}(NL/2+n_{\mathrm{win}}L/2)$，前提是 $N\gg n_{\mathrm{win}}$。这是 CED 与 decoder 回放共同带来的结果，不能只归功于磁盘缓存命中。[^v41-arch]

## 为什么节省的持久空间接近八分之七

令 V4 持久缓存中的全局 KV 为 $G$，SWA KV 为 $S$。论文称实际部署中 $S$ 接近 $G$；V4.1 一方面不再把 SWA 放进持久缓存，另一方面通过 CSA2 的跨层复用与更低精度，把全局 KV 压到约原来的 $G/4$。所以：

$$
\frac{\text{V4.1 持久 KV}}{\text{V4 持久 KV}}
\approx \frac{G/4}{G+S}
\approx \frac{1}{8}\quad(S\approx G).
$$

这个 **$1/8$ 是持久缓存总量的两个因素相乘**，不是 Bounded Replay 单独带来的八倍压缩，也不是 GPU HBM 中每 token 的压缩比。论文另报 V4.1 全局 KV 约 **890 B/token**、约为 V4-Flash 的 $1/4$；那个数字讨论的是全局 KV 布局。[^v41-cache] [^v41-conclusion]

持久 SWA 取消后，并非所有局部状态都立即重算。论文将 SWA 放入每台机器以 **10% host DRAM** 配置的分布式内存池，TTL 只有数分钟；全局 KV 则保留在有至少 **72 小时**生命周期保证的持久缓存里。短时间内 SWA 命中就直接续算；只在全局 KV 命中、SWA 已失效时使用 encoder 的有界回放。这个冷热分层与 SWA 的分钟级复用、全局 KV 的长尾复用相匹配。[^v41-cache]

## 与 SGLang tombstone 的关系和边界

两者共享一个观察：**长程历史的命中，不自动意味着局部窗口状态可用。** 但它们回答的问题不同。上一篇所述 SGLang `UnifiedRadixCache`，在给定源码 commit 的普通 paged SWA 设备缓存中，用 Full/SWA components、tombstone 与连续窗口覆盖判断一个命中能否**精确**复用；命中不够就退回安全边界。V4.1 论文讨论的是 DeepSeek 自身持久缓存中 SWA 过期后的**近似**恢复，以及 CED decoder 的局部状态构建。不能因为前者有 tombstone，就推断该 SGLang 版本已经实现 Bounded Replay。[^sglang-cache] [^v41-replay]

代价也必须留在结论里。论文报告回放对响应质量的影响很小，但明确承认结果不与完整 forward 数学等价，并把极端条件下的 SWA 恢复边界列为仍需测试的风险；文中没有给出一个可直接迁移到别的模型或工作负载的通用误差上界。对于必须保持逐 token 精确等价的场景，应继续使用保留 SWA 状态、checkpoint 或精确回放等方案。**Bounded Replay 的选择成立，是因为 V4.1 同时具备可复用的全局 KV、短窗口局部注意力，以及论文所报告的质量容忍度。**[^v41-replay] [^v41-limit]

## Reference

[^v4-layout]: [DeepSeek-V4 技术报告 §3.5.1：异构 KV、state cache 与压缩块](https://arxiv.org/html/2606.19348v1#S3.SS5.SSS1)。
[^v4-disk]: [DeepSeek-V4 技术报告 §3.5.2：磁盘 KV 与三种 SWA 策略](https://arxiv.org/html/2606.19348v1#S3.SS5.SSS2)。
[^v41-arch]: [DeepSeek-V4.1-Flash 技术报告 §2.1–2.2：40 层 CED 与 decoder SWA 成本](https://arxiv.org/html/2609.19969#S2.SS2)。
[^v41-cache]: [DeepSeek-V4.1-Flash 技术报告 §3.2.1：持久 KV 缓存管理](https://arxiv.org/html/2609.19969#S3.SS2.SSS1)。
[^v41-replay]: [DeepSeek-V4.1-Flash 技术报告 §3.2.2：Encoder/Decoder SWA Bounded Replay](https://arxiv.org/html/2609.19969#S3.SS2.SSS2)。
[^v41-conclusion]: [DeepSeek-V4.1-Flash 技术报告结论：890 B/token 与持久缓存比例](https://arxiv.org/html/2609.19969#S6)。
[^v41-limit]: [DeepSeek-V4.1-Flash 技术报告：近似恢复的未测边界](https://arxiv.org/html/2609.19969#S6)。
[^sglang-cache]: [SGLang `UnifiedRadixCache` 源码快照](https://github.com/sgl-project/sglang/blob/f47d89564dc2479be8e2e866a2b17f149318758e/python/sglang/srt/mem_cache/unified_radix_cache.py)、[Full/SWA component 匹配规则](https://github.com/sgl-project/sglang/blob/f47d89564dc2479be8e2e866a2b17f149318758e/python/sglang/srt/mem_cache/unified_cache/components/README.md)。
