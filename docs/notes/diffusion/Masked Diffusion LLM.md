---
title: Masked Diffusion Large Language Model
tags:
  - LLMInference
created: 2025-12-2
---

# Masked Diffusion Large Language Model

:book: 这里以 Ant Group 的 LLaDA 论文为例，介绍 Masked Diffusion LLM 的基本原理和实现方法；后面结合 SGLang 对该模型的推理实现进行说明。

## Traing Details

![Llada 训练](img/llada.png)

### 基于掩码的扩散模型(Masked Diffusion Model)
LLaDA 通过定义前向过程（Forward Process）和反向过程（Reverse Process）来建立模型分布 $p_\theta(x_0)$。
- 前向过程（破坏数据）：LLaDA 不像传统扩散模型那样添加高斯噪声，而是通过**掩码（Masking）** 来破坏数据。
  - 给定一个文本序列 $x_0$，过程从 $t=0$（无掩码）进行到 $t=1$（全掩码）。
  - 在任意时间步 $t \in (0, 1)$，序列 $x_t$ 中的每一个 Token 都有 $t$ 的概率被掩盖（变成 [MASK]），或者以 $1-t$ 
  - 使用的掩码比例是随机的（$0$ 到 $1$ 之间），而 BERT 使用的是固定的比例。这使得 LLaDA 在数学上成为一个有原则的生成模型，能够逼近最大似然估计。
- 反向过程（恢复数据）与训练目标：LLaDA 的核心是一个掩码预测器（Mask Predictor） $p_\theta(\cdot|x_t)$，它接收部分被掩盖的序列 $x_t$，并同时预测所有被掩盖的 Token。其损失函数定义为：
  $$L(\theta) \triangleq -E_{t,x_0,x_t} \left[ \frac{1}{t} \sum_{i=1}^{L} 1[x_i^t = M] \log p_\theta(x_i^0|x_t) \right]$$
  - $t$: 时间步/噪声水平。
  - $x_0$: 原始的、干净的数据样本（例如一句话或一张图的 Token 序列）。
  - $x_t$: 被破坏（加噪/掩码）后的数据样本。
  - $t \sim [0, 1]$ (Continuous Random Variable)这是一个连续的时间变量，从 0 到 1 均匀采样
  - $\mathbf{1}[x_t^i = \text{M}]$: 如果位置 $i$ 上的 Token 是掩码标记（$M$），则该项为 1；否则为 0。
  - $\log p_\theta(x_0^i | x_t)$: 给定被破坏的序列 $x_t$，模型预测位置 $i$ 处的原始 Token 是 $x_0^i$ 的概率。
  - 系数 $\frac{1}{t}$ 至关重要，它使得该损失函数成为模型负对数似然的上界，从而保证了 LLaDA 具有坚实的生成模型理论基础。
> [!NOTE]
> 这个公式引入了连续时间 $t$，实际上是在训练一个**生成**模型。在推理（生成）阶段，模型会从一个全 [MASK] 的序列开始，迭代地、一点一点地根据置信度把 [MASK] 填回去，最终生成完整的内容。这通常被称为 Iterative Decoding（迭代解码） 或 Discrete Diffusion（离散扩散）。

### 预训练 (Pre-training)
LLaDA 的预训练旨在让模型学会从不同程度的掩码中恢复文本。
- 模型架构：
  - 双向注意力：与 GPT 系列（自回归模型）不同，LLaDA 使用的 Transformer 不使用因果掩码（Causal Mask）。这意味着在预测时，模型可以看到上下文中的所有内容（包括“未来”的 Token，如果它们没被掩盖的话）。-
  - 组件调整：为了适应架构，LLaDA 做了一些微调。例如，它使用标准的**多头注意力（MHA）**而不是分组查询注意力（GQA），因为它不使用 KV CachE。为了保持参数量与 LLaMA3 8B 一致，它相应减少了前馈神经网络（FFN）的维度。
  - 数据与流程：训练时，对每个序列随机采样一个时间步 $t \in [0, 1]$，然后按概率 $t$ 独立掩盖 Token。
    - 输入是一段完整的文本（如图中上方彩色方块）。
    - 采样一个掩码比例 $t$（Mask ratio）。独立地掩盖所有 Token，无论是前面的还是后面的 Token，都有可能被换成 Mask token。
    - 预测：Mask predictor 接收这个残缺的序列，试图还原那些被掩盖的 Token。
### 监督微调 (Supervised Fine-Tuning, SFT)
这是让 LLaDA 具备对话和指令遵循能力的关键步骤。SFT 需要模拟条件分布 $p_\theta(r_0|p_0)$，即给定提示词（Prompt）生成回复（Response）。将 Prompt ($p_0$) 和 Response ($r_0$) 拼接在一起。
- Prompt 部分：保持完全不掩盖，作为已知条件。
- Response 部分：按照概率 $t$ 进行独立掩盖，生成 $r_t$。

模型需要根据完整的 Prompt 和残缺的 Response 来填补空缺。这使得模型学会了“根据指令说话”。

### 采样与生成 (Sampling)
LLaDA 的生成是一个迭代的去噪过程：
- 初始状态 ($t=1$)：Response 部分全都是 Mask token，Prompt 部分保持可见。
- 中间步骤 (Intermediate step)：模型预测出所有 Token 的一种可能填法，然后根据调度策略，保留置信度高的，重新掩盖 (Remask) 置信度低的。
  > [!NOTE]
  > 原则上，重掩码策略应当是纯随机的。然而，受 LLM 采样中退火技巧 的启发，Llada 采用了一种低置信度重掩码策略（Low-confidence remasking strategy）
  >
  > 时间 $t \in (0, 1]$ 到 $s \in [0, t)$ 的中间步骤中，基于预测结果，将置信度最低的那 $s/t$ 比例的 Token 重新掩盖
- 最终状态 ($t=0$)：经过多次迭代，所有 Mask 都被替换为具体的 Token，形成完整的回复。

## Problems Proposed in Paper

- **长度需指定**：GPT 生成时，遇到 <eos> 标记自动停止。但 LLaDA 基于扩散，生成前通常需要预设一个长度（比如“给我生成一个 100 个 Token 的回复”）。虽然作者说模型对长度不敏感，但这依然不如自动停止方便。

- **没有 KV Cache**：GPT 推理快是因为有 KV Cache（不用重复计算前面的词）。LLaDA 是并行生成的，传统的 KV Cache 优化不适用，需要新的优化技术。
  > [!NOTE]
  > Block-wise KV Cache 是可以使用的，正在生成的 block 不能使用，但是已经生成并冻结的 block 是可以使用 KV Cache 的，详情见下文“推理优化”部分。

- **没有 RLHF**：LLaDA 目前还是“预训练”状态，还没经过 RLHF 的 post-training。
## 推理优化

对于“正在生成的 Block”：依然不能使用 KV Cache。因为在这个 64 Token 的小窗口内，扩散过程依然是反复迭代、双向注意力的，每一步都要重算。

对于“已固定的前文”：可以使用 KV Cache。因为 Block 1 一旦生成完毕并冻结，它对于 Block 2 来说就是永远不变的上下文（Context）。Block 2 在迭代去噪时，虽然自己的 KV 会变，但它查询 Block 1 的 KV 是固定的。