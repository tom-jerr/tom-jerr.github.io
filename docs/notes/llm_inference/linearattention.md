# Linear Attention
## Linear Attention
传统 Transformer 的核心痛点是 **Softmax Attention** 的时间与空间复杂度随序列长度呈平方级增长，即 **$O(N^2)$**。

传统 Attention 的计算方式是：先算 Query 和 Key 的相似度矩阵，再乘 Value：

$$
O = \text{Softmax}(Q K^T) V
$$

Linear Attention 的**核心思想是去掉 Softmax，利用一个特征映射函数 $\phi(\cdot)$ 来代替，然后利用矩阵乘法的结合律**，改变计算顺序：

$$
O = (\phi(Q) \phi(K)^T) V = \phi(Q) (\phi(K)^T V)
$$


我们将 $\phi(K)^T V$ 提前计算好（它是一个固定大小的矩阵），这样复杂度就降到了 **$O(N)$**。在自回归（RNN）视角下，它可以表示为隐状态 $S_t$ 的不断累加：

$$
S_t = S_{t-1} + \phi(k_t) v_t^T
$$

$$
o_t = \phi(q_t) S_t
$$

## Gated DeltaNet
初代的 Linear Attention 虽然速度快，但语言建模能力一直不如标准 Transformer。研究界经过几年的探索，最终演进出了 Gated DeltaNet 这个“六边形战士”。
- 阶段一：Vanilla Linear Attention (2020)
  - 机制：单纯地把历史信息无脑累加到隐状态 $S_t$ 中。
  - 痛点：记忆像一个不断塞满的垃圾桶，模型无法区分重要和不重要的信息，导致长文本表现拉胯。
- 阶段二：Gated Linear Attention, GLA (2023-2024)
  - 进化：向 RNN / LSTM 借用了“遗忘门（Forget Gate）”的思想，引入了数据驱动的 Gating 机制。
  - 机制：$S_t = \alpha_t S_{t-1} + k_t v_t^T$。其中 $\alpha_t$ 是一个根据当前输入计算出的标量（或向量）。
  - 效果：模型拥有了“全局重启开关”，当遇到新话题时，可以通过让 $\alpha_t \to 0$ 快速衰减并清空无关的历史记忆。
- 阶段三：DeltaNet (2024)进化：引入了神经网络早期的 Delta Rule（增量学习法则）。
  - 机制：不再无脑写入 $k_t v_t^T$，而是先看当前记忆里已经有了什么（提取特征 $\hat{v}_t = S_{t-1}^T k_t$），然后只把目标值 $v_t$ 与当前预测值 $\hat{v}_t$ 的差值写入记忆。
  - 效果：模型拥有了“精准橡皮擦”，可以实现精确的记忆覆盖（Selective Overwriting），大幅提升了信息检索（如大海捞针）的能力。
- 阶段四：Gated DeltaNet (2024-2026)终极形态：结合了 Mamba/GLA 的 Gating 机制（全局遗忘） 和 DeltaNet 的 Delta Rule（精准修改）。
  - 机制：$S_t = \alpha_t S_{t-1} + \beta_t k_t (v_t - \hat{v}_t)^T$
  - 应用：以极小的性能损耗，在长文本、数学推理和检索上打平甚至超越了标准 Attention。 
- 目前业界的做法通常是混合架构（Hybrid）：比如 3 层 Gated DeltaNet 搭配 1 层标准 Attention（Qwen3-Next 模式），既要线性的推理速度和恒定缓存，又要全注意力的精度。

