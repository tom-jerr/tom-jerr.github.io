---

title: Attention & Transformers
created: 2025-09-30
updated: 2026-01-18
tags:

- LLMInference
description: 首先从概率论角度介绍语言生成模型最终训练目标然后介绍注意力机制（Attention）及其在 Transformer 架构中的应用，并详细解析 Transformer 的结构和工作流程。
cover: /img/transformer.png
---

# Attention & Transformers
## 概率论相关知识

### 最大似然估计

假设数据服从某个条件分布族
$$
   X \sim p(x \mid \theta)
$$
给定观测样本 $x_1,\dots,x_n$
构造联合条件分布：
 $$
   p(x_1,\dots,x_n \mid \theta)
$$
把它当作关于 $\theta$ 的函数并最大化：
$$
   \hat{\theta} = \arg\max_\theta p(x_1,\dots,x_n \mid \theta)
$$

所以：

> 最大似然估计 = 在参数空间中，寻找使条件分布下观测数据的联合概率(密度)最大的参数。

## Transfomrer 的数学原理

### 生成模型的训练目标

生成模型实际上需要学习的是数据分布本身：$p_{data}(x)$

我们使用 LLM 用一个带参数的分布族逼近数据分布：$p_\theta(x)$

所以我们需要最小化这两个分布的 KL 散度：

$$
\min_\theta \; \mathrm{KL}(p_{\text{data}}(x) \| p_\theta(x)) = \min_\theta \; \mathbb{E}_{p_{\text{data}}} \left[ \log \frac{p_{\text{data}}(x)}{p_\theta(x)} \right]
$$

展开：

$$
=\min_\theta \;\big(\mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}(x)] - \mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)]\big)
$$

注意：

- 第一项$\mathbb{E}_{p_{\text{data}}}[\log p_{\text{data}}(x)]$与模型参数 θ 无关（真实分布固定）
- 所以最小化 KL 等价于最大化第二项：$\min_\theta \mathrm{KL}(p_{\text{data}}(x) \| p_\theta(x)) \;\Longleftrightarrow\; \max_\theta \mathbb{E}_{p_{\text{data}}(x)} \log p_\theta(x)$

所以我们的训练目标就变成了：$\max_\theta \; \mathbb{E}_{p_{\text{data}}(x)} \log p_\theta(x)$

展开为经验形式就是我们熟悉的 token-level cross entropy loss：

$$
\max_\theta \sum_{i=1}^N \log p_\theta(x^{(i)})
$$

---

### ARM 预测

因为自回归分解：

$$
p_\theta(x) = \prod_{t=1}^T p_\theta(x_t \mid x_{<t})
$$

所以训练目标变成：

$$
\max_\theta \sum_{t=1}^T \log p_\theta(x_t \mid x_{<t})
$$

这就是每一步做的 softmax + cross entropy。

$$
  p_\theta(x_t \mid x_{<t}) = \text{Softmax}(f_\theta(x_{<t}))
$$

- 其中 $f_\theta$ 是由多层注意力 + MLP 组成的函数。
- 注意力是核函数加权平均：

  - 单头注意力：

  $$
  \text{Attn}(Q,K,V) = \text{Softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
  $$

  - 等价于：

  $$
  t = \sum_s \underbrace{ \frac{\exp(q_t^\top k_s)}{\sum_j \exp(q_t^\top k_j)} }_{\text{相似度核}} \, v_s
  $$

---

### DLLM 预测

目标函数：

$$
\min_\theta \; \mathbb E_t  \Big[ \mathrm{KL}\big( q(x_{t-1}\mid x_t, x_0) \;\|\; p_\theta(x_{t-1}\mid x_t)\big)\Big]
$$

含义：

- $x_0$：真实完整句子（干净文本）
- $x_t$：在噪声等级 t 下被破坏的句子
- $x_{t-1}$：稍微少一点噪声的句子

不是预测下一个 token，而是预测**整句在噪声时间轴上如何往真实句子流动一步**。

---

### Summary

- AR LLM 学的是

  $$
  abla_{x_t} \log p(x_t \mid x_{<t})
  $$
- Diffusion LLM 学的是

$$
\nabla_{x} \log p_t(x)
$$

- 二者都是在逼近同一个东西：

$$
\nabla_x \log p_{\text{data}}(x)
$$

| 模型          | 预测的是什么                                    | 训练目标公式                                                                                                          |
| ------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| AR LLM        | 下一个 token 的条件分布                         | $$\min_\theta \; \mathrm{KL}\big(p_{\text{data}}(x)\;\|\;p_\theta(x)\big)$$                                           |
| Diffusion LLM | 所有位置在去噪一步后的 token 分布（并行、全局） | $$\min_\theta \; \mathbb{E}_t\Big[ \mathrm{KL}\big(q(x_{t-1}\mid x_t, x_0)\;\|\;p_\theta(x_{t-1}\mid x_t)\big)\Big]$$ |


---
## Attention

### 传统的 self-attention

#### 传统的序列模型处理方式

在传统的序列处理模型（如 RNN、LSTM 和 GRU）中，模型是按顺序逐个处理序列中的元素（例如单词或字符），并且每个元素的处理依赖于前一个元素的隐藏状态。

> [!IMPORTANT]
> 这种方法在处理长序列时会面临**梯度消失或梯度爆炸的问题**，导致模型难以捕捉长距离的依赖关系。


#### 自注意力机制核心思想

对于序列中的每个元素，**模型可以同时考虑序列中所有其他元素的信息**，从而动态地计算每个元素与其他元素之间的**相关性（即“注意力”）**，并根据这些相关性对序列中的信息进行加权求和。这样，模型能够更高效地捕捉序列内部的长距离依赖关系，而不需要像 RNN 那样逐个处理序列元素。

## Transformer 推理过程

### 0. 符号设定和配置

假设模型配置如下：

- $B$: Batch Size
- $L$: 当前处理的序列长度

  - **Prefill 阶段**: $L = L_{prompt}$（提示词长度）
  - **Decode 阶段**: $L = 1$（当前生成的一个 token）
- $T_{past}$: KV Cache 中已存储的历史长度

  - **Prefill 阶段**: $T_{past} = 0$
  - **Decode 阶段**: $T_{past} \ge L_{prompt}$
- $d$: 隐藏层维度 ($d_{model}$)
- $H$: Query 头数
- $G$: KV 头数 (GQA, G < H)
- $d_h$: 单头维度 (d / H)
- $V_{vocab}$: 词表大小

---

### 输入与 Embedding

**输入**: Token IDs $I \in \mathbb{N}^{B \times L}$

Embedding 查找:

$$
X = \text{Embedding}(I)
$$

形状：$X \in \mathbb{R}^{B \times L \times d}$

> **差异点**:
>
> - **Prefill**: $L$是整个 prompt 长度（例如 512）。
> - **Decode**: $L$ 是 1。

---

### Transformer Layer (循环 N 层)

进入第 $i$ 层，输入为 $X$（残差流）。

#### 2.1.  Pre-Normalization (RMSNorm)

$$
X_{norm} = \text{RMSNorm}(X)
$$

形状：$X_{norm} \in \mathbb{R}^{B \times L \times d}$

#### 2.2. QKV 线性投影 (GQA)

$$
Q = X_{norm} W_Q, \quad K = X_{norm} W_K, \quad V = X_{norm} W_V
$$

**形状变化**:

$$
  W_Q \in \mathbb{R}^{d \times (H \cdot d_h)}
  $$
$$
  W_K, W_V \in \mathbb{R}^{d \times (G \cdot d_h)}
  $$
- 输出 reshape 后：

   $$
    Q \in \mathbb{R}^{B \times H \times L \times d_h}
    $$
   $$
    K \in \mathbb{R}^{B \times G \times L \times d_h}
    $$
   $$
    V \in \mathbb{R}^{B \times G \times L \times d_h}
    $$

#### 2.3. RoPE 旋转位置编码

根据当前 token 在全局序列中的绝对位置 $pos$ 进行旋转。

注意力机制只能看到语义相关性，而不能知道位置相关性。所以我们需要将位置相关性也编码到 Q*K 中，所以需要位置编码的存在。

将位置编码与词向量通过旋转矩阵相乘，使得词向量不仅包含词汇的语义信息，还融入了位置信息

$$
(R_mq)^T(R_nk) = q^TR_m^TR_nk = q^TR_{m-n}k
$$

给位置为 m 的向量 q 乘上矩阵$(R_m$)、位置为 n 的向量 k 乘上矩阵$(R_n$)用变换后的 Q,K 序列做 Attention，Attention 就自动包含相对位置信息


- **Prefill**: $pos$ 是向量 $[0, 1, \dots, L-1]$。
- **Decode**: $pos$ 是标量 $T_{past}$ (即当前是第几个 token)。

$$
Q_{rope} = \text{RoPE}(Q, pos)
$$

$$
K_{rope} = \text{RoPE}(K, pos)
$$

$$
V \text{ 不变}
$$

> **注意**: 此时 $K_{rope}$ 包含的是**当前步**计算出的 K。

#### 2.4. KV Cache 管理

这里是显存占用的核心来源，也是推理加速的关键。

- **场景 A: Prefill 阶段**

  - 历史 Cache 为空。
  - 直接将当前的 $K_{rope}, V$存入 Cache。
    $$
    K_{cache} = K_{rope}, \quad V_{cache} = V
    $$
  - **Cache 形状**: $(B, G, L, d_h)$
- **场景 B: Decode 阶段**

  - 历史 Cache 已有 $T_{past}$ 长度。
  - 将当前的 $K_{rope}$ (长度 1) 和 $V$ (长度 1) 拼接到 Cache 尾部。
    $$
    K_{cache} \leftarrow \text{Concat}([K_{old}, K_{rope}], \text{dim}=2)
    $$
    $$
    V_{cache} \leftarrow \text{Concat}([V_{old}, V], \text{dim}=2)
    $$
  - **Cache 形状**: $(B, G, T_{past}+1, d_h)$

> 总结: 进入注意力计算时，我们使用的是 完整的 Cache。
> 令 $T_{total} = T_{past} + L$。
>
> $
> K_{cache}, V_{cache} \in \mathbb{R}^{B \times G \times T_{total} \times d_h}
> $

#### 2.5. GQA Grouping & Broadcast

为了让 $H$ 个 Q 头能和 $G$ 个 KV 头计算，需要对 KV 进行广播。

$group\_size = H / G$。

1. Reshape Q:

  $$
  Q_{rope} \to (B, G, group\_size, L, d_h)
  $$

2. Reshape K, V:

  $$
  K_{cache} \to (B, G, 1, T_{total}, d_h)
  $$

  $$
  V_{cache} \to (B, G, 1, T_{total}, d_h)
  $$

#### 2.6. 注意力计算 (Scaled Dot-Product)

$$
\text{Scores} = \frac{Q_{rope} \cdot K_{cache}^T}{\sqrt{d_h}}
$$

- **形状分析**:
  $$
    Q: (B, G, group, L, d_h)
    $$
  $$
    K^T: (B, G, 1, d_h, T_{total})
    $$
  **Scores**(序列的相关性矩阵): $\in \mathbb{R}^{B \times G \times group \times L \times T_{total}}$

> **Mask 的区别**:
>
> - **Prefill**: 需要 **Causal Mask** (下三角矩阵)，防止 $t$位置看到 $t+1$。
> - **Decode**: 不需要 Mask，因为 $Q$ 只有 1 个 (当前)，它理应看到所有历史 $K$。

$$
\text{AttnWeights} = \text{Softmax}(\text{Scores} + \text{Mask})
$$

#### 2.7. 加权求和

$$
O_{heads} = \text{AttnWeights} \cdot V_{cache}
$$

- **计算**: $(B, G, group, L, T_{total}) \times (B, G, 1, T_{total}, d_h)$
- **结果**: $O_{heads} \in \mathbb{R}^{B \times G \times group \times L \times d_h}$
- $V$ 包含了这一层想要传递给下一层的特征信息

#### 2.8. 输出投影与残差

1. Flatten: 将 $G$ 和 $group$ 维度合并回 $H$。

$$
O_{heads} \to (B, L, H \cdot d_h) = (B, L, d)
$$

2. Output Linear:

$$
O = O_{heads} W_O
$$

3. Residual Add:

$$
X = X + O
$$
#### 总结
一致性的好处：
- 无缝残差： 每一层的输入输出维度由始至终保持不变，不需要 resize。
- 避免瓶颈： 保证输入的原始信息量足以支撑后续 32+ 层网络的深层处理。
- 工程友好： 在分布式训练（TP/PP）和推理优化（KV Cache）中，统一的维度让内存管理更容易。

4. 形状回归：$X \in \mathbb{R}^{B \times L \times d}$

---

### FFN / MLP Block (SwiGLU)

#### 3.1. Pre-Norm

$$
X_{norm} = \text{RMSNorm}(X)
$$

#### 3.2. SwiGLU 计算

LLaMA 使用三个矩阵：Gate ($W_1$), Up ($W_3$), Down ($W_2$)。

设中间层维度为 $d_{ff}$ (通常是 $4d$ 或 $\frac{8}{3}d$)。

$$
X_{gate} = \text{SiLU}(X_{norm} W_{gate}) \odot (X_{norm} W_{up})
$$

$$
X_{out} = X_{gate} W_{down}
$$

形状变化：

$$
(B, L, d) \xrightarrow{Up/Gate} (B, L, d_{ff}) \xrightarrow{Down} (B, L, d)
$$

#### 3.3. Residual Add

$$
X = X + X_{out}
$$

---

### 输出层 (Final Output)

经过 $N$ 层堆叠后，得到最终隐状态 $X_{final} \in \mathbb{R}^{B \times L \times d}$。

1. Final Norm:

$$
X_{final} = \text{RMSNorm}(X_{final})
$$

2. Logits 生成:

$$
\text{Logits} = X_{final} W_{head}
$$

其中 $W_{head} \in \mathbb{R}^{d \times V_{vocab}}$。

结果：$\text{Logits} \in \mathbb{R}^{B \times L \times V_{vocab}}$
- Logits 是模型最后一层输出的原始打分（Raw Scores），还没有被转换成概率。
---

### 采样与更新 (Prefill vs Decode)
在把 Logits 变成概率之前，我们通常会修改 Logits 的值。
- Temperature (温度) $T$：
  $$\text{New\_Logits} = \frac{\text{Logits}}{T}$$
  - $T < 1$ (如 0.1)： 差距拉大。大的更大，小的更小。模型变得保守、确定。比如 12.5 / 0.1 = 125。
  - $T > 1$ (如 1.5)： 差距缩小。分数高的和分数低的变接近了。模型变得发散、有创造力。比如 12.5 / 1.5 
- Penalty (惩罚)：如果设置了重复惩罚，模型会检查刚才生成的词，把它们对应的 Logits 强行减去一个值（比如减 2.0），让它们不容易再被选中。

然后使用 Softmax 将 Logits 变成 概率 (Probability)：
$$P_i = \frac{e^{logit_i}}{\sum_{j} e^{logit_j}}$$


有了概率 [0, 0.98, 0.019, 0, 0]，模型到底选哪个？
这取决于采样策略：
- Greedy Search (贪婪搜索):永远只选概率最大的那个。
  - 特点：最稳定，但容易车轱辘话，缺乏创造力。
- Random Sampling (随机采样):根据概率掷骰子。
  - 有 98% 的几率选 "blue"，但也有 1.9% 的几率选 "green"（比如特意想说“绿色的天空”）。
- Top-K / Top-P: 为了防止选到太离谱的词（比如 "apple"），我们会先截断，只在概率最高的几个词（K）或累积概率达到 P 的词里抽签。

这里以贪婪采样为例
#### **场景 A: Prefill 结束时**

我们通常只关心最后一个 token 的输出，因为我们要预测提示词后的第一个新词。

取 Logits[:, -1, :] 进行采样 (Argmax / Top-k / Top-p)。

得到 Next Token ID。

#### **场景 B: Decode 结束时**

输入本来就是 1 个 token，输出也是 1 个 token 的 Logits。

采样得到 Next Token ID。

循环:

将生成的 Next Token ID 作为下一轮 Decode 的输入 $I_{new}$，同时 $T_{past} \leftarrow T_{past} + 1$。

---

### 为什么不直接用输入 X 当作 V？

即为什么要有 $V = X W_V$ 这个投影？

- **输入 $X$ 是原始特征**：它可能包含了所有的信息（语法、语义、位置、情感）。
- **投影 $W_V$ 是特征提取器**：它允许当前这个注意力头（Head）**只提取它关心的信息**。

  - _Head 1_ 可能想关注语法关系，它的 $W_V$ 会提取词性特征。
  - _Head 2_ 可能想关注指代关系，它的 $W_V$ 会提取实体特征。

---

### 为什么 NLP 序列任务选 LayerNorm (LN) 而不是 BatchNorm (BN)？

1. BatchNorm 的本质假设被序列数据打破,对长短句子的泛化能力极弱
   BatchNorm 归一化的是：

   $$
   mu_{\text{batch}} = \frac{1}{B}\sum_{i=1}^B x_i,\quad \sigma_{\text{batch}}^2 = \frac{1}{B}\sum_{i=1}^B (x_i-\mu)^2
   $$

   它假设：

   - batch 内样本 **同分布**
   - 训练时用当前 Batch 的均值/方差，推理时用训练累积的全局移动平均值。
     但在序列模型中：
   - 每个时间步 t 的分布不同：
   - 不同长度、padding、mask
   - 自回归推理时 batch size = 1
     Batch 统计量完全不稳定，导致 **Train-Test Skew（训练-推理偏差）**。
     LN 直接拿当前输入的这个句子算均值方差，**随用随算**，不需要训练时的历史统计量，因此对长短句子的泛化能力极强
2. 条件概率链式结构被 BatchNorm 破坏
   语言模型建模的是：

   $$
   (x_1,\dots,x_T)=\prod_t p(x_t\mid x_{<t})
   $$

   但 BatchNorm 在第 t 步使用了：

   $$
   mu = \frac{1}{B}\sum_{b=1}^B h_t^{(b)}
   $$

   这意味着：

   - 当前样本的表示依赖于 batch 中**其他样本的未来 token 分布**
   - 引入了跨样本的信息泄漏
   - 如果在(B,T)维度进行归一化，还会违反因果性（causality）
     而 LayerNorm：

   $$
   mu_t = \frac{1}{d}\sum_{k=1}^d h_{t,k}
   $$

   只在特征维归一化：

   - 不依赖其他样本
   - 不依赖其他时间步
   - 保持条件独立结构
3. 小 Batch Size 问题

   - Transformer 模型极大，训练时显存吃紧，Batch Size 往往很小（甚至只有 1 或 2）。
   - BN 在小 Batch 下估计的均值/方差波动极大，导致训练不收敛。
   - LayerNorm 基于每一层自身的输入进行归一化，能够更好地控制每一层输入的范围和分布，避免梯度消失或梯度爆炸问题。

---

### Pre-Norm or Post-Norm ?

- Post-Norm（先加后归一）:

$$
x_{t+1} = \text{Norm}(x_t + F(x_t))
$$

- Pre-Norm（先归一后加）:

$$
x_{t+1} = x_t + F(\text{Norm}(x_t))
$$

#### Post-Norm (BERT/原始 Transformer)

- **优势**：

  - **理论上限略高**：如果能调教好，Post-Norm 在某些任务上的最终精度可能会比 Pre-Norm 稍微好一点点（因为**归一化后的输出对参数的 scale 更敏感**，保留了更多特征幅度信息）。
- **劣势**：

  - **训练极其困难（梯度消失/爆炸）**：这是最核心的问题。
  - 在深层网络中，梯度需要反向传播经过每一个 Norm 层。由于 Post-Norm 把 Norm 放在主干路上，层数越深，梯度被 Norm 操作反复缩放，导致靠近输入的层梯度极其不稳定。
  - **必须使用 Warm-up**：为了防止训练初期梯度爆炸，必须使用 Learning Rate Warm-up（热身），让学习率从 0 慢慢升上来，非常依赖调参技巧。

#### Pre-Norm (GPT/LLaMA)

- **优势**：

  - 训练稳定：$x_{final} = x_0 + F_1(\dots) + F_2(\dots) + \dots + F_N(\dots)$
  - **可以训得非常深**：Pre-Norm 是 100 层以上大模型（如 GPT-3）能够训练成功的基础。
  - **不需要 Warm-up (或依赖较小)**：训练初期就很稳定。
- **劣势（理论上的）**：

  - **表示坍塌风险**：有研究指出，在极深的网络中，Pre-Norm 结构会导致靠后的层 $F(x)$ 的贡献相对于主干 $x$ 越来越小，导致学不到太多新东西。
  - _注：虽然有这个理论劣势，但在几十亿、几千亿参数的 LLM 实践中，稳定性是第一位的，所以大家几乎全部倒向了 Pre-Norm。_

## KV Cache 存储哪些东西？

因果独立性：第 $t$ 步的注意力分布，完全独立于第 $t-1$ 步的注意力分布。

- 算第 6 步时，不需要知道第 5 步关注了谁。只要把 $K$ 和 $V$ 给我就行，我自己（$Q_6$）会去和它们重新计算关系。

**存 K**：因为 $q_t$ 需要和所有的历史 $k$ 做点积（计算我应该关注哪些 token）。

**存 V**：因为算出的权重需要作用在历史 $v$ 上（提取内容是什么）。

**不存 Q**：因为过去的 $Q$ 已经完成了生成过去 Token 的任务，对未来无影响。

**不存 Attention Matrix**：因为那是 $Q$ 和 $K$ 的临时交互产物，每次新的 $Q$ 都要重新交互。



### 为什么现代大模型（GPT/LLaMA）的 embedding dim 和 llm model dim 一致？

如果 $D_{emb} \neq D_{model}$，模型必须在第一层加一个线性投影层（Projection Layer），把维度从 $D_{emb}$ 强行映射到 $D_{model}$，之后所有的残差连接才能在 $D_{model}$ 维度上进行
> Add & Norm 操作要求输入输出维度一致，否则无法相加

#### 为什么不故意把 Embedding 做小一点？（为了省显存？）
ALBERT 的做法： 假设 $D_{model}=768$，但 $D_{emb}=128$。它认为词向量主要学的是“上下文无关”的浅层语义，不需要那么大维度。

结果： 虽然参数量巨幅下降（Embedding 层参数变少），但计算量并没有减少（模型内部还是要算 768 维），而且性能下降了。

现代大模型的共识： **Embedding 层不仅存储词义，还承载了模型的第一波“知识”**。将它压缩会造成信息瓶颈（Information Bottleneck），导致输入模型的信息“先天不足”。

#### 为什么不把 Embedding 做大一点？（为了更强表达？）
有些研究（如 Google 的 T5 或早期的 Transformer）尝试过让 $D_{emb} > D_{model}$ 或相反，但这就引入了额外的投影矩阵 $W_{proj}$。

在大模型时代（Scaling Law 时代），大家发现 **“架构越简单越好”**。增加投影层不仅增加了代码实现的复杂度，还增加了额外的计算开销（虽然不大）。

