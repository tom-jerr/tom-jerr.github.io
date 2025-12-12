---

title: Attention & Transformers
created: 2025-09-30
updated: 2025-12-12
tags:

- LLMInference

---

# Attention & Transformers

## Attention

### 传统的 self-attention

#### 传统的序列模型处理方式

在传统的序列处理模型（如 RNN、LSTM 和 GRU）中，模型是按顺序逐个处理序列中的元素（例如单词或字符），并且每个元素的处理依赖于前一个元素的隐藏状态。

> [!WARNNING]
> 这种方法在处理长序列时会面临**梯度消失或梯度爆炸的问题**，导致模型难以捕捉长距离的依赖关系。


#### 自注意力机制核心思想

对于序列中的每个元素，**模型可以同时考虑序列中所有其他元素的信息**，从而动态地计算每个元素与其他元素之间的**相关性（即“注意力”）**，并根据这些相关性对序列中的信息进行加权求和。这样，模型能够更高效地捕捉序列内部的长距离依赖关系，而不需要像 RNN 那样逐个处理序列元素。

### Attention 变种

- MHA 和 MQA 都是 GQA 的特殊表达形式
  三者可以用同一套代码，只需要修改 GQA 代码里面的 nums_key_value_head 参数就可
  nums_key_value_head 设置等于 1 就是 MQA
  nums_key_value_head 设置等于 nums_head 就是 MHA

![](img/gqa.png)

#### Multi-Head Attention

每个 Query 头都有独立的 Key 和 Value。实际上也额外增加了并行能力，后续的 Tensor Parallelism 主要基于这个特点进行通信量的减少

**现在 block-diffusion llm 因为无法在 block 内使用 KV Cache，GQA 失去了它的优势，所以常用 MHA**

> ![NOTE]
> 优势：
> 允许不同的 Query 头关注不同的 Key-Value 信息，提高模型的表达能力。
> 更适合复杂任务，如长序列建模和复杂推理任务。
>劣势：
>KV 头很多，导致 KV 缓存（KV Cache）非常大，占用大量显存和内存带宽。


#### Multi-Query Attention

所有 Query 头共享相同的 Key 和 Value。精度损失比较大，基本现在主流使用 GQA

> ![NOTE]
>优势：
>推理速度快，因为只需要存储和读取一个 Key-Value 组，而不是多个。
>显存占用低，适用于 大规模语言模型推理（如 ChatGPT）。
>劣势：
>不同 Query 头会关注相同的信息，导致模型表达能力下降，尤其是在长序列建模任务上（如机器翻译、摘要生成）。
>可能导致训练不稳定，特别是长序列输入时，训练容易出现 Loss spikes（损失值剧烈波动）。

#### Group-Query Attention

GQA 将 Key 和 Value 按组分配，每个组共享 Key 和 Value，而 Query 仍然是独立的。

> ![NOTE]
>高效性：相比 MHA，GQA 减少了 Key 和 Value 的存储需求，推理速度更快。
>高质量：相比 MQA，GQA 的 BLEU 得分接近 MHA，减少了信息冗余。
>灵活性：通过调整组的数量（num_groups），可以在质量和速度之间进行权衡


## Transformers 结构(decode only)

### 流程

#### 输入阶段 (Input Processing)
Tokenization (分词)将自然语言文本转换为整数索引（Token IDs）。
- Input: "Hello world" (String)
- Output: [15496, 995] (List of Ints)
- Tensor Shape: $(B, T)$ 这里存储的是整数索引 (Index)。

Embedding (嵌入层)将整数索引转换为稠密向量。
- 操作: 查表 (Lookup Table)。模型持有一个 $(V, D)$ 的大矩阵。
- 过程: 根据 input 的 ID，从大矩阵中把对应的行取出来。
- Tensor Shape:$$(B, T) \xrightarrow{\text{Lookup}} (B, T, D)$$
- 注意: 此时通常会加上 Positional Encodings (位置编码)。
  - 如果是绝对位置编码（如 GPT-2），直接加：$X = X + P$。
  - 如果是 RoPE (旋转位置编码，LLaMA)，这一步不加，而是**推迟到 Attention 层的 Q, K 计算时再做**。
#### Transformer Block (核心循环 N 层)

输入进入 N 个堆叠的 Block，每个 Block 输入和输出维度保持不变。
输入 $X$ 维度：$(B, T, D)$

LayerNorm / RMSNorm (归一化)
- 操作: 对最后一个维度 $D$ 进行归一化。
- Shape: $(B, T, D) \rightarrow (B, T, D)$ (形状不变)

Masked Self-Attention (注意力机制)这是维度变化最复杂的部分。

1. Q, K, V 投影 (Linear Projection):
   - 输入 $X$ 乘以三个矩阵 $W_Q, W_K, W_V$ (形状均为 $D \times D$)。
   - Shape: $(B, T, D)$
   - 注：如果是 LLaMA，在这里对 Q 和 K 应用 RoPE 旋转。
2. 分头 (Split Heads):
    - 将 $D$ 拆分为 $H \times D_h$ ($D_h = D/H$)。
    - Reshape: $(B, T, H, D_h)$
    - Transpose (转置): 为了让 $T$ 维度参与矩阵乘法，交换维度。
    - Shape: $(B, H, T, D_h)$
3. Attention Score (QKᵀ):
   - $Q$ 乘以 $K$ 的转置。
   - 计算：$(B, H, T, D_h) \times (B, H, D_h, T)$
   - Shape: $(B, H, T, T)$
   - 得到一个 $T \times T$ 的方阵，表示 token 之间的相关性。
4. Causal Masking (因果遮蔽):
    - 将 $T \times T$ 矩阵的上三角部分设为 $-\infty$。
    - Shape: $(B, H, T, T)$ (形状不变)
5. Softmax:归一化得到概率。
   - Shape: $(B, H, T, T)$
6. Weighted Sum (Score @ V):
   - Attention 矩阵乘以 $V$。
   - 计算：$(B, H, T, T) \times (B, H, T, D_h)$
   - Shape: $(B, H, T, D_h)$
7. Output Projection (合并头):
   - Transpose 回去：$(B, T, H, D_h)$
   - Reshape 合并：$(B, T, D)$
   - 最后经过一个线性层 $W_O$。
   - Output Shape: $(B, T, D)$
8. Residual Connection (残差连接):
   - $X_{new} = X_{input} + Attention(Norm(X_{input}))$
   - Shape: $(B, T, D)$

#### Feed Forward Network (MLP / FFN)
Attention 负责提取“相关性”，FFN 负责“知识处理”。
- Norm: $(B, T, D)$
- Up Projection (升维):
  - $X \times W_{up}$。维度通常放大 4 倍（或者 LLaMA 的 SwiGLU 结构）。
  - Shape: $(B, T, D) \rightarrow (B, T, D_{ff})$
- Activation: (ReLU / GELU / SiLU)，形状不变。
- Down Projection (降维):
  - $X \times W_{down}$。
  - Shape: $(B, T, D_{ff}) \rightarrow (B, T, D)$
- Residual Connection:
  - $X_{out} = X_{new} + FFN(Norm(X_{new}))$
  - Shape: $(B, T, D)$

#### 输出阶段 (Output Head)
经过 $N$ 层 Block 后，数据来到了最后一层。

- Final Norm
  - 最后再做一次 RMSNorm/LayerNorm。
  - Shape: $(B, T, D)$
- LM Head (Linear Un-embedding)这是把隐藏向量映射回“词表概率”的关键一步。
  - 操作: 乘以矩阵 $(D, V)$。**这个矩阵有时与 Input Embedding 矩阵共享参数（Weight Tying）**。
  - 计算: $(B, T, D) \times (D, V)$
  - Output (Logits): $(B, T, V)$

### Logits 的使用
Logits 是模型最后一层输出的原始打分（Raw Scores），还没有被转换成概率。

#### 前向传播 (Forward Pass)
输入经过 Embedding 和 N 层 Transformer Block 后，得到最后一个 Token 的隐藏状态向量（假设维度 $D=4096$）。
然后，经过输出层 (LM Head)，这是一个线性变换（矩阵乘法）：
$$\text{Hidden\_State} (1, 4096) \times W_{head} (4096, 5) \rightarrow \text{Logits} (1, 5)$$

Logits 是  一堆没有任何范围限制的浮点数。

#### Logits 的处理 (Post-processing)这
在把 Logits 变成概率之前，我们通常会修改 Logits 的值。
- Temperature (温度) $T$：
  $$\text{New\_Logits} = \frac{\text{Logits}}{T}$$
  - $T < 1$ (如 0.1)： 差距拉大。大的更大，小的更小。模型变得保守、确定。比如 12.5 / 0.1 = 125。
  - $T > 1$ (如 1.5)： 差距缩小。分数高的和分数低的变接近了。模型变得发散、有创造力。比如 12.5 / 1.5 
- Penalty (惩罚)：如果你设置了“重复惩罚”，模型会检查刚才生成的词，把它们对应的 Logits 强行减去一个值（比如减 2.0），让它们不容易再被选中。


#### Softmax (归一化)
我们需要把 Logits 变成 概率 (Probability)。使用 Softmax 
$$P_i = \frac{e^{logit_i}}{\sum_{j} e^{logit_j}}$$

#### 采样 (Sampling)
有了概率 [0, 0.98, 0.019, 0, 0]，模型到底选哪个？
这取决于采样策略：
- Greedy Search (贪婪搜索):永远只选概率最大的那个。
  - 特点：最稳定，但容易车轱辘话，缺乏创造力。
- Random Sampling (随机采样):根据概率掷骰子。
  - 有 98% 的几率选 "blue"，但也有 1.9% 的几率选 "green"（比如特意想说“绿色的天空”）。
  - Top-K / Top-P: 为了防止选到太离谱的词（比如 "apple"），我们会先截断，只在概率最高的几个词（K）或累积概率达到 P 的词里抽签。

### 为什么用 Transformer 中用 layer norm

| 特性             | Batch Norm         | Layer Norm             | RMSNorm                                            |
| ---------------- | ------------------ | ---------------------- | -------------------------------------------------- |
| 标准化维度       | 小批量内各特征维度 | 每个样本的所有特征维度 | 每个样本的特征维度的均方根                         |
| 计算开销         | 中等               | 较大                   | 较小                                               |
| 对小批量大小依赖 | 依赖               | 不依赖                 | 不依赖                                             |
| 应用场景         | CNN、MLP           | RNN、Transformer       | 各类神经网络，尤其在计算效率和稳定性要求高的任务中 |
| 正则化效果       | 有一定正则化效果   | 无显著正则化效果       | 无显著正则化效果                                   |

1. 列长度的灵活性：
   Transformer 处理的是序列数据，序列长度可能因输入样本而异。LayerNorm 对每个样本自身的一层神经元的输入进行归一化，与其他样本的序列长度无关，**能够很好地处理不同长度的输入序列**。而 batch norm 对长度大小不同的 NLP 任务计算的超参数泛化能力差。
2. 并行计算的适应性：
   Transformer 的多头注意力机制高度并行化，LayerNorm 只需要**对单个样本的一层进行计算，不需要等待其他样本的信息，因此更适合并行计算环境**。
3. 模型的稳定性：
   LayerNorm 基于每一层自身的输入进行归一化，能够更好地控制每一层输入的范围和分布，避免梯度消失或梯度爆炸问题。

### post-norm 与 pre-norm

- 原始的 transformer 中使用的是 post-norm，而 llm 中大多使用 pre-norm

| norm 位置 | 优点                                                                                                                                                                                                                                                     | 缺点                                                                                                                                                                                     |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| pre-norm  | **训练稳定**：在残差连接之前进行归一化，可以有效缓解梯度消失或爆炸的问题，使深层网络的训练更加稳定。 **收敛速度快**：梯度能够更直接地传递到前面的层，从而加快模型的整体收敛速度。**减少调参工作**：不需要像 Post-Norm 那样依赖复杂的学习率预热等优化技巧 | 潜在的表示塌陷问题：靠近输出位置的层可能会变得非常相似，从而对模型的贡献变小，限制了模型的上限。可能削弱层的贡献：由于先进行了归一化，可能会减弱每一层的实际贡献，导致模型的有效深度变浅 |
| post-norm | **保留输入特征**：更接近原始输入的特征，有助于信息的传递。 **潜在性能优势**：虽然训练不稳定，但有研究暗示其在效果上可能有更大的潜力                                                                                                                      | 训练不稳定：在深层模型中，梯度容易爆炸或消失，对学习率和权重初始化非常敏感，收敛困难。依赖优化技巧：需要使用学习率预热等复杂的优化方法来稳定训练                                         |
  

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
#### 总结
一致性的好处：
- 无缝残差： 每一层的输入输出维度由始至终保持不变，不需要 resize。
- 避免瓶颈： 保证输入的原始信息量足以支撑后续 32+ 层网络的深层处理。
- 工程友好： 在分布式训练（TP/PP）和推理优化（KV Cache）中，统一的维度让内存管理更容易。

