---
title: Attention & Transformers
date: 2025/9/30
update:
comments: true
katex: true
tags:
  - LLMInference
---

# Chapter 1 Attention & Transformers

## Attention

### 传统的 self-attention

#### 传统的序列模型处理方式

在传统的序列处理模型（如 RNN、LSTM 和 GRU）中，模型是按顺序逐个处理序列中的元素（例如单词或字符），并且每个元素的处理依赖于前一个元素的隐藏状态。

```admonish warning
这种方法在处理长序列时会面临**梯度消失或梯度爆炸的问题**，导致模型难以捕捉长距离的依赖关系。
```

#### 自注意力机制核心思想

对于序列中的每个元素，**模型可以同时考虑序列中所有其他元素的信息**，从而动态地计算每个元素与其他元素之间的**相关性（即“注意力”）**，并根据这些相关性对序列中的信息进行加权求和。这样，模型能够更高效地捕捉序列内部的长距离依赖关系，而不需要像 RNN 那样逐个处理序列元素。

### Attention 变种

- MHA 和 MQA 都是 GQA 的特殊表达形式
  三者可以用同一套代码，只需要修改【GQA】代码里面的 nums_key_value_head 参数就可
  nums_key_value_head 设置等于 1 就是 MQA
  nums_key_value_head 设置等于 nums_head 就是 MHA

![](img/gqa.png)

#### MultiHeadAttention

每个 Query 头都有独立的 Key 和 Value。

```admonish info
优势：
允许不同的 Query 头关注不同的 Key-Value 信息，提高模型的表达能力。
更适合复杂任务，如长序列建模和复杂推理任务。
劣势：
推理速度慢，因为在每一步都要存储和读取 所有 Query 头的 Key 和 Value，导致 KV 缓存（KV Cache）非常大，占用大量显存和内存带宽。
```

#### Multi-Query Attention

所有 Query 头共享相同的 Key 和 Value。

```admonish info
优势：
推理速度快，因为只需要存储和读取一个 Key-Value 组，而不是多个。
显存占用低，适用于 大规模语言模型推理（如 ChatGPT）。
劣势：
不同 Query 头会关注相同的信息，导致模型表达能力下降，尤其是在长序列建模任务上（如机器翻译、摘要生成）。
可能导致训练不稳定，特别是长序列输入时，训练容易出现 Loss spikes（损失值剧烈波动）。
```

#### Group-Query Attention

GQA 将 Key 和 Value 按组分配，每个组共享 Key 和 Value，而 Query 仍然是独立的。

```admonish info
高效性：相比 MHA，GQA 减少了 Key 和 Value 的存储需求，推理速度更快。
高质量：相比 MQA，GQA 的 BLEU 得分接近 MHA，减少了信息冗余。
灵活性：通过调整组的数量（num_groups），可以在质量和速度之间进行权衡
```

#### pytorch 示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_groups, dropout=0.1):
        super(GroupedQueryAttention, self).__init__()
        assert num_heads % num_groups == 0, "num_heads 必须是 num_groups 的整数倍"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, (embed_dim // num_heads) * num_groups, bias=False)
        self.v_proj = nn.Linear(embed_dim, (embed_dim // num_heads) * num_groups, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape
        _, seq_len_kv, _ = key.shape

        Q = self.q_proj(query)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        K = self.k_proj(key)
        V = self.v_proj(value)
        K = K.view(batch_size, seq_len_kv, self.num_groups, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len_kv, self.num_groups, self.head_dim).permute(0, 2, 1, 3)

        group_size = self.num_heads // self.num_groups
        Q_grouped = Q.view(batch_size, self.num_groups, group_size, seq_len, self.head_dim)

        attn_logits = torch.matmul(Q_grouped, K.transpose(-2, -1))
        attn_logits /= self.head_dim ** 0.5

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        O = torch.matmul(attn_weights, V)
        O = O.permute(0, 3, 1, 2, 4).contiguous().view(batch_size, seq_len, self.embed_dim)

        Y = self.o_proj(O)
        return Yclass GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head, dropout=0.1):
        super().__init__()
        assert hidden_dim % nums_head == 0
        assert nums_head % nums_key_value_head == 0

        self.nums_head = nums_head
        self.head_dim = hidden_dim // nums_head
        self.hidden_dim = hidden_dim
        self.nums_key_value_head = nums_key_value_head

        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.o_proj = nn.Linear(nums_head * self.head_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, X, attention_mask=None):
        batch_size, seq_len, _ = X.size()
        query = self.q_proj(X)
        key = self.k_proj(X)
        value = self.v_proj(X)

        # 分头
        query = query.view(batch_size, seq_len, self.nums_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.nums_key_value_head, self.head_dim).transpose(1, 2)

        # kv broadcast
        key = key.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        value = value.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        # 计算注意力权重
        attention_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 添加mask
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask == 0, float("-1e20"))

        # 归一化
        attention_weights = torch.softmax(attention_weights, dim=-1)
        # drpout
        attention_weights = self.attn_dropout(attention_weights)

        # 加权求和
        out = torch.matmul(attention_weights, value)

        # shape: (batch_size, num_head, seq_len, head_dim) -> (batch_size, seq_len, num_head, head_dim)
        # 合并头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 线性变换
        out = self.o_proj(out)
        return out
```

## Transformers 结构

### 输入 X 与 attention_mask 的 shape

- 输入 X 一般形状为`[batch_size, seq_len, embedding_dim]`
- attention_mask 是经过 tokenizer 后返回的 mask 结果，表示哪些样本需要忽略形状一般是`[batch_size, num_heads, seq_len]`

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
