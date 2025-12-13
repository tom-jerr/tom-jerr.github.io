---
title: Group Query Attention
created: 2025-10-12
tags:
  - LLMInference
---

# GQA: Group Query Attention

## Group Query Attention

这是一种针对多头注意力机制的优化技术，可以降低与键 (K) 和值 (V) 投影相关的计算和内存成本。与多头注意力机制 (MHA) 中每个查询 (Q) 头都有自己的 K 头和 V 头不同，多个 Q 头共享相同的 K 头和 V 头。多查询注意力机制 (MQA) 是 GQA 的一个特例，其中所有 Q 头共享一个 K/V 头对。

实际上在应用时，我们还会再计算 `q @ k` 后进行 mask 叠加，这里一般有两种情况：

- 一种就是我们自己设计的 mask 形式
- 一种就是 `causal mask(因果掩码)`，用于防止注意力机制关注序列中未来 tokens 的技术

### PyTorch 示例

```python
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
	    """PyTorch: 如果 S > L(即在推理阶段KV cache 累积的 key 更多),
	    那么 query 只能看到前 L 个 key, 这是不正确的
	    [[1, 0, 0, 0, 0],
		 [1, 1, 0, 0, 0],
		 [1, 1, 1, 0, 0]]
	    """
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value
```

## Causal Masking

因果掩码是一种防止注意力机制关注序列中未来 tokens 的技术。当  `mask`  设置为  `causal`  时，我们将应用因果掩码。

因果掩码是一个形状为  `(L, S)`  的方阵，其中  `L`  是查询序列长度， `S`  是键/值序列长度。掩码是一个下三角矩阵，对角线上和对角线下方的元素为 0，对角线上的元素为 -inf。例如，如果  `L = 3`  且  `S = 5` ，则掩码将为：

```shell
0   0   0   -inf -inf
0   0   0   0    -inf
0   0   0   0    0
```

:warning: **我们的实现实际上与 PyTorch API 不同**

```python
def causal_mask(L: int, S: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """
    PyTorch: 如果 S > L (即在推理阶段KV cache 累积的 key 更多),
    那么 query 只能看到前 L 个 key, 这是不正确的
    [[1, 0, 0, 0, 0],
     [1, 1, 0, 0, 0],
     [1, 1, 1, 0, 0]]

    Ours:
    [[1, 1, 1, 0, 0],
     [1, 1, 1, 1, 0],
     [1, 1, 1, 1, 1]]
    """
    offset = S - L
    mask = torch.tril(torch.ones((L, S), device=device), diagonal=offset)
    mask = torch.where(mask.bool(), 0.0, -torch.inf).to(dtype=dtype, device=device)
    return mask
```

## Qwen2 Grouped Query Attention

### Overview

整个 LLM 流程是：

- 输入 x 先经过 Embedding，`shape=(B, L, E)`
- 然后 Embedding 后的输入再经过 RoPE，`shape=(B, L, E)`
- 接着计算 Q，K，V，`Q shape=(B, L, E) -> (B, L, H_q, D)`，`K&V shape=(B, L, E) -> (B, L, H, D)`
- 然后进行 Grouped Query Attention 操作，得到 query 对 key 的计算相关性分数并归一化，然后对 V 的进行加权求和，每个注意力头都在做这件事`shape=(B, L, H_q, D)`
- 最后对拼接后的多个注意力头的结果通过最后一个线性层进行变换，得到多头注意力层的最终输出`shape=(B, L, E)`

### GQA

示例伪代码如下：

```python
x: B, L, E
q = linear(x, wq, bq) -> B, L, H_q, D
k = linear(x, wk, bk) -> B, L, H, D
v = linear(x, wv, bv) -> B, L, H, D

q = rope(q, offset=slice(0, L))
k = rope(k, offset=slice(0, L))
(transpose as needed)
x = scaled_dot_product_attention_grouped(q, k, v, scale, mask) -> B, L, H_q, D ; Do this at float32 precision
(transpose as needed)
x = linear(x, wo) -> B, L, E
```

### RoPE

Qwen2 模型采用了一种非传统的 RoPE 形式。在这种形式下，头部嵌入维度被拆分成两半，并且这两半以不同的频率应用。假设  `x1 = x[.., :HALF_DIM]` ， `x2 = x[.., HALF_DIM:]` 。

```python
output[0] = x1[0] * cos_freqs[0] + x2[0] * -sin_freqs[0]
output[HALF_DIM] = x1[0] * sin_freqs[0] + x2[0] * cos_freqs[0]

output[1] = x1[1] * cos_freqs[1] + x2[1] * -sin_freqs[1]
output[HALF_DIM + 1] = x1[1] * sin_freqs[1] + x2[1] * cos_freqs[1]
```

### transformers 的实现

需要先设置 Qwen2 的 config，然后设置 attention 的实现方式，这里需要设置成 `sdpa(scale dot product attention)` 才能进行 GQA 的计算

```python
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
	Qwen2Attention,
	Qwen2RotaryEmbedding,
)

config = Qwen2Config(
	hidden_size=hidden_size,
	num_hidden_layers=2,
	intermediate_size=hidden_size * 4,
	num_attention_heads=num_heads,
	num_key_value_heads=num_kv_heads,
	rms_norm_eps=1e-6,
	vocab_size=1000,
	rope_theta=theta,
	max_position_embeddings=max_seq_len,
)

config._attn_implementation = "sdpa"  # 这里可以使用causal mask
rotary_emb = Qwen2RotaryEmbedding(config)
torch_attention = Qwen2Attention(config, layer_idx=0).to(device=dev, dtype=dtype)

torch.manual_seed(42)
x = torch.rand(batch_size, seq_len, hidden_size, dtype=dtype, device=dev) * 2 - 1
position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)
position_embeddings = rotary_emb(x, position_ids) # 输入先进行 RoPE，然后再进行 attention

torch_output = torch_attention(
	x,
	position_embeddings
	attention_mask=None,
	is_causal=True if mask == "causal" else False,
)[0].to(device=dev, dtype=dtype)
```
