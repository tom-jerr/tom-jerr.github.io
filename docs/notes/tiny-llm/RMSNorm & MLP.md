---
title: RMSNorm & MLP
created: 2025-10-11
tags:
  - LLMInference
---

# RMSNorm & MLP

## RMSNorm

RMSNorm[^rmsnorm] 的定义：

$$
y = \frac{x}{\sqrt{mean(x^2) + \epsilon}} \cdot weight
$$

- `x`  是输入张量。
- `weight`  是一个可学习的缩放参数。
- `epsilon(eps)` 是为了数值稳定性而添加的一个小常数（例如，1e-5 或 1e-6）。
- $mean(x^2)$ 是平方和然后除以元素的数量
- LayerNorm 成功的关键在于其 **“重新缩放” (re-scaling)** 的不变性，而 **“重新中心化” (re-centering，即减去均值)** 这一步可能不是必需的

### 归一化方法

| **方法**                 | **归一化维度**                                    | **是否依赖批次** | **主要应用领域**           |
| ------------------------ | ------------------------------------------------- | ---------------- | -------------------------- |
| **BatchNorm**            | 跨批次 (`N`)，在每个特征 (`D`) 上独立             | **是**           | 计算机视觉 (CNN)           |
| **LayerNorm   层范数**   | 跨特征 (`D`)，在每个样本 (`N, S`) 上独立          | **否**           | 自然语言处理 (Transformer) |
| **GroupNorm   群体规范** | 跨特征分组，在每个样本上独立                      | **否**           | 计算机视觉 (小批次场景)    |
| **RMSNorm   均方根范数** | 跨特征 (`D`)，在每个样本上独立 (简化版 LayerNorm) | **否**           | **大语言模型 (LLM)**       |

### 为什么 LLM 使用 RMSNorm

- **更少的参数量**：LLM 参数量极大，RMSNorm 去掉一个可训练参数对 FLOPs 减少效果非常明显。可以实现更快的训练速度和更低的推理延迟。
- **性能相当**：这种简化并没有带来明显的性能损失。在 Transformer 这种结构中，去掉均值中心化这一步，模型的性能依然非常稳定和强大。

RMSNorm 提供了一个 **性价比极高** 的方案，用最小的改动换来了显著的效率提升，同时又没有牺牲模型性能，因此迅速成为像 Llama、Qwen 等主流大模型的标配。

## MLP

### FFN v1

在原始 Transformer（Vaswani et al., 2017[^transformer]）中，前馈层（FFN）通常是：
$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x)$$
后来很多模型（如 BERT）改成了 `GELU` 激活，但整体结构仍然是“线性 → 非线性 → 线性”，没有门控机制。

:warning: **存在问题**

> Transformer 的 FFN 层只使用一个单一的非线性激活（如 ReLU/GELU），\
> 限制了模型的表达能力和梯度流动。

### FFN v2

可以借鉴卷积网络中 **门控机制（Gated Linear Unit, GLU）** 的思想， 让前馈层的输出**由两条路径共同决定：一条主路径 + 一条门控路径**。

$$
GLU(X)=(xW_a + b_a)\odot \sigma(xW_b + b_b)
$$

$W_a$ 是主通道，$W_b$ 是门控通道，门控路径控制主路径中哪些特征可以“通过”

### FFN v3

Shazeer [^swiglu]系统地测试了 **各种 GLU 变体** 替换 Transformer 的 FFN 子层中的激活函数。

把传统 FFN：$W2\cdot ReLU(W1x)$

替换成：$W_3 \cdot \big( f(W_1 x) \odot g(W_2 x) \big)$

其中 f 和 g 是不同的激活函数组合（sigmoid, ReLU, GELU, Swish...）。

| 名称       | 公式                | 激活函数     | 特点               |
| ---------- | ------------------- | ------------ | ------------------ |
| **GLU**    | $a \odot \sigma(b)$ | sigmoid      | 原始版本           |
| **ReGLU**  | $a \odot ReLU(b)$   | ReLU         | 稀疏激活，计算简单 |
| **GeGLU**  | $a \odot GELU(b)$   | GELU         | 平滑梯度           |
| **SwiGLU** | $a \odot Swish(b)$  | Swish (SiLU) | 最平滑、效果最好   |

**用门控激活替换传统 FFN 激活能显著提升 Transformer 性能。**

#### 损失函数

- GELU: $\text{GELU}(x) = x \cdot \Phi(x)$，**是一个概率门控，对输入乘以一个介于 0 ～ 1 的概率（由高斯分布决定）**
  其中，$\Phi(x)$ 是标准正态分布的累积分布函数 (CDF)：
  $$
  \Phi(x) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
  $$
  等价的近似形式（便于计算）为：
  $$
  \text{GELU}(x) \approx 0.5x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
  $$
- SiLU: $SiLU(x)=x \cdot \sigma(x)$，**平滑可导，无零梯度死区**；
  - 对小负值仍然有部分响应
  - 对大正值趋近于线性；对大负值趋近于 0。

| 特性           | GELU                 | SiLU (Swish)                    |
| -------------- | -------------------- | ------------------------------- |
| 定义           | $x \cdot \Phi(x)$    | $x \cdot \sigma(x)$             |
| 来源           | BERT, GPT 等         | EfficientNet, LLaMA, Qwen2      |
| 平滑性         | 高                   | 高                              |
| 是否对称       | 否                   | 否                              |
| 是否有门控形式 | 否                   | 可直接门控（GLU 变体中常用）    |
| 计算复杂度     | 稍高（含 tanh/erf）  | 略低                            |
| 效果           | Transformer 经典选择 | 现代 LLM（如 Qwen2、LLaMA）偏爱 |

### Qwen2 MLP

Qwen2 的 MLP 中的所有线性投影通常都是无 bias 实现的。

- A gate linear projection ($W_{gate}$​).
- An up linear projection ($W_{up}$​).
- A SiLU activation function applied to the output of $W_{gate}$
- An element-wise multiplication of the **SiLU-activated $W_{gate}$ output and the$W_{up}$ output. This forms the "gated" part**.
- A final down linear projection ($W_{down}$​).
  $$
  MLP(x) = (SiLu(W_{gate}(x))\odot W_{up}(x))W_{down}
  $$

```python
N.. is zero or more dimensions for batches
E is hidden_size (embedding dimension of the model)
I is intermediate_size (dimension of the hidden layer in MLP)
L is the sequence length

input: N.. x L x E
w_gate: I x E
w_up: I x E
w_down: E x I
output: N.. x L x E
```

## Reference

[^transformer]: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
[^swiglu]: [GLU Variants Improve Transformer](https://arxiv.org/pdf/2002.05202v1)
[^rmsnorm]: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
