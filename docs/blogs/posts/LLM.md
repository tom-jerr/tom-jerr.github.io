---
title: LLM 笔记大杂烩
tags: [attention, position-embedding, transformer]
math: "true"
modified: 星期日, 二月 2日 2025, 2:59:04 下午
---

> [!warning] 施工中……

## Attention

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250202124530166.png)

输入序列

$$
X = (x_{0},x_{1},\dots,x_{n-1})^{\top} \in \mathbb{R}^{n\times d_{1}}
$$

通过可训练的权重矩阵 $W^{Q}\in \mathbb{R}^{d_{1}\times d_{k}}$，得到 queries

$$
Q = XW_{Q} \in \mathbb{R}^{n\times d_{k}}
$$

然后对外来的（或者 $X$ 自己的子序列）

$$
Y = (y_{1},y_{2},\dots,y_{m})\in \mathbb{R}^{m\times d_{2}}
$$

通过 $W^{K}\in \mathbb{R}^{d_{2}\times d_{k}}, W^{V}\in \mathbb{R}^{d_{2}\times d_{v}}$，有 keys 和 values

$$
\begin{align}
K & = YW^{K}\in \mathbb{R}^{m\times d_{k}} \\
V & = YW^{V}\in \mathbb{R}^{m\times d_{v}}
\end{align}
$$

之后

$$
\text{Attention}(Q,K,V)=\text{softmax}\left( \frac{QK^{\top}}{\sqrt{ d_{k} }} \right)V \in \mathbb{R}^{n\times d_{v}}
$$

可以认为是将一个 $n\times d_{k}$ 的序列 $Q$ 转化为了新的序列 $K\in \mathbb{R}^{n\times d_{v}}$

逐向量版本

$$
\text{Attention}(q_t,K,V) = \sum_{s=1}^m \frac{1}{Z}\exp\left(\frac{\langle q_t, k_s\rangle}{\sqrt{d_k}}\right)v_s
$$

### Multi-head Attention

也就是将 $Q$、$K$、$V$ 按照第二个维度拆分后，分别进行计算，然后拼接起来

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250202124541437.png)

$$
\text{head}_{i} = \text{Attention}(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})
$$

这里 $W_{i}^{Q},W_{i}^{K}\in \mathbb{R}^{d_{k}\times \bar{d}_{k}}, W_{i}^{V}\in \mathbb{R}^{d_{v}\times \bar{d}_{v}}$

之后 

$$
\text{MultiHead}(Q,K,V) = \text{concat}(\text{head}_{1},\text{head}_{2},\dots,\text{head}_{h})
$$

### Self-Attention

只需要将 $Q$、$K$、$V$ 都使用一个序列 $X$ 得到即可

$$
\begin{align}
Q_{i} & = XW_{i}^{Q}\in \mathbb{R}^{n\times \bar{d}_{k}} \\
K_{i} & = XW_{i}^{K}\in \mathbb{R}^{n\times \bar{d}_{k}} \\
V_{i} & = XW_{i}^{V}\in \mathbb{R}^{n\times \bar{d}_{v}}
\end{align}
$$

之后得到多头注意力

$$
\text{head}_{i} = \text{Attention}(Q_{i},K_{i},V_{i}) \in \mathbb{R}^{n\times \bar{d}_{v}}
$$

拼接后得到

$$
\text{MutilHead}(Q,K,V) = \text{concat}(\text{head}_{1}, \text{head}_{2},\dots,\text{head}_{n}) \in \mathbb{R}^{n\times d_{v}}
$$

可以取 $d_{v}$ 和 $X$ 序列的每一个向量维度（即 $X$ 的第二维度）相同

### Linear Attention

在最原始的 attention 中，计算复杂度是 $O(n^{2})$，主要是由 softmax 引起的，更一般的，attention 公式可以写作

$$
\text{Attention}(Q,K,V) = \frac{\sum_{j=1}^{n}\text{sim}(q_{i},k_{j})v_{j}}{\sum_{j=1}^{n} \text{sim}(q_{i},k_{j})}
$$

其中相似度函数 $\text{sim}$ 需要满足非负性，也因此无法简单的去除原式中的 softmax 以加快矩阵运算 $(n\times d_{k})\times(d_{k}\times m)\times(m\times d_{v})$

#### 核函数形式

将 $q_{i},k_{j}$ 用核函数映射到非负区域上，从而保证非负，即

$$
\text{sim}(q_{i},k_{j}) = \phi(q_{i})^{\top}\psi(k_{j})
$$

#### 利用 Softmax 特殊性质

只要先对 $Q$ 的 $d$ 那一维和 $K$ 的 $n$ 那一维进行 softmax，那么得到的结果自动满足归一化条件：

$$
\text{LinearAttention} = \text{softmax}_{2}(Q)\cdot \text{softmax}_{1}(K)^{\top} V
$$

## Position Embedding

为了表征 token 的相对位置，可以引入位置编码，否则 transformer 将无法辨认词语间的先后关系。

### Sinusoidal

Google 的原始论文中给出的编码为

$$
\begin{cases}
\text{PE}_{2i}(p)  = \sin\left( \frac{p}{10000^{2i/d_{P}}} \right) \\
\text{PE}_{2i+1}(p)  = \cos \left( \frac{p}{10000^{2i/d_{P}}} \right) \\
\end{cases}
$$

其中 $P$ 为序列总长度。之所以使用这个表达式，是为了便于表征相对位置，因为有

$$
\begin{align}
\sin(\alpha+\beta) & =\sin\alpha \cos \beta + \sin \beta \cos \alpha  \\
\cos(\alpha+\beta) & =\cos \alpha \cos \beta-\sin \alpha \sin \beta 
\end{align}
$$

即第 $p+k$ 位的表征可以由第 $p$ 位的表征加上 $k$ 位的偏移得到

## Reference

1. [《Attention is All You Need》浅读（简介+代码） - 科学空间\|Scientific Spaces](https://kexue.fm/archives/4765)
2. [线性Attention的探索：Attention必须有个Softmax吗？ - 科学空间\|Scientific Spaces](https://kexue.fm/archives/7546)
