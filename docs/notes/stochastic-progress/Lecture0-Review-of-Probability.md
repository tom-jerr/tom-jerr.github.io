---
title: Review of Probability
date created: 星期日, 九月 15日 2024, 1:54:47 下午
date modified: 星期三, 九月 25日 2024, 2:54:39 下午
tags: [stochastic-progress]
math: true
modified: 星期五, 二月 28日 2025, 4:52:02 下午
---

## The Probability Space

> [!note] $\sigma$-Algebra
> 对于集合 $\Omega$, 以及 $\mathcal{F} \in 2^{\Omega}$，如果满足：
> - $\emptyset, \Omega \in \mathcal{F}$
> - $\forall A \in \mathcal{F}, A^c \in \mathcal{F}$
> - 对任意可数（无限）集合序列 $A_1, \ldots, A_n, \ldots \in \mathcal{F}$，有 $\cup_i A_i\in \mathcal{F}$
> 
> 那么称 $\mathcal{F}$ 是一个 $\sigma$-algebra

> [!note] Probability Space
> 对于元组 $(\Omega, \mathcal{F}, \mathrm{P})$，如果满足：
> - The universe $\Omega$ is the outcomes
> - The set $\mathcal{F}$ is a $\sigma$-algebra, which represents all possible “events”.
> - The probability $\mathrm{P}: \mathcal{F}\rightarrow [0,1]$ assigns a real to each event and must satisfy
>   - $\mathrm{P}(\emptyset) = 0, \mathrm{P}(\Omega)=1$
>   - $\mathrm{P}(A^c) = 1 - \mathrm{P}(A)$ for every $A \in \mathcal{F}$
>   - For any **finite or countable** sequence of **disjoint** sets $A_1,A_2,\dots\in\mathcal{F}$, $\mathrm{P}(\cup_{i\ge 1}A_i) = \Sigma_{i=1}^\infty\mathrm{P}(A_i)$
> 
> 那么称该元组为一个 probability space。

- $\sigma$ 代数对求交是封闭的，对求并则不一定

## Random Variable

- 随机变量是一个**可测函数**（measurable function）$\mathbf{X}: \Omega \rightarrow \mathbb{R}$。这里可测函数满足，对任意 $A\in \mathcal{B}(\mathbb{R})$，有 $\mathbf{X}^{-1}(A) \in \mathcal{F}$（这里 $\mathcal{B}(\mathbb{R})$ 指 $\mathbb{R}$ 上的所有 Borel set[^1] 的集合）
- 对于任意一个 Borel set $A \subseteq \mathbb{R}$，定义 $X$ 在 $A$ 中取值的概率为 $\mathbf{Pr}[X \in A]:= \mathrm{P}(X^{-1}(A))$

> [!note] Law of $X$
> 任意一个从 $\Omega$ 映射到 $\mathbb{R}$ 上的随机变量 $\mathrm{X}$, 都可以 induce 出一个在 $(\mathbb{R}, \mathcal{B}(\mathbb{R}), \mu_x)$ 上的一个概率测度（样本点为 $\mathbb{R}$，事件空间为 $\mathcal{B}(\mathbb{R})$，概率为 $\mu_x$），其中 $\mu_x(A)$ 定义为 $\mathbf{Pr}(\mathrm{X} \in A)$

注意，在这里 $\mathbf{Pr}$ 和 $\mathrm{P}$ 是两个不同的函数。记随机变量 $\mathrm{X}$，则有具有如下关系：

$$
\mathbf{Pr}[\mathrm{X} \in A] = \mathrm{P}(\mathrm{X}^{-1}(A)) \\
\mathrm{X}^{-1}(A) = \left\{ \omega \in \Omega \mid \mathrm{X}(\omega) \in A \right\}
$$

特别的，当 $A$ 为单元素集 $\{a\}$ 时，有

$$
\mathbf{Pr}[\mathrm{X} = a] = \mathrm{P}(\mathrm{X}^{-1}(a)) \\
\mathrm{X}^{-1}(a) = \left\{ \omega \in \Omega \mid \mathrm{X}(\omega) = a \right\}
$$

[^1]: Borel 集：定义 $\mathcal{G}$ 为 $\mathbb{R}$ 上所有开区间 $(a,b)$ 的集合，并定义 $\mathcal{B} = \sigma(\mathcal{G})$，称为 $\mathbb{R}$ 上的 Borel 集。这里 $\sigma(\cdot)$ 表示包括 $\cdot$ 的最小 $\sigma$ 代数。
