---
title: Concentration Inequalities
tags: [stochastic-progress]
math: true
---

## Basic Concentration Inequalities

### Markov Inequality

> [!note] Theorem
> For any *non-negative* random variable $X$ and $a > 0$,
>
> $$
> \mathbf{Pr}[X \ge a] \le \frac{\mathbf{E}[X]}{a}
> $$

- 直白一点：如果一个非负随机变量的均值非常小，则该随机变量取较大值的概率也非常小。
- Markov 不等式只是用了 $X$ 的一阶矩，所以放缩精度不大；而使用了二阶矩的 Chebyshev 不等式就要精确一些

### Chebyshev’s Inequality

> [!note] Theorem
> For any random variable with bounded $\mathbf{E}[x]$ and  $a \ge 0$, it holds that:
>
> $$
> \mathbf{Pr}[\left|X - \mathbf{E}[X]\right| \ge a] \le \frac{\mathbf{Var}[X]}{a^2}
> $$

- 直白一点：如果一个随机变量的方差非常小，那么该随机变量取远离均值的值的概率也非常小
- 不要求随机变量非负，只要求均值有界
- 实质上是这样的：$\mathbf{Pr}[\left|\mathrm{X} - \mathbf{E}[\mathrm{X}]\right| \ge a] \Rightarrow \mathbf{Pr}[f(\mathrm{X} - \mathbf{E}[\mathrm{X}]) \ge f(a)]$，再使用 Markov Inequality，就得到一个上界 $\frac{\mathbf{E}[f(\mathrm{X} - \mathbf{E}[\mathrm{X}])]}{f(a)}$。这里 $f$ 是一个非减函数
- 尽管比 Markov 更精确，但是仍然不是非常精确；可以使用更高阶的矩来实现更高的精度；不过用矩母函数 $\mathbf{E}[e^{\alpha \mathrm{X}}]$ 就行了，这也是下面的 bound 的证明方法。

## Chernoff Bound

> [!note] Theorem (Chernoff Bound).
> Let $X_1, X_2, \ldots, X_n$ be independent random variables such that $X_i \sim \operatorname{Ber}(p_i)$ for each $i = 1, 2, \ldots, n$. Let $X = \sum_{i=1}^{n}X_i$ and denote $\mu := \mathbf{E}[X] = \sum_{i=1}^{n}p_i$, we have
>
> $$
> \Pr[X \geq (1 + \delta)\mu] \leq \left(\frac{e^{\delta}}{(1 + \delta)^{1 + \delta}}\right)^\mu
> $$
>
> If $0 < \delta < 1$, then we have
>
> $$
> \Pr[X \le (1 - \delta)\mu] \leq \left(\frac{e^{-\delta}}{(1 - \delta)^{1 - \delta}}\right)^\mu
> $$

- 注意 $X_i$ 必须满足是服从 Bernoulli 分布的

> [!note] **Proof**
> 由 Markov Inequality 有
>
> $$
> \mathbf{Pr}[X \ge (1 + \delta)\mu] = \mathbf{Pr}[e^{\alpha X} \ge e^{\alpha (1 + \delta)\mu}]
> \le \frac{\mathbf{E}[e^{\alpha X}]}{e^{\alpha (1 + \delta)\mu}}
> $$
>
> 而
>
> $$
> \begin{align*}
> \mathbf{E}[e^{\alpha X}] &= \mathbf{E}[e^{\alpha\sum_{i=1}^nX_i}] \\
> &= \prod_{i=1}^n\mathbf{E}[e^{\alpha X_i}] \\
> &= \prod_{i=1}^n\left(p_ie^{\alpha} + (1- p_i)\right) \\
> &\le \prod_{i=1}^ne^{p_i(e^\alpha - 1)} \\
> &= e^{\mu(e^\alpha - 1)}
> \end{align*}
> $$
>
> 代入后求导，易证。

> [!note] Corollary
> For any $0 < \delta < 1$, 
>
> $$
> \begin{align*}
> \Pr[X \geq (1 + \delta)\mu] &\leq \exp{\left(-\frac{\delta^2}{3}\mu\right)}\\
> \Pr[X \leq (1 - \delta)\mu] &\leq \exp{\left(-\frac{\delta^2}{2}\mu\right)}
> \end{align*}
> $$
>
> （求导易证）

## Hoeffding’s Inequality

> [!note] Theorem (Hoeffding’s inequality).
> Let $X_1, X_2, \ldots, X_n$ be independent random variables where each $X_i \in [a_i, b_i]$, for certain $a_i \le b_i$ with probability $1$. Assume $\mathbf{E}[X_i]=p_i$ for every $1 \le i \le n$. Let $X = \sum_{i=1}^n X_i$ and $\mu := \mathbf{E}[X] = \sum_{i=1}^{n}p_i$, then
>
> $$
> \Pr [| X - \mu | \geq t] \leq 2 \exp\left(-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)
> $$
>
> for all $t \ge 0$

> [!note] Lemma (Hoeffding’s lemma).
> Let $X$ be a random variable with $\mathbf{E}[X] = 0$ and $X \in [a, b]$. Then
>
> $$
> \mathbf{E}[e^{\alpha X}] \leq \exp\left(\frac{\alpha^2 (b-a)^2}{8}\right) \text{ for all } \alpha \in \mathbb{R}.
> $$

- 证明：使用和 Chernoff Bound 中一样的证明方法，另外还需要凸函数的技巧（即用线性取代非线性）

> [!info]- Hint
> ![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250228094641192.png)

## Multi-Armed Bandit

问题描述：对于 $k$ 臂老虎机，每抽一臂对应一个奖励，并且得到奖励的概率为 $p_i$，$i=1,2,\ldots,k$。现在不知道每个 $p_i$ 究竟是多少，给出一个策略使得 regret $R(T) = p^*T - \sum_{t=1}^Tp_{A_t}$ 最小。这里 $T$ 是抽的总次数，$A_t$ 是第 $t$ 轮抽的老虎臂编号，$p^*$ 是最好的那个老虎臂对应的概率。

为了简便，只考虑 $k=2$ 的情形。可以考虑 *Explore Then Commit* 算法：先每个老虎臂都分别抽 $L$ 次，之后根据抽出的结果来推测哪个老虎臂对应的概率更大，之后的轮数中就只抽那一个老虎臂。

不妨假设两臂对应的概率相差 $\epsilon$，则前 $2L$ 轮中，regret 为 $\epsilon L$；在后 $T-2L$ 轮中，如果推测正确（记概率为 $p_{correct}$），则 regret 为 $0$，否则为 $\epsilon (T - 2L)$，那么总的 regret 为

$$
R(T) = \epsilon L + p_{wrong} \epsilon (T- 2L)
$$

只需计算 $p_{wrong}$。

不妨假设 $p_1 \gt p_2$，记在第一臂上第 $i$ 抽获奖的随机变量为 $X_i$, 在第二臂上则为为 $Y_i$，并设 $Z_i = X_i - Y_i \in [-1,1]$，$Z = \sum_{t=1}^L Z_t$，则 $\mathbf{E}[Z_i] = p_1 - p_2 =\epsilon \ge 0，\mathbf{E}[Z] = L\epsilon$

$$
\begin{align*}
p_{wrong} &= \mathrm{P}(Z \le 0)\\
&= \mathrm{P}(Z \le \mathbf{E}[Z] - L\epsilon) \\
&= \mathrm{P}(Z - \mathbf{E}[Z] \le -L \epsilon) \\
&\le \mathrm{P}(|Z - \mathbf{E}[Z]| \ge L \epsilon
) \\
&\le \exp\left( - \frac{2(L\epsilon)^2}{\sum_{t=1}^L2^2} \right) = \exp\left( - \frac{L\epsilon^2}{2} \right)
\end{align*}
$$

那么

$$
\begin{align*}
    R(T) &\le \epsilon L + e^{-\frac{L\epsilon^2}{2}}\epsilon(T-2L) \\
    &\le \dots
\end{align*}
$$

当 $L$ 取值为 $O(T^{2/3})$ 时，可以让 $R(T)$ 尽可能小。