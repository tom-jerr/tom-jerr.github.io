---
math: false
tags:
  - CS61C
  - C
title: Floating Point
---

$$
f = (-1)^{S} \cdot  2^{E-127} \cdot  (1+M)
$$

- $S$: 1 位
- $E$: 8 位，有 127 的偏置，原始位置为 $[-126, 127]$，故偏置后的范围为 $[1, 254]$
    - 当 $E=M=0$ 时，表示 $0$
    - 当 `E=0xffffffff`、$M=0$ 时，表示 $\infty$
    - 当 `E=0xffffffff`、$M\neq 0$ 时，表示 `NaN`
- $M$: 23 位，默认省略 1，故需要加回来
- Range: $[2^{-126}, (2-2^{-23})\times_{2}^{127}]$
    - step size: $2^{E-127}\times(1+M+2^{-23}) - 2^{E-127\times(1+M)}=2^{E-150}$
- (Small + Big) + Big != Small + (Big + Big)
