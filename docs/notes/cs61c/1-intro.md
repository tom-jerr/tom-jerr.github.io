---
title: Intro
math: false
tags:
  - CS61C
---

## Great Ideas

1. **Abstraction**: Levels of Representation / Interpretation
2. **Moore's Law**: Designing through trends
3. **Principle of Locality**: Memory Hierarchy
4. **Parallelism & Amdahl's law**
5. **Dependability via Redundancy**

## Number Representations

- Bit: 0/1
- Byte: 8 bits
- Most Significant Bit (**MSB**)
- Least Significant Bit (**LSB**)
- One's Complement: `-x=~x`
    - Neg: $[-(2^{n-1}-1),-0]$
    - Pos: $[0, 2^{n-1}-1]$
- Two's Complement: `-x=~x+1`
    - Neg: $[-2^{n-1},-1]$
    - Pos: $[0, 2^{n-1}-1]$
- Bias encoding: 例如想要利用 4 位 bits 代表 $[-3,12]$，只需取 bias 为 $-3$，利用 $[0,15]$ 来表示
    - 对 two's complement，bias 为 $N=-2^{n-1}+1$，不过 $-2^{n-1} - N=-1$，而不是 $0$ 
    - 这是因为在 floating representations 中，全为 $1$ 的数有特殊用途
    - 例如： `0b1010` 的 bias 为 $-2^{4-1}+1=-7$, 这个数为 $10-(-7)=3$
