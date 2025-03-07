---
title: lecture15_Data_Level_Parallelism
date: 2023-09-11 22:17:49
tags:
  - CS61C
---

# Data-Level Parallelism

## SIMD

- single instruction, multiple data or vector instructions

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/vector_sum.png)

### SIMD 实现矩阵乘法

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/SIMD_Matrix.png)

### common mistake

- 直接使用 32-bit 的 SIMD vector（寄存器与内存不同）
- 使用\_mm_load or \_mm_store 时采用未对齐的内存
- 忘记处理尾部特殊情况
- 使用太多的 vector
