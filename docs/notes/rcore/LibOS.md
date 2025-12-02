---
title: LibOS
date: 2024/4/17 13:41
update:
comments: true
description: LibOS介绍
katex: true
tags:
  - Operator System
  - rust
categories: Project
---

# LibOS

![](../img/libos.png)

- 实现与 OS 无关的 OS 类型的程序

## 结构

![](../img/libos_structur.png)

- OS 在 APP 运行前进行初始化：建立栈空间 + .bss 段清零
- Bootloader 在 0x80000000 处加载，所以 OS 只能在 0x80200000 处加载

## 内存布局

![](../img/libos_mem.png)
