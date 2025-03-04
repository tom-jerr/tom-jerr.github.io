---
title: LibOS
date: 2024/4/17 13:41
update: 
comments: true
description: LibOS介绍
katex: true
tags: 
- rCoreOS
- rust
categories: Project

---

# LibOS

![](../img/libos.png)

- 实现与OS无关的OS类型的程序

## 结构

![](../img/libos_structur.png)

- OS在APP运行前进行初始化：建立栈空间 + .bss段清零
- Bootloader在0x80000000处加载，所以OS只能在0x80200000处加载

## 内存布局

![](../img/libos_mem.png)

