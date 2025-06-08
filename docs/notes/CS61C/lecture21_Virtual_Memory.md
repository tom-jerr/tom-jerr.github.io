---
title: lecture21_Virtual_Memory
date: 2023-09-11 22:17:49
tags:
  - System Arch
---

# Virtual Memory

直接使用物理内存问题

- 没有足够空间；RISC-V 只提供 32bit 空间即 4GB
- 地址空间有空洞
- 不能保证其他程序不访问同一块内存

### Benefits

- 允许运行自身大小比主存大得多的程序；只有工作的页放在主存，其他页放在磁盘；由 OS 进行页的请求和置换
- OS 可以共享内存并实现程序的互相保护

- 隐藏机器级的不同
