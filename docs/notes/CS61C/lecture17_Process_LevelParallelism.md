---
title: lecture17_Process_Level_Parallelism
date: 2023-09-11 22:17:49
tags:
  - System Arch
---

# Process-Level Parallelism

## Multiprocess Framework

- 不同的核不分享内存，但是共享一个文件系统
- 在多线程程序运行时，一个线程 crash，整个程序可能 crash；但是多核程序，每个核之间是独立的，如果一个核 crash，其他的核可以继续运行

- 不同的核交流占用大量时间
- 将一个大问题分解成独立的小问题；以此来减少数据的传输时间损耗
