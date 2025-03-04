---
title: 1-OS的隔离性
date: 2023-09-11 22:17:49
tags:
  - 6.s081
---

# OS 的隔离性

- 需要不同的应用程序之间有强隔离性
- 需要 OS 与应用程序间也有强隔离性

OS 隔离保证 multiplexing 和内存隔离

- multiplexing（CPU 在多进程同分时复用）：不论应用程序在执行什么操作，multiplexing 都会迫使应用程序时不时的释放 CPU，这样其他的应用程序才能运行。
- 不同应用程序之间的内存是隔离的，应用程序之间不会相互覆盖

## 硬件支持强隔离

- uer/kernel mode
  - 在处理器里面有一个 flag。在处理器的一个 bit，当它为 1 的时候是 user mode，当它为 0 时是 kernel mode。当处理器在解析指令时，如果指令是特殊权限指令，并且该 bit 被设置为 1
- virtual memory
  - 处理器包含了 page table，而 page table 将虚拟内存地址与物理内存地址做了对应
  - 每一个进程只能访问出现在自己 page table 中的物理内存
- 内核：Trusted Computing Base (TCB)
  - 宏内核：所有操作系统服务都在 kernel mode
  - 微内核：将大部分的操作系统运行在内核之外；使用 IPC 进行通信
    - 在 user/kernel mode 反复跳转带来的性能损耗。
    - 在一个类似宏内核的紧耦合系统，各个组成部分，例如文件系统和虚拟内存系统，可以很容易的共享 page cache。而在微内核中，每个部分之间都很好的隔离开了，这种共享更难实现。进而导致更难在微内核中得到更高的性能。

## OS 的防御性

- 必须具备防御性抵御恶意程序的攻击
