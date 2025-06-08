---
title: lecture16_Thread_Level_Parallelism
date: 2023-09-11 22:17:49
tags:
  - System Arch
---

# Thread-Level Parallelism

- SISD：线性执行指令，没有并行
  - RISC-V 单周期 CPU
- SIMD：单指令流，多数据流
  - Intel instrinsics
  - GPUS
- MISD：多指令流，单数据流
  - Deep learning acceleration chips
- MIMD：多指令流，多数据流
  - Modern processors

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/thread_parallelism.png)

## 多核执行模型

- 独立资源
  - Datapath (PC, registers, ALU)
  - Highest level caches (L1, L2 cache)
- 共享资源
  - Memory (DRAM)
  - 3rd level cache

## 线程

- 一个进程可以可以分裂或者 fork 出新线程；这些线程可以同时运行
- 对于单核来说，CPU 分时运行线程

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/singleCPU.png)

- 每个 CPU 核提供一个或多个硬件的线程来执行指令

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/hard_thread.png)

## 硬件多线程

- 在处理器硬件中有两个 PC 核对应的寄存器组
- simultaneous multithreading (SMT) or hyperthreading (HT)

## Big/little Processors

- 大核有更高的性能
  - higher frequency, more superscalar pipelines, larger caches
- 现代处理器设计
  - i9，8 performance cores + 16 efficiency cores
  - 8 Gen 2, 1 big core, 4 medium cores, 3 little cores
  - M1 Pro, 8 performance cores, 2 efficiency cores

## Thread-Level Parallelism (TLP)

- 不同的指令在同一个核上运行

  - 线程共享内存
  - 线程容易通信

- Multithreading Framework
  - Registers
  - PC
  - Stach
- Threads of same process share:

  - Heap

- 注意负载均衡

- 在循环中需要确定线程的执行顺序；需要保证 cache 的高命中率

  ```c
  for(int i = tid*250000;i<(tid+1)*250000;i++)
  ```

### Data Races

- 不同线程竞争共享的内存，可能出现错误
- 需要设置临界区，同一时间只能有一个线程执行该部分代码
