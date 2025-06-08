---
title: lecture18_Caches
date: 2023-09-11 22:17:49
tags:
  - System Arch
---

# Caches

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/computer_components.png)

## Memory Hierarchy (层次化内存)

- 寄存器离 CPU 最近，使用也最快；但是很贵；每个 CPU 只有少量寄存器
- DRAM 更适合存储大量数据，但是速度更慢
  - 需要数据总线进行传输

### 加速方式

- Hardware Multithreaing

  - 在等待数据的过程中，切换到另一个进行去执行任务
  - 每个物理核含有两个 PC 核寄存器组，所以上下文切换很快

- Prefetching

  - 每个周期预取指令和数据

- Caching

  - 利用空间局部性和时间局部性，减少和主存之间的数据传输

  - 缓存中存放被主存使用的数据副本
  - 主存中存放磁盘上的数据副本

### 层次化管理

- Registers <---> Memory
  - 通过编译器或者汇编级别编程
- Cache <---> 主存
  - 通过缓存控制硬件
- Main Memory <---> Disks
  - 通过操作系统（虚拟内存）
  - 通过编程者 (files)

### Cache 三种映射

- 全相联映射

  - 每个数据可以被映射到 n 个位置

  - 物理地址被分为 tag and offset

  ![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/fully_associate.png)

- 直接映射

  - 每个数据只能映射到固定的一个位置

  ![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/directed_mapped.png)

- 组相联映射

  - 分为 m 个组；每个数据可以被映射到每个组组内的任意位置
  - 每个地址被分为 tag + index（组的索引） + offset（块大小）

  ![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/set_associate.png)

### 替换策略

- LRU
  - 最近最少用的块被驱逐出缓存
- MRU
  - 最近使用过的块被驱逐
- FIFO
  - 最老的块被驱逐（queue）
- LIFO
  - 最新的块被驱逐（stack）

### 写策略

- 回写策略
  - 数据改变，马上更新 cache 和 memory 的数据
- 写回
  - 每次只更新 cache 的数据
  - 直到该块被驱逐，检查 dirty flag，将改变写回 main memory
  - 比回写快

## 多级缓存

- 每级缓存比前一级更大，缓存的数据更多

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/multi_caches.png)

### caching with multithreading

- 每个核有自己的 L1、L2 缓存

- 不同的核进行读取核写入时可能发生缓存的不一致问题

### MSI Protocol (缓存一致性)

- 读取时，检查其他的核心该块是否是脏块，如果是回写该改动
- 写入时，如果该块在其他核中可用，使其不可用（如果是脏块，回写数据）
- Invalid：该块不在缓存
- Shared：该块在其他核心；还未进行更改，如果要更改，驱逐每个人缓存中的该块，转化为 modified
- Modified：该块已经被读和修改；稍后其他缓存不在拥有该块

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/MSI.png)

### MOESI

- Exclusive: Same as valid bit on, dirty bit off in regular cache

  - The block has been read, but not modified

  - Further, no other cache has this block

- Owner:
  - The block is in some other cache
  - If we make modifications, it's our responsibility to tell all the other caches in shared state about thesechanges
  - Allows for writing while other threads read the same data.

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/MOESI.png)

### Coherence Missed (一致性 miss)

- 两个线程同时写相同的块
- 可能造成错误共享
  - The entire block is invalidated and must be reloaded, even though technically no data is shared.

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/false_sharing.png)
