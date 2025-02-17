---
title: 8 Memory Hierarchy
tags: [CS61C, memory]
math:
modified: 星期一, 二月 17日 2025, 9:34:17 上午
---

## Cache

### **1. 缓存的作用**

- **问题背景**：处理器与内存速度不匹配（CPU 性能年增长 55%，DRAM 速度提升缓慢）。
    - 例：1980 年 CPU 执行 1 条指令的时间≈DRAM 访问时间；2020 年 CPU 可执行 1000 条指令≈1 次 DRAM 访问时间。
- **解决方案**：引入缓存（Cache），作为内存的快速副本 (**copy**)，存储最近使用的数据。
    - 特点：容量小、速度快、价格高，位于 CPU 芯片内。
    - 大多数处理器对于 instructions 和 data 有不同的 cache

**类比**：图书馆写报告时，将常用书籍放在桌上（缓存），避免频繁去书库（内存）取书。

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250216150111327.png)

- the main memory is **DRAM**
- 关机时，在 disk 中的数据会被保存，但是其他的会被清楚
- main memory 由 processor 直接处理；而 disk 在使用 operating system 的时候被处理
- programmer-invisible

### **2. 局部性原理（locality）**

- **时间局部性**：近期访问的数据很可能再次被访问。
    - 例：循环变量在多次迭代中被频繁使用。
- **空间局部性**：访问某个地址后，其邻近地址可能被访问。
    - 例：遍历数组时，连续地址的数据会被依次加载。

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250216151105728.png)

以读取数据为例，首先向 cache 发送地址，确认地址中的值是否在 cache 中已经有存储了。如果有（**cache hit**），则直接从 cache 中读取，反之 (**cache miss**) cache 则到 memory 中去读取 data，然后再发送给 processor

### **3. 缓存类型与工作机制**

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250216153242474.png)

#### **(1) 全相联缓存（Fully Associative Cache）**

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250217084256537.png)

- **特点**：数据可存放在任意缓存行，通过标签（Tag）匹配查找。
- **Terminology**:
    -  **cache line / block**: a single entry in the cache
    -  **line / block size**: number of bytes in each cache line （这里是 4 bytes，注意不要包括 tag 和 valid bit）
    -  **tag**: identifies the data stored at a given cache line
    -  **valid bit**: tells you if the data stored at a given cache line is valid
    -  **Capacity**: The total number of data bytes that can be stored in a cache （这里是 16 btyes）

- **地址结构**：  

```  
| Tag（高位） | Byte Offset（低位） |  
```  

- 例：12 位地址，4 字节块大小 → Tag 占 10 位，Offset 占 2 位。
- **工作流程**：  
    - **命中**：标签匹配 → 直接读取数据。
    - **未命中**：从内存加载整个块（即使只需 1 字节），替换旧块（如 LRU 策略）。
    - 例：访问地址 `0x5E0`，若 Tag 匹配缓存行，则命中；否则加载整个块 `0x5E0-0x5E3`。
- tags 在 cache 中可能会占用非常多的空间
- **valid bit**: 由于在程序启动时，之前的 cache 无效，所以需要一个 indicator 来指示 tag entry 是否对当前程序有效，于是在 cache tag entry 中加入一个 valid bit，`0` 表示 cache miss（即使是由于错误读取到 0），`1` 表示 cache hit
- **Byte offset**: 用 line size 计算，`# byte offset bits = log2(line size)`
- **tag bits**: `# tag bits = # address bits - # byte offset bits`
- 由于 locality，读取数据时，即使只需要读一个 byte 的数据，也要将整个 Line 的数据全部读取出来
- **LRU (Least-Recent used)**: 硬件负责保存 access history，当 cache 满时，覆盖最久未使用过的数据，如下：

#### **(2) 直接映射缓存（Direct Mapped Cache）**

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250217091844594.png)

- **特点**：每个内存块只能映射到唯一缓存行（通过索引 Index，也等价于对地址取模操作，如上图）。
- **地址结构**：  
    - 例：12 位地址，4 行缓存 → Index 占 2 位，Tag 占 8 位，Offset 占 2 位。使用 write-through 策略

```  
| Tag | Index | Byte Offset |  
```  

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250217092204594.png)

- **冲突问题**：同一索引下的不同 Tag 导致频繁替换（miss 率高）。
    - 例：地址 `0xFE2` 和 `0xDF9` 映射到同一索引，导致相互覆盖。
- **硬件实现**：

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250217092337052.png)

#### **(3) 组相联缓存（Set Associative Cache）**

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250217093025769.png)

- **特点**：折中方案，每组（Set）包含多个缓存行（如 2 路组相联）。
- **地址结构**：  

```  
| Tag | Index | Byte Offset |  
```  

  - 例：12 位地址，2 组 → Index 占 1 位，Tag 占 9 位，Offset 占 2 位。
- **优势**：减少冲突，提升命中率。
  - 例：组内可存多个 Tag，访问 `0x8E8` 和 `0x8E9` 可能在同一组内命中。

### **4. 缓存替换策略**

- **LRU（Least Recently Used）**：替换最久未使用的块。
    - 例：全相联缓存中，记录每行的访问时间，优先替换最旧的（这里 1、2、3、0 仅仅是编号，比方说可以是在最前面的是最近访问的）

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250217090027494.png)

- **随机替换**：简单但命中率较低。

### **5. 写策略**

- **写直达（Write-Through）**：同时更新缓存和内存，简单但速度慢。
- **写回（Write-Back）**：仅更新缓存，标记为“脏”（Dirty Bit），替换时写回内存。
    - 例：频繁修改同一数据时，写回策略减少内存访问次数。

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250217090335444.png)

### **6. 缓存性能优化**

- **增大块大小**：利用空间局部性，但可能加载无用数据。
- **多级缓存**：L1（小且快）、L2（较大稍慢）、L3（更大更慢），层级化平衡速度与容量。


**总结**：缓存通过局部性原理和层级设计，弥补 CPU 与内存速度差距。全相联灵活性高但硬件成本大，直接映射简单但易冲突，组相联是两者的平衡。写策略和替换策略进一步优化性能。
