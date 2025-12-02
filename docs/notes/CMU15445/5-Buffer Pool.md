---
title: 5_Database_Storage
date: 2023-09-16 22:17:49
  - Database
---

# 5 Buffer Pool

- 在磁盘中将文件切成一个个页
- 在内存中开辟一个缓存池；加快对页的访问

## 5.1 Buffer Pool

### Organization

- 是一个有着固定页数的数组；每个数组元素叫 frame (帧)

- 通过 page table 去索引内存池中的页
- page table 可以 pin 某个页，也可以锁住某个索引

### Mete-Data

- 页表跟踪现在在内存中的页

- Dirty flag
- Pin/Reference Counter

### locks vs latches

- locks
  - 保护事务中的内容
  - 在事务期间持有锁
  - 需要回滚
- latches
  - 保护临界区数据结构
  - 在操作期间持有锁
  - 不必回滚改变

### page dictionary vs. page table

- page dic
  - 在磁盘中，标记每个页在那个文件中
- page table
  - 在内存中，标记页在 Buffer Pool 的什么位置

## 5.2 Allocation Policies

- 全局：同一安排空间
- 局部：为某个线程分配帧不考虑并发的其他线程

## 5.3 Buffer Pool Optimizations

### Multiple Buffer Pools

- DBMS 使用多个 buffer pool，每种 Buffer pool 可以针对不同目的

- Two approaches to mapping desired pages to a buffer pool are object IDs and hashing

### Pre-fetching

- 在数据执行计划时进行预取

#### 顺序预取

- ![](https://github.com/tom-jerr/Mybloghttps://github.com/tom-jerr/MyblogImg/raw/15445/raw/15445/pre-fetching.png)

### Scan-Sharing

- If a query wants to scan a table and another query is already doing this, then the DBMS will attach the second query's cursor to the existing cursor.

- 如果第二个查询与第一个查询的表相同，先跟着第一个查询一起查询；最后查询还未查询的 page

![](https://github.com/tom-jerr/Mybloghttps://github.com/tom-jerr/MyblogImg/raw/15445/raw/15445/scan_sharing.png)

![](https://github.com/tom-jerr/Mybloghttps://github.com/tom-jerr/MyblogImg/raw/15445/raw/15445/scan_sharing2.png)

![](https://github.com/tom-jerr/Mybloghttps://github.com/tom-jerr/MyblogImg/raw/15445/raw/15445/scan_sharing3.png)

### Buffer Pool Bypass

- 在一些特殊的情况下，我们可能并不需要 Buffer Pool，例如顺序扫描磁盘 page，如果我们需要加载的磁盘 page 的分布是连续的，我们可以直接加载磁盘数据，因为是顺序 IO，性能仍然能够不错，并且省去了 Buffer Pool 换入换出的开销。

## Buffer Pool 淘汰策略

- LRU
- CLOCK
- LRU-K

### LRU-K 算法

- 最近 K 次访问最少；数据在访问历史列表里后没有达到 K 次访问，则按照一定规则（[FIFO](https://so.csdn.net/so/search?q=FIFO&spm=1001.2101.3001.7020)，LRU）淘汰；历史队列中的数据访问次数达到 K 次后，将数据索引从历史队列删除，将数据移到缓存队列中，并缓存此数据，缓存队列重新按照时间排序；淘汰缓存队列末尾的元素
- LRU-K 需要多维护一个队列，用于记录所有缓存数据被访问的历史。只有当数据的访问次数达到 K 次的时候，才将数据放入缓存
- 使用 List 进行数据删除时，使用 erase 需要 list 的迭代器；同时需要一个 map 进行迭代器的存储

## 5.4 Buffer Replacement Policies

目标：正确、准确、快速、更新元数据

### Least-recently Used

- 保存每个页最近访问的时间戳
- 查询洪泛问题
  - 热点页多次被换入换出

### Clock

- 模糊的 LRU 不需要每个页有时间戳
  - 每个页有一个引用位
  - 当一个页被访问了，引用位置为 1
- 扫描整个缓存池；如果 ref bit = 1，置为 0；如果 ref bit = 0，直接驱逐

### Better polices: LRU-K

- LRU 和 CLOCK 方法仅仅只考虑了访问时间而没有考虑访问频率，易受到**顺序洪泛**的影响
- 最近 K 次访问最少；数据在访问历史列表里后没有达到 K 次访问，则按照一定规则（[FIFO](https://so.csdn.net/so/search?q=FIFO&spm=1001.2101.3001.7020)，LRU）淘汰；历史队列中的数据访问次数达到 K 次后，将数据索引从历史队列删除，将数据移到缓存队列中，并缓存此数据，缓存队列重新按照时间排序；淘汰缓存队列末尾的元素
- LRU-K 需要多维护一个队列，用于记录所有缓存数据被访问的历史。只有当数据的访问次数达到 K 次的时候，才将数据放入缓存
- 使用 List 进行数据删除时，使用 erase 需要 list 的迭代器；同时需要一个 map 进行迭代器的存储

#### mysql 的 LRU-K 的替代方式

- 从未出现过的 Page 放入 old list 的 HEAD，已经出现在 Old list 的 page 再次访问，放入 young list 的 HEAD
  ![](https://github.com/tom-jerr/MyblogImg/raw/15445/mysql_lru.png)

### Better polices: Localization

- 只驱逐自己使用的页，如果别人也使用该页，不会驱逐

### Better polices: Priority Hints

在 LRU 基础上进行的优化

- 在查询执行过程中，了解每个页的内容
- 提供 hint 去判断内存池中页是否重要
- 现代数据库一般启动时将根结点加入内存池

![](https://github.com/tom-jerr/MyblogImg/raw/15445/priority_hints.png)

## 5.5 Dirty pages

- Fast Path：如果 page 非 dirty，直接驱逐
- Slow Path：如果 page 是 dirty，必须先将脏页回写到磁盘
- WAL：先写日志到磁盘，再将脏页写入磁盘（使用 dirty flag）

### Background Writing

DBMS 定期将脏页回写到磁盘

> 如果脏页已经安全被写入，DBMS 会驱逐该页或者 unset dirty flag

> 在日志没有落盘前，不应该写入脏页中的任何数据

## 5.6 DISK I/O SCHEDULING

![](https://github.com/tom-jerr/MyblogImg/raw/15445/os_cache.png)

> OS/hardware 通过重排和对 IO 进行批量操作来分摊写入的成本

> 但是 OS 不确定哪个 IO 是更重要的；我们需要抛弃 OS 的数据写入控制而由 DBMS 来接管

![](https://github.com/tom-jerr/MyblogImg/raw/15445/diskIO_schedule.png)

### OS PAGE CACHE

正常的 IO 操作会经过 OS PAGE CACHE，但是 DBMS 希望直接对 IO 进行控制，所以采用 direct I/O，绕过 OS page cache

![](https://github.com/tom-jerr/MyblogImg/raw/15445/os_cache2.png)

### fsync problems

![](https://github.com/tom-jerr/MyblogImg/raw/15445/fsync_probs.png)
