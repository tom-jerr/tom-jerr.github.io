---

title: SST
created: 2025-05-15
update:
comments: true
description: Sorted String Table 的基本介绍
katex: true
tags:

- MiniLSM
- rust

# categories: Project

---

# Block

- 这里我们实现了 Sorted String Table 的 encode 和 decode 以及 block iterator
  ![](img/week1-04-overview.svg)

## Sorted String Table format

```shell
-------------------------------------------------------------------------------------------
|         Block Section         |          Meta Section         |          Extra          |
-------------------------------------------------------------------------------------------
| data block | ... | data block |            metadata           | meta block offset (u32) |
-------------------------------------------------------------------------------------------

-------------------------------------
|           Block Meta              |
-------------------------------------
| num of metas| Meta1 | Meta2 | ... |
-------------------------------------

---------------------------------
|           Meta i              |
---------------------------------
| offset | first key | last key |
---------------------------------

```

## Entry format

- 采用与 first key 公用前缀的方式

```shell
-----------------------------------------------------------------------
|                           Entry #1                            | ... |
-----------------------------------------------------------------------
| overlap_with_first_key (2B) | key_len (2B) | key (keylen) | value_len (2B) | value (varlen) | ... |
-----------------------------------------------------------------------

```

## The end of block

- store the offsets of each entry and the total number of entries

```shell
-------------------------------
|offset|offset|num_of_elements|
-------------------------------
|   0  |  12  |       2       |
-------------------------------

```

## Test Understanding of Block implement

### 1. What is the time complexity of seeking a key in the block?

- 使用二分查找，O(logn)

### 2. Where does the cursor stop when you seek a non-existent key in your implementation?

- cursor 会停在 block.offsets.len()处，此时返回的 key 是空的，value_range 是(0, 0)

### 3. So Block is simply a vector of raw data and a vector of offsets. Can we change them to Byte and Arc\<[u16]>, and change all the iterator interfaces to return Byte instead of &[u8]? (Assume that we use Byte::slice to return a slice of the block without copying.) What are the pros/cons?

#### 优点 ​​

- 零拷贝操作 ​：Bytes 类型天然支持零拷贝切片（通过 Bytes::slice），避免数据复制，尤其适用于大块数据或多处共享的场景。

- 线程安全 ​​：Arc\<[u16]> 是线程安全的不可变结构，允许多线程安全共享偏移量数组，无需额外同步。Bytes 内部通过原子引用计数管理数据，支持跨线程安全共享。

- 性能优化 ​​：偏移量数组 Arc\<[u16]> 在多次引用时共享同一内存，比 Vec<u16> 的独立存储更节省内存。
  ​

#### 缺点 ​​

- 内存碎片化 ​​：Arc\<[u16]> 的每个实例独立分配内存，可能导致内存碎片（尤其是大量小 Block 时）。对比 Vec<u16> 的连续内存布局，Arc\<[u16]> 的随机分配可能降低缓存局部性。
- 引用计数开销 ​​：Arc 和 Bytes 的原子操作（增减引用计数）在高并发场景下有轻微性能损耗。频繁创建/销毁 Block 时，原子操作的开销可能累积。
- 灵活性降低 ​​：不可变设计使得无法动态修改 Block 的数据或偏移量，若需原地更新则需重建整个 Block。若系统需要动态调整偏移量（如合并 Block），需额外的数据拷贝。

### 4. What is the endian of the numbers written into the blocks in your implementation?

- 大端存储

### 5. Is your implementation prune to a maliciously-built block? Will there be invalid memory access, or OOMs, if a user deliberately construct an invalid block?

- 在序列化和反序列化时进行检查

### 6. Can a block contain duplicated keys?

- 可以的，合并相同的 key 在 compaction 中进行

### 7. What happens if the user adds a key larger than the target block size?

- 会直接返回失败，一般应该压缩后存入

### 8. Consider the case that the LSM engine is built on object store services (S3). How would you optimize/change the block format and parameters to make it suitable for such services?

高延迟、低随机访问性能、按请求计费（PUT/GET）等特点进行优化

- 增大块大小
- 元数据外置 ​​
  > 设计 ​​：将块索引、布隆过滤器等元数据单独存储为小文件，或内联在块头部。
  > 优点 ​​：减少每次 GET 操作传输的数据量（先拉取元数据，再按需拉取数据块）。支持快速过滤无效查询（如布隆过滤器判断键不存在）。- 列式存储布局
- 尽可能压缩更多的 KV 到一个 block 中
- 如果是 scan，使用预取，将多个 block 的数据一次取出来
