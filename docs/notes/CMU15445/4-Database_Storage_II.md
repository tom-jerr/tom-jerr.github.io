---
title: 4_Database_Storage II
date: 2023-09-15 23:49:49
tags:
  - CMU15445
---

# 4 Database Storage II

- 数据库负责将非易失性存储中的数据和内存进行交互

## 4.1 problem with slotted page design

- Fragmentation (碎片)
- Useless Disk I/O
- Random Disk I/O (update 20 tuples on 20 pages)

## 4.2 Log-structed storage

**更多使用 KV 数据库上，只有一个键一个值**

- 不存数据，存放数据的操作（insert, delete, update）

- 直接在后面加上新操作，不检查前面的所有操作是否正确

![](https://github.com/tom-jerr/MyblogImg/raw/15445/log-structed.png)

- 读取一个记录，从后向前进行扫描记录；找到需要的数据

![](https://github.com/tom-jerr/MyblogImg/raw/15445/read_record.png)

- 考虑对相同 id 的 log record 进行索引

![](https://github.com/tom-jerr/MyblogImg/raw/15445/read_record1.png)

- 因为 Log 数据会非常大，需要周期性对页内进行压缩

![](https://github.com/tom-jerr/MyblogImg/raw/15445/read_record2.png)

### Log-Structured Compaction

- Level Compaction

  - 将不同的块中记录连接起来，将压缩后的结果存入下一层
  - 从 level0 开始向下读，一直到最后一层

  ![](https://github.com/tom-jerr/MyblogImg/raw/15445/level_compaction.png)

- Universal Compaction

  - 尽可能将周边的块合并；在同一层

  ![](https://github.com/tom-jerr/MyblogImg/raw/15445/universal_compaction.png)

### 优势

- 将随机写变成顺序写，IO 效率高，但是查找代价很高

## add: Log-Structured Merge Tree

- 首先是内存的 C0 层，保存了所有最近写入的 （k，v），这个内存结构是有序的，并且可以随时原地更新，同时支持随时查询。剩下的 C1 到 Ck 层都在磁盘上，每一层都是一个在 key 上有序的结构。

![](https://github.com/tom-jerr/MyblogImg/raw/15445/lsm_tree.png)

#### 写入流程

- 首先将写入操作加到写前日志中，接下来把数据写到 memtable 中，当 memtable 满了，就将这个 memtable 切换为不可更改的 immutable memtable，并新开一个 memtable 接收新的写入请求。而这个 immutable memtable 就可以刷磁盘了。这里刷磁盘是直接刷成 L0 层的 SSTable 文件，并不直接跟 L0 层的文件合并。

- 每一层的所有文件总大小是有限制的，每下一层大十倍。一旦某一层的总大小超过阈值了，就选择一个文件和下一层的文件合并。就像玩 2048 一样，每次能触发合并都会触发，这在 2048 里是最爽的，但是在系统里是挺麻烦的事，因为需要倒腾的数据多，但是也不是坏事，因为这样可以加速查询。

- 这里注意，所有下一层被影响到的文件都会参与 Compaction。合并之后，保证 L1 到 L6 层的每一层的数据都是在 key 上全局有序的。而 L0 层是可以有重叠的。

![](https://github.com/tom-jerr/MyblogImg/raw/15445/level_db.png)

#### 查询流程

- 先查 memtable，再查 immutable memtable，然后查 L0 层的所有文件，最后一层一层往下查。

- 为了加速查询，因为每个 key 在每层至多出现一次；所以查询可以使用布隆过滤器进行优化

## 4.3 Data Representation

![](https://github.com/tom-jerr/MyblogImg/raw/15445/data_repre.png)

- 浮点数的精确值有问题
- 数据库中存储数据，将数据的值变成字符串来存，保证精度

```c
// POSTGRES
typedef unsigned char NumericDigit;
typedef struct {
    int ndigits;			// 数据位数
    int weight;				// 权重
    int scale;				// 指数
    int sign;
    NumericDigit *digits;
}numeric;

// MYSQL
typedef int32 decimal_digit_t;
struct decimal_t {
  int intg, frac, len;	// 小数点前位数，小数点后的位数，总位数
  bool sign;
  decimal_digit_t *buf;
};
```

### Large Values

- 存储的值过大，使用 overflow page，原来存放值的位置存放指向该页的指针
- 溢出页可以继续指向溢出页，成为链表

![](https://github.com/tom-jerr/MyblogImg/raw/15445/large_val.png)

- 存储外部文件的指针；但是无法保证外部文件是否被修改

![](https://github.com/tom-jerr/MyblogImg/raw/15445/externfile.png)

### System Catalogs

DBMS 把数据库元数据存放在 internal catalog 中；数据库将其存为表，自己管理自己的元数据

- tables, colums, indexs, views
- users, permissions
- internal statistics

比如`information_schema`存放数据库元数据；将元数据存放成表

### Database workloads

- On-Line Transaction Processing (OLTP)

  - 快速读写小数据

- On-Line Analytical Processing (OLAP)
  - 复杂查询，对数据进行分析
- Hybrid (混合) Transaction + Analytical Processing
  - OLTP + OLAP
- OLTP 收集数据，提取后进行数据变换和加载；存入 OLAP，OLAP 进行分析后可以写回 OLTP

![](https://github.com/tom-jerr/MyblogImg/raw/15445/OLTP+OLAP.png)

## 4.4 Decompositon Storage Model (列存储)

- N-ary Storage Model (行存储)；对小数据更新和查询十分有效，效率很高

- 复杂查询大量数据，可以使用列存储；如果使用行存储，需要扫描所有数据，仅仅取出某几个属性；浪费大量时间进行扫描；
- 列存储更适合分析数据的情况

### tuple edentification

- Fixed-length Offsets：相同的偏移即存的是同一行的属性
- Embedded Tuple Ids：增加索引；造成存储的开销，但是方便存储

OLTP = 行存储

OLAP = 列存储
