---

title: 16 Multi-Version Concurrency
created: 2024-10-31
tags:

- Database

---

# Mutli-Version Concurrency

> DBMS 有多个物理版本和一个逻辑版本

> 一个事务写入时会创建一个新版本；读取时会读取该事务开始时最新的版本

**MVCC 实际上是维护多个版本的机制；而版本之间的并发正确性仍然需要并发控制协议来维护**

![](https://github.com/tom-jerr/MyblogImg/raw/15445/multi-version.png)

> writers do not block readers and readers do not block writers

> 只读事务可以不获取锁读取一个一致性的快照

## MVCC example

![](https://github.com/tom-jerr/MyblogImg/raw/15445/mvcc_example.png)

## Write Skew Anomaly

> 快照隔离并不能确保可串行性；可能出现这种写偏斜问题

![](https://github.com/tom-jerr/MyblogImg/raw/15445/write_skew_anomaly1.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/write_skew_anomaly2.png)

## Version Storage

![](https://github.com/tom-jerr/MyblogImg/raw/15445/version_storage.png)

### Append-Only Storage

> 可以看作是一个多版本的单链表

1. Oldest to Newest(O2N)
1. Newest to Oldest(N2O)

### Time-Travel Storage

> 为本来没有做 MVCC 的系统提供的一种 MVCC 机制

![](https://github.com/tom-jerr/MyblogImg/raw/15445/time-travel_storage.png)

### Delta Storage

> 增量存储：只存储更改后的列

![](https://github.com/tom-jerr/MyblogImg/raw/15445/delta_storage.png)

## Garbage Collection

![](https://github.com/tom-jerr/MyblogImg/raw/15445/gc.png)

### Tuple-level GC

> 后台清理：只扫描被修改的页，寻找到不在被引用的页的版本，将这些 tuple 通过后台进程处理掉

![](https://github.com/tom-jerr/MyblogImg/raw/15445/background_vacuuming.png)

> 协同清理：索引可能指向 O2N 的链，事务在找到当前需要的版本时会同时清理不再需要的版本号

![](https://github.com/tom-jerr/MyblogImg/raw/15445/cooperative_cleaning.png)

### Transaction-level GC

> 每个事务会跟踪他自己的读写集合并交给 GC worker

![](https://github.com/tom-jerr/MyblogImg/raw/15445/transaction_gc.png)

## Index Management

![](https://github.com/tom-jerr/MyblogImg/raw/15445/index_management.png)

### Secondary Indexes

1. Logical Pointers: 不会直接指向真实数据，而是指向一个间接层；一般指向主键索引

   ![](https://github.com/tom-jerr/MyblogImg/raw/15445/logical_pointer.png)

   ![](https://github.com/tom-jerr/MyblogImg/raw/15445/logical_pointer2.png)

1. Physical Pointers: 直接指向版本链的头部

   > 如果我要更新版本链，我会更新多个指针的指向；开销过大

   ![](https://github.com/tom-jerr/MyblogImg/raw/15445/physical_pointer.png)

### MVCC Indexes

1. 每个索引必须支持重复的键来指向不同的快照
   > 相同的 key 在不同的快照中指向不同的逻辑 tuples

![](https://github.com/tom-jerr/MyblogImg/raw/15445/mvcc_duplicated_key.png)

## MVCC Deletes

> 逻辑删除：\
> 在某些 MVCC 实现中，删除操作可能只是将数据标记为“已删除”，而不是物理移除。这意味着数据仍然存在于数据库中，但在查询时不会被返回。
> 这样可以保持数据的一致性和可追溯性。\
> 真正的物理删除由后台线程自动清理过期的版本

![](https://github.com/tom-jerr/MyblogImg/raw/15445/mvcc_deletes.png)
