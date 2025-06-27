---
title: RocksDB 使用指南
date: 2025/6/27
update:
comments: true
description: RocksDB 使用指南
katex: true
tags:
  - Database
---

# RocksDB 概述

# RocksDB 的 Column Family

RocksDB 的 Column Family 是一个重要的概念，它允许用户在同一个数据库实例中创建多个逻辑分区。每个 Column Family 都可以有独立的配置和数据存储方式，这使得 RocksDB 在处理不同类型的数据时更加灵活。

# RockDB 的 Write Batch

Write Batch 是 RocksDB 中用于批量写入操作的机制。它允许用户将多个写操作（如插入、更新和删除）组合成一个原子操作，从而提高写入性能并减少磁盘 I/O。WriteBatch 通过 WAL 机制保证一致性，如果通过 disableWAL 选项禁用 WAL，则 WriteBatch 的操作不会被记录到 WAL 中，这样可以提高性能，但会牺牲数据一致性，但是我们仍能保证内存中的数据一致性。

这一机制可以与 Column Family 结合使用，使得在同一个 Write Batch 中可以对不同的 Column Family 进行操作。这样，用户可以在一个事务中对多个逻辑分区进行批量写入，提高了数据处理的效率。实际上可能的代码如下:

```c++
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/write_batch.h"

// 假设你已经打开了数据库，并获取了多个 Column Family 的句柄
// rocksdb::DB* db;
// rocksdb::ColumnFamilyHandle* cf1_handle;
// rocksdb::ColumnFamilyHandle* cf2_handle;

// 创建一个 WriteBatch
rocksdb::WriteBatch batch;

// 1. 向默认 Column Family 写入
batch.Put("key_default", "value_default");

// 2. 向名为 "cf1" 的 Column Family 写入
batch.Put(cf1_handle, "key_cf1", "value_cf1");

// 3. 从名为 "cf2" 的 Column Family 删除一个键
batch.Delete(cf2_handle, "key_to_delete_in_cf2");

// 准备写入选项
rocksdb::WriteOptions write_options;

// 执行写入
rocksdb::Status s = db->Write(write_options, &batch);
```

在 db->Write(write_options, &batch) 调用内部，RocksDB 会执行以下操作：

1. RocksDB 会迭代 WriteBatch 内部的序列化数据。每一条记录都包含了：
   - 操作类型 (Put, Delete, Merge, etc.)
   - Column Family ID (一个内部整数 ID，与你的 ColumnFamilyHandle 对应)
   - Key
   - Value (如果是 Put 或 Merge)
2. 分发操作: 对于 WriteBatch 中的每一条记录，RocksDB 会根据其 Column Family ID 找到对应的 MemTable。
