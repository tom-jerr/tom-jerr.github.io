---
title: 9-Sort && Aggregation Algorithm
date: 2024-10-31
tags:
  - CMU15445
---

# 9-Sort && Aggregation Algorithm

## 9.0 Query Plan

![](https://github.com/tom-jerr/MyblogImg/raw/15445/query_plan.png)

> 操作符可以处理比内存更多的数据

> 尽可能高效地利用 buffer pool

> 访问磁盘用尽可能多的顺序 IO

## 9.1 Sort

### 9.1.1 Why do we need to sort

> 通常情况下，基于哈希的方式比基于排序的方式更优；但是对于预先排序的数据基于排序的方式可能更优

![](https://github.com/tom-jerr/MyblogImg/raw/15445/sort.png)

### 9.1.2 IN-MEMORY SORTING

![](https://github.com/tom-jerr/MyblogImg/raw/15445/in_memory_sorting.png)

### 9.1.3 TOP-N HEAP SORT

> 如果出现相等的值，就扩展堆数组的大小

![](https://github.com/tom-jerr/MyblogImg/raw/15445/top_n_heap_sort.png)

### 9.1.4 EXTERNAL MERGE SORT

#### sorted run

> 行存储一般采取早期物化，列存储选择延迟物化，先存储 record ID
> ![](https://github.com/tom-jerr/MyblogImg/raw/15445/sorted_run.png)

#### 2-way external merge sort

> 使用这种方式，可以通过删除排序前的源文件来清理磁盘空间

![](https://github.com/tom-jerr/MyblogImg/raw/15445/2_way_external.png)

#### general external merge sort

![](https://github.com/tom-jerr/MyblogImg/raw/15445/gemeral_external_sort.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/external_merge_example.png)

### 9.1.5 Double Buffering

> general external merge 在同一时刻，CPU 和磁盘总有一个会空闲，所以 Double buffering 将 Buffer pool 中的空闲帧分为两部分(buffer and shadow buffer)，可以提高并行度

![](https://github.com/tom-jerr/MyblogImg/raw/15445/double_buffering.png)

### 9.1.6 Comparison Optimizations

> 比较字符串可以使用前缀字符串编码比较；仅当前缀字符串编码相等才进行完整字符串的比较

![](https://github.com/tom-jerr/MyblogImg/raw/15445/comparison_optimizations.png)

### 9.1.7 using B+tree for sorting

如果排序的 Key 上有聚簇 B+Tree，使用它来排序；只需要对 B+树叶子节点扫描一遍

如果是非聚簇索引，需要多次访问同一页面并反复跳转页面，是随机 IO；应该使用外部排序

## 9.2 Aggregations

![](https://github.com/tom-jerr/MyblogImg/raw/15445/aggregations.png)

### 9.2.1 Sorting Aggregation

> 在排序后去重可以优化：在外部排序算法中进行去重

![](https://github.com/tom-jerr/MyblogImg/raw/15445/sorting_aggregation.png)

### 9.2.2 Hashing Aggregation

> 可以进行分区后自主进行选择基于 hash 还是基于 sort 进行后续操作

![](https://github.com/tom-jerr/MyblogImg/raw/15445/hash_aggregation.png)

#### partition

![](https://github.com/tom-jerr/MyblogImg/raw/15445/hash_partion.png)

#### rehash

![](https://github.com/tom-jerr/MyblogImg/raw/15445/rehash.png)

#### hashing summarization

![](https://github.com/tom-jerr/MyblogImg/raw/15445/hash_sum.png)
