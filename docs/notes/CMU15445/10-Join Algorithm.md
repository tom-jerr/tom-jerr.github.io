---
title: 10-Join Algorithm
date: 2024-10-31
tags:
  - CMU15445
---

# 10-Join Algorithm

## 10.1 join algorithms

> 尽可能选择较小的表作为外围表

![](https://github.com/tom-jerr/MyblogImg/raw/15445/join_algorithm.png)

## 10.2 Join opetators

### Output

#### data

![](https://github.com/tom-jerr/MyblogImg/raw/15445/early_materialization.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/later_meterialization.png)

### Cost Analysis Criteria

## Nested Loop Join

### Nested Loop Join

> 已知表非常小的情况下，可以使用 nested loop join，可以适配 L3 缓存

![](https://github.com/tom-jerr/MyblogImg/raw/15445/nested_loop_join.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/cost_nested_loop_join.png)

### Index Nested Loop Join

![](https://github.com/tom-jerr/MyblogImg/raw/15445/index_loop_join.png)

### Block Nested loop join

![](https://github.com/tom-jerr/MyblogImg/raw/15445/cost_block_nested.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/block_nested.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/cost2_block_nested.png)

### Nested Loop Join Summary

Key Takeaways

> 选择更小的表作为 Outer table  
> 尽可能在 bufferpool 中缓存 outer table  
> 使用索引快速访问 inner table

Algorithms

> Naive  
> Block  
> Index

## Sort-Merge Join

1. Sort

![](https://github.com/tom-jerr/MyblogImg/raw/15445/sort_join.png)

2. Merge

![](https://github.com/tom-jerr/MyblogImg/raw/15445/merge_join.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/sort_merge_joni.png)

此时 R 中 id 为 200；S 的 id 比 200 大，需要回溯，但是由于表是有序的，不必像 nested loop join 回溯到开头只需要回溯到上一个不大于 R.id 的值就可以
![](https://github.com/tom-jerr/MyblogImg/raw/15445/sort_merge_joni2.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/sort_merge_joni4.png)

### when is sort-merge join useful

1. 其中一个或多个表在 Join key 上已经排序
2. 输出必须在 join key 上排序

## Hash Join

![](https://github.com/tom-jerr/MyblogImg/raw/15445/hash_join.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/hash_join2.png)

### simple hash join algorithm

> 可以在 probe 阶段使用 bloom filter 进行优化；首先概率性地查找是否存在这个 Key

![](https://github.com/tom-jerr/MyblogImg/raw/15445/simple_hash_join.png)

### Partitioned hash join

1. partiion phase
2. probe phase

![](https://github.com/tom-jerr/MyblogImg/raw/15445/partition_hash_join.png)

如果其中的桶仍然溢出，使用第二个哈希函数进行 rehash
![](https://github.com/tom-jerr/MyblogImg/raw/15445/partition_hash_join2.png)

### Hybird hash join

仅仅在数据分布及其倾斜的时候使用；将大量使用的 hash 桶保存在内存
![](https://github.com/tom-jerr/MyblogImg/raw/15445/hybird_hash_join.png)

## Join algorithms summary

> hashing is almost always better than sorting for operator executino  
> sorting is better on non-uniform data  
> sorting si better when result needs to be sorted

![](https://github.com/tom-jerr/MyblogImg/raw/15445/join_summary.png)
