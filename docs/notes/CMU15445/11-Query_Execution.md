---
title: 11 Query Execution
date: 2024-10-31
tags:
  - CMU15445
---

# 11. Query Execution

## Processing Models

### Iterator Model

迭代器模型，也叫做火山或者流水线模型

> 大量函数调用，指令缓存会很快失效

![](https://github.com/tom-jerr/MyblogImg/raw/15445/iterator_model.png)

### Materialization Model

生成所有数据然后返回给上层

> 对于 OLTP 表现不错，因为没有很大的表需要传递

![](https://github.com/tom-jerr/MyblogImg/raw/15445/mater_model.png)

### Vectorized/Batch MOdel

> 在物化模型和火山模型间是一个良好的平衡  
> 可以使用 SIMD 指令加速

![](https://github.com/tom-jerr/MyblogImg/raw/15445/vector_model.png)

### Plan Processing Direction

> 自上而下对于上面的模型来说更加自然

![](https://github.com/tom-jerr/MyblogImg/raw/15445/plan_processing.png)

## Access Methods

### Sequential Scan

![](https://github.com/tom-jerr/MyblogImg/raw/15445/seqscan.png)

#### Optimization

- Prefetching
- Buffer Pool Bypass
- Parallelization
- Heap Clustering
  - 只是取回 RID，最后才取回真正的数据
- Late Materialization
- Data Skipping

##### Data Sipping

![](https://github.com/tom-jerr/MyblogImg/raw/15445/data_skipping.png)

###### ZONE MAPS

> one zone map in one zone，zone 的大小取决于我们的实现，一般为页  
> 当 zone map 存储在区域之外，与索引的工作流程很像

![](https://github.com/tom-jerr/MyblogImg/raw/15445/zone_maps.png)

### Index Scan

![](https://github.com/tom-jerr/MyblogImg/raw/15445/index_scan.png)

### Multi-Index Scan

![](https://github.com/tom-jerr/MyblogImg/raw/15445/multiindex_scan.png)

## Modification Queries

> Halloween problem: 跟踪已经修改过的 record id，一般在操作符内部使用数据结构来跟踪，避免下一次再次修改
> materailization 不会有这样的问题

![](https://github.com/tom-jerr/MyblogImg/raw/15445/modified_query.png)

## Expression Evaluation

> JIT compilation 可以高效地评估表达式；表达式会被编译成常量或函数

![](https://github.com/tom-jerr/MyblogImg/raw/15445/expression_evaluate.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/JIT.png)

## Scheduler

![](https://github.com/tom-jerr/MyblogImg/raw/15445/scheduler.png)

> quickstep 的 scheduler 原型；工作线程池是无状态的、弹性可伸缩的，可以对查询进行优先级的执行

![](https://github.com/tom-jerr/MyblogImg/raw/15445/scheduler1.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/scheduler2.png)

## Process Models

1. Process per DBMS Worker
   ![](https://github.com/tom-jerr/MyblogImg/raw/15445/process_worker.png)
2. Thread per DBMS Worker
   ![](https://github.com/tom-jerr/MyblogImg/raw/15445/thread_worker.png)
3. Embedded DBMS
   ![](https://github.com/tom-jerr/MyblogImg/raw/15445/embedded_worker.png)

## Execution Parallelism

![](https://github.com/tom-jerr/MyblogImg/raw/15445/parallelism.png)

### Inter-query Parallelism

![](https://github.com/tom-jerr/MyblogImg/raw/15445/inter_parallelism.png)

### Intra-query Parallelism

![](https://github.com/tom-jerr/MyblogImg/raw/15445/intra_parallelism.png)

#### Intra-Operator(Horizontal)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/intra_operator.png)

> 通过硬件并行性动态确定，现在一般通过调度器来实现

![](https://github.com/tom-jerr/MyblogImg/raw/15445/exchange_operator.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/intra_operator2.png)

#### Inter-Operator(Vertical)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/inter_operator.png)

> 投影运算符在下层运算符还没有执行完成时，就开始工作

![](https://github.com/tom-jerr/MyblogImg/raw/15445/inter_operator1.png)

#### Bushy

![](https://github.com/tom-jerr/MyblogImg/raw/15445/bushy.png)

## I/O Parallelism

![](https://github.com/tom-jerr/MyblogImg/raw/15445/IO_parallelism.png)

### Multi-Disk Parallelism

![](https://github.com/tom-jerr/MyblogImg/raw/15445/multiindex_scan.png)

### Database Partitioning

![](https://github.com/tom-jerr/MyblogImg/raw/15445/database_partition.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/database_partition2.png)
