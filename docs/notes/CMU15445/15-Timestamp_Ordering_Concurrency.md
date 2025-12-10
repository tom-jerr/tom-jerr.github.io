---

title: 15 Timestamp Ordering Concurrency Control
created: 2024-10-31
tags:

- Database

---

# Timestamp Ordering Concurrency Control

> 2PL 和时间戳排序算法是悲观并发控制算法\
> 还有乐观并发控制算法

![](https://github.com/tom-jerr/MyblogImg/raw/15445/concunrrency_control_approach.png)

## T/O Concurrency Control

> 使用时间戳来确保事务执行的顺序
> 如果 TS($T_I$) < TS($T_j$)，DBMS 需要确保执行的调度必须和$T_i$发生在$T_j$前的串行化调度相同

### Timestamp allocation

1. System/Wall Clock
1. Logical Counter
1. Hybrid

### Basic T/O

> 事务读取和写入对象不需要锁

> W-TS(X)：是最后成功写入的事务的时间戳

> R-TS(X)：是最后成功读取的事务的时间戳

> **_如果事务尝试获取时间戳在自己之后的对象，会 abort 然后 restart_**

> 为每个事务保存数据副本开销很大；长时间运行的事务很可能会被饿死，新事务可能会使得长时间运行的事务 abort 并 restart

#### Reads

> 会为 X 创建一个本地副本来确保$T_i$可重复读

![](https://github.com/tom-jerr/MyblogImg/raw/15445/basic_to_reads.png)

#### Writes

![](https://github.com/tom-jerr/MyblogImg/raw/15445/basci_to_write.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/thomas_write_rule.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/basic_to_example.png)

## Optimistic Concurrency Control

> 事务对数据的修改只发生在事务本地的 workspace

![](https://github.com/tom-jerr/MyblogImg/raw/15445/optimistic_concurrency_control.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/occ_phase.png)

### occ validation

![](https://github.com/tom-jerr/MyblogImg/raw/15445/occ_val.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/occ_val2.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/occ_val3.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/occ_val4.png)

### occ write phase

> 串行提交，每次只允许一个事务处于 validation/write 阶段

![](https://github.com/tom-jerr/MyblogImg/raw/15445/occ_write_phase.png)

> OCC 在冲突比较少的情况下工作比较好：\
> 所有的事务都是只读的(ideal)\
> 事务访问的数据没有交集

### OCC-performance issues

1. 本地复制数据的高成本
1. Validation/Write 阶段的瓶颈
1. abort 比 2PL 更加浪费(因为发生在事务已经执行后才进行 abort)

## The phantom Problem(幻读问题)

> $T_1$仅对已经存在的记录进行锁定，无法看到新插入的记录（**_使用 table-level 的锁无问题；使用 record-level 锁会出现上述问题_**）

> 2PL, OCC 都是创建本地副本，仍然无法看到事务运行后其它事务新插入的记录

![](https://github.com/tom-jerr/MyblogImg/raw/15445/phantom_prob.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/phantom_prob1.png)

### solution

1. 完成事务之前，重新读取查询指定的所有数据
1. 谓词锁：在查询真正开始运行之前逻辑上决定覆盖哪些谓词
1. 索引锁：类似谓词锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/phantom_solution.png)

#### Re-execute scans

![](https://github.com/tom-jerr/MyblogImg/raw/15445/re-execute-scans.png)

#### Preidcate Locking

![](https://github.com/tom-jerr/MyblogImg/raw/15445/predicate_locking.png)

谓词锁的工作原理可以概述为以下几个步骤：

> 1. 定义谓词：在执行查询或更新操作时，定义一个谓词条件（如 age > 18），描述想要锁定的条件。这一谓词会根据查询或操作涉及的数据范围而变化。
> 1. 应用谓词锁：数据库将这一谓词应用于当前数据表的范围，标记出符合条件的数据项（如年龄大于 18 的所有行）。在没有具体数据信息时，它并不是直接锁住某一行，而是锁住满足条件的数据区域。
> 1. 判断冲突：当其他事务尝试访问数据库时，系统会检查它们是否涉及到已经被谓词锁锁定的区域。如果其他事务的谓词条件与当前锁产生冲突，系统会阻止该事务继续，直至锁释放。
> 1. 锁释放：完成对被锁定区域的操作后，事务提交或回滚，系统随即释放谓词锁。

![](https://github.com/tom-jerr/MyblogImg/raw/15445/predicate_locking1.png)

#### Index Locking

![](https://github.com/tom-jerr/MyblogImg/raw/15445/key-lock.png)

> 通过间隙锁来防止其他事务插入出现问题

![](https://github.com/tom-jerr/MyblogImg/raw/15445/gap-lock.png)

> 下面这两种方式只能实现一种，如果两种都实现会发生死锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/key-range-lock.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/key-range-lock2.png)

> 层次锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/hierarchical-lock.png)

### Locking without an index

> 使用粗粒度的锁（page lock or table lock），损失并行度

![](https://github.com/tom-jerr/MyblogImg/raw/15445/locking-withou%20an%20index.png)

## Isolation levels

![](https://github.com/tom-jerr/MyblogImg/raw/15445/isolation_levels.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/isolation_levels2.png)

> 读已提交并没有严格遵守 2PL 协议，而是读取后直接释放共享锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/isolation_levels3.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/isolation_levels4.png)
