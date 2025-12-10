---

title: 13 Concurrency Control Theory
created: 2024-10-31
tags:

- Database

---

# Concurrency Control Theory

## Transaction in sql

![](https://github.com/tom-jerr/MyblogImg/raw/15445/transaction1.png)

## ACID

![](https://github.com/tom-jerr/MyblogImg/raw/15445/ACID.png)

### Mechanisms for ensuring atomicity

#### Logging

> LSM tree 是针对单个表文件配置的；而 logging 是针对全局跨文件设置的

![](https://github.com/tom-jerr/MyblogImg/raw/15445/logging_for_atomic.png)

#### Shadow Paging

> 即使改变几字节，也会复制整个页

![](https://github.com/tom-jerr/MyblogImg/raw/15445/Shadow_paging.png)

### Consistency

> 很多系统采用最终一致性，但在过程中可能出现不一致的情况

![](https://github.com/tom-jerr/MyblogImg/raw/15445/consistency.png)

### Mechanisms for ensuring isolation

![](https://github.com/tom-jerr/MyblogImg/raw/15445/universe_schedule.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/isolation1.png)

#### Formal Properties of schedules

![](https://github.com/tom-jerr/MyblogImg/raw/15445/isolation_schedulers.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/isolation_schedulers1.png)

#### Unreaptable Read

> 读写冲突的情况下，同一事务在两次读取同一个值时读到两个不同的值

![](https://github.com/tom-jerr/MyblogImg/raw/15445/unrepeatable_read.png)

#### Dirty Read

> 写读冲突下，另一个事务读取了其他事务并未提交的值并提交

![](https://github.com/tom-jerr/MyblogImg/raw/15445/dirty_read.png)

#### Lost Update

> 写写冲突导致事务更新值消失

![](https://github.com/tom-jerr/MyblogImg/raw/15445/lost_update.png)

#### Dependency Graph & Confict Serializable

> 大多数 DBMS 实现冲突可串行化
> 如果图中出现环路，说明这个调度是 bad schedule

![](https://github.com/tom-jerr/MyblogImg/raw/15445/dependency_graph.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/dependency_graph1.png)

### Transaction Durability

![](https://github.com/tom-jerr/MyblogImg/raw/15445/transaction_durability.png)
