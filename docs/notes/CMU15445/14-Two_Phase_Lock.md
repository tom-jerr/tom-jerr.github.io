---
title: 14 Two Phase Lock
date: 2024-10-31
tags:
  - CMU15445
---

# Two Phase Lock Concurrency Control

## Executing with locks

> 事务需要锁(upgrade)
> 锁管理器为请求申请锁  
> 事务释放锁  
> 锁管理器更新内部的 lock table  
> lock table 跟踪每个事务持有什么锁，并正在等待什么锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/executing_with_locks.png)

## Locks vs. Latches

locks 保护数据库磁盘上的对象；lock manager 是内存数据结构，该结构使用 latch 进行保护

![](https://github.com/tom-jerr/MyblogImg/raw/15445/locks_vs_latches.png)

## Basic Lock Types

- S-LOCK: shared locks for reads
- X-LOCK: Exclusive lock for writes

![](https://github.com/tom-jerr/MyblogImg/raw/15445/basic_locks_type.png)

## Two-Phase locking(2PL)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/universe_schedule2.png)

> 是一种并发控制协议  
> **该协议不需要知道事务将要执行的所有查询**  
> **_两阶段锁协议可能会发生死锁_**，但可以通过适当的策略（如死锁检测和预防）来处理和避免。

1. Growing

   > 每个事务向 DBMS 的锁管理器请求它所需要的锁  
   > 锁管理器同意或拒绝锁请求

2. Shrinking
   > 事务在该阶段仅仅释放或者对锁降级  
   > **不可以再次申请锁**

### Executing with 2PL

> 锁机制实际上是在打破 dependency graph 的循环依赖  
> 但是它会出现**级联终止问题**

![](https://github.com/tom-jerr/MyblogImg/raw/15445/executing_with_2pl.png)

### Cascading aborts

> 当 T1 事务释放锁后 abort，T2 事务已经获取了 T1 中的 A 锁，造成脏读

![](https://github.com/tom-jerr/MyblogImg/raw/15445/cascading_aborts.png)

### Strong strict two-phase locking

> 仅仅当事务 commit 或者 abort 才能释放锁

## 2PL Deadlocks

两种解决方案：死锁检测或者死锁避免

### Deadlock detection

> node 是事务  
> $T_{i}$到$T_{j}$的边表示$T_{i}$等待$T_{j}$释放锁  
> 这个系统检查 waits-for graph 的 cycle 并决定如何打破它

![](https://github.com/tom-jerr/MyblogImg/raw/15445/deadlock+detection.png)

> 当 DBMS 发现一个死锁，它会选择一个 victim 事务及逆行回滚来打破思索  
> victim 事务通常会重启或者 abort  
> 在检测死锁频率以及事务在死锁前等待时间是一个 trade-off 问题

#### Rollback length

![](https://github.com/tom-jerr/MyblogImg/raw/15445/rollback_length.png)

### Deadlock prevention

> Wait-Die: 如果等待锁的事务更早开始，可以等待持有锁的事务释放锁；否则等待锁的事务直接 abort  
> Wound-Die: 如果等待锁的事务更早开始，可以抢夺持有锁的事务的锁使其 abort；否则等待锁的事务 wait
> ![](https://github.com/tom-jerr/MyblogImg/raw/15445/deadlock_prevention.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/deadlock_prevention2.png)

## Lock Granularities(锁粒度)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/lock_granularity.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/lock_hierarchy.png)

### intention locks(意向锁)

> 如果只是一个只读事务，只会在表上获取 S 锁，而不是意向锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/intention_locks.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/intention_matrix.png)

#### example

> 使用意向锁，必须在 tuple 中获取最终的 S 或 X 锁；它的父节点必须是 IS、IX 或者 SIX 锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/two_level_hirearchy.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/three_transactions_lock.png)

> 升级锁的请求可能拥有更高的优先级

### Lock Escalation(锁升级)

> 在低层级的许多锁都是 X 锁；它的父节点的层级锁也会变为 X 锁

![](https://github.com/tom-jerr/MyblogImg/raw/15445/lock_escalation.png)

### PostgreSQL lock table

![](https://github.com/tom-jerr/MyblogImg/raw/15445/select_for_update.png)
