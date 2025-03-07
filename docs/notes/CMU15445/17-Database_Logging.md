---
title: 17 Database Logging
date: 2024-10-31
tags:
  - CMU15445
---

# Database Logging

## Crash Recovery

1. Actions during normal txn processing to ensure that the DBMS can recover from a failure
2. Actions after a failure to recovver the database to a state that ensures atomicity, consistency, and durability

## Failure Classification

### Transaction Failures

1. Logical Errors: 事务因为一些内部原因没有完成（完整性约束失效）
2. Internal State Errors: DBMS 必须终止一个活跃的事务因为一个错误的条件（死锁）

### System Failures

1. Software Failure: OS 或者 DBMS 实现的错误
2. Hardware Failure:
   > the computer hosting the DBMS crahsed(电源线被拉了)  
   > Fail-stop Assumption: 非易失性存储的内容被假设不会被 system crash 破坏

### Storage Media Failures

Non-Repairable（无法修复的） Hardware Failure

1. head crash 或者相似的磁盘 failure 破坏了所有或者部分的 non-volatile storage
2. 重建被假设可以被 detectable(磁盘控制器使用校验和来检测错误)

**_数据库不能从这种错误中恢复数据_**

## UNDO VS. REDO

![](https://github.com/tom-jerr/MyblogImg/raw/15445/undo_redo.png)

### Steal Policy

> 似乎否允许 DBMS 允许一个未提交的事务覆盖最近提交的数据

Steal: is allowed
No-steal: is not allowed

### Force Policy

> 是否一个 DBMS 要求事务的所有更新都在事务提交之前反应到 non-valaile 存储中

Force: is required
NO-force: is not required

### Combination

#### NO-STEAL + FORCE

> 不必 undo，因为改变还未写入磁盘；  
> 不必 redo，因为所有的改变在提交时已经确保写入磁盘

![](https://github.com/tom-jerr/MyblogImg/raw/15445/nosteal_force.png)

> Shadow Paging 实际上在副本上进行修改，事务 commit 后，将 master 清空并将 shadow 作为新的 master

![](https://github.com/tom-jerr/MyblogImg/raw/15445/Shadow_paging1.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/Shadow_paging_example.png)

#### STEAL + NO-FORCE

> 维护一个 log 包含了事务对数据库的改变

![](https://github.com/tom-jerr/MyblogImg/raw/15445/WAL.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/WAL2.png)

## WAL PROTOCOL

> 在数据写入磁盘前，对数据进行操作的日志必须先落盘

![](https://github.com/tom-jerr/MyblogImg/raw/15445/WAL_PROTOCOL.png)

> 每次事务提交需要等待 log 落盘，这可能会成为瓶颈；所以 DBMS 使用 group commit optimization 来 batch multiple log flushed；这会提高吞吐量
>
> 1. 如果 buffer 满了，写磁盘
> 2. 如果超时了，写磁盘（5ms）

![](https://github.com/tom-jerr/MyblogImg/raw/15445/WAL_PROTOCOL2.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/group_commit.png)

## Logging Schemes

![](https://github.com/tom-jerr/MyblogImg/raw/15445/logging_schemes.png)

> 逻辑日志需要重新执行语句，成本过于高昂；Physiological 在页级别是物理的，但是页内部只记录修改的数据

![](https://github.com/tom-jerr/MyblogImg/raw/15445/logging_schemes2.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/physical_logical_logging.png)

## Checkpoints

> 多频繁地进行检查点的设置是一个 tradeoff

### Blocking/Consistent Checkpoint Protocol

1. 暂停所有查询
2. 将所有的 WAL 记录落盘
3. 将所有在 Buffer Pool 中的更改的页落盘
4. 在 WAL 中写入<CHEKCKPOINT>并落盘
5. 恢复查询
