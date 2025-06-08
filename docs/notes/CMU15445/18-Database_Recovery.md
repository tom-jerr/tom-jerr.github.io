---
title: 18 Database Recovery
date: 2024-10-31
tags:
  - Database
---

# Database Recovery

## ARIES

![](https://github.com/tom-jerr/MyblogImg/raw/15445/aries.png)

## Log Sequence Numbers

![](https://github.com/tom-jerr/MyblogImg/raw/15445/LSN1.png)

> MasterRecord 被硬编码到 DBMS，所以我们恢复时这个页面会先被拉到内存中

![](https://github.com/tom-jerr/MyblogImg/raw/15445/LSN2.png)

> 仅仅当 pageLSN <= flushLSN，才能将 log 刷入磁盘  
> 所有的记录都有一个 LSN  
> 每次一个事务修改一个页上的 record，pageLSN 会改变  
> 每次 DBMS 将 WAL buffer 中的东西写入磁盘，flushedLSN 会更新

![](https://github.com/tom-jerr/MyblogImg/raw/15445/writing_log_records.png)

### Normal Execution

![](https://github.com/tom-jerr/MyblogImg/raw/15445/normal_execution.png)

### Transaction Commit

![](https://github.com/tom-jerr/MyblogImg/raw/15445/transaction_commit.png)

> 我们只需要保证在刷新 flushLSN 之前先将日志记录刷新到磁盘即可  
> TXN-END 写入后说明 commit 已经成功，所以 wal 可以清除没有用的 Log

![](https://github.com/tom-jerr/MyblogImg/raw/15445/transaction_commit2.png)

### Transaction Abort

> prevLSN 维护一个链表允许我们追踪 abort 事务的记录链表

![](https://github.com/tom-jerr/MyblogImg/raw/15445/transaction_abort.png)

> 我们在 abort 和 end 之间可能存在其他日志，我们需要维护这些日志；我们不会在 abort 时立即将这些记录刷写到磁盘

![](https://github.com/tom-jerr/MyblogImg/raw/15445/transaction_abort2.png)

### Compensation Log records

> 是对 update 的撤销操作  
> undoNextLSN 是一个效率优化，而不是一个核心优化

![](https://github.com/tom-jerr/MyblogImg/raw/15445/CLR.png)

## Abort Algorithm

![](https://github.com/tom-jerr/MyblogImg/raw/15445/abort_algorithm.png)

## Checkpoints

### Non-Fuzzy Checkpoints

1. 任何新事务开始都会被停止
2. 所有活跃的事务等待直到 checkpoint 执行完成
3. 将所有的脏页刷新到磁盘

### Slightly better Checkpoints

> 暂停事务，然后将部分提交的数据写入磁盘；  
> 缩短了等待时间，但是磁盘上存储的并非是稳定快照

![](https://github.com/tom-jerr/MyblogImg/raw/15445/slightly_checkpoint.png)

#### Active Transaction Table(ATT)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/ATT.png)

#### Dirty Page Table(DPT)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/DPT.png)

> ATT 和 DPT 必须刷进磁盘后再恢复事务执行

![](https://github.com/tom-jerr/MyblogImg/raw/15445/slightly_checkpoint2.png)

### Fuzzy Checkpoints

> 复制了 ATT 和 DPT 的副本在内存中

![](https://github.com/tom-jerr/MyblogImg/raw/15445/fuzzy_checkpoints.png)

> 任何在<CHECKPOINT-BEGIN>之后开始的事务会被<CHECKPOINT-END>的 ATT 排除

![](https://github.com/tom-jerr/MyblogImg/raw/15445/fuzzy_checkpoints2.png)

## ARIES - Recovery Algorithm

![](https://github.com/tom-jerr/MyblogImg/raw/15445/aries_recovery.png)

### Overview

![](https://github.com/tom-jerr/MyblogImg/raw/15445/aries_overview.png)

### Analysis Phase

> analysis 的过程仅仅是确定 ATT 和 DPT

![](https://github.com/tom-jerr/MyblogImg/raw/15445/analysis_phase2.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/analysis_phase.png)

### Redo Phase

![](https://github.com/tom-jerr/MyblogImg/raw/15445/redo_phase.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/redo_phase2.png)

### Undo Phase

![](https://github.com/tom-jerr/MyblogImg/raw/15445/undo_phase.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/aries_example.png)

## Additional Crash Issues

![](https://github.com/tom-jerr/MyblogImg/raw/15445/addtional_crash.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/addtional_crash2.png)
