---
title: Memtables
date: 2025/4/3
update:
comments: true
description: Memtable 的基本介绍
katex: true
tags:
  - MiniLSM
  - rust
# categories: Project
---

# Memtables

- 将 KV 对首先插入到 Memtable 中，该 Memtable 在内存中只有一个

![](img/memtables.png)

- 如果 Memtable 已经装满，会触发 frozen，将 memtable 转变为 immemtable，重新创建新的 memtable 来给 KV 对插入

![](img/frozen.png)

## Understanding of Memtable Implement

### 1. Why doesn't the memtable provide a delete API?

- lsm-tree 是 append-only 结构，delete 操作是把一个墓碑标记插入 memtable
- 这种方式将 delete 的随机写转变为顺序写，更有利于写

### 2. Does it make sense for the memtable to store all write operations instead of only the latest version of a key? For example, the user puts a->1, a->2, and a->3 into the same memtable.

- 有道理的，如果需要更新最新值，无法发挥顺序写的优势，后续的 write_batch 也就无从谈起了

### 3. Is it possible to use other data structures as the memtable in LSM? What are the pros/cons of using the skiplist?

- SLM-DB 论文中提出可以用 B+树作为内存索引；Oceanbase 使用的存储引擎也是 lsm-tree based，它的 memtable 使用 B+tree 或者 hash index
- 使用跳表优点
  > 插入、删除、查找都是 O(logn)级别，实现简单  
  > 天然有序  
  > 并发控制简单
- 跳表缺点
  > 写放大  
  > 删除操作可能级联操作多级指针

### 4. Why do we need a combination of state and state_lock? Can we only use state.read() and state.write()?

- 必须用组合才能保证真正的并发安全
- 在 force_freeze_memtable 中，考虑多个线程通过 ReadGuard 并发 put 或者 flush，判断 memtable 满了进行 frozen；这个逻辑如果没有一把 mutex 来保护，会出现问题
  > 可能 memtable 已经被其他线程给 frozen 了，又再次 frozen，后 frozen 的 memtable 没有达到大小限制

### 5. Why does the order to store and to probe the memtables matter? If a key appears in multiple memtables, which version should you return to the user?

- 因为我们始终需要保证后 Put 的数据可以首先被 Get 到，保证用户可以看到最新的数据
- 如果存在最新的版本，我们应该返回在该事务提交前的最新的数据

### 6. Is the memory layout of the memtable efficient / does it have good data locality? (Think of how Byte is implemented and stored in the skiplist...) What are the possible optimizations to make the memtable more efficient?

- 如果是短数据，Byte 会内联存储在结构体中，内存连续；对大值（如超过 1KB 的 Blob），Bytes 通过引用计数共享底层 Vec<u8>，避免复制
- 间接指针访问会增加内存间接性，影响预取效率
- Byte 元数据占用额外内存，如果使用内部内存分配，可能造成内存不连续
- 优化

  > 构造统一的内存分配器，同一分配内存  
  > 将 key 和 value 分开存储

### 7. So we are using parking_lot locks in this course. Is its read-write lock a fair lock? What might happen to the readers trying to acquire the lock if there is one writer waiting for existing readers to stop?

- 对写者并不公平，如果多个读者一直在读，写者会一直阻塞
- 如果有一个写者在写，就会阻塞所有读者的读操作

### 8. After freezing the memtable, is it possible that some threads still hold the old LSM state and wrote into these immutable memtables? How does your solution prevent it from happening?

- 可能会出现这种情况，两个线程同时在 put，都发现了当前超过限制，需要强制 frozen
- 我们的解决方案是：获取到 state_lock 后，再次判断是否超过大小限制，这样可以保证没拿到锁的线程一定会在新的 memtable 中进行更新

### 9. There are several places that you might first acquire a read lock on state, then drop it and acquire a write lock (these two operations might be in different functions but they happened sequentially due to one function calls the other). How does it differ from directly upgrading the read lock to a write lock? Is it necessary to upgrade instead of acquiring and dropping and what is the cost of doing the upgrade?

- 结论：需要根据不同的工作负载来确定，读多写少的负载就可以采用分步锁，写少读多的负载可以采用锁升级
- ​ 先释放读锁再获取写锁：

  > 在这两个操作之间，其他线程可能会修改状态，导致之前读取的信息失效。例如，线程 A 读取 memtable 未满，释放读锁，此时线程 B 可能获取写锁并填满 memtable，导致线程 A 再次获取写锁时发现需要 flush，但此时可能已经被处理过了，或者需要重新检查条件。  
  > 这可能导致竞争条件或需要额外的检查，比如在获取写锁后再次验证条件是否仍然满足（即“双重检查”）。

- ​ 直接升级读锁到写锁：
  > 升级锁的过程中，读锁不会被释放，因此其他线程无法在中间插入修改，保证了操作的原子性。这样，在读取状态后到写入的整个过程中，状态不会被其他线程改变。  
  > 但是，锁升级可能带来潜在的问题，比如死锁。例如，当多个线程尝试升级读锁时，可能形成循环等待。此外，某些锁实现可能不支持升级，或者升级操作本身需要一定的成本。

#### 并发性与性能

- acquire read -> drop read -> acquire write

  > ​ 优势：释放读锁后，其他线程可以继续获取读锁，提高并发读取性能。  
  > 劣势：写入线程可能需要等待更长时间获取写锁（因中间存在其他竞争）。

- ​ 锁升级
  > ​ 优势：减少写锁竞争，因为升级时已持有读锁，可能更快获得写锁。  
  > 劣势：若其他线程持有读锁，升级会阻塞直到所有读锁释放（类似直接获取写锁）。在高并发读场景下，可能导致长时间阻塞。
