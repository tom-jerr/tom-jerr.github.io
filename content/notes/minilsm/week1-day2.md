---

title: Merge Iterator
created: 2025-04-05
update:
comments: true
description: Merge Iterator 的基本介绍
katex: true
tags:

- MiniLSM
- rust

# categories: Project

---

# Merge Iterator

- 一个 lsm-tree 会有一个 memtable 和多个 immumemtable，在执行 get 或者 scan 时，我们需要对所有的 memtable 进行查询
- 一般我们会使用迭代器进行 scan，所以我们需要对所有的 memtable 进行 merge 的 iterator

## Understanding of Merge Iterator Implement

### 1. What is the time/space complexity of using your merge iterator?

- 创建的时候 O(n)，遍历所有的 iterator，然后 store
- next 的时候是 O(Clogn)，每次与堆上的最小值进行比较，向后迭代

### 2. Why do we need a self-referential structure for memtable iterator?

- 因为我们需要处理生命周期的问题，使用第三方库的跳表，里面的迭代器周期我们很难处理，所以我们使用自引用结构，规避这个问题

### 3. If a key is removed (there is a delete tombstone), do you need to return it to the user? Where did you handle this logic?

- 不应该，如果是 scan 的话，应该直接跳转到下一个未被删除的 key 进行查询
- 在`lsm_iterator`执行 next 中进行处理，还有刚刚创建迭代器进行处理
  > 可能扫描的第一个 key 就是已经被删除的 key

### 4. If a key has multiple versions, will the user see all of them? Where did you handle this logic?

- 不会，只会看到最新的一个
- 在`lsm_iterator`的 next 中进行处理，最前面的 iterator 中的记录是 lastest 的，会覆盖其他 iterator 的查询结果

### 5. if we want to get rid of self-referential structure and have a lifetime on the memtable iterator (i.e., MemtableIterator\<'a>, where 'a = memtable or LsmStorageInner lifetime), is it still possible to implement the scan functionality?

- 我对这个不是很熟悉，但是我的看法是 memtable 被刷写到磁盘后，iterator 仍然保存着对 memtable 的引用；这是 memtable 的生命周期已经结束，该 iterator 失效，但是 scan 正在进行迭代，出现意料之外的错误

### 6. What happens if (1) we create an iterator on the skiplist memtable (2) someone inserts new keys into the memtable (3) will the iterator see the new key?

- 使用的是 Arc<SkipMap>，所有的 clone 指向同一份原始数据，所以会看到新插入的数据
- 如果需要隔离，应该使用锁，或者提供版本机制

### 7. What happens if your key comparator cannot give the binary heap implementation a stable order?

- 对同一个 key 进行插入、删除、再插入；可能顺序会被搞混，造成看到已经删除的 key，或者看到旧版本数据

### 8. Why do we need to ensure the merge iterator returns data in the iterator construction order?

- 将多个有序数据源（如内存中的 MemTable 和磁盘上的 SSTable）合并成一个全局有序的视图。为了正确性和性能，合并迭代器必须遵循 ​​ 迭代器的构造顺序 ​​（即数据源的优先级顺序）来决定相同键（Key）的返回顺序
- 避免旧数据覆盖新数据，保证读取操作返回最新值。

### 9. Is it possible to implement a Rust-style iterator (i.e., next(&self) -> (Key, Value)) for LSM iterators? What are the pros/cons?

- 可行，但是状态变更必须通过 ​ 共享且可变的内存实现，而不是直接修改 self
  > single thread: 使用 Cell/RefCell\
  > multi thread: 使用 Mutex/RwLock
- 优点
  > 共享迭代器状态: &self 允许迭代器被多个线程或上下文共享（例如通过 Arc<LsmIterator>）\
  > lazy init: 迭代器状态可以按需初始化
- 缺点
  > 性能开销：内部可变性成本，RefCell 在运行时检查借用规则（可能导致 panic）；Mutex 引入线程同步开销（锁竞争）\
  > 可能存在悬垂引用：若外部数据（如 SSTable 文件）在迭代过程中被删除，可能引发未定义行为。

### 10. The scan interface is like fn scan(&self, lower: Bound\<&[u8]>, upper: Bound\<&[u8]>). How to make this API compatible with Rust-style range (i.e., key_a..key_b)? If you implement this, try to pass a full range .. to the interface and see what will happen.

- 实现了，需要我们提供新的 trait，转换为 Bound<Bytes>

### 11. The starter code provides the merge iterator interface to store Box<I> instead of I. What might be the reason behind that?

#### 生命周期

- 子迭代器的生命周期可能依赖于其来源（例如，从 MemTable 或 SSTable 中借用的迭代器）。若直接存储引用（如 &'a mut I），需要为 MergeIterator 添加复杂的生命周期参数。
- 使用 Box<I> 将子迭代器的所有权转移到 MergeIterator 中，确保迭代器在合并过程中始终有效，无需绑定外部生命周期。

#### 堆分配

- trait 对象（如 dyn Iterator）的大小在编译时无法确定，无法直接存储在栈上。
- 使用 Box 将 Trait 对象分配到堆上，通过指针间接引用。
