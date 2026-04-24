# Hicache In SGlang
## Overview
HiCache 即多级 KVCache，架构如下：

- HiRadixTree：单机 GPU-CPU 双层前缀缓存树
- Storage Backend：可插拔存储后端，集成 3FS、Mooncake、NIXL 等
  - 统一接口封装 batch_get / batch_set / batch exists
  - 零拷贝数据传输
- Global KVManager：提供分布式文件系统（FS）的元数据统一管理服务，具备高效的元数据组织、查询与协调能力，为全局 KVCache 提供一致性管理（KVCache 全局索引）
- 3FS/Mooncake Global Storage：存算分离架构，结合 RDMA 网络优化与 NVMe SSD，提供 TiB/s 级别的聚合读取带宽，作为 HiCache 的持久化存储底座

### 流程
1. Req 进入 Scheduler，先触发 L3 Storage 的 prefetch，将匹配的 hash page 预先加载到 host 内存（L2）
2. 然后加入 waiting queue，等待被 Scheduler 调度
3. 在第一次被调度时，进行 prefill 之前，hicachecontroller 进行 check & match prefix 操作；然后进行load back 操作，此时只是在操作 kv indices
   - Check：对 ack queue 里面的请求进行清空；中断所有正在进行的 prefetch；清空所有 L2 host cache 多分配的部分
   - Match prefix：在 HiRadixTree 上进行 prefix match，找到匹配的节点；如果节点的 GPU KV 可用（即 L1 存在），直接返回 GPU KV slot；如果 GPU KV 不可用但 host KV 可用（即 L2 存在），把 host KV 送到 GPU 上，并更新树节点状态
4. 然后执行 start_loading，异步地真正将 l2 host cache 的数据搬到 GPU 上；同时 forward_batch 也会被送到 GPU 上，准备进行 prefill 计算
5. 真正进入 prefill 阶段，此时 load kv cache 和 prefill 计算是按层重叠进行的：prefill 计算第 i 层时，load 正好把第 i+1 层的数据搬到 GPU 上；prefill 计算第 i+1 层时，load 把第 i+2 层的数据搬上来，以此类推。
6. Prefill 结束后，将 KV cache 写入 GPU，如果需要 write back，则把 GPU KV 写回 host，并更新树节点状态；如果还需要进行 write storage，则把 host KV 写回 storage，并更新树节点状态
7. 然后正常进行 decode 阶段

> [!NOTE]
> 这里的 L3 预取和 L2 load back 都需要匹配时超过一个阈值，如果没有超过不会进行这个过程
> L3 prefetch 预分配一大段的 host kv slot，实际上可能 IO 线程只填充了一部分，剩余的会放入 host_mem_release_queue 里面。

![](img/hicache_overview.png)

## Component Details

### HiRadixTree
**RadixAttention 原始模型**
普通 RadixAttention 用 radix tree 存 prefix KV。每个节点是一段连续 token：
```shell
  root
   └── [system prompt tokens]
        └── [user turns]
             └── [more tokens]
```
节点保存：
```python
  TreeNode.key        # 这段 token span
  TreeNode.value      # GPU KV cache indices
  TreeNode.children   # 后续 token span
  TreeNode.lock_ref   # 正在被请求引用，不能驱逐
```
匹配请求时，从 root 开始按 token prefix 往下走。命中的节点 KV 可以直接复用，未命中的 suffix 才需要 prefill 计算。

**HiRadixTree 增加的状态**
HiCache 在同一个 TreeNode 上增加分层状态：
```python
  node.value       # L1: GPU KV indices；None 表示 GPU 已驱逐
  node.host_value  # L2: host KV indices；None 表示 host 没有备份
  node.hash_value  # L3: 每个 page 的 hash key，用来查外部 storage
```
所以一个节点可以有几种状态：
```python
  value != None, host_value == None
    只在 GPU L1 中。

  value != None, host_value != None
    同时在 GPU L1 和 host L2 中。

  value == None, host_value != None
    GPU 已驱逐，但 host L2 还有备份；tree 节点保留。

  value == None, host_value == None
    本地已经没有可用数据，通常节点会被删除；L3 是否存在要实时查询 storage。
```
这就是分层 radix cache 的本质：tree 结构仍然按 token prefix 组织，但每个节点标记 KV 数据在哪一层。

---

### HiCacheController
- 初始化路径：Scheduler -> HiRadixCache -> HostKVCache -> HiCacheController

![](img/hicache_controller.png)
## 写入操作

### L1->L2 Write_backup
实际写入由 HiCacheController 来完成，write() 实际上是将 host_indices 放入 write_queue，然后 merge_ops 把队列中多个写操作合并成一个批量操作：
1. 拼接 tensor：把多个 host_indices 和 device_indices 张量拼接成一个大 tensor
2. 收集 node_ids：把所有请求的 node_id 汇总
3. 取最小 priority：使用最低优先级
4. 合并 pool_transfers：合并内存池传输信息







## Load 操作
H2D copy 时间和 transformer layer compute 时间重叠。如果 load 比 compute 慢，forward 会在某些层短暂停一下；如果 load 足够快，forward 基本感知不到完整 load 延迟。
举个简化例子，4 层模型：

  t0: scheduler enqueue load, device_indices 已经分配
  t1: start_loading 开始在 load_stream 搬 L0
  t2: L0 搬完，record event L0；load_stream 开始搬 L1
  t3: forward 到 L0，wait L0 立即通过，开始算 L0
  t4: forward 算 L0 的同时，load_stream 搬 L1/L2
  t5: forward 到 L1，如果 L1 已搬完就不等；否则只等 L1
  t6: forward 算 L1，同时 load_stream 继续搬后面的层

```python
scheduler:
  init_load_back()
    分配 GPU KV slot
    radix node.value 指向这些 slot
    load op 放进 load_queue

scheduler:
  ready_to_load_host_cache()
    merge load ops
    在 load_stream 上开始逐层 CPU->GPU copy
    返回 producer_id，也就是 hicache_consumer_index

worker:
  收到 ModelWorkerBatch
  set_hicache_consumer(hicache_consumer_index)

forward:
  layer 0 attention 读 KV
    wait layer 0 load event
    读 layer 0 KV
    计算 layer 0

  layer 1 attention 读 KV
    wait layer 1 load event
    读 layer 1 KV
    计算 layer 1

  
```


  3. Page 粒度
  HiCache 不是任意 token 粒度搬运，而是按 page_size 对齐。代码里：

  - page_size == 1 时按 token 匹配。
  - page_size > 1 时按 page 匹配。
  - child key 也变成一个 page tuple。

  例如 page_size = 4，请求 token 是：

  [1,2,3,4, 5,6,7,8, 9,10]

  真正用于 cache match / L3 查询的是前 8 个 token，最后 [9,10] 不满一页，不参与 page-level cache 复用。

  原因是 KV cache 在内存池和 L3 后端里按 page 管理，page 粒度可以降低元数据量，并提升批量 I/O 和 zero-copy 传输效率。

  4. L1/L2 本地命中流程
  请求进来后，scheduler 会调用 match_prefix()。

  它做几件事：

  1. 把 key 按 page 对齐。
  2. 从 root 遍历 HiRadixTree。
  3. 对 node.value != None 的节点，把 GPU indices 加入 device_indices。
  4. 如果继续命中到 node.value == None 但 node.host_value != None 的节点，记录 host_hit_length。
  5. 返回：
      - device_indices: 已经在 GPU 上的 prefix KV。
      - last_device_node: 最后一个 L1 命中节点。
      - last_host_node: 最后一个 L2 命中节点。
      - host_hit_length: L2 命中的 token 数。

  重要细节：本地匹配本身不搬数据，只走树和返回 indices，所以很快。真正把 L2 搬回 L1 是后面的 init_load_back()。

  5. L3 prefetch 流程
  如果启用了 L3 storage，scheduler 在本地 match 后会执行 _prefetch_kvcache()：

  matched_len = len(req.prefix_indices) + req.host_hit_length
  new_input_tokens = req.fill_ids[matched_len:]
  last_hash = last_host_node.get_last_hash_value()
  tree_cache.prefetch_from_storage(...)

  含义是：

  本地 L1/L2 已命中 matched_len 个 token。
  从 matched_len 之后的 suffix 开始，去 L3 查还能不能继续命中。

  prefetch_from_storage() 做：

  1. suffix 按 page_size 对齐。
  2. 如果 suffix 小于 prefetch_threshold，不查，避免小 I/O。
  3. 从 host memory pool 预分配 L2 空间。
  4. 把 PrefetchOperation 放进 cache_controller.prefetch_queue。

  后台 prefetch 线程做：

  1. 根据 last_hash 和后续 token page 计算链式 hash。
  2. 调用 storage backend 的 batch_exists() 查 L3 是否有连续 page。
  3. 多 TP rank 下用 all_reduce(min) 对齐命中长度，保证各 rank 状态一致。
  4. 如果命中 token 数小于 threshold，撤销 prefetch，释放 host pages。
  5. 如果命中足够多，调用 batch_get / batch_get_v1 把 L3 page 读入 host L2。
  6. scheduler 调 check_prefetch_progress()，根据策略决定是否等待：
      - best_effort: 不阻塞，能取多少算多少。
      - wait_complete: 等全部完成。
      - timeout: 等到完成或超时。

  读入完成后，_insert_helper_host() 会把 L3 命中的 token span 插入 HiRadixTree，但只填 host_value，不填 value。也就是说：

  L3 hit -> 先 materialize 成 L2 hit -> 再按需 load back 到 L1。

  这点很关键：HiRadixTree 不长期保存 L3 地址。L3 是否命中，是实时问后端；命中后才把结果变成本地 L2 tree 节点。

  6. L2 到 L1 的 load back
  请求真正进入 batch 前，调度器会看 host_hit_length。如果 L2 命中大于 0，就调用：

  tree_cache.init_load_back(...)

  内部 load_back() 会：

  1. 从 last_host_node 往父节点回溯，收集连续的 evicted == True 且 backuped == True 节点。
  2. 拼接这些节点的 host_value。
  3. 判断是否值得加载，比如长度必须超过 load_back_threshold。
  4. 从 GPU KV pool 分配 device_indices。
  5. 调 cache_controller.load()，把 host KV 异步搬回 GPU。
  6. 给这些节点重新填上 node.value = device_indices[...]。

  然后 ready_to_load_host_cache() 会真正启动异步加载。cache_controller.start_loading() 是按 layer 搬运的，配合 LayerDoneCounter 做 layer-level overlap：

  加载 layer N+1 的 KV
  同时计算 layer N

  这就是文档里说的 CPU-to-GPU transfer 与 prefill compute overlap。

  7. 新 KV 的插入
  没有命中的 suffix 会正常 prefill。prefill 完成后，新生成的 KV page 会通过 insert() 写回 radix tree：

  new_node.key = key
  new_node.value = value.clone()      # GPU indices
  new_node.hash_value = compute_node_hash_values(...)

  如果启用了 storage 或 KV events，就会计算 hash_value。这个 hash 是 page 级、位置相关的链式 hash：

  hash(page_i) = H(page_tokens_i, prior_hash=hash(page_{i-1}))

  所以同样的一页 token 出现在不同 prefix 下，会得到不同 key，避免把不同上下文位置的 KV 混用。

  8. L1 -> L2 -> L3 写回
  HiCache 有三种写回策略：

  write_through
    命中/插入后尽快写到 L2/L3，缓存收益最大，I/O 压力也最大。

  write_through_selective
    命中次数达到阈值后才写，偏向热点数据。

  write_back
    GPU 驱逐时才写到 host/L3，减少 I/O。

  代码路径是：

  write_backup()
    GPU L1 -> host L2

  write_backup_storage()
    host L2 -> L3 storage

  write_backup() 会把 node.value 对应的 GPU KV 拷到 host pool，完成后设置：

  node.host_value = host_indices.clone()

  如果启用了 L3，再调用 write_backup_storage()，按 node.hash_value 把 page 写入 storage backend。

  对 write_through 有一个重要约束：host 备份必须形成从 root 开始的连续 prefix，不能父节点没备份而子节点先备份。代码里如果 parent 没有 backuped，会跳过子节点备份。这样 L2/L3 prefix 语义才成立。

  9. 驱逐逻辑
  HiCache 的驱逐分两级。

  L1 GPU 驱逐：

  如果 node 已经 backuped:
    只释放 GPU KV，保留 tree 节点和 host_value。
    node.value = None
    以后可从 L2 load back。

  如果 node 没有 backuped:
    释放 GPU KV，并从 tree 删除节点。

  对应代码是：

  _evict_backuped()
    # GPU -> CPU demotion

  _evict_regular()
    # 未备份，直接删除

  L2 host 驱逐：

  只驱逐已经不在 GPU 的节点，也就是 evicted == True 的节点。
  释放 host_value 后，tree 节点被删除。

  这就是分层 radix tree 能工作的原因：L1 被驱逐不等于 prefix 元数据消失，只要 L2 有备份，tree 仍可命中这个 prefix。

  10. 一个例子
  假设 page_size = 2，已有 tree：

  root
   └── [A B][C D]    value=L1, host_value=L2
        └── [E F]    value=None, host_value=L2

  新请求：

  [A B C D E F G H]

  流程是：

  1. match_prefix()
     [A B C D] 命中 L1，返回 GPU indices。
     [E F] 命中 L2，host_hit_length = 2。
     [G H] 本地没命中。

  2. prefetch_from_storage()
     用 [A B C D E F] 的 last_hash 继续算 [G H] 的 page hash。
     查 L3。

  3. 如果 L3 有 [G H]
     把 [G H] 读到 host L2。
     插入 tree，形成：
        [G H] value=None, host_value=L2

  4. init_load_back()
     把 [E F] 和可能的 [G H] 从 L2 搬回 L1。
     请求实际 prefill 只需要计算更后面的未命中 token。

  最终效果是：

  L1 命中：无需搬运，直接用 GPU KV。
  L2 命中：需要 host -> GPU，但不用重新 prefill。
  L3 命中：storage -> host -> GPU，也不用重新 prefill。
  未命中：正常 prefill，并把新 KV 插入 tree。

  11. 为什么 L3 不直接放进 HiRadixTree
  这是一个重要设计选择。SGLang 文档明确说：HiRadixTree 对本地 L1/L2 维护精确元数据，但不会持续同步 L3 元数据。原因是 L3 是跨实例、跨机器共享的，持续同步会带来很大开销，而且状态容易过期。

  所以 L3 的语义是：

  本地 tree 保存 page hash。
  需要时用 hash 问 storage backend。
  命中后把数据拉到 L2，并把 L2 元数据插入本地 tree。

  这使得 HiCache 既能保持本地 radix match 很快，又能通过 L3 做跨实例 KV 复用。

  12. 后端接口
  L3 后端统一实现 HiCacheStorage，核心接口是：

  batch_exists(keys)
  batch_get(keys, target_locations)
  batch_set(keys, values)

  新接口还有 batch_get_v1 / batch_set_v1，可以直接传 host memory indices，减少一次 Python tensor 拷贝，Mooncake、NIXL、HF3FS 等后端会走这种 zero-copy 风格路径。

  所以完整链路可以总结为：

  HiRadixTree:
    负责 prefix 元数据和 L1/L2 位置。

  HiCacheController:
    负责 GPU<->host 搬运、storage 线程、异步 ack、TP 同步。

  HiCacheStorage backend:
    负责 L3 exists/get/set，例如 file、Mooncake、NIXL、HF3FS、AIBrix。

  这就是 SGLang HiCache 的分层 radix cache 实现：radix tree 管 prefix，节点状态表达 L1/L2，page hash 连接 L3，调度器把 L3 命中逐步 materialize 到 L2/L1，从而把“prefix KV 复用”扩展到 GPU 外


## 测试
这里通过 python 模拟了 4 个 request 的完整链路，验证了 HiCache 的 L1/L2/L3 hit/miss、write_through、load_back、evict 等核心功能。日志里可以看到每一步的状态变化和数据流向。

### Req 1

  - 先查 L1/L2，没命中
  - 尝试 L3 prefetch，发现 L3 也没有，结果 hit_tokens=0
  - 模型正常 prefill 后，把 2880 tokens 插入 L1 radix tree
  - 由于 write_through，马上触发 L1->L2 write
  - controller 把 device KV copy 到 host KV
  - L1->L2 完成后，节点状态变成 L1=Y、L2=Y、L3_hash_pages=45
  - 随后触发 L2->L3 write
  - 写入 L3 storage 的 backup 线程开始 SET
  - L2->L3 完成

### Req 2

  - radix tree 直接命中 L1
  - 结果是：L1_tokens=2880 host_hit_length=0
    - 即不需要搬数据，只复用 GPU KV。
  - 客户端也确认 device=2880
  - 然后测试脚本强制 L1 驱逐随后 queue drain 请求让异步写队列跑完，并触发：
  - L1 evict demote
  - device pool free 2880 tokens
  - 节点变成 L1=N、L2=Y

### Req 3
  - match_prefix 发现 L1 没有，但 L2 有 2880 tokens
  - 触发 L2->L1 load_back
  - controller 创建 load queue
  - load_back 后节点重新变成 L1=Y、L2=Y
  - start_loading 启动真实 copy
  - 客户端确认 host=2880

### Req 4
  - /flush_cache 成功
  - flush 后 L1/L2 radix tree 都清了，但 file backend L3 没清。
  - L3 prefetch 完成 2880 tokens
  - loaded_from_storage=2880
  - L3 读出来以后先进入 L2 host，然后再 L2->L1 load_back
  - 客户端确认 storage=2880

## HiCache 整体流程
HiCache 的核心结构是 HiRadixCache。每个 radix tree node 记录一段 token span 的缓存位置：

  - node.value：L1，GPU KV cache indices。
  - node.host_value：L2，host memory KV cache indices。
  - node.hash_value：L3，每个 page 对应的 hash key，用于 file/NIXL/Mooncake 等 storage backend。
  - node.lock_ref：保护正在被请求使用的节点，不允许被驱逐。
  - node.host_ref_counter：保护正在 storage prefetch/write 的 host KV，不允许被释放。


### 线程和协作
主要有三类执行单元：

  1. Scheduler 主线程
     它负责请求调度、radix tree 查询、决定是否走 L1/L2/L3、更新 tree node 状态。match_prefix、init_load_back、check_prefetch_progress 都在这条主路径上发生。
  2. GPU/host copy stream，不是 Python 线程
     L1->L2 和 L2->L1 不是普通阻塞 copy。controller 会在独立 device stream 上发起异步 copy：
      - L1->L2：python/sglang/srt/managers/cache_controller.py:681
      - L2->L1：python/sglang/srt/managers/cache_controller.py:749
  3. Storage 后台线程
     只有启用 L3 storage 时启动：
      - prefetch_thread：负责 L3 metadata 查询、判断 storage hit、提交实际读任务。
      - prefetch_io_aux_thread：实际执行 L3->L2 page transfer。
      - backup_thread：负责 L2->L3 写入。

  这些线程在 controller reset/start 时创建。prefetch 线程逻辑在 python/sglang/srt/managers/
  cache_controller.py:948，backup 线程逻辑在 python/sglang/srt/managers/cache_controller.py:1050。

  - L1->L2 write completion：ack_write_queue 里放 CUDA event，scheduler 后续通过 writing_check() 查询 event。完成后才把 node 从 ongoing_write_through 移出，并继续触发 L2->L3 write。见 python/sglang/srt/mem_cache/hiradix_cache.py:713。
  - L2->L1 load completion：ack_load_queue 里放 load event，loading_check() 查询完成后解除 node lock。见 python/sglang/srt/mem_cache/hiradix_cache.py:756。
  - L3 prefetch completion：scheduler 反复调用 check_prefetch_progress()；如果策略是 wait_complete，必须等 operation 的 completed_tokens == hash_pages * page_size 才继续。见 python/sglang/srt/mem_cache/hiradix_cache.py:1130。
  - TP 同步点：多 TP rank 时，storage hit 数和完成 token 数要做 all_reduce(MIN)，保证所有 rank 对 radix tree 做一致更新。相关逻辑在 python/sglang/srt/
    managers/cache_controller.py:963 和 python/sglang/srt/mem_cache/hiradix_cache.py:1151。
  - Storage control queue drain：backup ack、prefetch revoke、host memory release 都集中在 drain_storage_control_queues() 里处理，避免后台线程直接改 radix tree。见 python/sglang/srt/mem_cache/hiradix_cache.py:1051。

  整体可以理解成：scheduler 主线程拥有 radix tree 状态；copy stream 和 storage 线程只做数据搬运并把完成事件/ack 放回队列；scheduler 在安全点 drain/ack，然后更新 node 的 L1/L2/L3 状态。这样避免后台线程直接并发修改 tree。
