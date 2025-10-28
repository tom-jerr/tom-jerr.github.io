---
title: SGLang Schedular 技术变迁
date: 2025/10/28 18:01
tags:
  - LLMInference
---

# SGLang Schedular 技术变迁

- 最开始的 Schedular 中 CPU 和 GPU 是串行的，导致 GPU 的大量空闲
- 后面的 Schedular 允许 CPU 和 GPU overlap，实现了 zero overhead schedular

Schedular 整体的工作流程如下图所示：

![](img/sglang_scheduler.svg)

我们将结合代码分析一下整个 Schedular 的流程，并最后通过一个例子进行说明。但在分析整个流程之前，我们需要先了解一下调度中比较重要的数据结构以及这些结构之间的关系如何转换

## Key Structure

### Scheduler

`Scheduler`  组件负责管理 Active Request。以下核心全局状态用于在  `Scheduler`  中维护这些 Active Request。

#### `waiting_queue`

- **用途：** `waiting_queue`是一个数据结构，设计用于存放 Active Request。它根据优先级（Request 的最长前缀）或可用内存动态重新排序这些 Request，以优化批处理任务。
- **一些额外要点**
  - **入队**
    - 新到达的 Request。
    - 从  `retract_decode`  返回的 Request。
  - **出队**  当前有最高优先级的 Request 从队列中出队以形成 batch。

#### `new_batch`

- **用途：**：一批准备好进行 prefill/extend 阶段的 Request。
- **一些额外要点**
  - **分块预填充**：如果 Request 所需的 token 数超过可用内存（`remaining_tokens`），可能会被分块成较小的部分。
  - `new_batch`  中的 Request 将经历 prefill/extend。
  - prefill/extend 后，`new_batch`  将过渡到  **全局批次（Global Batch）**，用于下一次迭代。

#### `running_batch`

- **用途：**：一批准备好进行 decode 阶段的 Request。
  - 初始化为空批次：`ScheduleBatch(reqs=[], batch_is_full=False)`
  - 可以动态添加新完成 prefill 的请求
  - 可以移除已完成的请求
- **一些额外要点**
  - **Retracted**：如果 decode 期间可用内存不足，`Scheduler`可能会通过  `retract_decode`  从  `running_batch`  中撤回某些 Request，将其返回到  `waiting_queue`  以供后续处理。

#### `cur_batch`

- **用途：**：`Scheduler`  主循环（`run_batch`  函数）中当前正在处理的 Request 批次。**`prefill` 优先**
- **一些额外要点**
  - `cur_batch`  在  `event_loop_normal`  中分配。
  - 形成  `cur_batch`  的逻辑是：
    - 如果本轮有准备好 prefill 的 Request（`new_batch`），则使用  `new_batch`  作为  `cur_batch`。
    - 否则，`cur_batch`  将处理准备好 decode 的 Request，因此使用  `running_batch`  作为  `cur_batch`。

![](img/batch.png)

### 三个重要的 Batch

#### Overview

- ScheduleBatch 由 schedule.py::Scheduler 管理。它包含高级调度数据，大部分数据位于 CPU 上。
- ModelWorkerBatch 由 tp_worker.py::TpModelWorker 管理。它是 ScheduleBatch 的子集，只包含与 GPU 上模型 forward 相关的数据，它将从 CPU scheduler 转换到 GPU model runner。
- ForwardBatch 由 model_runner.py::ModelRunner 管理，**它包含低级 tensor 数据**

#### ScheduleBatch

```python
class ScheduleBatch:
    reqs: List[Req]  # 请求列表
    req_to_token_pool: ReqToTokenPool  # 请求到token的映射池
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator  # KV缓存分配器
    tree_cache: BasePrefixCache  # 前缀缓存树
    forward_mode: ForwardMode  # 前向模式

    # 批处理相关
    input_ids: torch.Tensor  # 输入token IDs
    seq_lens: torch.Tensor  # 序列长度
    extend_lens: List[int]  # 扩展长度(sel_len - prefix_len)
    prefix_lens: List[int]  # 前缀长度
```

#### ModelWorkerBatch

```python
class ModelWorkerBatch:
    forward_mode: ForwardMode
    input_ids: torch.Tensor # 输入token IDs
    req_pool_indices: torch.Tensor # req 对应的 out_cache_loc 的索引
    seq_lens: torch.Tensor # 序列长度
    out_cache_loc: torch.Tensor # 分配的 KV cache

    # 扩展相关(seq - prefix)
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
```

#### ForwardBatch

```python
class ForwardBatch:
    forward_mode: ForwardMode
    batch_size: int
    input_ids: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor  # 位置编码

    # 注意力相关
    attn_backend: AttentionBackend
    token_to_kv_pool: KVCache

    # 扩展信息(seq - prefix)
    extend_num_tokens: Optional[int]
    extend_start_loc: Optional[torch.Tensor]
    extend_prefix_lens: Optional[torch.Tensor]
```

#### Transpose

```python
# 1. Scheduler创建ScheduleBatch
def run_batch(self, batch: ScheduleBatch):
    # 2. 转换为ModelWorkerBatch
    model_worker_batch = batch.get_model_worker_batch()

# 3. TpModelWorker处理
def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # 4. 转换为ForwardBatch
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        # 5. ModelRunner执行前向传播
        logits_output, can_run_cuda_graph = self.model_runner.forward(forward_batch)

        # 6. 采样生成token
        next_token_ids = self.model_runner.sample(logits_output, forward_batch)

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
        )
```

### Cache

cache 在 sglang 中，相关的主要是``req_to_token_pool`, `token_to_kv_pool`,`tree_cache` 三个结构。

```python
req_to_token_pool[req_idx]:
┌─────────────────────────────────────────────────────────────┐
│  前缀部分 (1984 tokens)    │    新chunk部分 (2000 tokens)    │
├─────────────────────────────────────────────────────────────┤
│ [loc_1, loc_2, ..., loc_1984] │ [loc_1985, ..., loc_3984] │
└─────────────────────────────────────────────────────────────┘
位置:  0                    1984                         3984

KV Cache Pool:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│loc_1 │loc_2 │ ... │loc_1984│loc_1985│ ... │loc_3984│ ... │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ k1,v1│ k2,v2│ ... │k1984,v1984│k1985,v1985│ ... │k3984,v3984│ ... │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

#### ReqToTokenPool

- 管理 **req_idx 到 token 位置的映射关系**
- 为每个请求分配固定的内存槽位
- 维护请求的 token 序列在内存中的连续布局

```python
class ReqToTokenPool:
    def __init__(self, size: int, max_context_len: int, device: str, enable_memory_saver: bool):
        # 主要存储结构：[请求数量, 最大上下文长度]
        self.req_to_token = torch.zeros(
            (size, max_context_len),
            dtype=torch.int32,
            device=device
        )
        self.free_slots = list(range(size))  # 可用槽位列表
        self.size = size
        self.max_context_len = max_context_len
```

#### token_to_kv_pool

- 管理物理 KV 缓存的分配和释放
- 处理页对齐的内存分配（如果启用分页）
- 支持不同的分配策略（连续分配、分页分配等）

#### Tree Cache

其实是联系两个 pool 的组织结构，scheduler 调度过程中会频繁访问，并为请求分配 `req_to_token_pool` 和 `token_to_kv_pool` 中的 slot

- tree_cache 在调度策略中是个关键角色，根据 prefix match 的情况，会决定当前请求何时被 prefill
- `page_size` 决定前缀匹配的粒度，键匹配策略以及分页匹配算法

  - page_size = 1 是逐 token 精确匹配

    ```python
    # 初始化
    tree = RadixCache(req_pool, allocator, page_size=1)
    # 插入过程
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4, 5]))
    # 树结构（逻辑表示）
    root
    └── 1 (child_key=1)
        └── 2 (child_key=2)
            └── 3 (child_key=3)
                └── 4 (child_key=4)
                    └── 5 (child_key=5)
    # 前缀匹配示例
    match_result = tree.match_prefix(RadixKey(token_ids=[1, 2, 3, 7, 8]))
    # 结果: 匹配到[1, 2, 3]，长度为3
    # 可以匹配任意长度的前缀
    match_result = tree.match_prefix(RadixKey(token_ids=[1, 2]))
    # 结果: 匹配到[1, 2]，长度为2
    ```

  - page_size > 1 是按页进行前缀匹配(使用 tuple(tokens) 为 key

    ```python
    # 初始化
    tree = RadixCache(req_pool, allocator, page_size=4)
    # 插入过程
    tree.insert(RadixKey(token_ids=[1, 2, 3, 4, 5, 6, 7, 8]))
    # 树结构（逻辑表示）
    root
    └── (1,2,3,4) (child_key=(1,2,3,4))
        └── (5,6,7,8) (child_key=(5,6,7,8))
    # 前缀匹配示例
    match_result = tree.match_prefix(RadixKey(token_ids=[1, 2, 3, 4, 5, 6, 9, 10]))
    # 结果: 匹配到[1, 2, 3, 4]，长度为4（page对齐）
    # 无法匹配非对齐长度
    match_result = tree.match_prefix(RadixKey(token_ids=[1, 2, 3]))
    # 结果: 空匹配，因为长度3不是page_size=4的倍数
    ```

- page_size > 1 匹配和缓存请求的特殊处理

```python
def match_prefix(self, key: RadixKey, **kwargs) -> MatchResult:
    if self.page_size != 1:
        # 将长度对齐到page_size的倍数
        page_aligned_len = len(key) // self.page_size * self.page_size
        key = key[:page_aligned_len]

    # 例子: 如果key长度为23，page_size为16
    # page_aligned_len = 23 // 16 * 16 = 16
    # 只处理前16个tokens，忽略后面7个

def cache_finished_req(self, req: Req, is_insert: bool = True):
    if self.page_size != 1:
        page_aligned_len = actual_kv_len // self.page_size * self.page_size
        page_aligned_kv_indices = kv_indices[:page_aligned_len]
        # 释放未对齐的尾部
    self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
```

#### RelationShip

```python
# 假设有一个新请求进入系统
req = Req(rid="request_1", origin_input_ids=[1, 2, 3, 4, 5], ...)

# 1. 前缀匹配 - tree_cache
prefix_result = tree_cache.match_prefix(
    RadixKey(token_ids=[1, 2, 3, 4, 5], extra_key=req.extra_key)
)
# 假设匹配到前缀[1, 2, 3]，对应KV缓存索引[100, 101, 102]
req.prefix_indices = torch.tensor([100, 101, 102])
req.extend_input_len = 2  # 需要计算[4, 5]这两个tokens

# 2. 分配请求槽位 - req_to_token_pool
req_pool_idx = req_to_token_pool.alloc(1)[0]  # 分配槽位，比如得到索引7
req.req_pool_idx = req_pool_idx

# 3. 分配新token的KV缓存位置 - token_to_kv_pool_allocator
new_kv_indices = token_to_kv_pool_allocator.alloc(2)  # 为tokens [4, 5]分配，比如得到[203, 204]

# 4. 更新映射关系
req_to_token_pool.req_to_token[req_pool_idx, :5] = torch.tensor([100, 101, 102, 203, 204])

# 5. 实际使用时的查找
def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        # 获取该请求的所有token索引
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # 将CPU缓存加载到对应的KV位置
         token_to_kv_pool_allocator.load_cpu_copy(self.kv_cache_cpu, token_indices)
```

![](img/batch_transpose.jpg)

## Normal (No overlap)

### Overview

一个请求 Request 进入 SGLang Schedular，会经过如下阶段

```shell
Req -> Pre Schedule(CPU) -> Schedule(GPU compute) -> Sample(GPU) -> Post Schedule(CPU) -> next Schedule ...
```

#### Pre Schedule

一个 request 在 Schedule 中的流向如下：

```shell
First Round:
tokenizer(frontend) -> Schedule::waiting_queue(Prefill) -> Schedule::new_batch -> Schedule::cur_batch -> GPU compute -> Detokenizer(frontend)

Second Round - N Round:
Schedule::running_batch(Decode) -> Schedule::cur_batch -> GPU compute -> Detokenizer(frontend)
```

**Pre Schedule**：在 request 进入 Scheduler 真正调度之前会经过如下步骤：

- 收集前端传入的请求，并将其放入等待队列。(`Schedule::recv_request()` & `Schedule::process_input_requests()`)
- 从等待队列和 running_batch 中进行调度 (`Schedule::get_batch_to_run()`)
  - Prefill 中涉及 Radix Tree 和最长前缀匹配（Longest Prefix Matching）算法。(`Req::init_next_round_input()`)
- 为每个请求分配 Token 所需的内存资源。。(`ScheduleBatch::prepare_for_extend()` & `ScheduleBatch::prepare_for_decode()`)

#### Compute batch

新的请求会进入 Prefill 阶段，Prefill 阶段结束后进入 Decode 阶段

##### Prefill Schedule

1. `get_next_batch_to_run`：这里是 Prefill 优先，所以会调用 `get_new_batch_prefill`，并直接把函数返回的`new_batch`**作为这一轮执行的 batch(即 `cur_batch`)**
2. `get_new_batch_prefill`:
   - 创建 PrefillAdder 来管理批次构建，从 waiting_queue 中选择请求
   - 更新 radix tree cache 的前缀
     - 会重新插入新的，并删除冗余的 cache
   - 创建新的 ScheduleBatch
   - 调用 `prepare_for_extend`:
     - 分配`req_pool_indices`：为每个请求在请求池中分配一个唯一的索引位置，这个索引用于在  `req_to_token_pool`  中存储该请求的 token-to-KV 映射。
       - allocate kv cache
       - 将 req 与 kv cache 的映射写入到 `req_to_token_pool`
3. `run_batch()`：执行 Prefill 推理，调用 `TpModelWorker::forward_batch_generation()` -> `ModelRunner::forward()` -> `ModelRunner::_foward_raw()` -> `ModelRunner::forward_extend()`，后面执行 backend 的算子等待返回结果

##### Decode Schedule

1. `get_next_batch_to_run()`：处理上一批次的完成请求，然后与 running_batch 进行 merge
2. `update_running_batch()`：
   - 调用  `prepare_for_decode()`：
     - `Schedular.input_ids = Schedular.output_ids`
     - `Schedular.output_ids = None`
       - `req_to_token_pool`：不变
       - `token_to_kv_pool`
         - 为  `out_cache_loc`  分配（batch size \* 1）个 slot，因为在 decode 模式下我们对每个 batch 一次只生成一个 token
3. `run_batch()`：执行 Decode 推理，调用 `TpModelWorker::forward_batch_generation()` -> `ModelRunner::forward()` -> `ModelRunner::_foward_raw()` -> `ModelRunner::forward_decode()`后面执行 backend 的算子等待返回结果

![](img/kv-cache-request-lifecycle.png)

#### Sample

`TpModelWorker::forward_batch_generation()`：

- 如果不是 overlap 模式，立即进行 Sample，否则重叠 CPU 和 GPU 进行延迟采样

#### Post Schedule

- Prefill： `process_batch_result_prefill()`
  - 获取结果，调用 tree_cahce 的 `cache_unfinished_req()` 保留该 req 的 cache
- Decode： `process_batch_result_decode()`：
  - 对每个 Req 进行判断，如果已经完成就释放 tree_cache 中对应的 KV cache(**循环解释后批量释放**)
  - 将 batch 的结果通过 `stream_out()` 返回给 detokenizer 进行下一步处理

### 事件循环

一直在循环：接收请求 -> 处理请求 -> 获取 cur_batch 来执行 -> 执行 batch -> 处理 batch 的结果(backend 返回)

- 请求都会先执行 prefill 阶段然后执行 decode 阶段，直到获得 EOS 或其他原因退出

```python
def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()
            self.last_batch = batch
```

### Pre Schedule

#### Req -> Waiting_queue

- 首先执行 `recv_requests`：

  - 只有 Pipeline rank = 0 的可以从 zmq 中获取 tokenizer 传来的 requests
  - 其他 pipeline rank 从前一个 pipeline 获取 requests

        ```python
        if self.attn_tp_rank == 0:
                        dp_offset = self.attn_dp_rank * self.attn_tp_size
                        recv_reqs = point_to_point_pyobj(
                            [],
                            self.pp_rank * self.tp_size + dp_offset,
                            self.world_group.device_group,
                            (self.pp_rank - 1) * self.tp_size + dp_offset,
                            self.pp_rank * self.tp_size + dp_offset,
                        )
        """
        def point_to_point_pyobj(
            data: List[Any],
            rank: int,           # 当前进程的全局rank
            group: ProcessGroup, # 通信组（world_group.device_group）
            src: int,           # 源rank（前一个pipeline阶段）
            dst: int,           # 目标rank（当前pipeline阶段）
        ):
        """
        ```

  - work_reqs 在 attn_tp 中进行广播；系统的 control_reqs 在整个 tp_size 中广播

- 然后执行 `process_input_requests`

  1.  **提取 worker_id**: [worker_id = recv_req.worker_id]
  2.  **解包请求**: [recv_req = recv_req.obj]
  3.  **分发处理**: [output = self._request_dispatcher(recv_req)]  调用请求分发器
      - 将 recv_req 构建为新的 `Req` 对象
      - 调用 `_add_request_to_queue()` 将 `Req` 插入 waiting_queue 中
  4.  **发送回应**: 将输出发送回对应的 tokenizer 工作进程

- 在 `recv_requests` 中涉及到一个特殊的变量 `self.input_blocker`，这个东西在顺序 tokenization 才生效，Schedular 在 Tokenizer 批处理阶段暂停处理新请求

##### Blocker

只在顺序 tokenization 模式下生效

- **请求流控制**：在批量处理期间暂停接收新请求
- **原子性保证**：确保批处理操作不被新请求中断
- **全局同步**：协调多进程间的请求处理状态

##### Why Blocker?

**问题场景：没有 input_blocker 的情况**

1.  TokenizerManager 开始批量处理 10 个请求
2.  处理到第 3 个时，新请求突然到达
3.  Scheduler 立即处理新请求，打断批量流程
4.  导致批处理效率降低，资源利用不均匀

**有 input_blocker 的情况**

1. TokenizerManager 发送 BLOCK 信号
2. Scheduler 暂停接收新请求
3. TokenizerManager 完整处理完 10 个请求
4. TokenizerManager 发送 UNBLOCK 信号
5. Scheduler 释放暂存的请求，恢复正常

##### 工作流程

```python
# 1. 初始状态：UNBLOCKED
self._state = _State.UNBLOCKED

# 2. TokenizerManager 开始批量处理
with input_blocker_guard_region(send_to_scheduler=self.send_to_scheduler):
# 自动发送 BlockReqInput(BlockReqType.BLOCK)

# 3. Scheduler 收到 BLOCK 信号
def _execute_block_req(self):
	self._change_state(original=_State.UNBLOCKED, target=_State.BLOCKED)

# 4. 在 BLOCKED 状态下，新请求被暂存
if self._state == _State.UNBLOCKED:
	return [recv_req]  # 正常传递
else:
	self._pending_reqs.append(recv_req)  # 暂存请求
	return []

# 5. 批量处理完成后，自动发送 UNBLOCK 信号
# BlockReqInput(BlockReqType.UNBLOCK)

# 6. Scheduler 收到 UNBLOCK 信号
def _execute_unblock_req(self):
    self._change_state(original=_State.BLOCKED, target=_State.GLOBAL_UNBLOCK_BARRIER)
    self._global_unblock_barrier.local_arrive()  # 本地到达屏障

# 7. 等待所有进程都到达屏障
def poll_global_arrived(self) -> bool:
    # 使用 all_reduce 检查所有进程是否都 local_arrive
    torch.distributed.all_reduce(global_arrived, ReduceOp.MIN)

# 8. 全局屏障满足后，释放暂存请求
def _handle_arrive_unblock_barrier(self):
    self._change_state(original=_State.GLOBAL_UNBLOCK_BARRIER, target=_State.UNBLOCKED)
    output_reqs = [*self._pending_reqs]  # 释放所有暂存请求
    self._pending_reqs.clear()
    return output_reqs
```

#### Waiting_queue/running_batch -> cur_batch

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    # 步骤1: 处理上一批次的结果
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        # 过滤完成的请求
        self.last_batch.filter_batch()
        # 合并到运行批次中
        if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
            if self.running_batch.is_empty():
                self.running_batch = self.last_batch  # 直接替换
            else:
                self.running_batch.merge_batch(self.last_batch)  # 合并批次

    # 步骤2: 获取新的prefill批次
    new_batch = self.get_new_batch_prefill()  # 从 waiting_queue 选择请求，prefill 完成返回 None

    # 步骤3: 决定返回哪个批次(Prefill First)
    if new_batch is not None:
        return new_batch
    else:
        return self.update_running_batch(self.running_batch)
```

##### 获取 prefill batch `get_new_batch_prefill`

- 创建 PrefillAdder，对新来的请求进行分块，然后每次处理多个分块
  - `init_next_round_input()`：
    - 构建完整填充序列：原始输入 token + 已生成的输出 token
    - 最大前缀长度计算：最多缓存到倒数第二个 token（`input_len - 1`），保留最后一个 token 为预测目标
    - 前缀匹配：
      - 当 Request `ABC`到达时，假设当前 radix cache 里存在一个节点`AFG`
      - `match_prefix`  会尝试在当前 radix cache 里找到现存的`ABC`的最长前缀，也就是说它会在`AFG`节点里找到`A`
      - Radix cache 会把这个节点`AFG`拆分成`A`和`FG`，`A`节点成为当前 Request 的最后一个节点

```python
输入: 10000 tokens 的请求
chunk_size: 2000 tokens
page_size: 16 (假设)

1-init_next_round_input
req.fill_ids: [t1, t2, ..., t10000] (10000 tokens)
req.prefix_indices: [] (空)
req.extend_input_len: 10000
req.last_node: root node

1-add_one_req
req.fill_ids: [t1, t2, ..., t1984] (截断到1984, 16 page_size * 16)
req.extend_input_len: 1984
req.is_chunked = 1
调度器设置 [self.chunked_req = req]
return req

2-init_next_round_input
req.fill_ids: [t1, t2, ..., t10000] (重新构建为完整序列)
req.prefix_indices: [cache_loc_1, ..., cache_loc_1984] (前1984个已缓存)
req.extend_input_len: 8096(剩余需要处理)
req.last_node: 指向1984位置的节点

2-add_chunked_req
req.fill_ids: [t1, t2, ..., t3984] (radix 1984 + new chunk 2000)
req.extend_input_len: 2000 (当前 chunk 大小)
req.prefix_indices: [cache_loc_1, ..., cache_loc_1984] (前1984个已缓存)
return req

//...

5-init_next_round_input
self.fill_ids = [t1, t2, ..., t10000]              # 完整序列
self.prefix_indices = [cache_loc_1, ..., cache_loc_7984]  # 前7984个已缓存
self.extend_input_len = 10000 - 7984 = 2016        # 最后2016个

5-add_chunked_req
truncated = 2016 > 2000 = False                    # 最后一个chunk！
req.extend_input_len = 2016                        # 处理剩余全部
req.fill_ids = [t1, t2, ..., t10000]              # 恢复完整序列
self.prefix_indices = [cache_loc_1, ..., cache_loc_7984]  # 前7984个已缓存
return None # 标志 prefill finished
```

- 创建一个新的 ScheduleBatch
- 调用 `ScheduleBatch::prepare_for_extend()`
  - 分配`req_pool_indices`为每个请求在请求池中分配一个唯一的索引位置，这个索引用于在  `req_to_token_pool`  中存储该请求的 token-to-KV 映射。
    - allocate kv cache：[每个 Request 的总 input token 数 - match 到的 prefixtoken 数] 个`out_cache_loc`
    - 将 req 与 kv cache 的映射写入到 `req_to_token_pool`

```python
req_to_token_pool[req_idx]:
┌─────────────────────────────────────────────────────────────┐
│  前缀部分 (1984 tokens)    │    新chunk部分 (2000 tokens)    │
├─────────────────────────────────────────────────────────────┤
│ [loc_1, loc_2, ..., loc_1984] │ [loc_1985, ..., loc_3984] │
└─────────────────────────────────────────────────────────────┘
位置:  0                    1984                         3984

KV Cache Pool:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│loc_1 │loc_2 │ ... │loc_1984│loc_1985│ ... │loc_3984│ ... │
├──────┼──────┼──────┼──────┼──────┼──────┼──────┼──────┤
│ k1,v1│ k2,v2│ ... │k1984,v1984│k1985,v1985│ ... │k3984,v3984│ ... │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

###### PrefillAdder

是 SGLang Scheduler 中负责**智能批处理预填充请求**的核心组件。它的主要作用是从等待队列中选择合适的请求，组装成一个可以高效执行的预填充批次

```python
class PrefillAdder:
    def __init__(self, ...):
	    self.page_size = page_size # 内存页大小
		self.tree_cache = tree_cache  # radix kv cache
		self.token_to_kv_pool_allocator = token_to_kv_pool_allocator # kv cache pool
		self.running_batch = running_batch # 当前正在运行的 decode batch
		self.new_token_ratio = new_token_ratio # 新 token 生成比例
        self.can_run_list = []        # 可以运行的请求列表
        self.preempt_list = []        # 被抢占的请求列表  
        self.new_chunked_req = None   # 新的分块请求
        self.log_hit_tokens = 0       # 缓存命中的token数
        self.log_input_tokens = 0     # 输入token统计
       
	@property
	def rem_total_tokens(self):
	    """计算剩余可用的总token数"""

	def add_one_req(self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]):
	    """添加一个请求到批次中"""

	def add_chunked_req(self, req: Req):
	    """处理分块预填充请求"""
	   
	def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
	    """抢占低优先级请求为高优先级请求让路"""
```

##### 获取 decode batch

- 先从 batch 中删除已经完成或者已经出错的 batch，然后将上一轮的 decode batch 与 running_batch 合并
  - 实际上就是将 `seq_lens`, `orig_seq_lens`, `output_ids` 等进行 `torch.cat` 拼接
- 调用 `update_running_batch()` 获取 decode batch

  - 先检查是否需要回退请求

  ```python
  # 当前运行batch包含5个请求
  batch.reqs = [req1, req2, req3, req4, req5]

  # 检查内存：下一轮需要5个新页，但只有2个可用页
  check_result = batch.check_decode_mem(buf_multiplier=1.5)  # False

  # 触发回退
  retracted_reqs, new_ratio, aborted = batch.retract_decode(server_args)

  # 回退结果：
  # - retracted_reqs = [req4, req5]  # 回退2个请求
  # - new_ratio = 0.75               # 新的token比率
  # - aborted = []                   # 没有中止的请求

  # 更新调度器状态
  self.new_token_ratio = 0.75       # 从1.0降到0.75
  self.num_retracted_reqs = 2

  # 重新排队
  for req in [req4, req5]:
      self._add_request_to_queue(req, is_retracted=True)
  ```

  - 调用 `ScheduleBatch::prepare_for_decode()`

    - 上一轮输出作为这一轮的输入

    ```python
    # Prefill结束时
    self.input_ids = [1, 2, 3, ..., 100]    # 原始输入序列
    self.output_ids = [101]                   # 第一个生成的token

    # 第1轮decode准备
    self.input_ids = [101]                    # 使用上一轮输出
    self.output_ids = None                    # 等待新生成

    # 第1轮decode完成
    self.output_ids = [102]                   # 生成第二个token

    # 第2轮decode准备
    self.input_ids = [102]                    # 使用上一轮输出
    self.output_ids = None                    # 等待新生成
    ```

    - 内存分配 + 序列长度更新

    ```python
    # 假设batch有3个请求，每个请求当前长度分别为[10, 15, 8]
    # decode需要为每个请求的下一个位置分配空间

    # 分配前的KV cache映射
    req_to_token_pool[req1_idx, 0:10] = [loc1, loc2, ..., loc10]
    req_to_token_pool[req2_idx, 0:15] = [loc11, loc12, ..., loc25]
    req_to_token_pool[req3_idx, 0:8] = [loc26, loc27, ..., loc33]

    # 执行 alloc_for_decode 后
    req_to_token_pool[req1_idx, 10] = loc34    # 为位置10分配
    req_to_token_pool[req2_idx, 15] = loc35    # 为位置15分配
    req_to_token_pool[req3_idx, 8] = loc36     # 为位置8分配

    # out_cache_loc = [loc34, loc35, loc36]

    # A faster in-place version
    self.seq_lens.add_(1)
    self.seq_lens_cpu.add_(1)
    self.orig_seq_lens.add_(1)
    # update all length
    self.seq_lens_sum += bs  # bs = batch_size
    ```

### Compute batch & Sample

- 判断是 generation 还是 embedding，这里我们暂时只考虑 generation 的情况
- 将 `ScheduleBatch` 转化为 `ModelWorkerBatch`
- 调用 `TpModelWorker::forward_batch_generation()`
  - 将 `ModelWorkerBatch` 转化为 `ForwardBatch`
  - 调用 `ModelRunner::forward()`，最后调用后端 flashinfer 的算子
    - Prefill 执行 `ModelRunner::forward_extend()`
    - Decode 执行 `ModelRunner::forward_decode()`
  - **立即进行 Sample，根据 logits 获得下一个 token**
  - 返回结果 `GenerationBatchResult`

### Post Schedule

#### Prefill

- 解包 `GenerationBatchResult`
- 执行 `synchronize()` 等待 GPU→CPU 拷贝完成，保证之后访问的数据已在 CPU 上可用。
- 遍历 batch 中每个请求
  - 更新生成结果
  - 更新 logprob
  - Chunked 请求的特殊处理：`is_chunked > 0` 表示 prefill 尚未全部完成，需要递减计数并跳过流式输出。
- 输出流式结果：调用 `stream_output()` 将结果（token、logprob 等）发送给客户端（例如 WebSocket / API response）

| 成员变量                                                        | 来源于 `GenerationBatchResult`      | 在 `process_batch_result_prefill()` 中的用途 |
| --------------------------------------------------------------- | ----------------------------------- | -------------------------------------------- |
| `logits_output`                                                 | 模型 forward 输出的 logits/logprobs | 提供 next token 的 logprob、hidden states    |
| `next_token_ids`                                                | 模型采样出的下一个 token ID         | 转成 list，更新每个请求的 output             |
| `extend_input_len_per_req` / `extend_logprob_start_len_per_req` | 长度元信息                          | 计算 logprob 范围                            |
| `copy_done`                                                     | CUDA 异步事件                       | 同步 GPU→CPU 拷贝完成                        |
| `hidden_states`                                                 | 如果用户需要返回 hidden states      | 从 batch 中切出对应部分                      |

#### Decode

```python
[GPU forward kernel]
   ↓ (结果写入 GenerationBatchResult)
[Scheduler.process_batch_result_decode()]
   ├── copy_done.synchronize()
   ├── next_token_ids.tolist()
   ├── 更新 req 状态
	   ├── req.output_ids.append(next_token_id)
	   ├── if finished, self.tree_cache.cache_finished_req(req) # 释放 kv cache
   ├── stream_output()
   └── free KV pages
```

| 字段名                                     | 来源                     | 在 `process_batch_result_decode()` 中的使用 |
| ------------------------------------------ | ------------------------ | ------------------------------------------- |
| `logits_output`                            | 模型 forward 输出        | 提供 token 概率、logprob、hidden states     |
| `next_token_ids`                           | 模型采样输出             | 转成 list，写入 req.output_ids              |
| `copy_done`                                | CUDA Event               | 同步 GPU→CPU 拷贝完成                       |
| `accept_lens` / `last_batch_allocate_lens` | speculative decoding     | 决定哪些 token 被接受，释放对应 KV 页       |
| `num_accepted_tokens`                      | speculative metrics      | 统计推测解码 token 接受数量                 |
| `can_run_cuda_graph`                       | CUDA graph compatibility | 影响调度日志与复用逻辑                      |
| `hidden_states`                            | optional                 | 提供 hidden states 返回值                   |

## Why Overlap?

运行实际调度算法仅占总体调度开销的一小部分。**大部分开销来自于模型输入的准备和模型输出的后处理**。具体而言，最大的开销来自于构建输入张量、执行输出去标记化以及准备每个请求的元数据

**构建模型输入张量和采样元数据方面的开销主要源于 Python**

多步调度降低了总体调度开销，但也存在一些弊端。例如，在两次调度调用之间，即使某些请求提前完成，也无法将新的请求添加到批次中。

- **vLLM** 的 multi-step decode
- **SGLang** 的 speculative group execution

## Zero-Overhead Schedule(Overlap)

### 原理

在介绍原理之前我们需要回忆一下上面推理过程的 4 个大步骤，考虑哪些步骤可以进行 Overlap，减少 GPU Bubble

#### Inference Overview

SGLang 的推理过程主要分为以下四个阶段：

1. **Pre schedule：**
   - 收集前端传入的请求，并将其放入等待队列。(`Schedule::recv_request()` & `Schedule::process_input_requests()`)
   - 从等待队列和 running_batch 中进行调度 (`Schedule::get_batch_to_run()`)
     - Prefill 中涉及 Radix Tree 和最长前缀匹配（Longest Prefix Matching）算法。(`Req::init_next_round_input()`)
   - 为每个请求分配 Token 所需的内存资源。。(`ScheduleBatch::prepare_for_extend()` & `ScheduleBatch::prepare_for_decode()`)
2. **Compute batch：**
   - 将 batch 发送到 GPU 上进行一步（即 Continue Batch 的一个 iter）推理(`Schedule::run_batch()`)
3. **Sample：**
   - 根据模型输出的 Logit 进行采样，生成下一步的 Token。
4. **Post schedule：**
   - 在一步推理完成后，动态检查请求是否满足结束条件（Check Finish Condition）。
   - 将已完成的请求从批次中移除，并送入 Detokenizer 处理，最终将结果返回给前端。

#### Overlap Overview[^overlap]

Compute batch 和 Sample 这两个挨在一起的阶段是 GPU heavy 的，而 schedule 的两个阶段是 CPU heavy 的。当多个 batch 流水线化时，我们可以用 **GPU 的 Compute 和 Sample 来重叠上一个 batch 的 post scheduler 与当前 batch 的 pre scheduler**。

实际上，如果想让 GPU **异步的**执行代码，我们必须要 CPU 有一个函数，将 Kernel launch 到 GPU 上；并且当 kernel 返回时，我们也需要一个函数来帮助处理 logits

![](img/overlap.png)

在流水线中，Scheduler 可以在不等待 Compute 阶段返回结果的情况下，继续调度下一个批次。然而，它不能一次性连续调度多个批次（例如，在 Compute Batch1 计算时，直接连续调度 5 个批次）。

1. Check Finish Condition：在每次迭代（iter）完成后，需要对当前批次中的每个请求进行 Check Finish Condition。如果有请求已经完成，则需要立即将其结果返回给用户。
2. **更新约束采样(个人认为是主因）**：在每次迭代完成后，还需要更新 vocab_mask，以便在下一轮采样中应用约束条件

### 初始化 Overlap

- **forward_stream**：专门用于 GPU 前向计算，与默认流并行
- **copy_stream**：处理 GPU->CPU 数据传输
- **future_map**：管理异步计算结果，使用**负索引作为 future 标识符**

```python
def init_overlap(self):
    if not self.enable_overlap:
        return
    # 创建专用的CUDA流
    self.forward_stream = torch.cuda.Stream()      # GPU前向计算流
    self.copy_stream = torch.cuda.Stream()         # 数据拷贝流
    # Future映射管理异步结果
    self.future_map = FutureMap(max_running_requests, device, spec_algorithm)
    # 批次记录缓冲区（防止GPU张量被GC回收）
    self.batch_record_buf = [None] * 2
    self.batch_record_ct = 0
```

#### FutureMap

- `future_ct`：当前环形计数器（指针），用于生成新的 future indices（并非“尚未完成的数量”）。
- `future_limit`：环形指针的模（用来做 `% self.future_limit`）。代码里用 `*3` 的因子来 **减小索引冲突概率**（防止 `future_ct` 快速回绕覆盖尚未写回的 slot）。
- `future_buffer_len`：实际缓冲区物理长度（`*5`），比 `future_limit` 更长以保证写入区间有足够空间（防止 slice 越界或回绕写入与读冲突）。

- 这两个因子（3 和 5）是工程经验值，用来增加安全裕量；你可以根据并发量和 outstanding futures 调整。

```python
class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
    ):
        self.future_ct = 0
        # A factor of 3 is used to avoid collision in the circular buffer.
        self.future_limit = max_running_requests * 3
        # A factor of 5 is used to ensure the buffer is large enough.
        self.future_buffer_len = max_running_requests * 5
```

### Overlap 事件循环

![](img/overlap.png)

```python
def event_loop_overlap(self):
    self.result_queue = deque()  # 存储(batch, result)对
    while True:
        # === Pre Schedule 2 ===
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        batch_result = None
        if batch:
            # === Launch Compute Batch 2 ===
            batch_result = self.run_batch(batch)  # 非阻塞启动
            self.result_queue.append((batch.copy(), batch_result))

        # === Post Schedule 1 ===
        # compute batch 2 与 post schedule 1 并行执行
        if self.last_batch:
            # 处理队列中的结果（此时GPU在并行计算当前批次）
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
        elif batch is None:
            self.self_check_during_idle()

        # === Launch Sample 2 ===
        self.launch_batch_sample_if_needed(batch_result) # update vocab mask
        self.last_batch = batch
```

实际上所有函数中与 normal event loop 不同的函数如下：

- `Schedule::run_batch()`
  - `Schedule::record_batch_in_overlap()`：在两个 overlap 的 batch 中交替存储 model_worker_batch 引用，**避免在 overlap 的 forward 尚未结束时，CUDA kernel 访问野指针或已释放的显存**
  - `FutureMap::resolve_future()`：用**真实的 token 替换负索引的占位符**
- `TpModelWorker::forward_batch_generation()`，该函数仅仅将 model_runner.sample 函数 delay 执行，先返回 batch_result

  ```python
  class GenerationBatchResult:
      # For overlap scheduling
      copy_done: Optional[torch.cuda.Event] = None # run_batch() 判断 GPU 是否已经完成计算
      delay_sample_func: Optional[callable] = None # forward_batch_generation() 设置
      future_indices: Optional[FutureIndices] = None # run_batch() 将负数索引替换为 token
  ```

- 增加了 `Scheduler::launch_batch_sample_if_needed()`：
  - 执行真正的 Sample 操作，并拷贝到 CPU 上；**next round** 的 Compute Batch 与 this **round** 的 Post Schedule 并行
  - `ModelRunner::sample()` 函数中调用 `SamplingBatchInfo::update_regex_vocab_mask()`
    - 屏蔽非法 token，分配 vocab_mask 张量大小 [batch, vocab_size]
    - 移动 mask 到计算设备

### 流同步 & 延迟采样

- 延迟采样必须等待上一个 batch 的 grammar 状态更新完成
- 与 [^overlap] 中的代码版本不同，现在的代码将 vocab mask 放在 sample batch 中获取然后 copy 到 cpu 上，

![](img/lazy_sampling.png)

### 延迟采样

1. **分配阶段**：`batch.output_ids = -future_indices.indices`（负数标识）
2. **计算阶段**：GPU 异步计算，结果存储到[future_map](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)
3. **解析阶段**：[future_map.resolve_future()](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)将负索引替换为实际 token_ids

```python
def launch_batch_sample_if_needed(self, batch_result):
    if batch_result is None or batch_result.delay_sample_func is None:
        return

    # 在专用流中处理延迟采样
    with self.forward_stream_ctx:
        self.forward_stream.wait_stream(self.default_stream)
        _batch_result = batch_result.delay_sample_func()  # 执行采样
        self.future_map.store_to_map(batch_result.future_indices, batch_result)
        batch_result.copy_to_cpu()  # 异步拷贝到CPU
```

## Reference

[^overlap]: [# Zero-Overhead Batch Scheduler](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/zero-overhead-scheduler/zero-overhead-batch-scheduler.md)
