---

title: Batching Inference & KV Cache
created: 2025-10-13
tags:

- LLMInference

---

# Batching Inference & KV Cache

## KV Cache

### Question without KV Cache

- Attention 计算实际上是与 `seq_len` 平方成正比的，所以 prompt 变长的话，我们的 FLOPs 会很快增长到超过 Single GPU 的计算能力(**Compute-Bound**)
- **“Aha!”时刻：** 当我们去预测第 9 个词时，我们需要：
  - $Q$ = 第 8 个 token 的 Query 向量。
  - $K$ = `["The", ..., "and", "the"]`（所有 8 个 token）的 Key 向量。
  - $V$ = `["The", ..., "and", "the"]`（所有 8 个 token）的 Value 向量。
  - \*\*`Key` 和 `Value` 向量一旦为某个 token 计算出来，就永远不会改变了，我们不需要重复计算”

### Definition

**KV Cache（键值缓存）** 是一个在显存（VRAM）中开辟的存储空间，用于**保存模型每一层**计算过的所有历史 token 的 `Key` 和 `Value` 向量。

- **Prefill (阶段一)**: 处理 $N$ 个 token（提示词长度）。
- **Decode (阶段二)**: 每一步都**只处理 1 个 token**。
- 生成 $M$ 个词的总计算量现在是 $O(N + M)$，这是一个**线性**复杂度

### New Question with KV Cache

我们实际上是在用**空间**换**计算** (Compute-Bound => Memory-Bound)

- **大小**: 缓存的大小与 `seq_len x num_layer x hidden_size` 成正比。
- **问题**: 这是一个巨大的显存消耗。当你说一个模型支持 128k 的上下文窗口时，不仅需要显存来加载模型权重，还需要**额外**的、海量的显存来**容纳这个 128k 长度的 KV Cache**。
- **优化**: 这也催生了新的优化技术，比如**Grouped-Query Attention (GQA)** 和 **Multi-Query Attention (MQA)**，它们的核心目的就是通过减少 K 和 V 头的数量，来**减小 KV Cache 的显存占用**，从而在不牺牲太多性能的情况下支持更长的上下文。

### Implement

实际上 KV Cache 的最简单实现就是简单的进行 keys 和 values 的拼接

- Prefill：初始化的 KV Cache
- Decode：在原有的 KV Cache 后面增加新 token 的 KV Cache
- 这里使用的是动态构建，实际上可以进行预分配减少 **`torch.cat` 重新分配显存和拷贝数据的开销**；或者像 vllm 或者 SGlang 使用 `page kv cache` 类似 OS 中的页表管理，减少内部碎片

```python
class SimplestKVCache:
    def __init__(self):
        self.key_values = None
        self.offset = 0
    @torch.no_grad()
    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        key = key.detach()  # do not keep autograd
        value = value.detach()
        B, H, S, D = key.shape
        if self.key_values is None:
	        # prefill
            self.key_values = (key, value)
            self.offset = S
        else:
	        # decode
            prev_key, prev_value = self.key_values
            new_key = torch.cat([prev_key, key], dim=2)
            new_value = torch.cat([prev_value, value], dim=2)
            self.key_values = (new_key, new_value)
            self.offset += S
        return self.key_values[0], self.key_values[1], self.offset
```

- 但实际上 KV Cache 通常与 Batch Inference 技术一同出现，所以我们先介绍 Continuous Batching 技术和 Chunked Prefill 技术，最后介绍如何改进 KV Cache 可以适配这两个技术

## Continuous Batching

### Why Continuous Batching

在这种 Batching 技术出现之前，一般批处理有两种模式：

#### Static Batching

静态批处理是最简单的批处理形式。在这种模式下，服务器会等待，直到积累了固定数量的请求，然后将它们作为一个整体进行处理  。

- **机制**：所有请求被组合成一个批次，并**被填充（gpaddin）到该批次中最长序列的长度**（包括 prompt 和预期的最大生成长度）。然后，这个批次从头到尾被作为一个不可分割的单元进行处理  。

- **缺陷**：这种方法导致了严重的 GPU 资源浪费和性能瓶颈，主要源于两个问题：

  1. **队头阻塞（Head-of-Line Blocking）**：批次中的所有请求必须等待最长的那个请求完成生成。这意味着那些早已完成生成（例如，生成了较短回复）的请求所占用的 GPU 资源将处于闲置状态，无法被释放或用于处理新请求  。
  1. **填充开销（Padding Overhead）**：为了将不同长度的序列组合成一个规整的张量，系统必须添加大量的填充令牌。GPU 在这些填充令牌上执行的计算是完全无效的，不产生任何有意义的输出，构成了巨大的计算浪费  。

#### Dynamic Batching

作为静态批处理的改进，动态批处理引入了更灵活的成批策略。它通常**基于一个时间窗口或批次大小上限，以先到者为准** 。

- **机制**：服务器在收到第一个请求后启动一个短暂的计时器。如果在计时器结束前批次大小达到了上限，则立即开始处理；否则，在计时器结束后，无论批次是否已满，都将当前积累的请求作为一个批次进行处理。这种方式在延迟和吞吐量之间取得了更好的平衡  。
- **Head-of-Line Blocking**：尽管有所改进，动态批处理仍然在**请求级别**上运作。这意味着一旦一个批次开始处理，它仍然是一个同步的、不可分割的单元。**整个批次必须等待其中最慢的请求完成，才能开始下一个批次的处理**。对于输出长度极不确定的 LLM 推理任务，队头阻塞问题依然存在，只是程度有所减轻  。

为了解决上面的队头阻塞问题，提出了 Continuous Batching 技术

### What is Continuous Batching[^orca]

![](img/continuous_batching.png)

1. **单次迭代调用**：调度器每次只调用执行引擎运行模型的一个**迭代步骤**，而不是运行整个请求的生成过程  。
1. **late-arrived enter**：这种细粒度的调度使得**新到达的请求几乎可以立即加入到当前正在运行的批次**中，只需等待当前迭代（通常是毫秒级）结束即可。这极大地缩短了请求在队列中的等待时间  。
1. **early exit**：一旦某个请求完成，它可以在下一次迭代开始前就立即将结果返回给客户端，彻底解决了“早完成、晚返回”的问题  。

| **特性**           | **静态批处理 (Static Batching)**                 | **动态批处理 (Dynamic Batching)**                | **连续批处理 (Continuous Batching)**                 |
| ------------------ | ------------------------------------------------ | ------------------------------------------------ | ---------------------------------------------------- |
| **调度粒度**       | 请求级 (Request-level)   请求级（Request-level） | 请求级 (Request-level)   请求级（Request-level） | 迭代级 (Iteration-level)   迭代级（Iteration-level） |
| **调度逻辑**       | 等待固定数量的请求                               | 等待时间窗口或数量上限                           | 在每次令牌生成后动态调整                             |
| **GPU 利用率模式** | 锯齿状，大量空闲                                 | 锯齿状，空闲时间减少                             | 持续高位，接近饱和                                   |
| **延迟特征**       | 队头阻塞严重，首令牌延迟高                       | 队头阻塞有所缓解                                 | 首令牌延迟显著降低                                   |
| **吞吐量特征**     | 变化负载下吞吐量低                               | 吞吐量有所提升                                   | 各种负载下均能实现高吞吐                             |
| **主要弱点**       | 队头阻塞和填充开销                               | 队头阻塞问题依然存在                             | 内存管理复杂性高                                     |

### Chanllenges & Resolutions

不同请求由于已生成的 tokens 数量不同，其 KV Cache 的长度也不同。Attention 计算中需要查询完整的 KV Cache

- 无法直接用规则张量批处理（维度不匹配）
- 如果 padding 到最大长度，会浪费大量内存和计算
  MLP 层的计算只依赖于**当前时间步**的 hidden state，不需要访问历史 KV Cache
  所以 Orca[^orca] 使用了对 **MLP 进行 batching 加速，而 Attention 则是逐个进行计算**

### Implement

这里实际上也结合了 Chunked Prefill

- 这里使用的是简化的方法：预填充一个请求，然后为每个正在进行的请求解码一个令牌

```python
while requests_in_queue_or_in_progress:
    if prefill_request exists:
        prefill_request.try_prefill()  # perform a chunk of chunked prefill
        if prefill_request.ready:
            if kv_cache.try_add(prefill_request):
                prefill_request = next(requests)
    tokens = decode(model, kv_cache)
    requests.append(tokens)
```

## Chunked Prefill

### Why Chunked Prefill？

通过 Continuous Batching，我们已经能够实现 token-level 的调度，但是 prefill 是一个漫长且不可分割的工作单元，它会阻塞整个系统，带来队头阻塞问题：

1. **队头阻塞（Head-of-Line, HOL Blocking）**：在服务队列中，一个长上下文请求的 Prefill 任务会长时间独占 GPU 资源，导致后续所有请求（即使是处理时间很短的请求）都必须排队等待，极大地增加了它们的 TTFT 。这对于在多租户环境中维持服务质量（QoS）是致命的。

解决此问题的唯一途径是使这个长任务变得**可分割**和**可抢占**

### What is Chunked Prefill?

将一个完整的 Prefill 任务分解为一系列更小的“块”（chunks），它将一个漫长、不可中断的任务转变为一系列短暂、可中断的子任务。这使得调度器能够在处理长请求的各个 chunk 之间，穿插执行其他短请求，从而有效缓解队头阻塞，实现更公平的资源共享

- **实现 Prefill 与 Decode 操作的更优并行与交错**：通过将 Prefill 任务细粒度化，调度器可以在执行一个 chunk 的 Prefill 计算后，立即转去执行其他请求的 Decode 步骤，然后再回来处理下一个 chunk 。

- **处理超长上下文**：当输入提示的长度超过 GPU 显存单次所能处理的极限时，分块处理成为一种必要手段  。

- **创建可预测的调度单元**：将长度不定的用户输入转化为固定大小的计算任务，使调度更加规整和可预测

### Chanllenge & Resolutions

- **计算效率低**：小 chunk 会降低注意力计算核（kernel）的执行效率，导致吞吐量下降  。
  > 这是因为注意力计算的性能对输入矩阵的规模很敏感，小 chunk 意味着查询（Query）矩阵很小，而早期的注意力核并未针对这种“长 Key、短 Query”的场景进行优化  。
- **“读放大”（read amplification）**：在计算每个新 chunk 的注意力时，系统需要从速度较慢的 HBM（高带宽内存）中重新读取之前所有 chunk 累积的完整 KV Cache

#### Resolutions

- **PD 分离**：Prefill 和 Decode 任务被物理地分配到不同的 GPU 集群或“工作节点”（workers）上  。一个专用的 GPU 池负责处理计算密集的 Prefill 任务，生成 KV Cache，然后将计算好的 KV Cache 传输给另一个专用于处理 Decode 任务的 GPU 池。

  > 计算密集的 Prefill 任务不再与延迟敏感的 Decode 任务在同一个 GPU 上争抢资源，从而极大地提升了 ITL 的稳定性和系统的可预测性  。

- **Flash Attention**：FlashAttention 是一种 I/O 感知的注意力算法，它可以在不将完整的注意力矩阵写入 HBM 的情况下，计算出完全相同的精确结果

  > 将小的 Query 块加载到 SRAM，然后高效地迭代处理庞大的 Key/Value 矩阵的各个分块，避免了性能退化。

  > 其次，通过在 SRAM 内部完成大部分计算，FlashAttention 极大地减少了对 HBM 的访问，从而有效中和了因重复读取 KV Cache 而产生的“读放大”问题

### Implement

- 在整个 prefill 任务结束之前，我们仅仅只是更新 model 中的 KV Cache
- 在 prefill 最后一个 chunk 结束时，我们需要生成一个新的 token(即**第一个 output token**)

```python
def try_prefill(self):
    new_token = update_kv_cache(key, value)
	self.offset += tokens_to_prefill
	if self.offset == self.prefill_tokens.size(-1):
		self.is_prefill_done = True
		self.decode_done(token.item(), False)

def decode_done(self, token, update_offset=True):
	if token == self.eos_token_id:
		self.is_done = True
		return
	self.generated_tokens.append(token)
	self.next_token = token
	if update_offset:
		self.offset += 1
```

## Combine KV Cache with Batching

- 我们这里的实现并不高效，只是简单地将所有 KV Cache 都 padding 到 `max seq_len` 上，方便 GPU 做矩阵计算
- 这里的调度规则是 prefill 优先，但是 prefill 按 chunk 调度
  - 如果此时没有空闲的 slot，prefill 就一直 pending

```python
def batch_generate(
    model: any,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    decode_requests: list[Request] = [None] * batch_size
    is_idle = [True] * batch_size
    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]
    result = []
    pending_prefill_request = None
    next_request_idx = 0
    progress_cnt = 0
    start_time = datetime.now()

    while True:
        # prefill until no idle slots
        if len(prompts) > 0 and pending_prefill_request is None:
            prompt = prompts.pop(0)
            pending_prefill_request = Request(
                model, tokenizer, prompt, prefill_step, next_request_idx
            )
            next_request_idx += 1

        # In every iteration, we do a prefill first
        if pending_prefill_request is not None:
            if not pending_prefill_request.is_prefill_done:
                pending_prefill_request.try_prefill()
            if pending_prefill_request.is_prefill_done:
                prefill_kv_cache = pending_prefill_request.kv_cache
                found_slot = False
                # found an idel slot in batch size
                for i in range(batch_size):
                    if is_idle[i]:
                        # Add this request to the decode requests
                        is_idle[i] = False
                        for prefill_cache, batch_cache in zip(
                            prefill_kv_cache, kv_cache
                        ):
                            batch_cache.add_request(prefill_cache, i)
                        decode_requests[i] = pending_prefill_request
                        found_slot = True
                        break
                if found_slot:
                    pending_prefill_request = None

        # After the prefill request moves forward one step, we do the decode
        if not all(is_idle):
            next_tokens = []
            offsets = []
            for req in decode_requests:
                if req is not None:
                    next_tokens.append(req.next_token)
                    offsets.append(req.offset)
                else:
                    next_tokens.append(0)
                    offsets.append(0)
            # batch decode
            next_tokens = _step(model, next_tokens.reshape(-1, 1), offsets, kv_cache)
            for i in range(batch_size):
                if not is_idle[i]:
                    req = decode_requests[i]
                    remove_reason = None
                    if req.is_done:
                        for batch_cache in kv_cache:
                            batch_cache.remove_request(i)
                        is_idle[i] = True
                        result.append((req.prompt_idx, req.text()))
                        decode_requests[i] = None
                        continue
                    # decode a new token
                    req.decode_done(next_tokens[i].item())
    return result
```

## A Better Design

使用 Page Attention + Continuous Batching + Chunked Prefill[^vllm]

### 调度器：连续批处理 (Continuous Batching)

vLLM 的调度器每一步都会检查所有正在运行的序列。它会把所有准备好生成下一个 token 的序列（无论它们处于什么阶段，长度是多少）动态地组合成一个 "批次"（Batch）。

- 这个 batch 里可能包含：
  - 序列 A（刚处理完 Prompt，长度 500，等待生成第 501 个 token）
  - 序列 B（已经生成了 80 个 token，等待生成第 81 个 token）
  - 序列 C（一个新请求，正在进行 Preill，处理它的 200 个 prompt token）

### 自定义 Attention 核 (Custom Kernel)

vLLM 的 PagedAttention **自定义 CUDA 核** 被设计为可以直接处理这种 "异构" 批次。

- **Prefill 请求**使用  `context_attention_fwd`  函数处理（支持多 token 的上下文注意力）
- **Decode 请求**使用 `PagedAttention kernel` 处理（优化的单 token 注意力）

### Prons

- **无需 Padding：** Kernel 直接操作不同长度的数据，因为 Block Table 明确告诉了它每个序列到底有多少 K/V，以及它们的位置。
- **统一处理：** 无论是预填充（Prefill）还是解码（Decoding），在 PagedAttention Kernel 看来都是一样的：都是一个 Query 和一个指向其历史 K/V 的 Block Table。这使得 vLLM 可以**将 Preill 和 Decoding 请求混合在同一个批次中处理**，极大地提高了 GPU 利用率。

## Reference

\[^orca\]: [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)
\[^vllm\]: [vllm](https://github.com/vllm-project/vllm/tree/main)
