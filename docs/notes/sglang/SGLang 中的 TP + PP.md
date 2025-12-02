---
title: SGLang 中的 TP + PP
tags:
  - LLMInference
date: 2025/11/22
---

# SGLang 中的 TP+ PP

这里我们以 `Qwen2` 模型为例，开启 PP + TP 分析一下 SGLang 是如何实现模型推理的并行的

## 不同节点的职责

| 节点       | 工作内容                                       |
| -------- | ------------------------------------------ |
| Rank 0   | tokenizer、detokenizer、HTTP服务、调度器、模型 worker |
| Rank > 0 | **只运行调度器 + worker，不处理前端服务**                |

## Initiallize Server

1. `launch_server.py` 中根据 `grpc_mode` 参数决定执行 `serve_grpc()` 或者 `launch_server()`
	> 后续以 `launch_server()` 为例进行讲解

2. 调用 `engine.py` 中的 `_launch_subprocesses()`
	> 这里所有的子进程以 **spawn** 方式派生全新 Python 进程

	- 按照 pp 和 tp 重新映射 GPU Id，然后为每个 GPU 创建一个`mp.Pipe()`
	- 接着启动一个 Scheduler **子进程**，传入 pipe 的写端（使用 `run_scheduler_process()`），然后**父进程保留子进程引用和 pipe 的读端**
	- 多机场景下非 0 节点不参与前端服务，仅负责启动 scheduler 进程并**保持节点健康**，避免重复运行 tokenizer / detokenizer / 接入服务。
	- 0 号节点启动 detokenizer 子进程 by `run_detokenizer_process()`，
	- 启动 TokenizerManager，等待所有的 GPU 都加载了 model 并拥有相同的 `scheduler_info`(By `mp.Pipe()`)
3. 从 `_launch_subprocesses()` 中获得 `tokenizer_manager`  和 `scheduler_info`，直接在主进程启动 HTTP 服务 TokenizerManager 进行请求的接收

😆现在，Tokenizer 进程，Scheduler 进程，Detokenizer 进程可以通过事件循环不停驱动，实现用户请求的处理

### Scheduler 

先创建 Scheduler 对象，在 `__init__` 中进行 TpModelWorker 初始化，DraftWorker 初始化，memory pool 和 memory cache 初始化

然后根据 `server_args` 不同，启动不同的事件循环

```python
if disaggregation_mode == DisaggregationMode.NULL:
	if scheduler.enable_pdmux:
		scheduler.event_loop_pdmux()
	elif server_args.pp_size > 1:
		scheduler.event_loop_pp()
	elif scheduler.enable_overlap:
		scheduler.event_loop_overlap()
	else:
		scheduler.event_loop_normal()
```

### TpModelWorker & ModelRunner

TpModelWoker 的 `__init__()` 中进行了 ModelRunner 的初始化

ModelRunner 的 `__init__()` 调用了 `self.init_torch_distributed()`
- 确认使用的通信后端，这里以 NCCL 为例
- 最终调用 parallel_state.py 中的 `initialize_model_parallel()`。
- 创建全局的 `_TP` (Tensor Parallelism) 进程组。假设 TP=4，GPU 0-3 会被划入同一个 NCCL 通信组。
- 创建全局的 `_PP`(Pipeline Parallelism) 进程组，为每个流水线 stage 创建 1 个独立的通信 group

| i   | ranks = range(i, 8, 4) | PP Stage |
| --- | ---------------------- | -------- |
| 0   | 0,4                    | [0,4]    |
| 1   | 1,5                    | [1,5]    |
| 2   | 2,6                    | [2,6]    |
| 3   | 3,7                    | [3,7]    |
|     |                        |          |
### Detokenizer

实际上 detokenizer 会一直事件循环，从 Scheduler 得到 TODO，解码成 `BatchTokenIDOutput` 传递给 Tokenizer 子进程

```python
def event_loop(self):
"""The event loop that handles requests"""
	while True:
		recv_obj = self.recv_from_scheduler.recv_pyobj()
		output = self._request_dispatcher(recv_obj)
		if output is not None:
			self.send_to_tokenizer.send_pyobj(output)
```

### Parallel Linear in SGLang

在 SGLang中，`Parallel Linear` 是实现 Tensor Parallel (TP) 的核心组件。它通过将巨大的权重矩阵切分到多个 GPU 上，使得每个 GPU 只需计算一部分数据，从而减少显存占用并加速计算。

主要有两种并行方式：**Column Parallel (列并行)** 和 **Row Parallel (行并行)**。通常它们会成对出现（例如在 MLP 中：先 Column 后 Row），以最小化通信开销。

#### ColumnParallelLinear

**数学原理**：
假设线性层运算为 $Y =XA$ 
我们将权重矩阵 $A$ 按**列**切分为 $[A_1​,A_2​,...,A_p​]$。  
每个 GPU 持有 $A_i$​。  
输入 $X$ 是完整的（复制在所有 GPU 上）。  
每个 GPU 计算 $Y_i=XA_i$​。  
输出 $Y$被 切分为 $[Y_1,Y_2,...,Y_p]$，即每个 GPU 得到输出向量的一部分特征。

**典型应用**:

- Attention 的 QKV Projection (`QKVParallelLinear`)。
- MLP 的 Gate / Up Projection (`MergedColumnParallelLinear`)。

#### RowParallelLinear

**数学原理**：
假设线性层运算为 $Y =XA$ 。  
我们将权重矩阵 $A$ 按**行**切分为 $$\begin{bmatrix} A_1 \\ A_2 \\ \dots \\ A_p \end{bmatrix}  $$​​​  
为了匹配矩阵乘法规则，输入 $X$ 也必须按**列**切分为 $[X_1​,X_2​,...,X_p​]$（这正好是 ColumnParallelLinear 的输出格式）。  
每个 GPU 计算 $Y_i=XA_i$​。注意，$Y_i$ 的形状与最终输出 $Y$ 相同，但它只是部分和。  
最终输出 $Y = \sum Y_i$​ ，需要一次 **All-Reduce (Sum)** 操作

**典型应用**：
- Attention 的 Output Projection (`o_proj`)。
- MLP 的 Down Projection (`down_proj`)。

#### ColumnParallelLinear + RowParallelLinear

 **计算量减半，多了一次集群通信（allReduce），中间值的存储大小减半，Input, Weight 减半**

![](img/llm_dp.jpg)
### Qwen2 Model


每个 GPU 进程都会实例化一个 `Qwen2ForCausalLM` 对象，但根据其所在的 **PP Rank** 和 **TP Rank**，加载的内容不同

#### Embedding 层     
- 只有 **PP Rank 0** 的进程会初始化 `VocabParallelEmbedding`(**Row Parallel**)。
- **TP 处理**: 词表 (Vocab) 被切分到 TP 组的各个 GPU 上。每个 GPU 只持有 `VocabSize / TP_Size` 大小的权重。
- **其他 PP Rank**: 初始化为 PPMissingLayer (占位符，不占用显存)。

```python
# perform weight tying for PP
if self.pp_group.world_size > 1 and config.tie_word_embeddings:
	if self.pp_group.is_first_rank:
		self.pp_group.send(
			self.model.embed_tokens.weight, dst=self.pp_group.last_rank
		)
	elif self.pp_group.is_last_rank:
		emb_token_weight = self.pp_group.recv(
			size=(config.vocab_size, config.hidden_size),
			dtype=next(self.model.parameters()).dtype,
			src=self.pp_group.first_rank,
		)
		self.lm_head.weight.copy_(emb_token_weight)
```

#### Transformer Layers (所有 PP Rank)
- 使用 `make_layers` 进行构建。**所有层都会构建，只有本地层会加载权重(即占用显存)**
- **PP 切分**: 总层数（例如 32 层）会被均匀分配给 PP 组的各个 Rank。
	- 例如 PP=4，Rank 0 负责 0-7 层，Rank 1 负责 8-15 层，以此类推。
- **本地层**: 当前 Rank 只实际初始化它负责的那部分 `Qwen2DecoderLayer`。
- **缺失层**: 不属于当前 Rank 的层被初始化为 `PPMissingLayer`。

```python
modules = torch.nn.ModuleList(
	[PPMissingLayer(return_tuple=return_tuple) for _ in range(start_layer)]
	+ get_offloader().wrap_modules(
		(
			layer_fn(idx=idx, prefix=add_prefix(idx, prefix))
			for idx in range(start_layer, end_layer)
		),
		**(offloader_kwargs or {}),
	)
	+ [
		PPMissingLayer(return_tuple=return_tuple)
		for _ in range(end_layer, num_hidden_layers)
	]
)
```

Qwen2DecodeLayer 实际上主要是由 Qwen2Attention，Qwen2MLP，RMSNorm 组成的
1. **Qwen2Attention**:

	- **输入 -> 中间**: 使用 QKVParallelLinear (继承自 **ColumnParallelLinear**)。将输入投影到 Q, K, V。
	- **中间 -> 输出**: 使用 RowParallelLinear。将 Attention 的输出投影回 hidden size。
2. **Qwen2MLP**:
    - **输入 -> 中间**: 使用 MergedColumnParallelLinear (继承自 **ColumnParallelLinear**)。将输入投影到 Gate 和 Up 状态。
    - **中间 -> 输出**: 使用 RowParallelLinear。将激活后的状态投影回 hidden size。
#### LMHead 层 
是一个 **按词表维度切分 (Column Parallel)** 的线性层
- 只有 **最后一个 PP Rank** 会初始化 `RMSNorm` 和 `ParallelLMHead`。
- **其他 PP Rank**: 初始化为 PPMissingLayer。

---

### Inference Process

当一个 Batch 的请求到来时，数据流会在 GPU 之间通过流水线传递。

#### A. 输入处理

- **请求分发**：当请求进入系统时，Scheduler 会将请求对象（包含 `input_ids`）广播给所有的 Rank（包括不同的 PP Rank 和 TP Rank）。
- **Batch 构建**：每个 Rank 上的 Scheduler 都会独立构建 `ForwardBatch` 对象。

只有 PP Rank 0 使用 `input_ids` 进行计算，对于 PP Rank > 0 的节点，它们的输入数据不是来自 `input_ids`，而是来自**上一个 Rank 通过 P2P 通信发送的中间结果**。
```python
if self.pp_group.is_first_rank:
	# Rank 0: 真正使用 input_ids 进行 Embedding 查找
	if input_embeds is None:
		hidden_states = self.embed_tokens(input_ids)
	else:
		hidden_states = input_embeds
	residual = None
else:
	# Rank > 0: 忽略 input_ids，直接使用上一个阶段传过来的 hidden_states
	assert pp_proxy_tensors is not None
	hidden_states = pp_proxy_tensors["hidden_states"]
	residual = pp_proxy_tensors["residual"]
```

#### B. 流水线传递 (Pipeline Forward)

流程按 PP Rank 顺序依次执行：

1. **PP Rank 0 (起始阶段)**:
    
    - **Embedding**:
        - 输入 `input_ids`。
        - 执行 `VocabParallelEmbedding`。
	        -  每个 GPU 都在其本地的小权重矩阵上进行查表操作。
			- 对于属于该 GPU 的 Token，查出的向量是正确的。
			- 对于不属于该 GPU 的 Token，查出的向量是无意义的（垃圾值）。
        - **TP 动作**: 各 TP Rank 计算部分 Embedding，然后通过 **AllReduce** 聚合，使得每个 TP Rank 获得完整的 Embedding 向量。
	        ```python
	        output_parallel = tensor_model_parallel_all_reduce(output_parallel)
	        ```
    - **Layers (0 ~ N)**:
        - 顺序执行分配给该 Rank 的 Transformer 层。
    - **输出**: 将计算出的 `hidden_states` 发送给 PP Rank 1。
1. **PP Rank i (中间阶段)**:
    
    - **输入**: 通过 P2P `recv` 接收上一级 (Rank i-1) 发来的 `hidden_states`。
    - **Layers (M ~ K)**:
        - 顺序执行分配给该 Rank 的 Transformer 层。
    - **输出**: 将更新后的 `hidden_states` 发送给下一级 (Rank i+1)。
2. **PP Rank Last (最终阶段)**:
    
    - **输入**: 接收上一级发来的 `hidden_states`。
    - **Layers (X ~ End)**: 执行剩余的 Transformer 层。
    - **Final Norm**: 执行 `RMSNorm`。
    - **Logits**:
        - 执行 `ParallelLMHead`。
        - **TP 动作**: 类似于 Embedding，输出 Logits 通常需要聚合（或者在采样时处理）。
    - **采样**: 计算概率并采样下一个 Token。

#### C. 层内并行细节 (Inside a Layer with TP)

在每一个 Transformer 层内部，TP 是这样工作的：

1. **Qwen2Attention**:
    
    - **输入**: 完整的 `hidden_states` (所有 TP Rank 都有副本)。
    - **QKV Proj (Column Parallel)**: 每个 TP Rank 只计算一部分 Head 的 Q/K/V。
    - **Attention 计算**: 每个 Rank 独立计算自己那部分 Head 的 Attention。
    - **Output Proj (Row Parallel)**: 每个 Rank 计算部分输出。
    - **AllReduce**: 在 Output Proj 之后，执行一次 **AllReduce (Sum)**，让所有 TP Rank 重新获得完整的 Attention 输出，并加到 Residual 上。
2. **Qwen2MLP**:
    
    - **Gate/Up Proj (Column Parallel)**: 输入完整，输出被切分（每个 Rank 计算一部分中间特征）。
    - **Activation**: 在切分的数据上独立执行 (如 SiLU)。
    - **Down Proj (Row Parallel)**: 输入是列切分的（来自 Activation 的输出），权重按行切分。每个 Rank 计算出部分输出。
    - **AllReduce**: 执行 **AllReduce (Sum)**，聚合最终结果。
![](img/llm_dp.jpg)