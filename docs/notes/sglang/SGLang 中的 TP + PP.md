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

### Qwen2 Model

每个 GPU 进程都会实例化一个 `Qwen2ForCausalLM` 对象，但根据其所在的 **PP Rank** 和 **TP Rank**，加载的内容不同：

1. **Embedding 层 (PP Rank 0)**:
    
    - 只有 **PP Rank 0** 的进程会初始化 `VocabParallelEmbedding`。
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
1. **Transformer Layers (所有 PP Rank)**:
    
    - 使用 [make_layers](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 函数构建层。
    - **PP 切分**: 总层数（例如 32 层）会被均匀分配给 PP 组的各个 Rank。
        - 例如 PP=4，Rank 0 负责 0-7 层，Rank 1 负责 8-15 层，以此类推。
    - **本地层**: 当前 Rank 只实际初始化它负责的那部分 `Qwen2DecoderLayer`。
    - **缺失层**: 不属于当前 Rank 的层被初始化为 [PPMissingLayer](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
    - **TP 处理 (层内)**:
        - **Attention**: QKV 投影层 (`QKVParallelLinear`) 按 Head 切分。
        - **MLP**: Gate/Up 投影层 (`MergedColumnParallelLinear`) 按中间维度切分。
2. **Norm 和 Head 层 (PP Last Rank)**:
    
    - 只有 **最后一个 PP Rank** 会初始化 `RMSNorm` 和 [ParallelLMHead](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
    - **其他 PP Rank**: 初始化为 [PPMissingLayer](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
3. **权重加载与共享**:
    
    - 每个 Rank 只加载自己负责的层和切片的权重。
    - **权重绑定 (Weight Tying)**: 如果模型配置了 `tie_word_embeddings` (即 Embedding 和 Head 共享权重)，且 PP > 1：
        - PP Rank 0 会通过 P2P 通信将 Embedding 权重发送给 PP Last Rank，用于初始化 Head。

---

### 2. 推理执行阶段 (Inference / Forward Pass)

当一个 Batch 的请求到来时，数据流会在 GPU 之间通过流水线传递。

#### A. 输入处理

- 所有 Rank 都会接收到输入元数据（如 [input_ids](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html), `positions`, `forward_batch` 信息），但只有 PP Rank 0 真正拥有原始的 Token 输入数据用于计算。

#### B. 流水线传递 (Pipeline Forward)

流程按 PP Rank 顺序依次执行：

1. **PP Rank 0 (起始阶段)**:
    
    - **Embedding**:
        - 输入 [input_ids](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
        - 执行 [VocabParallelEmbedding](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
        - **TP 动作**: 各 TP Rank 计算部分 Embedding，然后通过 **AllReduce** 聚合，使得每个 TP Rank 获得完整的 Embedding 向量。
    - **Layers (0 ~ N)**:
        - 顺序执行分配给该 Rank 的 Transformer 层。
    - **输出**: 将计算出的 [hidden_states](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) (中间结果) 通过 P2P [send](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 发送给 PP Rank 1。
2. **PP Rank i (中间阶段)**:
    
    - **输入**: 通过 P2P [recv](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 接收上一级 (Rank i-1) 发来的 [hidden_states](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
    - **Layers (M ~ K)**:
        - 顺序执行分配给该 Rank 的 Transformer 层。
    - **输出**: 将更新后的 [hidden_states](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 发送给下一级 (Rank i+1)。
3. **PP Rank Last (最终阶段)**:
    
    - **输入**: 接收上一级发来的 [hidden_states](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
    - **Layers (X ~ End)**: 执行剩余的 Transformer 层。
    - **Final Norm**: 执行 `RMSNorm`。
    - **Logits**:
        - 执行 [ParallelLMHead](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)。
        - **TP 动作**: 类似于 Embedding，输出 Logits 通常需要聚合（或者在采样时处理）。
    - **采样**: 计算概率并采样下一个 Token。

#### C. 层内并行细节 (Inside a Layer with TP)

在每一个 Transformer 层内部，TP 是这样工作的：

1. **Attention 模块**:
    
    - **输入**: 完整的 [hidden_states](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) (所有 TP Rank 都有副本)。
    - **QKV Proj (Column Parallel)**: 每个 TP Rank 只计算一部分 Head 的 Q/K/V。
    - **Attention 计算**: 每个 Rank 独立计算自己那部分 Head 的 Attention。
    - **Output Proj (Row Parallel)**: 每个 Rank 计算部分输出。
    - **AllReduce**: 在 Output Proj 之后，执行一次 **AllReduce (Sum)**，让所有 TP Rank 重新获得完整的 Attention 输出，并加到 Residual 上。
2. **MLP 模块**:
    
    - **Gate/Up Proj (Column Parallel)**: 输入完整，输出被切分（每个 Rank 计算一部分中间特征）。
    - **Activation**: 在切分的数据上独立执行 (如 SiLU)。
    - **Down Proj (Row Parallel)**: 输入是切分的，输出是部分的。
    - **AllReduce**: 执行 **AllReduce (Sum)**，聚合最终结果。