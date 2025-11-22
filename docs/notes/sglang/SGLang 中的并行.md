---
title: SGLang 中的并行
tags:
  - LLMInference
date: 2025/11/22
---

# SGLang 中的并行

## Server Process

| 节点       | 工作内容                                       |
| -------- | ------------------------------------------ |
| Rank 0   | tokenizer、detokenizer、HTTP服务、调度器、模型 worker |
| Rank > 0 | **只运行调度器 + worker，不处理前端服务**                |

### Initiallize Server with Parrallelism(No DP and EP)

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

先创建 Scheduler 对象，在 `__init__`

实际上会根据 `server_args` 不同，启动不同的事件循环

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

## TP





## PP

## DP