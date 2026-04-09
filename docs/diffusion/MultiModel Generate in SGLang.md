---

title: MultiModel Generate in SGLang
tags:

- LLMInference
  created: 2025-12-2

---

# MultiModel Generate in SGLang

## Overview

SGLang Diffusion 采用了 **Client-Server** 架构，结合 **多进程 (Multi-Process)** 和 **模块化流水线 (Modular Pipeline)** 设计。

- **Client (DiffGenerator)**: 用户接口，负责发送请求和接收结果。
- **Server (Scheduler & Workers)**: 后端推理服务，由多个 GPU Worker 进程组成。
  - **Rank 0 (Scheduler)**: 主节点，负责接收 Client 请求，并通过分布式广播将任务分发给所有 Worker。
  - **Rank 0 ~ N-1 (GPUWorker)**: 计算节点，负责执行具体的模型推理。

### Component

- DiffGenerator: 用户使用的主要入口类。提供 generate() 接口，处理参数封装、请求发送和结果后处理（如保存视频/图片）。

- Scheduler: 运行在 Rank 0 进程中。它监听 ZMQ 端口接收请求，利用 torch.distributed 将请求广播给所有 GPU Worker。

- GPUWorker: 每个 GPU 对应一个 Worker 进程。它负责初始化分布式环境（TP/SP/CFG Parallel）和构建推理流水线。

- ComposedPipelineBase: 模块化的推理流水线基类。它将推理过程拆分为多个 Stage。

- PipelineStage: 流水线的具体阶段，如 TextEncodingStage (文本编码), DenoisingStage (去噪/DiT), DecodingStage (VAE 解码)。

- ParallelExecutor: 负责执行流水线中的各个 Stage，并处理 Stage 级别的并行策略（如 CFG Parallel）。

- Req: 一个包含请求所有状态（Prompt, Latents, Embeddings 等）的数据类，在流水线各 Stage 间传递。

## 工作流程

#### 第一阶段：初始化 (Initialization)

1. **启动 Server**:
   - 用户初始化 `DiffGenerator`。如果是本地模式，它会调用 `launch_server()` 启动多进程服务。
   - `launch_server()` 根据 GPU 数量启动 N 个进程，每个进程运行 `run_scheduler_process()`。
1. **Worker 初始化**:
   - 每个进程实例化 `GPUWorker]`。
   - **分布式环境**: 初始化 NCCL 通信组，配置 Tensor Parallel (TP)、Sequence Parallel (SP) 和 CFG Parallel。
   - **构建流水线**: 调用 `build_pipeline()`，根据模型路径加载对应的 Pipeline（如 `HunyuanVideoPipeline`）。
   - **加载模型**: Pipeline 加载各个子模块（VAE, Transformer, TextEncoder 等）。
     ```json
     {
       "_class_name": "StableDiffusionPipeline",
       "_diffusers_version": "0.9.0",
       "feature_extractor": [
         "transformers",
         "CLIPImageProcessor"
       ],
       "safety_checker": [
         "stable_diffusion",
         "StableDiffusionSafetyChecker"
       ],
       "scheduler": [
         "diffusers",
         "PNDMScheduler"
       ],
       "text_encoder": [
         "transformers",
         "CLIPTextModel"
       ],
       "tokenizer": [
         "transformers",
         "CLIPTokenizer"
       ],
       "unet": [
         "diffusers",
         "UNet2DConditionModel"
       ],
       "vae": [
         "diffusers",
         "AutoencoderKL"
       ]
     }
     ```
1. **Scheduler 就绪**:
   - Rank 0 进程额外初始化 `Scheduler` 对象，绑定 ZMQ 端口，进入事件循环 `event_loop()` 等待请求。

#### 第二阶段：推理执行 (Inference Execution)

1. **发送请求**:
   - 用户调用 `DiffGenerator.generate(prompt="...")`。
   - `DiffGenerator` 将参数封装为 `Req` 对象列表。
   - 调用 `_send_to_scheduler_and_wait_for_response()`，实际上通过 `sync_scheduler_client.forward()` 将请求序列化并通过 ZMQ 发送给 Rank 0 的 Scheduler。
1. **任务分发**: `recv_reqs()`
   - **Rank 0**: `Scheduler` 收到请求。
   - **广播**: `Scheduler` 调用 `broadcast_pyobj()`，利用 `torch.distributed.broadcast` 将 `Req` 对象广播给所有 GPU Worker（包括它自己）。
1. **流水线执行**:
   - 所有 Worker 调用 `GPUWorker.execute_forward(reqs)]`。
   - **Pipeline Forward**: 进入 `Pipeline.forward()`，委托给 `ParallelExecutor`。
   - **Stage 执行**: `ParallelExecutor` 按顺序执行各个 Stage（如 `TextEncodingStage` -> `DenoisingStage` -> `DecodingStage`）。
     - **并行处理**: 对于支持 CFG Parallel 的 Stage，Executor 会协调不同 Rank 分别计算 Positive 和 Negative Prompt，然后同步结果。对于 TP/SP，则在模型层内部处理。
   - **状态更新**: 每个 Stage 修改 `Req` 对象（例如填充 `req.latents`）。
1. **结果返回**:
   - 推理完成后，生成的结果（如生成的 Tensor）存储在 `Req` 对象中。
   - **Rank 0**: `Scheduler` 将结果封装为 `OuputBatch`，通过 ZMQ 发回给 Client。

#### 第三阶段：后处理 (Post-Processing)

1. **Client 接收**: `DiffGenerator` 收到 `OutputBatch`。
1. **保存结果**: 调用 `post_process_sample` 将 Tensor 转换为图像或视频帧，并保存到磁盘。

## 并行策略 (Parallelism)

SGLang Diffusion 支持多种并行方式以加速生成：

- **Tensor Parallel (TP)**: 将大模型（如 DiT）的权重切分到多卡。
- **Sequence Parallel (SP)**: 将长序列（视频 Token）切分到多卡。
- **CFG Parallel**: 将 Classifier-Free Guidance 的两次前向计算（有条件和无条件）分配到不同的 GPU 组并行执行，几乎能带来 2 倍加速。
- **Data Parallel(DP)**: 处理不同的图片
