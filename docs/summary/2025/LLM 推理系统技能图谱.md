# LLM Inference System Technique Guided Book

## Overview

| 层级     | 名称                                       | 说明                     | 代表技术 / 模块                                                                                                              |
| ------ | ---------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **L0** | **Hardware Layer（硬件层）**                  | 提供算力与带宽的物理基础           | GPU (H100, MI300, Ascend), HBM, NVLink, PCIe, InfiniBand                                                               |
| **L1** | **CUDA Runtime & Kernel Layer（计算运行时层）**  | 封装硬件资源的编程接口与并行执行模型     | CUDA Core, Tensor Core, CUDA Graph, Stream, Kernel Launch, Memory Hierarchy (global/shared/register)                   |
| **L2** | **Distributed Communication Layer（通信层）** | 提供多 GPU/多节点协同机制        | NCCL, RCCL, UCX, RDMA, NVLink/NVSwitch, collective ops (AllReduce/AllGather/Broadcast)                                 |
| **L3** | **Model Execution Layer（模型执行层）**         | 负责执行 Transformer 等模型计算 | FlashAttention, Fused MLP, Rope, kv-cache, Quantization kernel, Triton / TileLang / CUTLASS                            |
| **L4** | **System Algorithms Layer（系统算法层）**       | 高层推理与优化算法              | Speculative decoding, Paged KV cache, Continuous batching, Prefill/decode pipeline, Dynamic batching                   |
| **L5** | **Parallelism Strategy Layer（并行策略层）**    | 负责大模型在多 GPU 上的划分与同步    | TP（Tensor Parallelism）, PP（Pipeline Parallelism）, DP（Data Parallelism）, MoE Parallel, ZeRO/FSDP, sequence parallel     |
| **L6** | **Serving & Scheduler Layer（服务调度层）**     | 推理系统编排、资源调度、请求队列管理     | vLLM / TensorRT-LLM / TGI / SGLang, token scheduler, memory pool, request batching, graph capture, speculative serving |
| **L7** | **Application Layer（应用层）**               | 面向最终用户或下游系统的接口层        | API, RESTful service, chat interface, RAG integration, caching system                                                  |
## Details

## **L0. Hardware Layer（硬件层）**

> 目标：理解底层算力结构、访存瓶颈与通信拓扑，为性能分析和优化奠定基础。

### 📚 核心知识点

- GPU 计算单元：SM、Warp、Thread、Block、Tensor Core
    
- 内存层次结构：Register / Shared / L2 / HBM
    
- Memory Bandwidth 与 Compute Bound 分析
    
- GPU 拓扑：PCIe / NVLink / NVSwitch / InfiniBand
    
- NVLink/NVSwitch 拓扑对 AllReduce 性能的影响
    
- GPU 性能建模：Roofline、Occupancy、Throughput、Latency
    

### 🧩 工具

- `nvidia-smi`、`nvcc --ptxas-options=-v`
    
- Nsight Compute / Nsight Systems / CUPTI / perf
    
- DCGM（NVIDIA GPU telemetry）
    
- NVIDIA-SMI topo 查询 GPU 互联结构
    

### 🔍 推荐阅读

- NVIDIA GPU Architecture Whitepapers (Volta → Hopper)
    
- “Roofline Model for GPUs” (Williams et al., 2009)
    

---

## **L1. CUDA Runtime & Kernel Layer（计算运行时层）**

> 目标：能编写/优化 GPU kernel，理解 CUDA 编程模型、并行执行与内存管理。

### 📚 核心知识点

- CUDA 编程模型（thread/block/grid，SIMT）
    
- Warp 执行、分支发散、同步机制（`__syncthreads()`、barrier）
    
- CUDA Stream 与 Graph 执行机制（减少 launch overhead）
    
- Kernel fusion、kernel launch overhead
    
- Memory management：Pinned memory / Unified memory / Async copy / Memory pool
    
- Tensor Core MMA 指令、CUDA WMMA、CUTLASS 基础
    
- Kernel profiling & warp stall 分析
    

### 🧩 工具

- Nsight Compute, Nsight Systems
    
- CUDA Graph capture API
    
- CUPTI event counters
    

### 🔍 源码参考

- [CUTLASS](https://github.com/NVIDIA/cutlass)（官方算子模板库）
    
- [Tiny-CUDA-NN](https://github.com/NVlabs/tiny-cuda-nn)
    
- [FlashAttention CUDA kernel](https://github.com/Dao-AILab/flash-attention)
    

### 🔍 推荐论文

- **FlashAttention**: “Fast and Memory-Efficient Exact Attention” (Dao et al., 2022)
    
- **Kernel fusion**: “A study of Deep Learning Operator Fusion” (Google XLA team)
    

---

## **L2. Distributed Communication Layer（通信层）**

> 目标：理解多 GPU / 多节点间通信机制，掌握高效 AllReduce / Scatter / Gather。

### 📚 核心知识点

- NCCL 通信原语：AllReduce / AllGather / ReduceScatter / Broadcast
    
- Ring vs Tree 拓扑算法
    
- Overlap compute & comm（通信与计算重叠）
    
- CUDA-aware communication & RDMA
    
- Hierarchical communication（Node-local + Inter-node）
    
- NCCL group、stream group、collective fusion
    
- InfiniBand, UCX, GPUDirect RDMA
    

### 🧩 工具

- `nccl-tests` / NCCL Debug env vars (`NCCL_DEBUG=INFO`)
    
- Nsight Systems 查看通信重叠
    
- `ibstat`, `nvidia-smi topo -m`
    

### 🔍 源码参考

- [NCCL](https://github.com/NVIDIA/nccl)
    
- [DeepSpeed ZeRO](https://github.com/microsoft/DeepSpeed)
    
- [Megatron-LM communication utils](https://github.com/NVIDIA/Megatron-LM)
    

### 🔍 推荐论文

- **Baidu Ring AllReduce** (Li et al., 2017)
    
- **Megatron-LM** (Shoeybi et al., 2019)
    

---

## **L3. Model Execution Layer（模型执行层）**

> 目标：理解 Transformer 的推理计算图及其算子优化。

### 📚 核心知识点

- Transformer 结构（Attention、FFN、LayerNorm、Rope）
    
- FlashAttention / Fused MLP / QKV projection 优化
    
- KV Cache 管理（paged、dynamic、hierarchical）
    
- 量化与算子融合：INT8 / FP8 / SmoothQuant / AWQ
    
- Triton / TileLang / CUTLASS 编写高性能算子
    
- Kernel autotuning（meta-scheduler）
    

### 🧩 工具

- Triton profiler (`triton.testing`)
    
- PyTorch Profiler / TensorRT Profiler
    
- vLLM memory profiler
    

### 🔍 源码参考

- [TileLang](https://github.com/tilelang/tilelang)（比 Triton 更底层的调度 DSL）
    
- [TVM](https://github.com/apache/tvm)（算子编译框架）
    
- [vLLM kernel repo](https://github.com/vllm-project/vllm)
    
- [TensorRT-LLM kernels](https://github.com/NVIDIA/TensorRT-LLM)
    

### 🔍 推荐论文

- **FlashAttention-2** (Dao et al., 2023)
    
- **PagedAttention** (vLLM, 2023)
    
- **SmoothQuant** (Xiao et al., 2022)
    
- **AWQ** (Lin et al., 2023)
    

---

## **L4. System Algorithms Layer（系统算法层）**

> 目标：理解推理阶段的调度算法与 memory 管理逻辑。

### 📚 核心知识点

- Continuous batching（动态批处理）
    
- Speculative decoding / draft verification
    
- Prefill & Decode pipeline
    
- KV cache eviction / PagePool / Cache table 管理
    
- CUDA graph 复用（graph capture）
    
- Dynamic shape inference
    
- Context parallel / attention mask 动态构建
    
- Memory manager / Arena allocator
    

### 🧩 工具

- Nsight Systems timeline 分析
    
- vLLM profiling hooks (`--enable-memory-profile`)
    
- PyTorch CUDA Graph API
    

### 🔍 源码参考

- [vLLM scheduler](https://github.com/vllm-project/vllm/blob/main/vllm/core/scheduler.py)
    
- [SGLang batch manager](https://github.com/sgl-project/sglang)
    
- [TensorRT-LLM runtime](https://github.com/NVIDIA/TensorRT-LLM/tree/main/tensorrt_llm/runtime)
    

### 🔍 推荐论文

- **vLLM** (Kwon et al., 2023)
    
- **SpecInfer** (2023)
    
- **SGLang** (2024) – dynamic serving with speculative decoding
    

---

## **L5. Parallelism Strategy Layer（并行策略层）**

> 目标：掌握大模型推理的并行策略、通信模式和张量划分方案。

### 📚 核心知识点

- Tensor Parallelism (intra-layer)
    
- Pipeline Parallelism (inter-layer)
    
- Data Parallelism (batch-level)
    
- Expert / Mixture-of-Experts Parallelism
    
- Sequence Parallel / Context Parallel
    
- ZeRO Stage 1–3、FSDP、Parameter Sharding
    
- Load balancing 与 activation checkpointing
    

### 🔍 源码参考

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
    
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
    
- [TensorRT-LLM parallel modules](https://github.com/NVIDIA/TensorRT-LLM)
    

### 🔍 推荐论文

- **Megatron-LM** (Shoeybi et al., 2019)
    
- **DeepSpeed ZeRO** (Rajbhandari et al., 2020)
    
- **GSPMD (Google)** (Xu et al., 2021)
    
- **Alpa** (Zheng et al., 2022)
    

---

## **L6. Serving & Scheduler Layer（推理服务层）**

> 目标：理解 LLM 推理服务如何调度请求、管理内存与资源。

### 📚 核心知识点

- Token-level scheduler（prefill/decode overlap）
    
- Batch padding 与 token streaming
    
- Async engine & request queue
    
- CUDA Graph & Stream Reuse
    
- Memory pool 分配与碎片回收
    
- Multi-model multiplexing（多模型共存）
    
- RESTful API / GRPC / Streaming Output
    
- Profiling 与 SLA 监控
    

### 🔍 源码参考

- [vLLM serving entry (`engine.py`)](https://github.com/vllm-project/vllm)
    
- [TensorRT-LLM Inference Server](https://github.com/NVIDIA/TensorRT-LLM)
    
- [TGI (Hugging Face)](https://github.com/huggingface/text-generation-inference)
    
- [SGLang runtime](https://github.com/sgl-project/sglang)
    

### 🔍 推荐论文

- **vLLM: Easy, Fast, and Cheap LLM Serving** (2023)
    
- **SGLang** (2024)
    
- **SpecInfer** (2023)
    

---

## **L7. Application Layer（应用层）**

> 目标：理解 LLM 推理系统如何与上层应用对接。

### 📚 核心知识点

- Prompt cache / embedding cache / RAG pipeline
    
- Token streaming protocol（WebSocket / HTTP chunk）
    
- Multi-turn session & memory context
    
- Load balancing / autoscaling / failover
    
- Monitoring & observability（Prometheus, Grafana）
    
- Cost optimization & resource scheduling
    

### 🔍 示例项目

- OpenAI API serving architecture
    
- ChatGPT / Claude session management
    
- vLLM + FastAPI / RAG Fusion（LangChain, LlamaIndex）