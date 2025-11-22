#### Stage 1: 基础 

- [ ] PMPP Ch 1-6 + Ch 10
    
- [ ] CUDA Basics: Tiling, Shared Mem, Coalescing
    
- [ ] Kernel 1: Tiled GEMM (FP16)
    
- [ ] Kernel 2: Warp Reduction
    

#### Stage 2: LLM 特有算子 (新增)

- [ ] **Kernel 3: Fused RMSNorm** (重点：学习读取数据 -> 归一化 -> 写回)
    
- [ ] **Kernel 4: RoPE** (重点：处理复杂的索引计算)
    
- [ ] **概念理解:** KV Cache 的显存布局 (PagedAttention 的物理基础)
    

#### Stage 3: Hopper/H100 进阶 (区分高手的分水岭)

- [ ] **TMA 机制:** 阅读 NVIDIA Hopper Tuning Guide，理解异步拷贝。
    
- [ ] **Triton 实战:** 用 Triton 重写上面的 RMSNorm 和 RoPE（Triton 会自动帮你处理很多 Hopper 的优化）。
    
- [ ] **FP8 概念:** 理解为什么 FP8 需要 Scaling Factor。

#### SGLang PR

**阶段 1：热身 (11月-12月中)**

- **动作：** 在你的 `MiniInfer` 里手写一个简单的 **Speculative Decoding Demo**。
    
- **目的：** 搞懂 Draft Model 怎么跑，Verification 怎么做，KV Cache 怎么回滚。不用太快，跑通逻辑即可。
    

**阶段 2：切入 SGLang (12月中-1月)**

- **动作：** 寻找 SGLang Issues 中关于 **Eagle / Medusa** 或者 **Tree Attention** 的相关任务。
    
- **贡献点：** 可能是用 Triton 写一个更快的 `Tree Verification Kernel`（树状验证算子）。这个算子非常适合用 GPU 并行加速，且正好用上你学的 CUDA 知识。
    

**阶段 3：简历包装 (2月)**

- **简历亮点：** “Implemented a Triton-based Tree Verification Kernel for Speculative Decoding in SGLang, achieving X% speedup.”