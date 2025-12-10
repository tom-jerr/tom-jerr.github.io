---
title: DiT Generate Model in SGLang
tags:
  - LLMInference
created: 2025-12-2
# include:
#    - ai-summary
# ai-summary-config:
#     api: "tongyi"
#     model: "qwen-turbo"
#     prompt: "帮我把下面的内容总结为200字以内的摘要："
---

# DiT Generate Model in SGLang

## 常见 DiT Generate 模型

### Stable Diffusion 3(Image Generate)

> [!NOTE]
> 文本（Text）和图像（Image）是两种本质不同的模态，因此应该使用两套独立的权重来分别处理它们，但在注意力机制（Attention）阶段允许两者进行交互。

1. 双流架构：文本流 $c$ 与图像流 $x$ 各自用独立权重处理，仅在注意力处交互，以更好保留各自模态特征。
2. 全局调制：由时间步 $t$ 与汇聚文本向量构成的 $y$ 经 SiLU+Linear，为两流各自产生 6 组调制参数（$\alpha,\beta,\gamma,\delta,\epsilon,\zeta$），在各子模块中注入时序与语义。
3. 自适应归一化：进入 Attention/MLP 前先做 LayerNorm，再按 $\text{Mod}(u)=\alpha\cdot\text{LayerNorm}(u)+\beta$ 调制，实现可控的缩放与平移。
4. 联合注意力：两流分别生成 Q/K/V，沿序列拼接后做自注意力，完成跨模态信息交换，随后再拆回两流（可选 QK RMS-Norm 提升稳定性）。
5. 门控残差：Attention 用 $\gamma$、MLP 用 $\zeta$ 缩放后与输入做残差相加，便于深层训练并选择性抑制/增强模块贡献。
6. 独立 MLP：两流各自使用 Linear-激活-Linear 的 MLP，并配合 $\delta,\epsilon$ 等进行 AdaLN 调制。

![](img/mmdit.png)

### Qwen-Image(Image Generate)

> [!NOTE]
> 利用 Qwen2.5-VL 实现了极强的语义理解和多模态指令遵循能力  
> 通过 特制的 VAE 解码器 解决了“AI 生成文字乱码”的常见痛点

**整体流程**

```shell
多路输入（System/User prompt，选填图像）→ Qwen2.5-VL 提取条件特征
→ 初始噪声与潜在特征结合并注入时间步 t → 多层 MMDiT 处理
→ UnPatchify -> VAE 解码成图像
```

**条件编码器（Qwen2.5-VL）**
该模型语言-视觉空间已对齐；保留强语言理解；原生支持图像输入（适合编辑/图生图）。用特制 System Prompt，引出最后一层 hidden state 作为条件向量。
**VAE Encoder**
VAE Encoder 将图像从“像素空间”映射到“潜空间（Latent Space）”。这极大地降低了数据的维度，使得 MMDiT 能够高效地进行去噪训练。
**MM-DiT**
类似 SD 模型的 MMDiT，双流（文本/图像）各自独立的 Gate/MLP/Norm，仅在自注意力中交互；时间步 t 经 MLP 生成 Scale/Shift 注入各层 Norm，控制去噪进度。
**MS-RoPE**
解决文本/图像位置编码混淆；将文本视作 2D，并在位置空间与图像做“对角线拼接”，既保留图像分辨率可扩展性，又保持文本侧 1D-RoPE 等价性，提升高分辨率稳定性。

![](img/msrope.png)

![](img/qwen-image.png)

### CogVideoX(Video Generate)

**核心流程**

- 压缩：用 3D Causal VAE 把视频压到潜在空间，得到长序列 z_vision；文本经 T5 编码为 z_text。
- 融合：在序列维度拼接 $z_text ⊕ z_vision$，作为 input 传给多个 Expret Transforme r 块。
- 还原：输出先 Unpatchify 恢复潜在形状，再用 3D Causal VAE 解码成视频。

**关键组件**

- 3D Causal VAE：同时压缩空间与时间，使用因果卷积保证生成的时间顺序一致；通过上下文并行支撑长视频训练；以重建+感知+KL 为主，后期少量 GAN 提升细节。

  ![](img/3dvae.png)

- Expert Transformer：
  - Patchify（切片化）： 3D 因果 VAE 编码出形状为 $T \times H \times W \times C$ 的视频潜在变量（分别代表帧数、高、宽、通道）。这些潜在变量被切片，生成长度为 $\frac{T}{q} \cdot \frac{H}{p} \cdot \frac{W}{p}$ 的序列 $z_{vision}$。
  - 3D-RoPE：将位置编码扩展到(x,y,t)三维，分别用 1D-RoPE 并按通道拼接，提升时空建模能力。
  - Expert Adaptive LayerNrom：文本与视频两套独立调制归一化，缓解跨模态尺度与分布差异。
  - 3D Full Attention：统一在时空维度做混合注意力，借助 FlashAttention 与并行技巧把算力压力降到可接受。

![](img/cogvideox.png)

**为什么要这样设计**

- 3D Causal VAE 的作用：压缩视频数据，减小计算量，同时保证时间顺序一致性，便于后续 Transformer 处理长序列。
- 3D-RoPE 的作用：3D-RoPE 专门为视频设计，告诉模型每个块在空间 $(x, y)$ 和时间 $(t)$ 上的具体位置。
- 3D Full Attention 的作用：同时计算所有帧、所有位置的关系。这让模型能直接捕捉大幅度的运动，保证物体在运动中不走样

### HunyuanVideo(Video Generate)

**架构概览**

- Latent DiT：输入为噪声潜变量 + 文本/图像条件，主干为约 13B 参数的 Transformer。
- 训练目标：采用 Flow Matching，相比扩散损失，生成轨迹更直更稳

![](img/hunyuanvideo.png)

**核心组件**

- 3D Causal VAE：时空压缩率为 $4 \times 8 \times 8$，潜在通道数 16；因果 Conv3D 确保第 t 帧不依赖未来帧，提升时序一致性。
- Text Encoder：使用增加了 bidirectional attention 的 MLLM
  ![](img/hunyuantextencoder.png)
- Diffusion Backbone：先双流 20 层（文本/视频各自建模，无跨流注意力），再单流 40 层（拼接后 Full Attention 深度融合）。
  ![](img/hunyuandit.png)

## DiT 模型的常见组件

### VAE Encoder/Decoder

VAE 负责在像素域与潜在域之间往返映射：Encoder 将图像/视频从像素空间压缩到潜空间（Latent Space），显著降维以便 DiT 高效建模；Decoder 将潜变量还原为像素。

- 压缩比例：图像常见 $8\times8$ 或 $16\times16$；视频常见时空压缩如 $4\times8\times8$（Time×H×W）。
- 2D vs 3D：图像用 2D VAE；视频更偏向 3D VAE（含 3D 卷积）。为保证生成时序正确，常用因果卷积（Temporally Causal Conv），仅依赖当前及过去帧，缓解闪烁（flickering）。
- 损失组合：重建损失（L1/L2）+ 感知损失（LPIPS）+ KL 正则，部分工作会在后期加入少量 GAN 损失以增强细节且抑制网格伪影。
- 训练与工程：编码器常冻结以稳定潜空间；对文本密集场景（海报/PDF）可微调解码器以提升文字还原；Patchify/Unpatchify 用于潜在张量与序列之间的高效转换。

### Text Encoder(文本编码器)

将自然语言转为高维向量（如 $z_{text}$/token embeddings），捕捉对象、属性、关系、风格与指令语义，用于约束/引导生成。

- **条件注入方式**：
  - 拼接式（Concatenation/Joint Attention）：将文本与视觉序列拼接到同一 Transformer 中做自注意力（如 MM-DiT）。
  - 交叉注意力（Cross-Attn）：文本作为 K/V，视觉为 Q，实现条件注意力（常见于 U-Net 时代）。
  - 全局调制（AdaLN）：将池化后的文本向量与时间步 $t$ 合成调制向量 $y$，产出缩放/平移参数注入各层归一化与门控。
- 指令与多语：MLLM（如 LLaVA/Llama、Qwen-VL 等）更擅长长指令、多轮与多模态条件；字形/多语种增强（glyph-aware）有助于中文等含文字场景的一致性。
- CFG 与负提示：通过 classifier-free guidance 与 negative prompt 调节文本对齐强度与多样性。

**主流模型**

- CLIP (OpenAI)：物体/风格/构图表现好，逻辑与长文本较弱。
- T5 (Google)：对长句与复杂逻辑友好，适合严格语义约束。
- MLLM（LLaVA/Llama、Qwen-VL 等）：更强指令遵循与跨模态理解，适合图文混合、编辑与高语义一致性需求。

### DiT (Diffusion Transformer)

基于 Transformer 的扩散/流匹配骨干，负责在潜在空间内逐步去噪或拟合流场，生成与条件一致的潜变量。

- 条件组成：文本/图像（或视频）嵌入 + 时间步嵌入 $t$。时间步经 MLP 产出调制参数（与文本池化向量合成 $y$）用于 AdaLN；支持负提示/CFG。
- 结构：多头自注意力 + 前馈网络，配合门控残差（$\gamma,\zeta$）与分层调制（$\alpha,\beta,\delta,\epsilon$）稳定深层训练。
- 位置编码：图像常用 2D RoPE；视频常用 3D RoPE $(x,y,t)$ 提升长时空依赖；可选 QK-Norm/RMSNorm 提升训练稳定性。
- **优势**：U-Net 以卷积/跳连为主，感受野与全局建模受限；DiT 全局注意力更利于大场景关系、长文本与高分辨率扩展（Sora 以后主流）。

> [!NOTE]
> 以前主流是 U-Net（SD 1.5），如今 DiT（Transformer）因更强的全局建模与可扩展性成为主流（Sora 以后）。

## DiT 生成模型的一般推理流程

### 1. 条件编码(Condition Encoding)

- 文本编码 (Text Encoding):
  用户输入的 Prompt 被送入文本编码器（如 T5-XXL 或 CLIP）。
- 图像编码 (Image Encoding, 可选):
  如果是“图生视频”，首帧图片会经过 VAE Encoder 被压缩成 Latent（潜变量）。

### 2. 去噪循环(Denoising Loop)

> [!WARNING]
> 整个流程中最耗时的部分

- 初始化噪声: 随机生成一个高斯噪声张量，形状通常为 [Batch, Channels, Frames, Height, Width]。

- 迭代去噪 (Scheduler Loop):

  - 标准流程: 循环 50 次（Steps）。每次将当前 Latent 输入 DiT/UNet 模型，预测噪声并减去。

### 3. 解码还原(Decoding)

- VAE 解码 (VAE Decoding):

  去噪完成后的 Latent 是高度压缩的。需要通过 3D VAE Decoder 将其还原为像素空间的视频帧。

- 后处理: 视频帧的拼接、插帧（如有需要）、转码（保存为 MP4）。

## DiT 生成模型优化

### Condition Encoding 优化

文本编码器量化 (Text Encoder Quantization):

fastvideo 支持将 T5-XXL 或 CLIP 以 FP8 或 INT8 精度加载。这可以将 T5 的显存占用从 ~22GB 降低到 ~6GB 左右，且对生成质量影响极小。

CPU Offload (卸载):

由于文本编码只在第一步运行一次，计算完拿到 Embedding 后，fastvideo 会立即将 T5 模型从 GPU 移回 CPU 内存，为后续繁重的 DiT 腾出宝贵的 VRAM。

### Denoising Loop 优化

A. 序列并行 (Sequence Parallelism / SP)这是 fastvideo 的杀手锏。当视频 Token 序列太长（比如 5 秒视频可能有 100k+ tokens），单张卡放不下。DeepSpeed Ulysses (DS-Ulysses): fastvideo 集成了这一技术。它将长序列（Video Tokens）在 Attention 计算维度上切分，分配到多张 GPU 上并行计算。效果： 以前单卡只能跑 2 秒视频，现在 8 卡并行可以跑 16 秒视频。Ring Attention: 对于超长上下文，使用环状通信传递 Key/Value 块，进一步突破显存墙。B. 算子优化 (Kernel Optimization)Flash Attention 2/3: 强制使用最新的 Flash Attention 库，极大加速 Attention 层的计算（$O(N^2)$ 复杂度优化），并显著降低显存占用。Fused Kernels (算子融合): 将 Layernorm、GeLU、Add 等琐碎的小算子合并成一个大算子（Kernel），减少 GPU 读写内存的次数。C. 低精度推理 (FP8 Inference)使用 FP8 (Float8) 格式进行矩阵乘法。相比 BF16/FP16，FP8 的吞吐量翻倍，显存占用减半。D. 时间步蒸馏 (Timestep Distillation)虽然这是模型层面的，但 fastvideo 常配合 LCM 或 Rectified Flow 蒸馏模型，将 50 步循环减少到 4-8 步，直接带来 6-10 倍的速度提升。

### Decoding 优化

Tiled Decoding (空间分块解码):原理： 不一次性解码整张图，而是将 Latent 在空间上切成小块（Tiles，比如 $512 \times 512$），一块块解码，最后拼起来。优化： 为了防止拼接处出现接缝（Seams），fastvideo 会在切块边缘进行重叠处理（Overlap blending）。Temporal Slicing (时间切片解码):原理： 现在的 Video VAE 具有时间压缩性。fastvideo 不会一次解码整个视频的所有帧，而是按“组”（比如每 8 帧一组）进行解码，解码完一组释放显存，再解下一组。VAE Tiling Parallelism (并行分块):如果有多张卡，可以将不同的 Tiles 分发到不同的 GPU 上解码，再汇聚结果，利用多卡加速这一过程。

## sgl-diffusion 的 DiT 推理流程

calculate_shift 和 prepare_mu 的用途
这两个函数用于计算 Flow Matching 调度器 (Scheduler) 的时间步偏移量 (Time Shift,
μ
μ)。

背景: 在基于 Flow Matching 的扩散模型（如 Qwen-Image, Flux 等）中，为了更好地处理不同分辨率的图像生成，通常需要调整时间步（Timestep）的采样分布。
calculate_shift: 这是一个线性插值函数。它根据当前的图像序列长度 (image_seq_len)，在基准长度 (base_seq_len) 和最大长度 (max_seq_len) 之间进行插值，计算出一个对应的偏移值 mu。
较小的图像（序列短）对应较小的 mu (接近 base_shift=0.5)。
较大的图像（序列长）对应较大的 mu (接近 max_shift=1.15)。
prepare_mu: 这是一个辅助函数，用于从请求 (batch) 中提取图像的高宽，计算序列长度，然后调用 calculate_shift 得到 mu 值。
返回值: ("mu", mu)。这个键值对会被传递给 Scheduler 的 set_timesteps 方法，从而动态调整生成过程中的时间步调度。

### Stage Pipeline 设置

```python
def create_pipeline_stages(self, server_args: ServerArgs):
    """Set up pipeline stages with proper dependency injection."""

    self.add_stage(
        stage_name="input_validation_stage", stage=InputValidationStage()
    )

    self.add_stage(
        stage_name="prompt_encoding_stage_primary",
        stage=TextEncodingStage(
            text_encoders=[
                self.get_module("text_encoder"),
            ],
            tokenizers=[
                self.get_module("tokenizer"),
            ],
        ),
    )

    self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

    self.add_stage(
        stage_name="timestep_preparation_stage",
        stage=TimestepPreparationStage(
            scheduler=self.get_module("scheduler"),
            prepare_extra_set_timesteps_kwargs=[prepare_mu],
        ),
    )

    self.add_stage(
        stage_name="latent_preparation_stage",
        stage=LatentPreparationStage(
            scheduler=self.get_module("scheduler"),
            transformer=self.get_module("transformer"),
        ),
    )

    self.add_stage(
        stage_name="denoising_stage",
        stage=DenoisingStage(
            transformer=self.get_module("transformer"),
            scheduler=self.get_module("scheduler"),
        ),
    )

    self.add_stage(
        stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
    )
```

### InputValidationStage

### TextEncodingStage

支持多重编码器 (Multi-Encoder Support)
现在的先进模型（如 SDXL, Flux）通常不只使用一个文本编码器

### DenoisingStage
