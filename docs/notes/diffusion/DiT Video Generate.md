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

## SGLang Diffusion 推理过程

### UniPC(Unified Predictor-Corrector) Scheduler
在log-SNR空间下，求解下面的 ODE；实际上它是给定一个当前状态，求解下一个 step 向哪个方向迭代的问题
- 将所有 predication 方式（ε/x₀/v/flow）统一成 $\hat{x}_0$，方便后续计算
- 求解 $\hat f_\theta(x, \lambda)$
**ODE（log-SNR）**
UniPC 不直接在 t 或 σ 上走，而是用  $\lambda(t) = \log(\alpha_t / \sigma_t)$

实际上，无论模型预测什么（ε / x₀ / v / flow），都可以统一成：
$$
\frac{dx(\lambda)}{d\lambda} = f_\theta(x(\lambda),\lambda)
$$
其中：
$$
x = \alpha x_0 + \sigma \varepsilon
$$
- $\lambda$ 单调递减（去噪过程）
- $x(\lambda)$ 是 $\lambda$ 时刻的 latent
- $f_\theta$ 由模型输出换算得到

**Predictor（UniP）** 

predication 的统一
  - ε-pred（DDPM 经典）: ε-pred 是**噪声空间**的坐标。
    - 模型预测：  
  $$
  \varepsilon_\theta
  $$

    - 正向过程：  
  $$
  x = \alpha x_0 + \sigma \varepsilon
  \Rightarrow
  x_0 = \frac{x - \sigma \varepsilon_\theta}{\alpha}
  $$


  - v-pred（Imagen / SDXL）: v-pred 是 $(x,\varepsilon)$ 的正交旋转坐标  
    - 定义：  
  $$
  v := \alpha \varepsilon - \sigma x_0
  \Rightarrow
  x_0 = \alpha x - \sigma v
  $$

    - 优点：高/低噪区数值更平衡，训练更稳定。

  - flow-pred（Flow Matching）: flow-pred 是**几何意义最直接的速度场**。
    - 模型预测：  
  $$
  v_\theta \approx \frac{dx}{d\sigma}
  $$

    - 正向线性插值：  
  $$
  x = (1 - \sigma)x_0 + \sigma z
  \Rightarrow
  x_0 = x - \sigma v_\theta
  $$


实际上，这些 predication 是**同一个向量场在不同基下的坐标表示**，我们只需要得到 $\hat{x}_0$，后续 ODE 积分完全一致。

$$
\text{Any prediction} \longrightarrow \hat{x}_0
$$
**Predictor 实际预测**
$$
x_{n+1}=x_n+\int_{\lambda_n}^{\lambda_{n+1}}f(x(\lambda),\lambda)\,d\lambda
$$

我们无法直接计算这个积分，因此用数值方法近似它，用离散点进行插值：
$$
x_{n+1} = \frac{\sigma_n}{\sigma_{n+1}} x_n - \alpha_{n+1}\Phi(h) - \alpha_{n+1}B(h)\sum_i \rho_i D_i
$$

然后算：
$$
x_{n+1}^{(p)}=x_n+\int_{\lambda_n}^{\lambda_{n+1}}\tilde f(\lambda)\,d\lambda
$$


**Corrector（UniC）**
用新得到的离散值 $m_{n+1}$ 和 历史离散值 $m_i$ 构造一个更好的插值函数$\int_{\lambda_n}^{\lambda_{n+1}} f(\lambda)\,d\lambda$，修正用旧信息构造的近似积分


$$
x_{n+1}^{(c)} = x_{n+1}^{(p)} - \int_{\lambda_n}^{\lambda_{n+1}} \bigl(\tilde f(\lambda)-f(\lambda)\bigr)\,d\lambda
$$

### UniPCMultistepScheduler 代码走读
- `timesteps`：离散索引
- `sigmas`：通过 $x = \alpha x_0 + \sigma \varepsilon$ 求出的噪声尺度，后续用于求解 `solver` 的时间步 $\Delta \lambda$，其中 $\lambda = \log(\frac{\alpha}{\sigma})$
- `solver`：在 log-SNR 空间下做 Predictor-Corrector 积分

**step 函数**
- 初始化 scheduler 当前的 `step_index`(当前 scheduler 的索引)
- 通过 `convert_model_output()` 将各类模型(flow-pred etc.)输出转换为统一的 $\hat{x}_0$ 格式
- 使用数组维护一个滑动窗口，存储最近 `solver_order` 个时间步的模型输出和时间步，用于后续 UniP 的显示多步插值以及 UniC 的滑动窗口 + 隐式新点的修正
  - 刚开始，滑动窗口未填满，使用 `self.lower_order_nums` 记录当前已填满的数量
  - 需要注意，靠近最后几步的时候，传入 UniP 的阶数不能比剩余步数大
- `prev_sample` 维护新的 latent
```python
def step(
    self,
    model_output: torch.Tensor,
    timestep: int | torch.Tensor,
    sample: torch.Tensor,
    return_dict: bool = True,
) -> SchedulerOutput | tuple:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
    the multistep UniPC.

    Args:
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`int`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        return_dict (`bool`):
            Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

    Returns:
        [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to call 'set_timesteps' after creating the scheduler"
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    use_corrector = (
        self.step_index > 0
        and self.step_index - 1 not in self.disable_corrector
        and self.last_sample is not None
    )

    model_output_convert = self.convert_model_output(model_output, sample=sample)
    if use_corrector:
        sample = self.multistep_uni_c_bh_update(
            this_model_output=model_output_convert,
            last_sample=self.last_sample,
            this_sample=sample,
            order=self.this_order,
        )

    for i in range(self.config.solver_order - 1):
        self.model_outputs[i] = self.model_outputs[i + 1]
        self.timestep_list[i] = self.timestep_list[i + 1]

    self.model_outputs[-1] = model_output_convert
    self.timestep_list[-1] = timestep

    if self.config.lower_order_final:
        this_order = min(
            self.config.solver_order, len(self.timesteps) - self.step_index
        )
    else:
        this_order = self.config.solver_order

    self.this_order = min(
        this_order, self.lower_order_nums + 1
    )  # warmup for multistep
    assert self.this_order > 0

    self.last_sample = sample
    prev_sample = self.multistep_uni_p_bh_update(
        model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
        sample=sample,
        order=self.this_order,
    )

    if self.lower_order_nums < self.config.solver_order:
        self.lower_order_nums += 1

    # upon completion increase step index by one
    self._step_index += 1

    if not return_dict:
        return (prev_sample,)

    return SchedulerOutput(prev_sample=prev_sample)
```

**multistep_uni_p_bh_update 函数**
实际上是上述 UniP 公式的实现