---
title: "OneLLM: One Framework to Align All Modalities with Language"
tags:
  - LLM
  - MoE
  - multimodal
---

> [!note] 文章信息
> - 作者：Jiaming Han1,2, Kaixiong Gong1,2, Yiyuan Zhang1,2, Jiaqi Wang2, Kaipeng Zhang2, Dahua Lin1,2, Yu Qiao2, Peng Gao2, Xiangyu Yue1†, (1MMLab, The Chinese University of Hong Kong, 2Shanghai Artificial Intelligence Laboratory)
> - arXiv：[:octicons-link-24:](https://arxiv.org/abs/2312.03700)
> - 项目主页：[:octicons-link-24:](https://onellm.csuhan.com/)
> - 代码：[:octicons-link-24:](https://github.com/csuhan/OneLLM)

> 特别之处在于，使用**同一个框架**对多达**八种** [^1] 模态进行了对齐。采用的方法是渐进式的对齐，即先实现 vision 和 language 的对齐，然后再逐个对齐其他模态

> [!note]- Abstract
> Multimodal large language models (MLLMs) have gained significant attention due to their strong multimodal understanding capability. However, existing works rely heavily on modality-specific encoders, which usually differ in architecture and are limited to common modalities. In this paper, we present **OneLLM**, an MLLM that aligns **eight** modalities to language using a unified framework. We achieve this through a **unified multimodal encoder** and **a progressive multimodal alignment pipeline**. In detail, we first train an image projection module to connect a vision encoder with LLM. Then, we build a **universal projection module (UPM)** by mixing multiple image projection modules and dynamic routing. Finally, we progressively align more modalities to LLM with the UPM. To fully leverage the potential of OneLLM in following instructions, we also curated **a comprehensive multimodal instruction dataset**, including 2M items from image, audio, video, point cloud, depth/normal map, IMU and fMRI brain activity. OneLLM is evaluated on 25 diverse benchmarks, encompassing tasks such as multimodal captioning, question answering and reasoning, where it delivers excellent performance.

## Structure

![模型架构](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241208182224803.png)

=== "结构说明"

    1. **Modality Tokenizer**: (lightweight) 2D/1D convolution layer，将输入信号转换为 tokens
    2. **Universal Encoder**: a *frozen* VLM，如 CLIP 等，提取高维特征
    3. **Universal Projection Module (UPM)**: several projection experts and modality routers，将输入信号和 language 对齐

=== "稍详细的解释"

    1. **Lightweight Modality Tokenizers**：对于 image、video 等，直接使用一层 2D convolution layer；其他的先转化为 2D 或者 1D 的序列，再使用相应的 convolution layer。输入记为 $x_{m \in \mathbb{R}^{L\times D}}$，$m \in \mathcal{M}$ 为某种模态
    2. **Universal Encoder**：一个 *frozen* pretrained VLM，这里是 CLIP-ViT。对于 video，将每帧依次输入之后，得到的结果取平均 [^2]。
    3. **Universal Projection Module**：
        1. 一部分是 MoE 结构 [^3] $\{ P_{k} \}_{k=1,2,\dots,K}$，每个 expert 由几层 transformer layers 组成，且都用 image-text data 做预训练。当每次扩展更多的模态时，只需要新增加几个 experts 训练即可。
        2. 另一部分是 router $R$，是一个 MLP，用来控制每个 expert 的贡献大小，因此是 *soft* 的 [^4]。
        3. 此外还需要加上可训练的 modality tokens $\{ q_{m} \}_{m\in \mathcal{M}}$ 来表征模态种类，每个模态都有几个 tokens。
        4. 最后得到 $\mathbf{\bar{q}}_{m}$ 作为所有模态的信息汇总输入到 LLM 中
    4. **LLM**：使用 LLaMA2，输入为 $\mathbf{\bar{q}}_{m}$ 和 text prompt，并且总是将 $\bar{\mathbf{q}}_{m}$ 放在开头

    $$
    \begin{align}
    [\bar{\mathbf{q}}_{m},\bar{\mathbf{x}}_{m}] & =\text{UPM}([\mathbf{q}_{m},\mathbf{x}_{m}]) = \sum_{k=1}^{K} \mathbf{w}_{m,k} \cdot P_{k}([\mathbf{q}_{m},\mathbf{x}_{m}]) \\
    \mathbf{w}_{m} & =\sigma \circ R_{m}([\mathbf{q}_{m},\mathbf{x}_{m}]) \in \mathbb{R}^{N\times K}
    \end{align}
    $$

## Train

=== "大致过程"

    1. **Alignment Stage**: 只训练 tokenizers 和 UPM，冻结 LLM
    2. **Instruction Tuning Stage**: 只训练 LLM，冻结其他部分

=== "稍详细的过程"

    1. **Begin with vision LLM**：使用 pretrained CLIP-ViT 作为 image encoder，几层 transfomer layers 作为 image projection module，LLaMA2 作为 LLM。使用大量 image-text data 预训练之后，projection module 学会将 visual representations 映射到 LLM 的 embedding space 里面
    2. **Align with more modalities**：将 pretrained CLIP-ViT 作为 *universal encoder*；使用记过 image projection experts 作为 *universal X-to-language interface*，并设计一个 *dynamic router* 来控制不同 expert 的权重，从而搭建出 UPM。然后逐个的加入新的模态进行对齐。
    1. 在加入新模态时，为了避免 *catastrophic forgetting*，从之前的数据和新模态的数据进行等概率的采样输入
    2. 分为几个阶段进行训练：image $\to$ video, audio, point cloud $\to$ depth/normal map, IMU, fMRI
    3. **Instruction Tuning**：参考原文。

=== "更详细的过程"

    参考原文

## Results

> [!note]- Results
> ![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241208213544698.png)
> ![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241208213615615.png)

## Ablation Study

- Joint training 更有利于 *data-scarce* 的模态
- Image-text alignment 有助于其他模态和 text 的对齐，例如 UPM 在随机初始化权重时效果更差
- 三个 experts 就可以同时处理所有模态了，experts 更少或更多不会带来明显的提升
- soft router 表现更好

> [!note]- Results
> ![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241208213647682.png)

[^1]: 包括 image, video, audio, point clouds, depth/normal map, Inertial Measurement Unit (IMU), functional Magnetic Resonance Imaging (fMRI)，它们都和 language 进行对齐
[^2]: 原文提到，concatenation 等其他方法可能效果更好，这里是为了效率考虑使用取平均的策略
[^3]: 原文提到，尽管单个 expert 也可以承担相同的功能，但是经验表明 MoE 更有效且 scalable
[^4]: 原文提到，也尝试了使用 constant router 和 sparse router

