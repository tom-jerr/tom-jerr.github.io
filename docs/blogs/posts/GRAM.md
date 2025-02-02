---
title: "GRAM: Multimodal Representation Learning and Alignment"
tags:
  - alignment
  - contrastive-loss
  - LLM
  - multimodal
  - representaion
---

> [!info] 文章信息
> - 文章题目：Gramian Multimodal Representation Learning And Alignment
> - 作者：Giordano Cicchetti∗, Eleonora Grassucci∗, Luigi Sigillo, Danilo Comminiello
> - 机构：Dept. of Information Engineering, Electronics, and Telecomm., Sapienza University of Rome, Italy
> - arXiv：[:octicons-link-24:](https://arxiv.org/abs/2412.11959)
> - 项目主页：[:octicons-link-24:](https://ispamm.github.io/GRAM/)
> - 代码：[:octicons-link-24:](https://github.com/ispamm/GRAM)

> [!note]- Abstract
> Human perception integrates multiple modalities—such as vision, hearing, and language—into a unified understanding of the surrounding reality. While recent multimodal models have achieved significant progress by aligning pairs of modalities via contrastive learning, their solutions are unsuitable when scaling to multiple modalities. These models typically align each modality to a designated anchor without ensuring the alignment of all modalities with each other, leading to suboptimal performance in tasks requiring a joint understanding of multiple modalities. In this paper, we structurally rethink the pairwise conventional approach to multimodal learning and we present the novel Gramian Representation Alignment Measure (GRAM), which overcomes the above-mentioned limitations. GRAM learns and then aligns n modalities directly in the higher-dimensional space in which modality embeddings lie by minimizing the Gramian volume of the k-dimensional parallelotope spanned by the modality vectors, ensuring the geometric alignment of all modalities simultaneously. GRAM can replace cosine similarity in any downstream method, holding for 2 to n modality and providing more meaningful alignment with respect to previous similarity measures. The novel GRAM-based contrastive loss function enhances the alignment of multimodal models in the higher-dimensional embedding space, leading to new state-of-the-art performance in downstream tasks such as video-audio-text retrieval and audio-video classification. The project page, the code and the pretrained models are available at <https://ispamm.github.io/GRAM/>.

> [!note]- Conclusion
> In conclusion, we presented GRAM, a fundamentally new measure for multimodal representation
>learning and alignment that operates in the higher-dimensional space spanned by all modality embeddings. By modeling the alignment through the volume of a parallelotope formed by k modality vectors, GRAM captures richer semantic relationships than traditional pairwise methods like cosine similarity. Furthermore, we introduced a novel GRAM-based contrastive loss, which leverages this geometric alignment to guide multimodal models in shaping a unified embedding space. The model pretrained using our loss outperform state-of-the-art methods by significant margins across multiple tasks, confirming that GRAM generalizes well to a wide range of modalities and tasks. This work represents a significant advancement in multimodal representation learning and alignment by addressing the limitations of pairwise alignment and providing a mathematically grounded, flexible solution for aligning any number of modalities. We believe that GRAM opens up new directions for the field, offering a more powerful and general framework for multimodal understanding and providing new insights into the structure of multimodal spaces.

## 提出背景

- 现有多模态对齐基于某个 **anchor modality**，使用余弦相似度，不能保证 **non-anchor modalities** 的对齐
- 当需要多种模态进行任务时，因为 **non-anchor modalities** 的没有精确对齐，在复杂的场景下会失败

## 主要贡献

- 提出 **GRAM**，一种用于**多模态表征学习和对齐**的**相似度评估方法**。使用由*所有*模态向量张成的超平行体体积进行计算，并可以在任何下游任务中代替余弦相似度
- 提出一种**基于 GRAM 的对比损失函数**，可用于预训练或微调现存任意多模态模型，构建出一个统一的、对齐的嵌入空间
- 还可以用作**量化的 metrics** 来评估多模态模型
- 数学上证明了 **GRAM 从 2 维到 $n$ 维的扩张**，并在实验上展示了 GRAM 的能力
- 超越了 SOTA 的模型，并且在多个下游任务上取得进步

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241221090818449.png)

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241221094018937.png)

## 余弦相似度

对于两种模态 $\mathcal{M}_{i},\mathcal{M}_{j}$：

$$
\cos(\theta_{ij})=\frac{\langle M_{i},M_{j} \rangle}{\lVert M_{i} \rVert \lVert M_{j} \rVert }
$$

而对于数量更多的模态，假设 anchor modality 为 $A$，使用对比损失：

$$
\begin{align}
\mathcal{L}_{M2A} & = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathrm{m}_{i}^{\top}\mathrm{a}_{i}/\tau )}{\sum_{j=1}^{B} \exp(\mathrm{m}_{i}^{\top}\mathrm{a}_{j} / \tau ) } \\
\mathcal{L}_{A2M} & = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\mathrm{a}_{i}^{\top}\mathrm{m}_{i}/\tau )}{\sum_{j=1}^{B} \exp(\mathrm{a}_{i}^{\top}\mathrm{m}_{j} / \tau ) }
\end{align}
$$

## GRAM

对于向量 $v_{1},v_{2}\dots,v_{k}$ 其张成的超平行体的体积为

$$
\text{Vol}(v_{1},v_{2},\dots,v_{k})=\sqrt{ \det \mathrm{G}(v_{1},v_{2},\dots v_{k}) }
$$

这里 $\mathrm{G}$ 为 [Gram 矩阵](https://en.wikipedia.org/wiki/Gram_matrix)

$$
\begin{align}
& \mathrm{G}(v_{1},v_{2},\dots,v_{k})  = A^{\top}A \\
& A  = (v_{1},v_{2},\dots,v_{k})
\end{align}
$$

由于 $k$ 往往较小（3-4），所以计算 $\text{Vol}$ 并不会带来昂贵的计算代价

从而构建对比损失函数：

$$
\begin{align}
\mathcal{L}_{D2A} & = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(-\text{Vol}(a_{i},m_{2i},\dots,m_{ki}) / \tau)}{\sum_{j=1}^{K} \exp(-\text{Vol}(a_{j},m_{2i},\dots,m_{ki}) / \tau )} \\
\mathcal{L}_{A2D} & = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(-\text{Vol}(a_{i},m_{2i},\dots,m_{ki}) / \tau)}{\sum_{j=1}^{K} \exp(-\text{Vol}(a_{i},m_{2j},\dots,m_{kj}) / \tau )} \\
\end{align}
$$

- 每个模态的表征 $m_{xj}$ 模长都为 1，$x$ 为模态编号，$j$ 为样本编号
- $a_{i}$ 为 anchor modality 表征，可以是单一模态，也可以是多个模态组合

除此之外，还引入 **Data-Anchor Matching loss**：

$$
\mathcal{L}_{DAM}=\mathbb{E}_{(a,m_{2},\dots,m_{k}) \sim (A,M_{2},\dots,M_{k})} [y\log p_{dam}+(1-y)\log(1-p_{dam})]
$$

最后，总损失函数：

$$
\mathcal{L}_{TOT}=\frac{1}{2}(\mathcal{L}_{D2A} + \mathcal{L}_{A 2 D}) + \lambda \mathcal{L}_{DAM}
$$

论文中取 $\lambda=0.1$

由于 GRAM 的天然性质，它也可以用于评测模型，如：

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241221100156675.png)

## 实验结果

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241221100415018.png)

使用 t-SNE 可视化的结果：

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241221101841153.png)
