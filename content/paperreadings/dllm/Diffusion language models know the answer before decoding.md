---
title: DIFFUSION LANGUAGE MODELS KNOW THE ANSWER  BEFORE DECODING 论文精读
created: 2025-12-29
update:

tags:

- LLMInference

---
# DIFFUSION LANGUAGE MODELS KNOW THE ANSWER  BEFORE DECODING
香港理工大学、达特茅斯学院、Google DeepMind 等组织发表
## Abstract

随着扩散语言模型（DLM）在各个领域的快速发展，其已成为自回归（AR）模型有力的替代方案。与 AR 模型相比，DLMs 的主要优势包括但不限于：高效的**并行解码和灵活的生成顺序**。

然而 DLMs 推理时需要双向注意力计算，而且为了高质量的 token 需要多步的 refine，这使得 DLMs 在推理速度上远远落后于 AR 模型，限制了其在实际应用中的使用。

本文，来自香港理工大学、达特茅斯学院等机构的研究者尝试从一个不同的角度来加速 DLMs 推理，这一思路源于一个长期被忽视却极具潜力的现象：**早期答案收敛**。

通过深入分析，研究者观察到：无论是半自回归重掩码还是随机重掩码场景下，有极高比例的样本在一半左右的解码 step 即可获得正确解码。这一趋势在随机重掩码中尤为显著，以 GSMK 和 MMLU 数据集为例，仅需半数优化步骤即可分别实现 97% 和 99% 的样本正确解码。

受此发现启发，该研究提出了 Prophet，一种无需训练的快速解码策略，该策略专为利用早期答案收敛特性而设计。Prophet 通过持续监控解码过程中**top-2 答案候选之间的置信度差距**，自适应地判断是否可安全地一次性解码剩余所有 token。



实验表明，该方法在保持高质量生成效果的同时，实现了显著的推理加速（最高达 3.4 倍）。

## Introduction 总体介绍

## Background

## Method

## Experiments/Evaluation/Results

## Conclusion

## My Summary

## Reference
