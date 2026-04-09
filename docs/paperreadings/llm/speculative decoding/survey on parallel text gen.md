# A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models

AR-Based
遵循 Draft-and-Verify 范式
[图片]
最大目标是最大化期望的吞吐率
- A 是 accept tokens
- $$L(\mathcal{M})$$denote the latency of a single forward pass for model M.
[图片]
两个关键因素会影响整体吞吐率
- Speculation Accuracy：This focuses on maximizing the numerator, $$\mathbb{E}[A]$$. It requires the  draft model $$\mathcal{M}_p$$to generate token sequences with a high probability of being validated by  the target model $$\mathcal{M}_q$$.
- Drafting Efficiency: This aims to minimize the denominator, particularly the drafting  latency $$L({\mathcal{M}_p})$$. The drafting process must be substantially more lightweight than a full  forward pass of $$\mathcal{M}_q$$.
我们希望 SD 的并行性大于正常的 AR 过程
[图片]
Speeding up Drafting and Verifying
- 高效 Drafter 减少 $$L(\mathcal{M}_p)$$
- 设计先进的 verification structures(token trees) 进行最大化并行，减少$$L(\mathcal{M}_q)$$
- 实现流水线执行，overlap computation and minimize idle time，减少 $$L(\mathcal{M}_p)+ L(\mathcal{M}_q)$$
Efficient Drafting
- 采用单独的较小模型
  - from the same family as the target model. 它不需要额外的培训或架构修改，可以在各种预先训练的模型中快速采用
- 更有效的方式利用目标模型(Self-drafting)
  - 利用目标模型本身提供了双模型系统的替代方案。核心策略涉及“浅”前向传递（仅执行层的子集）以快速生成草稿令牌。然后通过原始的、未修改的模型的完整前向传递来验证这些候选者，从而在重用参数的同时显着减少延迟
  - EAGLE, Medusa: 通过轻量级预测头增强了目标模型，使起草阶段几乎是瞬时的。这是通过避免重新计算昂贵的 transformer 层来实现的；在初始前向传递生成一个令牌后，其隐藏状态将被馈送到这些辅助头以快速生成一系列 draft tokens。
Efficient Verifying
- 虽然通过目标模型的单个前向传递的延迟在很大程度上是固定的，但通过增加该单个步骤中并行处理的 token 数量可以显着提高其效率。  因此，主要目标是设计验证结构（例如 token tree），以最大限度地提高并行性，允许目标模型同时评估大量令牌，从而减少每个生成令牌的有效延迟
- 将验证结构从线性路径发展为并行、多分支格式。基于树的验证将具有公共前缀的多个候选序列合并到单个令牌树中。然后，目标模型在一次前向传递中处理该树，使用专门的树注意掩码在并行计算期间维护因果依赖性。
Efficient Pipeline
标准推测解码 (SD) 会依次执行 draft model 和 target model，从而造成一个模型空闲而另一个模型工作的效率瓶颈。流水线方法通过重叠起草和验证阶段来解决这个问题。核心思想是同时执行它们，将起草延迟 L(Mp ) 隐藏在验证延迟 L(Mq) 后面，从而最大化资源利用率并最小化每个解码步骤的 wall-clock 时间
Improving Acceptance Rate
最大限度地减少计算延迟至关重要，但高效但不准确的系统将无法提供有意义的加速。
- 最大化接受令牌的预期数量 E[A]。这里的中心目标是提高起草令牌的质量和一致性，以便它们更有可能被目标模型验证。
Accurate Drafting
- 静态对齐，通过在推理之前显式训练或微调 drafter 来提高准确性，以更好地模仿目标模型的分布。最初的方法侧重于专门的 draft 模型。
- 向目标模型添加辅助头。这种方法由 Medusa 首创，在冻结模型上训练多个非自回归头，后来由 EAGLE 改进，使用更连贯的自回归头，重用模型的特征。进一步的研究重点是通过更复杂的头部设计来提高草稿质量。
Accurate Verifying
- 准确验证侧重于优化决策标准Verify(·)，使其不那么保守，同时又不影响输出分布的完整性。即使草案看似合理，过于严格的验证规则也可能导致其被拒绝，从而降低 E[A]。因此，我们的目标是设计更具概率的接受标准，可以批准更广泛的有效代币，从而最大化每个验证步骤的收益
- 使用概率验证。它根据目标模型和草稿模型下的概率比接受或拒绝代币，显着提高接受率，同时保留输出分布。

Decomposition-and-Fill
将复杂的生成任务分解为多个独立的子任务，然后并行执行每个组件的生成。此过程通常涉及两个主要阶段
- Task Decomposition，LLM 将整个任务分解为一组结构化的较小的独立组件，以便并行处理
- Parallel Content Filling，multiple LLM calls are made concurrently to flesh  out the details of each component.
不仅通过用较短的并行解码过程替换较长的顺序解码过程来显着减少端到端延迟，而且还可以通过从一开始就强制执行逻辑结构来提高最终输出的质量和一致性
Query-Level Decomposition
系统识别并提取用户初始提示中固有的可并行子任务
侧重于识别和提取用户查询中已存在的可并行子任务。许多现实世界的提示自然包含多个独立的子任务，例如分析项目列表或翻译多个句子的请求。

Answer Structure Planning
模型首先为其响应生成高级大纲，然后同时并行执行多个填充操作
结构规划查询级分解侧重于发现用户提示中现有的可并行结构，而答案结构规划将重点转移到模型输出中的此类结构。
- LLM 首先生成一个计划，通常是一个框架或大纲作为响应的支架。然后可以并行扩展该支架内的每个点。
Multiple Token Prediction(MTP)
通过允许 LLM 在单个步骤中预测多个令牌来实现并行解码。尽管这些模型可以同时预测后续的 N 个令牌
- Draft: 启用 MTP 的 LLM 同时生成 N 个未来代币的序列
- Verify: 同一 LLM 验证起草的令牌是否可接受。可以使用各种策略来执行此验证，例如线性或基于树的验证。
[图片]
Acquiring MTP Capability via Post-training Optimization
这种方法是通过 post-training 来释放其潜在潜力。核心思想是预训练的 LLM 的内部表示可能已经包含有关未来 token 的信息。为了明确提取此信息，将一个或多个新的预测头附加到 LLM 的架构，然后仅在 MTP 目标上微调这些头
Building Native MTP Capability via Pre-training
MTP 直接集成到预训练阶段。  其动机是 LLM 从一开始就预测多个标记可以训练出更高效、更强大的 LLM
- ProphetNet，它专为序列到序列任务而设计，并引入了预训练目标来预测未来的 n 元语法，而不仅仅是下一个单个标记 
- DeepSeek-V3 技术报告详细介绍了如何使用 MTP 目标作为其预训练的核心组成部分
- MiMo 模型还在预训练期间集成了 MTP 目标，这表明这种方法的好处不仅仅是加速，还可以增强 LLM 的核心能力。
Non-AR-Based
不像 ARM，一般是每个 step 并行生成多个 token
- One-shot Generation：一次生成整个序列
- Masked Generation: 从部分或完全屏蔽的序列开始，并通过多个步骤逐步填充 tokens
- Edit-based Generation: 通过一系列学习的编辑操作（例如插入、删除、替换）逐步修改初始序列，从而实现更有针对性的调整，并可能减少局部更改的解码步骤
[图片]
One-shot Generation
并行生成整个序列，但是这个是基于条件独立性假设的，每个 token 是在不知道其他邻居 token 情况下生成的，导致诸如标记重复、遗漏等错误，以及与自回归相比普遍缺乏连贯性，一般有两种改善方法
- 重新引入 token 依赖
- Refine Training Objectives
[图片]
- 这种生成方式通过开发复杂的机制（从高级对齐和依赖建模到新颖的损失函数和训练范例）来弥合与 ARM 的质量差距，同时努力保持并行解码的基本速度优势。
Masked Generation
从完全屏蔽的输入开始迭代填充屏蔽位置来生成内容。
[图片]
Summary
[图片]
多种并行技术的融合兼容矩阵：
[图片]
Combinations Involving Draft-and-Verify
- 多令牌预测 (MTP)。几乎所有 MTP 技术都包含草稿和验证，以减轻与同时预测多个标记相关的准确性损失。在此设置中，草稿模型每步生成 k 个令牌，而主模型有选择地接受或拒绝它们。这种混合通过将解码步骤减少 k 倍来实现显着的加速，同时通过验证保持保真度。例
- 第二个有希望的组合是与 Masked Generation 结合，特别是在基于扩散的 LLM 背景下，其中不存在自回归依赖性，质量控制变得至关重要。集成草稿和验证可以实现推测性令牌起草，然后进行屏蔽生成细化以纠正被拒绝或低置信度的跨度
- 草稿和验证还可以与基于编辑的细化集成，以进一步提高生成质量。在此管道中，推测性验证快速过滤掉低置信度标记，然后编辑模型迭代地纠正残留错误。
Combinations Involving Decomposition-and-Fill
由于分解填充本质上将生成划分为独立的单元，因此它可以与几乎所有其他并行文本生成范例灵活地结合。这是因为它的基于分段的策略充当了自然的包装器：一旦输入被分为 n 个语义连贯的片段（由轮廓、关键短语或结构线索引导），任何下游并行解码方法都可以独立应用于每个片段。
Masked Generation + Edit-Based Refinement
- 利用屏蔽生成，通过并行跨度填充快速生成初始草稿，然后进行基于编辑的细化，逐步纠正错误并提高流畅性。掩码生成，特别是在受扩散启发的方法中，可以快速生成文本的粗糙结构，但由于其非自回归性质，常常会留下不一致或局部不准确的情况。基于编辑的细化通过应用有针对性的插入、删除和替换来补充这一点，逐步将草稿转换为完善的最终输出，而无需重新解码整个序列。
- 该方法是资源密集型的：掩码生成轮次需要全序列处理，并且多个编辑通道进一步增加了计算和内存开销。此外，协调两种不同的解码机制会增加实现的复杂性，尽管有明显的潜在好处，但这可能会阻碍现实世界的部署。
Chanllenges
[图片]
- 系统复杂性和推理速度的权衡
- 质量和速度之间的基本权衡 
- 高熵场景中忽略的依赖性
- 现有优化生态系统的冲突
  - Dllm 使用 block-wise 推理方法复用 kv cache