# My Idea of Vector Search

1. 分层：底部是一个类似 NSG 的图，上面只有一层通过聚类等方式构建的图。
   > 类似粗粒度索引先获取候选子集，然后用这些子集作为入口点到底部图中进行精确搜索。
2. 并行搜索：类似 iQAN 的思想，使用多入口点进行并行搜索。
   - 考虑优化 iQAN:[Efficient Graph-Based Approximate Nearest Neighbor Search Achieving: Low Latency Without Throughput Loss](http://arxiv.org/abs/2504.20461)
3. 邻居选择：通过新的方式，比如角度距离等新发表的文章来选择邻居，而不是简单的欧氏距离。
4. 使用量化：1-bit 量化
5. 减少计算量：比如通过概率路由或其他不等式剪枝方式来实现

## Vertical layout for HNSW from [PDX: A Data Layout for Vector Similarity Search](https://dl.acm.org/doi/10.1145/3725333)

- dimension-by-dimension search strategy that operates on multiple-vectors-at-a-time in tight loops.

## Hubs idea from [Down with the Hierarchy: The ‘H’ in HNSW Stands for “Hubs”](https://arxiv.org/abs/2412.01940v2)

动态的枢纽识别和边管理策略。

1. 动态枢纽判断: 要判断一个节点是否是枢纽，你需要知道它的 k-occurrence，即有多少个其他节点将其作为 K 近邻之一。在增量构建中，这需要动态更新。

- 维护反向邻居列表 (Reverse Neighbor List)： 对于每个节点 P，不仅存储它的 K 近邻 N(P)，还要维护一个列表 R(P)，记录所有将 P 作为其 K 近邻的节点。

  > 当一个新节点 Q 加入并计算其 K 近邻时，对于 Q 的每个近邻 $P_j$，将 Q 添加到 $P_j$ 的 $R(P_j)$ 列表中。

  > 一个节点的 k-occurrence 值就是其 R(P) 列表的大小。

- Top-N 枢纽： 也可以维护一个当前图中最活跃的 Top-N 个枢纽列表。
  > 频率更新： 随着新节点的加入，定期（例如每加入 M 个节点后）重新评估所有节点的 k-occurrence 值，并更新枢纽列表。

2. 枢纽优先保留策略

- 当一个新节点 Q 加入时，它会连接到现有图中的一些节点。在决定保留哪些边时，可以应用枢纽优先策略：

  1.  初始边连接： 当 Q 加入时，首先为其找到 L 个初始近邻（L 通常大于最终需要的 R）。
  2.  枢纽连接偏好： 在剪枝过程中，如果你有多个候选边可以选择保留，而其中一些边连接到已识别的枢纽，
      - 提高权重： 赋予连接到枢纽的边更高的“优先级”或“权重”，使其在最终的 R 个出度限制中更容易被保留。
      - 额外连接预算： 可以为枢纽分配一个额外的连接预算，允许它们连接更多的边，或者确保它们至少有一定数量的入边（从非枢纽连接到它们的边）。
  3.  双向验证： 如果 Q 的一个近邻 P 是枢纽，并且 P 也将 Q 识别为其近邻，则这条边被认为更重要，更倾向于保留。

- 优化剪枝算法：

  > 在剪枝阶段，当决定从一个节点的邻居列表中移除哪些边时，枢纽优先策略可以介入。例如，在移除边之前，**检查这些边是否连接到枢纽。如果一条边连接到一个高 k-occurrence 的枢纽，即使它的距离略远，也可能被优先保留**，而不是被那些距离更近但连接到“普通”节点的边替代。

  > 角度与枢纽结合： 可以将角度剪枝（确保邻居在空间中均匀扩散）与枢纽偏好结合。例如，即使一条边导致角度较小，如果它连接到一个关键枢纽，也可能被例外保留。
