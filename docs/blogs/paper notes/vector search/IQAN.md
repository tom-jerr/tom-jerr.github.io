---
title: iQAN: Fast and Accurate Vector Search with Efficient  Intra-Query Parallelism on Multi-Core Architectures 论文阅读笔记
date: 2025/7/3
update:
comments: true
description: iQAN: Fast and Accurate Vector Search with Efficient  Intra-Query Parallelism on Multi-Core Architectures：
  - Performing in-depth studies to reveal the root causes of the poor scalability of state-of-the-art vector search algorithms on multi-core architectures
  - Introducing intra-query parallelism optimizations, i.e., path-wise parallelism, staged expansion, and redundancyaware synchronization to accelerate the search.

katex: true
tags:
  - C++
---

# Highlights

# iQAN Algorithm

## Overview

iQAN employs a **hybrid parallelism strategy**:

1. **Global Step (Rough Synchronization)**: The search progresses in global "steps." At the beginning of each global step, the current set of candidate nodes (in the global priority queue S) is divided among the available T workers.

2. **Local Best-First Search (Fine-Grained Parallelism)**: Each worker then independently performs a best-first search on its assigned subset of candidates, using its own local priority queue LS[t]. This local search continues until a condition is met (e.g., the checker indicates a sync, or the worker runs out of local candidates).

3. **Merging**: After local searches, all local priority queues are merged back into the global priority queue.

## Details

This algorithm uses **multi entry points**, by this way, it can achieve **intra-query parallelism**. The algorithm is designed to efficiently utilize multiple cores by allowing each core to work on different parts of the search space simultaneously.

![](img/iqan.png)

1. **Divide all unchecked vertices from S into LS**: This is the **global synchronization step**. The candidates currently in the global queue S that haven't been "checked" (expanded) yet are distributed among the T local queues LS[t]. The "unchecked" status is important to avoid redundant work.
2. **Parallel workers search their local queues LS[t]**: Each worker independently performs a best-first search on its local queue LS[t]. This is the **local best-first search** phase, where each worker explores its assigned candidates without needing to synchronize with others.
3. **Decide whether to continue or synchronize**: The checker(also a worker) computes some **average of the workers' update positions $\overline{u}$**.

   > If the average progress **$\overline{u} \ge L \cdot R$**, it means enough work has been done locally, and a global merge should occur. **doMerge** is set to true to signal this.

   > The checker role rotates among workers to distribute the overhead (**Round-Robin way**).

4. **Merge local queues into global queue S**: If the checker indicates that a merge is needed, all local queues LS[t] are merged back into the global queue S. This is the **global merge step**.
5. **Stop**: Once the while true loop breaks (meaning no more unchecked candidates globally), the algorithm returns the top K nearest neighbors from the final global priority queue S.

# Potential Problems

1. **Terrible Experiment Settings**: iQAN use 64 cores and Parlay use 128 cores, but this paper only use 24 cores, so the performance may not be optimal.
