---
title: PANNS: Enhancing Graph-based Approximate Nearest  Neighbor Search through Recency-aware Construction  and Parameterized Search
date: 2025/7/9
update:
comments: true
description: PANNS: Enhancing Graph-based Approximate Nearest  Neighbor Search through Recency-aware Construction  and Parameterized Search
  - Providing an in-depth analysis of the graph-based  ANNS workload, highlighting its optimization space.
  - Offering a more detailed parameterized search strategy that allows for flexible trade-offs between search speed and accuracy.
  - To incorporate hidden dimensions not embedded into  vectors (e.g., temporal information), we propose a new  proximity graph construction algorithm and a graph  memory layout optimization.

katex: true
tags:
  - Vector Search
---

# Perspectives

1. This paper reveals that it is not worthwhile to exploit intra-iteration and intra-query parallelism.
2. Not every iteration in searching contributes equally to the final result, so it is not necessary to wait for all iterations to finish.
   ![](img/profile_anns.png)
3. The paper reveals that the input queries and their ANNS results exhibit a notable temporal correlation in many real-world datasets.

# Core Ideas

## Parameterized Beam Search

![](img/PBS.png)

## Recency-aware Graph Construction

![](img/RAC.png)

## Graph Layout Optimization

### Selective Neighbor Scan

This early termination of neighbor scanning has two advantages:

1. it avoids fetching unpromising vector data from the memory;
2. unpromising distance computations are also circumvented.

![](img/selective_neighbor_scan.png)

### Vertex reordering

To avoid random access and simplify the selection logic, we propose a graph layout optimization. The idea is that, after the graph is constructed, we **sort each vertex’s out-neighbor list based on the timestamp**.

Then, at each iteration of the beam search, only a fraction of the vertex’s out-neighbors is fetched and evaluated.

# Potential Problems
