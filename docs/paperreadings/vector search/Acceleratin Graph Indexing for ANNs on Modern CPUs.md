---

title: Acceleratin Graph Indexing for ANNs on Modern CPUs 论文阅读笔记
created: 2025-06-10
update:
comments: true
description: Acceleratin Graph Indexing for ANNs on Modern CPUs：

- Write optimized LSM-Tree for Graph Index, just the bottom layer stored in disk, other layers in memory
- Locality-Aware graph reordering in disk
- Sampling-Guided traversal by probabilistic routing

katex: true
tags:

- Paper Notes
- Vector Search

---

# Highlights

# Potential Problems

1. **Terrible Experiment Settings**: iQAN use 64 cores and Parlay use 128 cores, but this paper only use 24 cores, so the performance may not be optimal.
