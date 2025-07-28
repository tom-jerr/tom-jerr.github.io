---
title: PDX: A Data Layout for Vector Similarity Search
date: 2025/7/9
update:
comments: true
description: PDX: A Data Layout for Vector Similarity Search
  - The design of PDX, a new data layout for vectors alongside PDXearch: a framework to perform pruned VSS dimension-by-dimension.
  - The design and evaluation of PDX-BOND: A VSS algorithm that leverages the PDX layout to visit first the most relevant dimensions relative to the incoming query.
  - To incorporate hidden dimensions not embedded into  vectors (e.g., temporal information), we propose a new  proximity graph construction algorithm and a graph  memory layout optimization.

katex: true
tags:
  - Vector Search
---

# Perspectives

# Core Ideas

## PDX Data Layout

PDX stores dimensions in **a vertical layout**, allowing efficient dimension-by-dimension distance calculation, more opportunities for SIMD execution, and better memory locality for search algorithms that prune dimensions.

![](img/PDX.png)

### Distance kernels auto-vectorization

![](img/pdx_calc.png)
