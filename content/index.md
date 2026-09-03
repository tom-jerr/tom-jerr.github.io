---
title: 刘芷溢
description: 刘芷溢的个人主页，记录大模型推理、CUDA、数据库与分布式系统方向的学习与实践。
hide:
  - date
  - navigation
  - toc
home: true
nostatistics: true
comments: false
icon: material/home
---

<section class="home-profile">
  <img src="/img/touxiang.png" alt="刘芷溢" class="home-avatar">
  <div class="home-intro">
    <p class="home-kicker">你好，我是</p>
    <h1>刘芷溢</h1>
    <p class="home-role">电子科技大学 · 2024 级硕士研究生</p>
    <p>我在电子科技大学计算机科学与工程学院攻读计算机体系结构方向硕士，所在实验室为 <a href="https://github.com/uestc-ndssl/">NDSL</a>。主要关注大语言模型推理、CUDA、数据库与分布式系统，喜欢从系统实现和性能瓶颈出发理解问题，并把学习与实践记录在这里。</p>
  </div>
</section>

> [!INFO] 正在寻找大模型推理加速、机器学习系统相关的秋招机会。如果你有相关岗位或线索，欢迎通过邮件或社交平台联系我！

<div class="home-contact">
  <p><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><rect x="3" y="7" width="18" height="14" rx="2"/><path d="M8 7V4h8v3M3 12l9 4 9-4M10 14h4"/></svg><span>工作邮箱：<a href="mailto:lzy_CS_LN@163.com"><strong>lzy [underline] CS [underline] LN [at] 163 [dot] com</strong></a></span></p>
  <p><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 6 9 7 9-7"/></svg><span>个人邮箱：<a href="mailto:tomlzy213@gmail.com"><strong>tomlzy213 [at] gmail [dot] com</strong></a></span></p>
  <p><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" aria-hidden="true"><path d="M14 2H5v20h14V7l-5-5Zm0 0v6h5M8 12h8M8 16h8"/></svg><span>简历：<a href="/assets/lzycv.pdf"><strong>点击查看</strong></a></span></p>
  <nav class="home-social" aria-label="社交平台">
    <a href="https://github.com/tom-jerr/"><svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.55.1.76-.24.76-.54v-2.07c-3.1.67-3.75-1.31-3.75-1.31-.5-1.28-1.24-1.62-1.24-1.62-1.01-.69.08-.68.08-.68 1.12.08 1.71 1.15 1.71 1.15 1 1.72 2.62 1.22 3.26.93.1-.72.39-1.22.71-1.5-2.47-.28-5.07-1.24-5.07-5.49 0-1.22.44-2.22 1.15-3-.11-.28-.5-1.41.11-2.94 0 0 .94-.3 3.05 1.14a10.6 10.6 0 0 1 5.55 0c2.12-1.44 3.05-1.14 3.05-1.14.61 1.53.23 2.66.12 2.94.71.78 1.14 1.78 1.14 3.01 0 4.26-2.6 5.2-5.08 5.48.4.35.75 1.02.75 2.06v3.04c0 .3.2.65.76.54A11.1 11.1 0 0 0 12 .9Z"/></svg>GitHub</a>
    <a href="https://x.com/tom_jerry_jack"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M4 3h4l12 18h-4L4 3ZM20 3l-7 8M4 21l7-8"/></svg>X</a>
    <a href="https://www.zhihu.com/people/chen-wen-de-jian-ke"><svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><text x="12" y="20" text-anchor="middle" font-family="Microsoft YaHei, sans-serif" font-size="22" font-weight="700">知</text></svg>知乎</a>
  </nav>
</div>

很高兴在互联网上与你相遇！欢迎交流系统实现和性能优化，也欢迎来信讨论博客中的问题。我在成都，平时喜欢音乐、篮球和阅读；如果你也在这里，可以一起聊天、学习。

## 研究方向

- **大语言模型推理与优化**：推理框架、KV Cache、调度、并行与推测解码。
- **GPU 系统与 CUDA**：Kernel 优化、内存层次与软硬件协同。
- **数据库与分布式系统**：存储引擎、向量检索、事务与一致性协议。

## 教育经历

- **电子科技大学 · 硕士**，计算机科学与工程学院，2024 年 9 月至今。
- **电子科技大学 · 本科**，计算机科学与工程学院，2020 年 9 月至 2024 年 6 月。

## 项目经历与近况

- **大模型推理与 GPU 优化**：维护教学型推理框架 MiniInfer，实践调度、KV Cache 和 CUDA Graph；在 SGLang 的个人 feature 分支探索混合批次与推测解码，并在 cuda-learn 中记录算子实现与基准测试。
- **Nebula Graph 开源之夏**（2025 年 7—10 月，11 月结题）：参与原生向量类型、nGQL DDL/DML 扩展和 ANN 索引接入，打通向量存储与相似度查询。实现复盘见 [Nebula 向量检索系列](./notes/nebula-vsearch/index.md)。
- **BusTub 数据库内核**（2025 年 3—6 月）：实现缓冲池、并发 B+ 树、查询执行和 MVCC/OCC 事务等课程模块，记录于 [BusTub 通关指北](./notes/CMU15445/bustub通关指北.md)。
- **TinyKV 分布式存储**（2024 年 12 月—2025 年 2 月）：实践 Raft、Multi-Raft、Region 分裂，以及 BadgerDB 上的日志、状态与快照管理。

## 荣誉奖项

- 2024 年 OceanBase 数据库大赛初赛（四川赛区）一等奖：[证书](./img/oceanbase初赛.pdf)。
- 2024 年 OceanBase 数据库大赛总决赛第 11 名：[证书](./img/oceanbase国赛.pdf)。

## GitHub 项目

以下卡片覆盖主页置顶项目，并补充数据库与存储方向的实践。架构图依据项目源码、设计文档和站内复盘整理，只展示核心模块；点击卡片进入对应仓库。

<div class="project-grid">
  <a class="project-card" href="https://github.com/tom-jerr/sglang/tree/feature/eagle-mixed-fa3-cuda-graph" target="_blank" rel="noopener noreferrer">
    <img src="/img/projects/sglang-architecture.svg" width="800" height="400" loading="lazy" alt="SGLang：Scheduler 的混合批次交给 EAGLEWorkerV2 中的 MixedWorker，复用已有 runner 与 KV 资源">
    <div class="project-caption"><strong class="project-title">SGLang · EAGLE 混合批次</strong><p>MixedWorker 职责拆分与模型内 attention 融合。</p><small>feature/eagle-mixed-fa3-cuda-graph</small></div>
  </a>
  <a class="project-card" href="https://github.com/tom-jerr/cuda-learn" target="_blank" rel="noopener noreferrer">
    <img src="/img/projects/cuda-learn-architecture.svg" width="800" height="400" loading="lazy" alt="cuda-learn：PyTorch binding 通过 TVM FFI 调用 CUDA kernels，测试与 benchmark 复用统一接口">
    <div class="project-caption"><strong class="project-title">cuda-learn</strong><p>手写 CUDA 算子、统一绑定、正确性校验与性能测试。</p></div>
  </a>
  <a class="project-card" href="https://github.com/tom-jerr/MiniInfer" target="_blank" rel="noopener noreferrer">
    <img src="/img/projects/miniinfer-architecture.svg" width="800" height="400" loading="lazy" alt="MiniInfer：LLMEngine 内部的 Scheduler、ModelRunner 与分页和前缀 KV 缓存管理">
    <div class="project-caption"><strong class="project-title">MiniInfer</strong><p>教学型推理框架：连续批处理、分页缓存与 CUDA Graph。</p></div>
  </a>
  <a class="project-card" href="https://github.com/tom-jerr/nebula" target="_blank" rel="noopener noreferrer">
    <img src="/img/projects/nebula-architecture.svg" width="800" height="400" loading="lazy" alt="Nebula 向量检索：graphd 查询、metad 元数据与 storaged 中 ANN 索引及 RocksDB 映射">
    <div class="project-caption"><strong class="project-title">Nebula · 原生向量检索</strong><p>向量类型、查询语言扩展与 HNSW / IVF 索引接入。</p></div>
  </a>
  <a class="project-card" href="https://github.com/tom-jerr/tom-jerr.github.io" target="_blank" rel="noopener noreferrer">
    <img src="/img/projects/blog-architecture.svg" width="800" height="400" loading="lazy" alt="个人博客：Markdown 与本地图片经 Quartz 插件生成页面、全文索引与标签">
    <div class="project-caption"><strong class="project-title">个人博客 · Quartz</strong><p>以本地 Markdown 和图解积累可检索的系统学习笔记。</p></div>
  </a>
  <a class="project-card" href="https://github.com/tom-jerr/bustub" target="_blank" rel="noopener noreferrer">
    <img src="/img/projects/bustub-architecture.svg" width="800" height="400" loading="lazy" alt="BusTub：查询执行访问 B+ 树与表存储，缓冲池管理磁盘页，MVCC 管理版本">
    <div class="project-caption"><strong class="project-title">BusTub</strong><p>关系型数据库内核：存储、索引、执行与事务。</p></div>
  </a>
  <a class="project-card" href="https://github.com/tom-jerr/tinykv" target="_blank" rel="noopener noreferrer">
    <img src="/img/projects/tinykv-architecture.svg" width="800" height="400" loading="lazy" alt="TinyKV：TinyScheduler 调度存储节点，同一 Region 的 Raft 副本通过日志复制保持一致，底层使用 BadgerDB">
    <div class="project-caption"><strong class="project-title">TinyKV</strong><p>Raft 共识、Region 分片与分布式键值存储。</p></div>
  </a>
  <a class="project-card" href="https://github.com/tom-jerr/MiniLSM" target="_blank" rel="noopener noreferrer">
    <img src="/img/projects/minilsm-architecture.svg" width="800" height="400" loading="lazy" alt="MiniLSM：内存表迭代器合并为有序视图；SST 编解码和表迭代器为独立学习模块">
    <div class="project-caption"><strong class="project-title">MiniLSM</strong><p>Rust 存储引擎学习：Memtable、SST 与合并迭代器。</p></div>
  </a>
</div>
