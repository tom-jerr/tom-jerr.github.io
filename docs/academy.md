---
hide:
  - navigation
  - toc
  - feedback
nostatistics: true
comments: false
---


# 刘芷溢 (Zhiyi Liu) ![](img/touxiang.png){ align=right style="width:7.5em; margin-left: 7.5em; margin-top: 0.5em; border-radius: 1em;"}

<!--:fontawesome-solid-building: Office: [424, 60 5th Ave, New York, NY 10011](https://maps.app.goo.gl/N7m2fM5EbM3TToB79)-->

:fontawesome-solid-inbox: 工作邮箱: [lzy [underline] CS [underline] LN [at] 163 [dot] com](mailto:lzy_CS_LN@163.com)

:fontawesome-solid-inbox: 个人邮箱: [tomlzy213 [at] gmail [dot] com](mailto:tomlzy213@gmail.com)

<!-- :material-file: 旧版简历: [点击查看](assets/CV_zh.pdf) -->

:material-file: 简历: : [点击查看](assets/lzycv.pdf)

<span style=font-size:2em;">[:fontawesome-brands-github:](https://github.com/tom-jerr/) [:fontawesome-brands-x-twitter:](https://x.com/tom_jerry_jack) [:fontawesome-brands-zhihu:](https://www.zhihu.com/people/chen-wen-de-jian-ke)</span>

---

## 个人简介

我是在 [电子科技大学 (UESTC)](https://www.uestc.edu.cn/) 攻读计算机体系结构方向的硕士二年级学生，目前就读于实验室 [NDSL](https://github.com/uestc-ndssl/)。我主要关注大语言模型推理 (LLM Inference)、数据库与分布式系统等方向。

> [!INFO] 目前**正在寻找与大模型推理加速相关的实习机会**，如果您有相关岗位或线索，欢迎随时联系我！谢谢！🥰🥰🥰

---

## 研究兴趣

- **大语言模型推理与优化**：正在深入学习大模型推理框架的体系结构，其中包括 SGLang。并维护一个大模型推理的教学项目 [MiniInfer](https://github.com/tom-jerr/MiniInfer)。 

<!-- - **Distributed Database**: I focus on Distributed Database. Last year, I and my partners have won the 11th in 2024 OceanBase Database Competition. Now I am working on designing vector search for nebula. -->

<!-- - **Vector Database**: I also foceus on vector database and Graph RAG on this field. Now I am working for adding the ability of vector search on a graph database. I want to do some optimization on vector search with hnsw, if you have any idea about hnsw, welcome to contact with me. -->

---

## News

<!-- === "2025" -->
<!---->
<!--     [01/2025] I became an intern in [GAIR](https://plms.ai/index.html), advised by Prof. [Pengfei Liu](http://pfliu.com/). -->

=== "2025" 

    [11/2025] :party_popper: 参与的 Nebula Graph 社区开源之夏项目——为 NebulaGraph 支持向量近似近邻检索顺利结题（项目周期：2025.07 - 2025.10）

=== "2024"

    [12/2024] :party_popper: NODDL 团队在 2024 年 OceanBase 数据库大赛总决赛中获第 11 名（**第 10/1212 支队伍**）

    [11/2024] :party_popper: NODDL 团队获得 2024 年 OceanBase 数据库大赛初赛（四川赛区）一等奖

---

## 教育经历

### 电子科技大学 计算机科学与工程学院（硕士）![Image title](img/UESTC.png){ align=right style="height:6em; border-radius: 0.5em;"}

2024年9月 —— 至今

<br>

### 电子科技大学 计算机科学与工程学院（本科）![Image title](img/UESTC.png){ align=right style="height:6em; border-radius: 0.5em;"}

2020年9月 —— 2024年6月

---

<!-- ## Publications & Manuscripts

> Coming soon...

--- -->

<!-- ## Experience -->

<!-- **GAIR, SJTU & Qing Yuan Research Institute**![Image title](img/gair.png){ align=right style="height:5em; border-radius: 0.5em;"} -->
<!---->
<!-- Jan. 2025 - Present -->
<!---->
<!-- Research Intern -->

<!-- > Coming soon... -->

---

## 项目经历

[**开源暑期项目 — Nebula Graph (2025.07 – 2025.10)**](img/开源之夏2025-刘芷溢-结项证书zh.pdf)

作为核心贡献者参与 Nebula Graph 社区开源之夏项目，负责为分布式图数据库引入原生向量检索（Vector Search）能力。

- **向量存储引擎设计**：设计并实现了原生向量数据类型（Vector Type）及其底层存储布局，优化了高维向量的序列化与压缩存储机制，实现了图元素（Vertex/Edge）与向量数据的统一管理。
- **查询语言与执行层扩展**：扩展了 nGQL（Nebula Graph Query Language）的 DDL 与 DML 语法，支持向量索引的定义与管理；解决了向量属性在分布式架构下的元数据同步与一致性挑战。
- **高性能 ANN 索引集成**：设计了向量索引接口抽象，集成了 HNSW 等近似最近邻搜索算法，实现了基于向量相似度的 Top-K 查询算子，大幅提升了图谱语义检索的性能与准确度。

**BusTub 关系型数据库内核实现 (2025.03-2025.06)**

实现了一个教学关系型数据库内核的一些关键功能，涵盖存储、索引、执行与事务四大核心模块。

- **存储引擎与缓冲池管理**：设计并实现了基于 LRU-K 算法的缓冲池管理器 (Buffer Pool Manager)，支持高并发下的页面置换与脏页刷盘；实现了磁盘调度器 (Disk Scheduler)，优化了磁盘 I/O 请求的处理效率。
- **并发 B+ 树索引**：实现了支持高并发读写的 B+ 树索引，采用乐观锁 (Optimistic Latch Crabbing) 策略与 Context 机制优化锁竞争，显著提升了多线程环境下的索引访问性能；支持范围查询迭代器 (Index Iterator)。
- **查询执行引擎**：实现了基于火山模型 (Volcano Model) 的查询执行器，包括 Hash Join 算子与外部归并排序 (External Merge Sort) 算法，支持大规模数据的连接与排序操作。
- **事务与并发控制**：构建了基于 MVCC (多版本并发控制) 与 OCC (乐观并发控制) 的事务管理系统，实现了可串行化 (Serializable) 隔离级别；设计了 Undo Log 版本链管理与垃圾回收 (GC) 机制，有效解决了读写冲突与版本回滚问题。

**TinyKV 分布式键值存储系统 (2024.12-2025.02)**

设计并实现了一个基于 Raft 共识算法的高可用、强一致性分布式 Key-Value 存储系统（架构参考 TiKV），支持水平扩展、多副本容错及自动故障恢复。

- **Raft 共识算法实现**：构建 Raft 核心模块，实现了 Leader 选举、日志复制、心跳保活机制；通过设计模块化的 RawNode 接口，实现了算法逻辑与上层应用的解耦。
- **Multi-Raft 架构与水平扩展**：实现了基于 Region 的 Multi-Raft 架构，支持数据按范围（Range）自动分片；开发了 Region Split（分裂）机制，使系统能够随着数据量增长自动进行水平扩展。
- **存储引擎与持久化**：封装 BadgerDB 作为底层单机存储引擎；实现了 PeerStorage 模块，负责 Raft 日志（Log Entries）与硬状态（HardState）的持久化存储，保证了系统崩溃后的数据安全性。
- **日志压缩与快照机制**：设计并实现了 Raft 日志压缩（Log Compaction）和快照（Snapshot）功能，解决了日志无限增长导致的存储压力问题，并支持落后节点通过快照快速恢复状态。

---

<!-- ## 媒体 / 个人输出

我会在 [知乎](https://www.zhihu.com/people/Kinnariya) 定期分享技术文章与思考，目前已有 100+ 关注者。 -->

## 荣誉奖项

- [2024 年 OceanBase 数据库大赛初赛（四川赛区）一等奖](img/oceanbase初赛.pdf), 2024
- [2024 年 OceanBase 数据库大赛总决赛第 11 名](img/oceanbase国赛.pdf), 2024

---

<!-- ??? tip "Some of My Friends"

    Umass: [Haoyu Zhen](https://haoyuzhen.com/)

    UC Berkeley: [Junyi Zhang](https://www.junyi42.com/) -->

<div align="center">
    <div align="center" style="width:20%">
        <script type="text/javascript" id="clustrmaps" src="//clustrmaps.com/map_v2.js?d=_1g20YoX1boCjXuxcNhGdbnRQiA2LG8IlLZwCYTAPUQ&cl=ffffff&w=a"></script>
    </div>
</div>
