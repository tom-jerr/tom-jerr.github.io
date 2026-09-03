# 项目卡片架构图依据

核对日期：2026-09-04。生成器为 `draw_project_architectures.py`，输出到 `content/img/projects/*-architecture.svg`。
所有图均为 800 × 400、2:1 的原生可编辑 SVG；颜色沿用 HiCache 图解：蓝色计算/执行、绿色数据/存储、灰色边界/控制。虚线表示元数据或协调接口，不是性能时间线。

## 项目选择

当前 [GitHub 主页](https://github.com/tom-jerr?tab=overview) 置顶为 SGLang、cuda-learn、MiniInfer、Nebula、tom-jerr.github.io；另补 BusTub、TinyKV，并保留原首页 MiniLSM。图仅概括有证据的结构，不表达 benchmark 结果或实现完成度。

## SGLang

- 分支：`feature/eagle-mixed-fa3-cuda-graph`，核对时 tree commit `0227acb584ed5d539c835440c07b8783c9665d0e`。
- [实现与设计文档](https://github.com/tom-jerr/sglang/blob/0227acb584ed5d539c835440c07b8783c9665d0e/benchmark_results/eagle_mixed_fused_4090/IMPLEMENTATION_AND_BENCHMARK.md)：§1、§2、§3、§6，文档 blob `afb6395a8bd613771519c9fe424ac26bf6a42c00`。
- [MixedWorker 源码](https://github.com/tom-jerr/sglang/blob/0227acb584ed5d539c835440c07b8783c9665d0e/python/sglang/srt/speculative/eagle_mixed_worker_v2.py)：文件说明、构造函数、`bind_memory_pools`。
- 核心关系：Scheduler 创建 MIXED parent 与 Prefill/Verify 子视图；EAGLEWorkerV2 拥有资源与委托对象；MixedWorker 不创建重复 model、graph runner、KV pool。
- 不画成 target/draft 并行；融合发生在各自 model-local composition 内。Fused attention 有后端/形状门控，图不宣称所有请求都走融合路径。

## cuda-learn

- [README](https://github.com/tom-jerr/cuda-learn/blob/main/README.md)，核对时 blob `cddb5af75a1680341292c9cc3bb411ef260e6f1a`。
- `tests + @bench` → `ops.py` → TVM FFI 全局函数注册 → CUDA kernels；DLPack 与 torch stream 是 binding 边界，而不是另一份 tensor 存储。
- 未把 CUDA allocator/graph 的独立演示错误并入统一 tensor-op benchmark。

## MiniInfer

- [v2 README](https://github.com/tom-jerr/MiniInfer/blob/v2/README.md)：架构、LLMEngine、Scheduler、ModelRunner、KVCacheManager。
- KVCacheManager 的 Paged KV / Radix prefix 表达为共享管理模块，不为 Scheduler 与 Runner 重复创建两份缓存。
- 图中 FA2 / FlashInfer 为可选 attention 后端；不把默认禁用的 Piecewise CUDA Graph 宣称为默认路径。

## Nebula

- 本地 `content/notes/nebula-vsearch/` 上、中、下三篇；尤其下篇的存储设计总结、`AnnIndexScan`、`VectorIndexManager`、ID 映射说明。
- `graphd`（nGQL/查询计划）、`metad`（schema/索引元数据）、`storaged`（ANN 与 KVStore）是服务边界。
- ANN 为内存索引（HNSWlib / Faiss IVF，可序列化为本地文件）；RocksDB 保存向量属性与 VectorID→VID 映射。未将 ANN 图结构画进 RocksDB，也未宣称 ANN 分布式容灾已经完成。
- [项目仓库](https://github.com/tom-jerr/nebula)。

## 个人博客

- 当前本地 `quartz.config.yaml`、`quartz.ts` 与 `quartz/build.ts`：内容解析、Transformer、Emitter，以及 ContentIndex/TagPage/ContentPage 和资源复制。
- 卡片明确展示此次 Quartz 迁移后的本地架构，不把尚未发布的本地改动描述成 GitHub 默认分支现状。

## BusTub 与 TinyKV

- `content/academy.md` 中文项目经历；`content/notes/CMU15445/bustub通关指北.md` 的 BufferPool、B+Tree、MVCC 复盘。
- [BusTub 仓库](https://github.com/tom-jerr/bustub) 与 [TinyKV README](https://github.com/tom-jerr/tinykv/blob/master/README.md)。
- BusTub 图为课程内核的分层概览，事务虚线表示协调，不表示日志直接写入某个索引节点。
- TinyKV 只示意同一 Region 的两个 Raft peer；不是两节点生产部署推荐。TinyScheduler 为调度控制面，BadgerDB 属于各存储节点。

## MiniLSM

- [lsm_storage.rs](https://github.com/tom-jerr/MiniLSM/blob/b206e68ecc84b636a245bc7778cdea54e5d7121d/src/lsm_storage.rs)，`scan`、`force_flush_next_imm_memtable`；本地 `content/notes/minilsm/week1-day1.md` 至 `week1-day4.md`。
- 源码 `scan` 当前仅合并 Memtable 迭代器，flush 仍为 `unimplemented!()`。因此图分开画内存读取和 SST 模块，不添加持久化 flush 边或完整 Compaction/MVCC 架构，避免把 starter 设计当作已完成实现。

## 复现与验收

```powershell
python scripts/draw_project_architectures.py
python scripts/draw_project_architectures.py tmp/project-diagram-check
python C:/Users/lzy/.codex/skills/drawing-toolkit/scripts/validate_svg.py content/img/projects/sglang-architecture.svg --forbid-raster --forbid-external --forbid-gradients --aspect 2:1
```

所有图需 XML 校验、重复生成哈希一致性检查，并渲染检查文字、边界和箭头。
