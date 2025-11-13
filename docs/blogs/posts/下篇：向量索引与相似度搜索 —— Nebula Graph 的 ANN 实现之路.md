---
title: 中篇：Vector 类型的 DDL & DML 适配
date: 2025/11/5
tags:
  - Database
---

# 中篇：Vector 类型的 DDL & DML 适配

📚 本系列文章分为上中下三篇，记录了我在开源之夏项目中，开发 Nebula Graph 向量搜索功能的一些复盘和思考，希望可以给大家学习和开发类似系统时有一定的样本参考。希望大家多多关注和交流，大家一起进步 😊 欢迎订阅我的个人网站:rocket: [tom-jerr.github.io](https://tom-jerr.github.io/)

> 本篇主要介绍如何支持 Ann 索引和 Ann Search。

在[nebula graph 的上篇](https://tom-jerr.github.io/blogs/posts/%E4%B8%8A%E7%AF%87%EF%BC%9A%E5%88%9D%E8%AF%86%20Nebula%20Graph%20%E2%80%94%E2%80%94%20%E5%90%91%E9%87%8F%E7%B1%BB%E5%9E%8B%E6%94%AF%E6%8C%81/)和[中篇](https://tom-jerr.github.io/blogs/posts/%E4%B8%AD%E7%AF%87%EF%BC%9AVector%20%E7%B1%BB%E5%9E%8B%E7%9A%84%20DDL%26DML%20%E9%80%82%E9%85%8D/)，我们已经实现了 Vector 类型的存储以及对 DDL 和 DML 的适配。现在我们需要实现 Ann Index 构建和 Ann Search，这里分三个步骤来实现：

- 构建 Ann Index Adapter，将 HNSWlib 和 faiss 封装成统一的接口
- 实现 Ann Index 的 DDL，支持创建和删除 Ann Index
- 利用 Ann Index 进行 Ann Search

这里面也有一些关键问题需要解决：

1. Ann Index 的生命周期管理谁来负责？
2. Ann Index 的数据存储在哪里？
3. Ann Search 生成的计划如何使用 Ann Index 进行搜索？

我们会在下面的章节中结合三个步骤逐一进行说明，并且分享我们在实现过程中得到的经验教训。

> :warning: 这里为了简化，我们假设使用 Tag Schema 进行说明

## Ann Index Interface

### Ann Index Lifecycle

向量索引的生命周期由存储服务器管理。执行创建索引命令后，向量索引将被插入；执行删除索引命令后，向量索引将被删除。在其他情况下，存储服务器将继续维护此索引。

> 在 Ann Index DDL 章节中会详细介绍创建和删除索引的实现。实际上是通过 `VectorIndexManager` 来管理向量索引的生命周期，这个单例会维护一个内存中的向量索引映射表，Key 是 `GraphSpaceID + PartID + IndexID`，Value 是具体的向量索引实例。

> 这个 `VectorIndexManager` 在存储守护进程**启动时初始化(加载持久化的向量索引)**，在进程退**出前持久化**所有的向量索引到磁盘。

### Memory Tracked

暂时我的实现是使用 Nebula 内置的 MemoryTracker 定期查询内存索引的大小。如果超过限制，则无法插入新的 Vector

### Ann Index Interface

为了支持不同的向量索引库，我们需要定义一个统一的向量索引接口 `AnnIndex`，并且实现不同的向量索引适配器。这个接口主要包含以下方法：

```cpp
class AnnIndex {
 public:
  AnnIndex() = default;

  AnnIndex(GraphSpaceID graphID,
           PartitionID partitionID,
           IndexID indexID,
           const std::string &indexName,
           bool propFromNode,
           size_t dim,
           const std::string &rootPath,
           MetricType metricType,
           size_t minTrainDataSize = 3);

  virtual ~AnnIndex() = default;
  AnnIndex(const AnnIndex &) = delete;
  AnnIndex &operator=(const AnnIndex &) = delete;

  [[nodiscard]] virtual Status init(const BuildParams *params) = 0;
  // add data to index incrementally
  [[nodiscard]] virtual Status add(const VecData *data) = 0;
  // upsert data to index
  [[nodiscard]] virtual Status upsert(const VecData *data) = 0;
  // soft delete data from index, return number of deleted vectors
  [[nodiscard]] virtual StatusOr<size_t> remove(const IDSelector &sel) = 0;

  // ann search
  [[nodiscard]] virtual Status search(const SearchParams *params, SearchResult *res) = 0;
  // reconstruct vector by id
  [[nodiscard]] virtual StatusOr<Vector> reconstruct(VectorID id) = 0;

  // load index file from disk
  // flush index to disk
  [[nodiscard]] virtual Status read(const std::string &file) = 0;
  [[nodiscard]] virtual Status write(const std::string &dir, const std::string &file) = 0;
  virtual AnnIndexType indexType() const = 0;
  virtual std::string toString() const = 0;
};
```

### Concurrent Ann Index

通过实现 `AnnIndex` 接口，我们构建了两个向量索引：一个底层使用 Faiss IVF 索引，另一个底层使用 HNSWlib HNSW 索引。为了实现并发，我们在向量索引中使用了读写锁。这允许多个查询执行 Ann Search，但只允许执行一个查询执行对索引的 DML 操作（添加或删除）。

### Ann Index Utils

为了简化向量索引的使用，我们定义了一些辅助数据结构和枚举类型：

- `MetricType`：表示向量距离度量类型，如 L2 距离和内积。
- `AnnIndexType`：表示向量索引类型，如 IVF 和 HNSW。
- `IDSelector`：用于选择要删除的向量 ID 列表。
- `VecData`：表示向量数据，包括向量数量、维度、数据和 ID。
- `BuildParams`：表示向量索引构建参数的基类，以及其派生类 `BuildParamsIVF` 和 `BuildParamsHNSW`。
- `SearchParams`：表示向量搜索参数的基类，以及其派生类 `SearchParamsIVF` 和 `SearchParamsHNSW`。
- `SearchResult`：表示向量搜索结果，包括向量 ID、距离和向量数据。

```cpp
enum MetricType : int8_t { L2, INNER_PRODUCT };
enum AnnIndexType : int8_t { IVF, HNSW };

// faiss used
struct IDSelector {
  size_t cnt;
  VectorID* ids;  // vector of IDs to select
};

struct VecData {
  size_t cnt;     // number of vectors
  size_t dim;     // dimension of each vector
  float* fdata;   // float type vector data source
  VectorID* ids;  // int64 identifier of each vector
};

struct OwnedVecData {
  std::vector<float> flat;
  std::vector<VectorID> ids;
  VecData view;
};

// ANN index build parameters
struct BuildParams {
  MetricType metricType{MetricType::L2};
  AnnIndexType indexType{AnnIndexType::IVF};
};

struct BuildParamsIVF final : public BuildParams {
  size_t nl{3};  // number of lists
  size_t ts{3};  // train size
};

struct BuildParamsHNSW final : public BuildParams {
  size_t maxDegree{16};      // the maximum degrees
  size_t efConstruction{8};  // expansion in construction time
  size_t capacity{10000};    // capacity of the index
};

struct SearchParams {
  size_t topK{10};        // number of nearest neighbors to search
  float* query{nullptr};  // query vector data
  size_t queryDim{0};     // dimension of query vector
};

struct SearchParamsIVF final : public SearchParams {
  size_t nprobe{10};  // number of lists to probe
};

struct SearchParamsHNSW final : public SearchParams {
  size_t efSearch{16};  // expansion factor at search time
};

// ANN search result
struct SearchResult {
  std::vector<VectorID> IDs;
  // distances of the result vectors
  std::vector<float> distances;
  // result vectors
  std::vector<float> vectors;
};
```

## Ann Index DDL

### Create Ann Index Syntax

- Tag Ann Index Creation Syntax

```shell
CREATE TAG ANNINDEX <index_name> ON <tag_name_list>::(<field_name>) [IF NOT EXISTS] ann_index_params}[COMMENT '<comment>']
```

- Ann Index Parameters
  - `ANNINDEX_TYPE`: Index type, support `IVF` and `HNSW`
  - `DIM`: Vector dimension
  - `METRIC_TYPE`: Metric type, support `L2` and `INNER_PRODUCT`
  - `NLIST`: Number of lists, only for `IVF` index
  - `TRAINSIZE`: Training size, only for `IVF` index
  - `MAXDEGREE`: Maximum degree, only for `HNSW` index
  - `EFCONSTRUCTION`: Expansion factor at construction time, only for `HNSW` index
  - `MAXELEMENTS`: Capacity of the index, only for `HNSW` index

```shell
{ANNINDEX_TYPE: "IVF", DIM:128, METRIC_TYPE:"L2", NLIST:3, TRAINSIZE:3}
{ANNINDEX_TYPE: "HNSW", DIM:128, METRIC_TYPE:"L2", MAXDEGREE:15, EFCONSTRUCTION:200, MAXELEMENTS:10000}
```

### Create Ann Index Implementation

## Ann Search

## 踩过的坑

## 总结
