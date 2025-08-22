# Create Ann Index

## Ann Index Manager

- 在 Storage Daemon 中设计一个`VectorIndexManager`单例对整个 Storaged 中的 Ann Index 进行管理。
- Ann Index 的生命周期：
  - 创建：通过 `CreateTagAnnIndex` 请求创建 Ann Index。
    > 除非删除，否则会一直在内存中维护，在退出时需要持久化到磁盘，同时重启系统后需要从磁盘中加载已经存在的 Ann Index。
  - 使用：在查询时使用 Ann Index 进行加速。
  - 删除：通过 `DropTagAnnIndex` 请求删除 Ann Index。
  - 更新：通过 `UpdateTagAnnIndex` 请求更新 Ann Index。

```cpp
class VectorIndexManager final {
 public:
  static VectorIndexManager& getInstance();
  // Initialize the manager with necessary dependencies
  Status init(meta::IndexManager* indexManager, std::string annIndexPath);
  // Start the manager (background tasks, cleanup threads, etc.)
  Status start();
  // Stop the manager gracefully
  Status stop();
  // Wait until the manager stops (similar to StorageServer::waitUntilStop)
  void waitUntilStop();
  // Notify the manager to stop (used for signal handling)
  void notifyStop();
  // Create a vector index for a specific partition and index ID
  Status createOrUpdateIndex(GraphSpaceID spaceId,
                             PartitionID partitionId,
                             IndexID indexId,
                             const std::shared_ptr<meta::cpp2::AnnIndexItem>& indexItem);
  // Get an existing vector index
  StatusOr<std::shared_ptr<AnnIndex>> getIndex(GraphSpaceID spaceId,
                                               PartitionID partitionId,
                                               IndexID indexId);
  // Remove a vector index
  Status removeIndex(GraphSpaceID spaceId, PartitionID partitionId, IndexID indexId);
  // Add vectors to an index
  Status addVectors(GraphSpaceID spaceId,
                    PartitionID partitionId,
                    IndexID indexId,
                    const VecData& vecData);
  // Search vectors in an index
  StatusOr<SearchResult> searchVectors(GraphSpaceID spaceId,
                                       PartitionID partitionId,
                                       IndexID indexId,
                                       const SearchParams& searchParams);
  // Get all managed indexes for a partition
  std::vector<std::shared_ptr<AnnIndex>> getIndexesByPartition(GraphSpaceID spaceId,
                                                               PartitionID partitionId);
  // Check if an index exists
  bool hasIndex(GraphSpaceID spaceId, PartitionID partitionId, IndexID indexId) const;
  // Rebuild an index (called during BuildVectorIndexTask)
  Status rebuildIndex(GraphSpaceID spaceId,
                      PartitionID partitionId,
                      IndexID indexId,
                      const std::shared_ptr<meta::cpp2::AnnIndexItem>& indexItem);
};
```

## Ann Index Interface

### Ann Index Item

- 通用的 Index Item 是对单个 Schema 的多个 field 的索引定义，同时索引参数只有`s2_max_level`和`s2_max_cells`，不能满足 Ann Index 对多个 Schema 同名属性进行索引创建的要求，所以我们设计了新的 Ann Index Item。
- Ann Index Item 是对多个 Schema 的同名属性进行索引创建的定义，包含了所有需要索引的 Schema 的信息。同时通过一个`list<binary>`来存储 ann index 创建的参数。

  > ```cpp
  > ann index params:
  > [IVF/*ann type*/, 128 /*dim*/, L2/*metric type*/, 3/*nlist*/, 3/*train size*/]
  > [HNSW/*ann type*/, 128 /*dim*/, L2/*metric type*/, 16/*max degree*/, 200/*ef construction*/, 100000/*max elements*/]
  > ```

```cpp
struct IndexItem {
    1: common.IndexID       index_id,
    2: binary               index_name,
    3: common.SchemaID      schema_id
    4: binary               schema_name,
    5: list<ColumnDef>      fields,
    6: optional binary      comment,
    7: optional IndexParams index_params,
}

struct AnnIndexItem {
    1: common.IndexID           index_id,
    2: binary                   index_name,
    3: binary                   prop_name,
    4: list<common.SchemaID>    schema_ids,
    5: list<binary>             schema_names,
    6: list<ColumnDef>          fields,
    7: optional binary          comment,
    8: optional list<binary>    ann_params,         // ANN specific parameters
}
```

### Ann Index Request & Response

- 之前的 `CreateTagIndexReq` 只支持单个 Schema 的索引创建，而 Ann Index 需要支持多个 Schema 的**同名属性**进行索引创建，所以我们设计了新的 `CreateTagAnnIndexReq`。

```cpp
struct CreateTagAnnIndexReq {
    1: common.GraphSpaceID      space_id,
    2: binary                   index_name,
    3: list<binary>             tag_names,
    4: IndexFieldDef            field,
    5: bool                     if_not_exists,
    6: optional binary          comment,
    7: optional list<binary>    ann_params,
}
```

- Storaged 需要通过 meta client 执行`listTagAnnIndexes`和`listEdgeAnnIndexes`获取 Ann 索引信息，Ann Index 的条目由 Ann Index Item 存储与之前的 Index Item 结构不同，所以设计新的 Resp。

```cpp
struct ListTagAnnIndexesResp {
    1: common.ErrorCode         code,
    2: common.HostAddr          leader,
    3: list<AnnIndexItem>	items,
}
struct ListEdgeAnnIndexesResp {
    1: common.ErrorCode         code,
    2: common.HostAddr          leader,
    3: list<AnnIndexItem>    	items,
}
```

## Create Ann Index Syntax Design

`CREATE TAG ANNINDEX <index_name> ON <tag_name_list>::(<field_name>) [IF NOT EXISTS] ann_index_params}[COMMENT '<comment>'] `

> ```cpp
> ann_index_params:
> {ANNINDEX_TYPE: "IVF", DIM:128, METRIC_TYPE:"L2", NLIST:3, TRAINSIZE:3}
> {ANNINDEX_TYPE: "HNSW", DIM:128, METRIC_TYPE:"L2", MAXDEGREE:15, EFCONSTRUCTION:200, MAXELEMENTS:10000}
> ```

## Create Ann Index Plan

1. Graphd 生成 `Start->CreateTagAnnIndex->SubmitJob` 的计划
2. 执行时，metad 先在内部创建 Ann 索引条目，创建成功后，会进行 AdminJob 的提交
3. Metad 将 Job 的参数打包传给 Storaged 的 AdminTask，Storaged 通过内部的 AdminTaskManager 处理这些任务，完成任务。这里涉及到分布式的时序问题：
   > Meta Client 缓存更新机制：Storage 节点的 IndexManager 通过 MetaClient 获取索引信息，但 MetaClient 的缓存是通过心跳周期更新的:
   >
   > - 时序问题：Meta 服务创建索引后，Storage 节点需要等待下一个心跳周期才能看到新索引
   > - 同步问题：getTagAnnIndex 直接从缓存读取，如果缓存还没更新就会返回 IndexNotFound

- 所以我们在 Storaged 中使用重试机制，如果多次重试失败直接向 meta server 请求数据强制刷新缓存

![](../create_ann_index_plan.png)

### Storaged BuildTagVectorIndexTask Plan

对每个 Partition 做相同的操作：

1. 扫描 KVStore(RocksDB)中的所有符合 property name 的 vector 属性的数据
2. 将 Vertex ID/Edge Type 与 VectorID 存入 KVStore 的 id-vid column family
   > Vertex ID 是 std::string 类型，这里通过 hash 计算 VectorID，所以需要存入两份映射`VectorID->VertexID`和`VertexID->VectorID`
3. 通过这些数据构建内存中的 Ann Index

![](../create_ann_index_storaged.png)
