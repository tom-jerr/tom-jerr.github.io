# Stroage Implementation of Vector Type

Vector 类型的属性存储的流程：

![](../vector_storage.png)

实际上真正向 RocksDB 中插入数据是在`Part::commitLogs(
    std::unique_ptr<LogIterator> iter, bool wait, bool needLock)`中，处理伪代码如下：

```c++
auto batch = engine_->startBatchWrite();
 while (iter->valid()) {
    switch (log[sizeof(int64_t)]) {
      case OP_MULTI_PUT: {
        auto kvs = decodeMultiValues(log);
        for (size_t i = 0; i < kvs.size(); i += 2) {
          auto code = batch->put(kvs[i], kvs[i + 1], cfName);
        }
        break;
      }
    }
    ++(*iter);
  }
engine_->commitBatchWrite(
      std::move(batch), FLAGS_rocksdb_disable_wal, FLAGS_rocksdb_wal_sync, wait);
```

为了最小化修改，不改变整体 RaftLog 的处理流程，我们在 Log 中增加了 cfName 字段——表示 column family，
并在 decodeMultiValues 中解析出该字段。

为了能够支持我们的 VECTOR type，我们需要修改全流程：

1. AddVerticesProcessor 对 VECTOR 类型属性存放到 RocksDB 的“vector”column family 中。
2. BaseProcessor 的 doPut 方法需要支持对“vector” column family 的操作。
3. KVStore 中的 asyncMultiPut 方法需要支持对“vector” column family 的操作。
4. Part 中的 asyncMultiPut 方法需要支持对“vector” column family 的操作，commitLogs 方法需要支持对“vector” column family 的操作。
5. encodeMultiValues 方法需要支持对“vector” column family 的操作。
6. WriteBatch 接口需要支持对“vector” column family 的操作。

## Addition of WriteBatch Interface

- 在现有的 WriteBatch 接口中添加对特定 column family 的操作方法

```c++
class WriteBatch {
 public:
  virtual nebula::cpp2::ErrorCode put(folly::StringPiece key,
                                      folly::StringPiece value,
                                      const std::string& cfName) = 0;

  virtual nebula::cpp2::ErrorCode remove(folly::StringPiece key, const std::string& cfName) = 0;

  virtual nebula::cpp2::ErrorCode removeRange(folly::StringPiece start,
                                              folly::StringPiece end,
                                              const std::string& cfName) = 0;
};
```

- 对 WriteBatch 的基类方法不做更改，新增 RocksWriteBatch 对特定 column family 的操作方法
- 对 KVEngine 的基类方法不做更改，新增 RocksEngine 对特定 column family 的操作方法

# vector encoding

## Key

- NebulaKeyUtils::vectorTagKey
  > type(1) + partId(3) + vertexId(\_) + tagId(4) + propId(4)
- NebulaKeyUtils::vectorVertexKey:
  > type(1) + partId(3) + vertexId(\_) + tagId(4) + propId(4)
- NebulaKeyUtils::vectorEdgeKey
  > type(1) + partId(3) + srcId(_) + edgeType(4) + edgeRank(8) + dstId(_) + propId(4) +placeHolder(1)

## Write

StatusOr<std::string> BaseProcessor<RESP>::encodeRowVal ->
WriteResult RowWriterV2::setValue(const std::string& name, const Value& val) ->
WriteResult RowWriterV2::setValue(ssize_t index, const Value& val) ->
WriteResult RowWriterV2::write(ssize_t index, const Vector& vector)

## Read

RowReaderWrapper() ->
static StatusOr<nebula::Value> readValue->
RowReaderWrapper::getValueByName

# KNN Search

## Syntax Design

`vector_distance(vector1: src1, vector2: src2, metric: L2;)`
