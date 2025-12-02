# WAL for Vector Type

## WAL Scenarios

### 场景 1: 新节点加入集群

#### 初始状态

```
集群状态：
- Leader: Node1 (term=5, lastLogId=100)
- Follower: Node2 (term=5, lastLogId=100)
- Follower: Node3 (term=5, lastLogId=100)
- 新节点: Node4 (term=0, lastLogId=0)
```

#### 重放流程

##### 步骤 1: 新节点启动

```cpp
// Node4 启动
void RaftPart::start(std::vector<HostAddr>&& peers, bool asLearner) {
  // Node4 设置：
  // - role_ = Role::FOLLOWER
  // - term_ = 0
  // - lastLogId_ = 0
  // - committedLogId_ = 0

  VLOG(1) << "Node4 started as follower with " << peers.size() << " peers";
}
```

##### 步骤 2: Leader 发现新节点落后太多

```cpp
// Node1 (Leader) 检测到 Node4 需要 snapshot
void RaftPart::sendHeartbeat() {
  // 发现 Node4 的 lastLogId=0，而 Leader 的 firstLogId=10
  // 差距过大，需要发送 snapshot

  if (node4_lastLogId < wal_->firstLogId()) {
    // 触发 snapshot 发送
    snapshot_->sendSnapshot(shared_from_this(), node4_addr);
  }
}
```

##### 步骤 3: Snapshot 发送过程

```cpp
// SnapshotManager::sendSnapshot()
folly::Future<StatusOr<std::pair<LogID, TermID>>> SnapshotManager::sendSnapshot(
    std::shared_ptr<RaftPart> part, const HostAddr& dst) {

  // 扫描所有数据
  accessAllRowsInSnapshot(spaceId, partId,
    [&](LogID commitLogId, TermID commitLogTerm,
        const std::vector<std::string>& data,
        int64_t totalCount, int64_t totalSize,
        SnapshotStatus status) -> bool {

      // 分批发送数据
      auto resp = send(spaceId, partId, termId, commitLogId, commitLogTerm,
                      localhost, data, totalSize, totalCount, dst,
                      status == SnapshotStatus::DONE);

      return resp.get().get_error_code() == nebula::cpp2::ErrorCode::SUCCEEDED;
    });
}
```

##### 步骤 4: Node4 接收 Snapshot

```cpp
// Node4 接收 snapshot
void RaftPart::processSendSnapshotRequest(const cpp2::SendSnapshotRequest& req,
                                          cpp2::SendSnapshotResponse& resp) {
  // 验证 Leader 身份
  auto err = verifyLeader(req);
  if (err != nebula::cpp2::ErrorCode::SUCCEEDED) {
    resp.error_code_ref() = err;
    return;
  }

  // 重置状态，准备接收 snapshot
  reset();
  status_ = Status::WAITING_SNAPSHOT;

  // 提交数据到状态机
  auto ret = commitSnapshot(req.get_rows(),
                           req.get_committed_log_id(),
                           req.get_committed_log_term(),
                           req.get_done());

  if (req.get_done()) {
    // Snapshot 接收完成
    committedLogId_ = req.get_committed_log_id();  // = 100
    committedLogTerm_ = req.get_committed_log_term();  // = 5
    lastLogId_ = committedLogId_;
    lastLogTerm_ = committedLogTerm_;
    status_ = Status::RUNNING;

    VLOG(1) << "Node4 snapshot completed, now at logId=" << lastLogId_;
  }
}
```

##### 步骤 5: 正常日志复制

```cpp
// Node4 现在可以接收正常的日志复制
void RaftPart::processAppendLogRequest(const cpp2::AppendLogRequest& req,
                                       cpp2::AppendLogResponse& resp) {
  // Node4 状态: lastLogId=100, term=5
  // Leader 发送: prevLogId=100, prevLogTerm=5, logs=[101, 102, 103...]

  // 验证日志连续性
  if (req.get_last_log_id_sent() == lastLogId_ &&
      req.get_last_log_term_sent() == lastLogTerm_) {
    // 连续，可以追加
    for (auto& log : req.get_log_str_list()) {
      wal_->appendLog(++lastLogId_, req.get_current_term(), clusterId_, log);
    }

    // 提交可提交的日志
    if (req.get_committed_log_id() > committedLogId_) {
      auto walIt = wal_->iterator(committedLogId_ + 1, req.get_committed_log_id());
      commitLogs(std::move(walIt), false, true);
    }
  }
}
```

---

### 场景 2: 故障节点重新加入

#### 初始状态

```
Node2 在 logId=80 时故障，现在重新启动
集群当前状态：
- Leader: Node1 (term=5, lastLogId=120)
- Follower: Node3 (term=5, lastLogId=120)
- 重启节点: Node2 (term=3, lastLogId=80, committedLogId=75)
```

#### 重放流程

##### 步骤 1: Node2 启动恢复

```cpp
void RaftPart::start(std::vector<HostAddr>&& peers, bool asLearner) {
  // 从本地存储恢复状态
  auto logIdAndTerm = lastCommittedLogId();
  committedLogId_ = logIdAndTerm.first;    // = 75
  committedLogTerm_ = logIdAndTerm.second; // = 3

  lastLogId_ = wal_->lastLogId();     // = 80
  lastLogTerm_ = wal_->lastLogTerm(); // = 3
  term_ = lastLogTerm_;               // = 3

  // 检查一致性
  if (lastLogId_ < committedLogId_) {
    // 异常情况，重置 WAL
    lastLogId_ = committedLogId_;
    lastLogTerm_ = committedLogTerm_;
    wal_->reset();
  }

  // 先重放本地未提交的日志
  if (lastLogId_ > committedLogId_) {
    auto walIt = wal_->iterator(committedLogId_ + 1, lastLogId_);
    commitLogs(std::move(walIt), true, false);
    committedLogId_ = lastLogId_;
    committedLogTerm_ = lastLogTerm_;
  }
}
```

##### 步骤 2: 接收 Leader 的日志

```cpp
void RaftPart::processAppendLogRequest(const cpp2::AppendLogRequest& req,
                                       cpp2::AppendLogResponse& resp) {
  // Node2 收到: term=5, prevLogId=120, 但本地只有到 80

  // 验证 Leader 身份
  if (req.get_current_term() > term_) {
    // 更新 term，转为 follower
    term_ = req.get_current_term();
    role_ = Role::FOLLOWER;
    leader_ = HostAddr(req.get_leader_addr(), req.get_leader_port());
  }

  // 检查日志连续性
  if (req.get_last_log_id_sent() > lastLogId_) {
    // 有 gap，请求从 lastLogId_ 开始发送
    resp.last_matched_log_id_ref() = lastLogId_;        // = 80
    resp.last_matched_log_term_ref() = lastLogTerm_;    // = 3
    return;
  }

  // 找到 prevLogId 在本地的位置
  auto termInWal = wal_->getLogTerm(req.get_last_log_id_sent());
  if (termInWal != req.get_last_log_term_sent()) {
    // term 不匹配，需要回滚
    wal_->rollbackToLog(req.get_last_log_id_sent() - 1);
    resp.last_matched_log_id_ref() = req.get_last_log_id_sent() - 1;
  }
}
```

##### 步骤 3: Leader 重新发送正确的日志

```cpp
// Leader 收到 Node2 的响应，知道需要从 logId=80 开始发送
void RaftPart::replicateLogs() {
  // 为 Node2 准备从 logId=81 开始的日志
  auto walIt = wal_->iterator(81, 120);

  // 发送日志 [81, 82, ..., 120]
  while (walIt->valid()) {
    // 构造 AppendLogRequest
    // prevLogId = 80, prevLogTerm = 3
    // logs = [81, 82, ..., 90] (分批发送)
    ++(*walIt);
  }
}
```

##### 步骤 4: Node2 重放追赶日志

```cpp
void RaftPart::processAppendLogRequest(const cpp2::AppendLogRequest& req,
                                       cpp2::AppendLogResponse& resp) {
  // Node2 收到从 81 开始的日志
  // prevLogId=80, prevLogTerm=3 ✓ 匹配

  // 追加日志到 WAL
  for (size_t i = 0; i < req.get_log_str_list().size(); ++i) {
    LogID logId = req.get_last_log_id_sent() + 1 + i;
    wal_->appendLog(logId, req.get_current_term(), clusterId_,
                   req.get_log_str_list()[i]);
  }

  lastLogId_ = req.get_last_log_id_sent() + req.get_log_str_list().size();
  lastLogTerm_ = req.get_current_term();

  // 提交可以提交的日志
  LogID canCommit = std::min(lastLogId_, req.get_committed_log_id());
  if (canCommit > committedLogId_) {
    auto walIt = wal_->iterator(committedLogId_ + 1, canCommit);
    auto result = commitLogs(std::move(walIt), false, true);

    if (std::get<0>(result) == nebula::cpp2::ErrorCode::SUCCEEDED) {
      committedLogId_ = std::get<1>(result);
      committedLogTerm_ = std::get<2>(result);
    }
  }
}
```

---

### 场景 3: Vector 数据的重放

#### Vector 操作的 WAL 记录

```cpp
// 原始操作：插入 vector 数据
auto vectorKey = NebulaKeyUtils::vectorTagKey(8, 1, "vertex1", 1, 1);
auto vectorValue = encodeVector({1.0, 2.0, 3.0, 4.0});

// WAL 中记录为 OP_PUT
std::string logEntry;
logEntry.append(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));
logEntry.append(1, OP_PUT);  // 操作类型
logEntry.append(encodeKeyValue(vectorKey, vectorValue));
```

#### Vector 数据的重放过程

```cpp
std::tuple<nebula::cpp2::ErrorCode, LogID, TermID> Part::commitLogs(
    std::unique_ptr<LogIterator> iter, bool wait, bool needLock) {

  while (iter->valid()) {
    auto log = iter->logMsg();

    switch (log[sizeof(int64_t)]) {
      case OP_PUT: {
        auto pieces = decodeMultiValues(log);

        if (NebulaKeyUtils::isVector(pieces[0].str())) {
          // Vector 数据的特殊处理
          VLOG(3) << "Replaying vector data: " << folly::hexlify(pieces[0]);

          // 1. 先写入 VectorIndexWal
          if (vectorIndexWal_) {
            auto vectorId = static_cast<VectorID>(
                std::hash<std::string>{}(pieces[0].str()));

            auto appendResult = vectorIndexWal_->appendEntry(
                vector::VectorWalOp::INSERT, vectorId, pieces[1]);

            if (!appendResult.ok()) {
              LOG(ERROR) << "Failed to append to VectorIndexWal during replay: "
                         << appendResult.status();
              return {nebula::cpp2::ErrorCode::E_UNKNOWN, kNoCommitLogId, kNoCommitLogTerm};
            }

            VLOG(3) << "Vector entry replayed to VectorIndexWal with LogID: "
                    << appendResult.value();
          }

          // 2. 然后写入 RocksDB vector column family
          code = batch->put(NebulaKeyUtils::kVectorColumnFamilyName, pieces[0], pieces[1]);
        } else {
          // 普通数据直接写入默认 column family
          code = batch->put(pieces[0], pieces[1]);
        }
        break;
      }

      case OP_REMOVE: {
        auto key = decodeSingleValue(log);

        if (NebulaKeyUtils::isVector(key.str())) {
          // Vector 删除操作
          if (vectorIndexWal_) {
            auto vectorId = static_cast<VectorID>(
                std::hash<std::string>{}(key.str()));

            auto appendResult = vectorIndexWal_->appendEntry(
                vector::VectorWalOp::DELETE, vectorId, folly::StringPiece{});

            if (!appendResult.ok()) {
              LOG(ERROR) << "Failed to append delete to VectorIndexWal: "
                         << appendResult.status();
              return {nebula::cpp2::ErrorCode::E_UNKNOWN, kNoCommitLogId, kNoCommitLogTerm};
            }
          }

          code = batch->remove(NebulaKeyUtils::kVectorColumnFamilyName, key);
        } else {
          code = batch->remove(key);
        }
        break;
      }
    }

    ++(*iter);
  }

  // 批量提交所有操作
  auto commitCode = engine_->commitBatchWrite(std::move(batch));
  return {commitCode, lastId, lastTerm};
}
```

#### Vector Index 重建

```cpp
// 节点重启后，可以通过 VectorIndexWal 重建 vector index
void rebuildVectorIndexFromWal() {
  if (!vectorIndexWal_) return;

  auto iterator = vectorIndexWal_->iterator(0, -1);  // 所有日志

  while (iterator->valid()) {
    auto entryResult = iterator->entry();
    if (entryResult.ok()) {
      const auto& entry = entryResult.value();

      switch (entry.op) {
        case vector::VectorWalOp::INSERT:
          // 重建向量索引项
          rebuildVectorIndexEntry(entry.vectorId, entry.vectorData);
          break;

        case vector::VectorWalOp::DELETE:
          // 从索引中删除
          removeFromVectorIndex(entry.vectorId);
          break;

        case vector::VectorWalOp::UPDATE:
          // 更新索引项
          updateVectorIndexEntry(entry.vectorId, entry.vectorData);
          break;
      }
    }

    iterator->next();
  }

  VLOG(1) << "Vector index rebuilt from VectorIndexWal";
}
```

---

### 监控和调试

#### 重要的日志输出

##### 启动时的状态恢复

```
I0809 10:30:15.123456 RaftPart.cpp:425] [Space 1, Part 1]: There are 2 peer hosts, and total 3 hosts. The quorum is 2, lastLogId 100, lastLogTerm 5, committedLogId 95, committedLogTerm 5
```

##### 日志重放进度

```
I0809 10:30:16.789012 Part.cpp:234] [Space 1, Part 1]: Committing logs from 96 to 100
I0809 10:30:16.789123 Part.cpp:245] [Space 1, Part 1]: Vector entry written to VectorIndexWal with LogID: 1001
I0809 10:30:16.789234 Part.cpp:278] [Space 1, Part 1]: Successfully committed logs up to 100
```

##### Snapshot 传输

```
I0809 10:30:20.456789 SnapshotManager.cpp:85] [Space 1, Part 1]: Finished snapshot, totalCount 10000, totalSize 104857600
I0809 10:30:20.567890 RaftPart.cpp:2031] [Space 1, Part 1]: Receive all snapshot, committedLogId 100, committedLogTerm 5
```

#### 性能指标监控

```cpp
// 重放延迟统计
stats::StatsManager::addValue(kCommitLogLatencyUs, elapsedTime);

// 复制延迟统计
stats::StatsManager::addValue(kReplicateLogLatencyUs, replicationTime);

// Vector WAL 统计
auto stats = vectorIndexWal_->getStats();
VLOG(1) << "VectorWal stats: entries=" << stats.totalEntries
        << ", files=" << stats.totalFiles
        << ", size=" << stats.totalSize;
```

这些示例展示了 Nebula 在不同场景下如何通过 WAL 重放机制确保数据一致性，特别是我们新增的 Vector Index WAL 如何与传统的重放流程集成。

## Snapshot 传输数据过程

### 概述

当新节点加入集群或节点重启后需要大量数据同步时，Nebula 使用 Snapshot 机制进行全量数据传输。对于 vector 数据，这个过程确保了数据的正确传输和后续 VectorIndexWal 的一致性维护。

### 完整流程图

```
Leader 节点                                新节点 (Follower)
     |                                         |
     |-- 检测到新节点落后太多 ------------------>|
     |                                         |-- 节点启动，WAL 为空
     |                                         |
     |==== Snapshot 发送阶段 ====              |==== Snapshot 接收阶段 ====
     |                                         |
     |-- accessAllRowsInSnapshot() ----------->|
     |    扫描所有数据（包括 vector）            |
     |                                         |-- processSendSnapshotRequest()
     |-- 分批发送数据 ----------------------->|-- 验证 Leader 身份
     |    (SendSnapshotRequest)                |-- reset() 清理状态
     |                                         |-- status_ = WAITING_SNAPSHOT
     |                                         |
     |-- 发送 vector 数据批次 ----------------->|-- commitSnapshot() 接收数据
     |    key: vector key                     |    直接写入 vector column family
     |    value: vector 向量数据               |    ⚠️ 不触发 VectorIndexWal
     |                                         |
     |-- 发送完成标记 ----------------------->|-- finished=true 完成 snapshot
     |    (finished=true)                     |-- 更新 committedLogId/Term
     |                                         |-- status_ = RUNNING
     |                                         |
     |==== 增量日志复制阶段 ====               |==== 增量日志接收阶段 ====
     |                                         |
     |-- appendLogs() 发送新日志 -------------->|-- processAppendLogRequest()
     |    包含新的 vector 操作                 |-- 接收新 vector 操作日志
     |                                         |
     |                                         |-- commitLogs() 重放日志
     |                                         |    ✅ 触发 VectorIndexWal 写入
     |                                         |    ✅ 写入 vector column family
```

### 代码实现详解

#### 1. Leader 端：数据扫描和发送

##### 1.1 触发 Snapshot 发送

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/Host.cpp:349`

```cpp
nebula::cpp2::ErrorCode Host::startSendSnapshot() {
  CHECK(!lock_.try_lock());
  if (!sendingSnapshot_) {
    VLOG(1) << idStr_ << "Can't find log " << lastLogIdSent_ + 1 << " in wal, send the snapshot"
            << ", logIdToSend = " << logIdToSend_
            << ", firstLogId in wal = " << part_->wal()->firstLogId()
            << ", lastLogId in wal = " << part_->wal()->lastLogId();

    sendingSnapshot_ = true;
    stats::StatsManager::addValue(kNumSendSnapshot);

    // 调用 SnapshotManager 发送 snapshot
    part_->snapshot_->sendSnapshot(part_, addr_)
        .thenValue([self = shared_from_this()](auto&& status) {
          // 处理发送结果
        });
  }
}
```

##### 1.2 扫描所有数据（包括 vector 数据）

**文件位置**: `/home/lzy/my-nebula/src/kvstore/NebulaSnapshotManager.cpp:32`

```cpp
void NebulaSnapshotManager::accessAllRowsInSnapshot(GraphSpaceID spaceId,
                                                    PartitionID partId,
                                                    raftex::SnapshotCallback cb) {
  // 获取分区和 RocksDB snapshot
  auto partRet = store_->part(spaceId, partId);
  auto snapshot = store_->GetSnapshot(spaceId, partId);

  // 获取 committed log 信息
  std::string val;
  auto commitRet = part->engine()->get(NebulaKeyUtils::systemCommitKey(partId), &val, snapshot);
  LogID commitLogId;
  TermID commitLogTerm;
  memcpy(&commitLogId, val.data(), sizeof(LogID));
  memcpy(&commitLogTerm, val.data() + sizeof(LogID), sizeof(TermID));

  // 扫描所有前缀的数据，包括 vector 数据
  std::vector<std::string> prefixes{
    NebulaKeyUtils::systemPrefix(),       // 系统数据
    NebulaKeyUtils::tagPrefix(partId),    // Tag 数据
    NebulaKeyUtils::edgePrefix(partId),   // Edge 数据
    NebulaKeyUtils::vectorTagPrefix(partId),  // ⭐ Vector Tag 数据
    NebulaKeyUtils::vectorEdgePrefix(partId), // ⭐ Vector Edge 数据
    NebulaKeyUtils::indexPrefix(partId),  // 索引数据
    // ... 其他前缀
  };

  for (const auto& prefix : prefixes) {
    bool hasData = accessTable(spaceId, partId, snapshot, prefix, cb,
                              commitLogId, commitLogTerm, data, totalCount, totalSize, rateLimiter);
    if (!hasData) break;
  }
}
```

##### 1.3 分批发送数据

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/SnapshotManager.cpp:121`

```cpp
folly::Future<raftex::cpp2::SendSnapshotResponse> SnapshotManager::send(
    GraphSpaceID spaceId, PartitionID partId, TermID termId,
    LogID committedLogId, TermID committedLogTerm,
    const HostAddr& localhost, const std::vector<std::string>& data,
    int64_t totalSize, int64_t totalCount,
    const HostAddr& addr, bool finished) {

  // 构造发送请求
  raftex::cpp2::SendSnapshotRequest req;
  req.space_ref() = spaceId;
  req.part_ref() = partId;
  req.current_term_ref() = termId;
  req.committed_log_id_ref() = committedLogId;
  req.committed_log_term_ref() = committedLogTerm;
  req.leader_addr_ref() = localhost.host;
  req.leader_port_ref() = localhost.port;
  req.rows_ref() = data;  // ⭐ 包含 vector 数据的行
  req.total_size_ref() = totalSize;
  req.total_count_ref() = totalCount;
  req.done_ref() = finished;

  // 发送到目标节点
  return client->future_sendSnapshot(req);
}
```

#### 2. Follower 端：接收和安装 Snapshot

##### 2.1 接收 Snapshot 请求

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/RaftPart.cpp:1956`

```cpp
void RaftPart::processSendSnapshotRequest(const cpp2::SendSnapshotRequest& req,
                                          cpp2::SendSnapshotResponse& resp) {
  std::lock_guard<std::mutex> g(raftLock_);

  // 检查状态和验证 Leader 身份
  if (status_ != Status::WAITING_SNAPSHOT) {
    VLOG(2) << idStr_ << "Begin to receive the snapshot";

    // 验证 Leader 身份
    auto err = verifyLeader<cpp2::SendSnapshotRequest>(req);
    if (err != nebula::cpp2::ErrorCode::SUCCEEDED) {
      resp.error_code_ref() = err;
      return;
    }

    // ⭐ 重置状态，准备接收 snapshot
    reset();  // 清理所有本地状态，包括 WAL
    status_ = Status::WAITING_SNAPSHOT;
    lastSnapshotCommitId_ = req.get_committed_log_id();
    lastSnapshotCommitTerm_ = req.get_committed_log_term();
  }

  // ⭐ 提交 snapshot 数据到状态机
  auto ret = commitSnapshot(req.get_rows(),
                           req.get_committed_log_id(),
                           req.get_committed_log_term(),
                           req.get_done());

  if (req.get_done()) {
    // Snapshot 接收完成
    committedLogId_ = req.get_committed_log_id();
    committedLogTerm_ = req.get_committed_log_term();
    lastLogId_ = committedLogId_;
    lastLogTerm_ = committedLogTerm_;
    status_ = Status::RUNNING;  // ⭐ 恢复运行状态

    VLOG(1) << idStr_ << "Receive all snapshot, committedLogId " << committedLogId_
            << ", committedLogTerm " << committedLogTerm_;
  }
}
```

##### 2.2 提交 Snapshot 数据（关键部分）

**文件位置**: `/home/lzy/my-nebula/src/kvstore/Part.cpp:519`

```cpp
std::tuple<nebula::cpp2::ErrorCode, int64_t, int64_t> Part::commitSnapshot(
    const std::vector<std::string>& rows,
    LogID committedLogId,
    TermID committedLogTerm,
    bool finished) {

  auto batch = engine_->startBatchWrite();
  int64_t count = 0;
  int64_t size = 0;

  for (auto& row : rows) {
    count++;
    size += row.size();

    // ⭐ 解码 key-value 对
    auto kv = decodeKV(row);

    // ⚠️ 重要：这里直接调用 batch->put()，不会触发 VectorIndexWal 写入
    // 因为这是 snapshot 安装过程，不是日志重放过程
    auto code = batch->put(kv.first, kv.second);
    if (code != nebula::cpp2::ErrorCode::SUCCEEDED) {
      VLOG(3) << idStr_ << "Failed to call WriteBatch::put()";
      return {code, kNoSnapshotCount, kNoSnapshotSize};
    }
  }

  if (finished) {
    // 记录 committed log 信息
    auto code = putCommitMsg(batch.get(), committedLogId, committedLogTerm);
    if (code != nebula::cpp2::ErrorCode::SUCCEEDED) {
      return {code, kNoSnapshotCount, kNoSnapshotSize};
    }
  }

  // ⭐ 提交到 RocksDB（包括 vector column family）
  auto code = engine_->commitBatchWrite(std::move(batch),
                                       FLAGS_rocksdb_disable_wal,
                                       FLAGS_rocksdb_wal_sync,
                                       true);
  return {code, count, size};
}
```

#### 3. Snapshot 完成后的增量日志复制

##### 3.1 Leader 发送增量日志

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/RaftPart.cpp:925`

```cpp
void RaftPart::replicateLogs(folly::EventBase* eb, AppendLogsIterator iter, /* ... */) {
  // 向所有 Follower 发送日志
  collectNSucceeded(
    gen::from(hosts) |
    gen::map([=](std::shared_ptr<Host> hostPtr) {
      return hostPtr->appendLogs(eb, currTerm, lastLogId, committedId, prevLogTerm, prevLogId);
    }),
    quorum_,  // 需要多数派确认
    [](auto& resp) { return resp.get_error_code() == nebula::cpp2::ErrorCode::SUCCEEDED; }
  );
}
```

##### 3.2 Follower 接收增量日志并重放

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/RaftPart.cpp:1787`

```cpp
void RaftPart::processAppendLogRequest(const cpp2::AppendLogRequest& req,
                                       cpp2::AppendLogResponse& resp) {
  // 追加日志到 WAL
  for (size_t i = 0; i < req.get_log_str_list().size(); ++i) {
    LogID logId = req.get_last_log_id_sent() + 1 + i;
    wal_->appendLog(logId, req.get_current_term(), clusterId_, req.get_log_str_list()[i]);
  }

  // ⭐ 提交可以提交的日志（这里会触发 VectorIndexWal 写入）
  LogID canCommit = std::min(lastLogId_, req.get_committed_log_id());
  if (canCommit > committedLogId_) {
    auto walIt = wal_->iterator(committedLogId_ + 1, canCommit);
    auto result = commitLogs(std::move(walIt), false, true);  // ⭐ 调用我们修改的 commitLogs
  }
}
```

##### 3.3 commitLogs 处理 Vector 数据（我们的修改）

**文件位置**: `/home/lzy/my-nebula/src/kvstore/Part.cpp:234`

```cpp
std::tuple<nebula::cpp2::ErrorCode, LogID, TermID> Part::commitLogs(
    std::unique_ptr<LogIterator> iter, bool wait, bool needLock) {

  auto batch = engine_->startBatchWrite();

  while (iter->valid()) {
    auto log = iter->logMsg();

    switch (log[sizeof(int64_t)]) {
      case OP_PUT: {
        auto pieces = decodeMultiValues(log);

        if (NebulaKeyUtils::isVector(pieces[0].str())) {
          // ⭐ Vector 数据的特殊处理：先写 VectorIndexWal，再写 RocksDB
          if (vectorIndexWal_) {
            auto vectorId = static_cast<VectorID>(std::hash<std::string>{}(pieces[0].str()));
            auto appendResult = vectorIndexWal_->appendEntry(
                vector::VectorWalOp::INSERT, vectorId, pieces[1]);

            if (!appendResult.ok()) {
              LOG(ERROR) << idStr_ << "Failed to append vector entry to VectorIndexWal: "
                         << appendResult.status().toString();
              return {nebula::cpp2::ErrorCode::E_UNKNOWN, kNoCommitLogId, kNoCommitLogTerm};
            }

            VLOG(3) << idStr_ << "Vector entry written to VectorIndexWal with LogID: "
                    << appendResult.value();
          }

          // 然后写入 vector column family
          code = batch->put(NebulaKeyUtils::kVectorColumnFamilyName, pieces[0], pieces[1]);
        } else {
          code = batch->put(pieces[0], pieces[1]);
        }
        break;
      }
      // ... 其他操作类型的处理
    }
    ++(*iter);
  }

  // 批量提交到 RocksDB
  auto code = engine_->commitBatchWrite(std::move(batch));
  return {code, lastId, lastTerm};
}
```

### 关键差异对比

#### Snapshot 阶段 vs 增量日志阶段

| 阶段              | Vector 数据处理方式                     | VectorIndexWal 写入 | 代码位置                 |
| ----------------- | --------------------------------------- | ------------------- | ------------------------ |
| **Snapshot 传输** | 直接写入 vector column family           | ❌ **不写入**       | `Part::commitSnapshot()` |
| **增量日志复制**  | 先写 VectorIndexWal，再写 column family | ✅ **写入**         | `Part::commitLogs()`     |

#### 为什么 Snapshot 阶段不写入 VectorIndexWal？

1. **性能考虑**: Snapshot 是全量数据传输，如果每个 vector 都写 VectorIndexWal，会产生大量额外 I/O
2. **一致性保证**: Snapshot 代表的是一个一致性状态点，不需要额外的 WAL 记录
3. **重建机制**: 后续的增量日志会确保 VectorIndexWal 的完整性

### Vector 索引状态的最终一致性

#### 1. Snapshot 完成后的状态

```
新节点状态：
- RocksDB vector column family: ✅ 包含所有 vector 数据
- VectorIndexWal: ❌ 为空（没有历史记录）
- Vector Index: ❌ 需要重建
```

#### 2. 增量日志复制后的状态

```
新节点状态：
- RocksDB vector column family: ✅ 包含最新 vector 数据
- VectorIndexWal: ✅ 包含 snapshot 后的所有 vector 操作
- Vector Index: ✅ 可以从 VectorIndexWal 重建 + 增量更新
```

#### 3. Vector Index 重建策略

```cpp
// 建议的重建逻辑
void rebuildVectorIndexAfterSnapshot() {
  // 方案 1: 从 RocksDB 重建整个索引
  rebuildVectorIndexFromRocksDB();

  // 方案 2: 从 VectorIndexWal 重建增量部分
  rebuildVectorIndexFromWal();

  // 方案 3: 混合方案
  if (vectorIndexWal_->getStats().totalEntries < threshold) {
    // WAL 记录较少，从 RocksDB 全量重建
    rebuildVectorIndexFromRocksDB();
  } else {
    // WAL 记录较多，增量重建
    rebuildVectorIndexFromWal();
  }
}
```

## Vector WAL 的设计

### 概述

当新节点加入集群或节点重启后需要大量数据同步时，Nebula 使用 Snapshot 机制进行全量数据传输。对于 vector 数据，这个过程确保了数据的正确传输和后续 VectorIndexWal 的一致性维护。

### 完整流程图

```
Leader 节点                                新节点 (Follower)
     |                                         |
     |-- 检测到新节点落后太多 ------------------>|
     |                                         |-- 节点启动，WAL 为空
     |                                         |
     |==== Snapshot 发送阶段 ====              |==== Snapshot 接收阶段 ====
     |                                         |
     |-- accessAllRowsInSnapshot() ----------->|
     |    扫描所有数据（包括 vector）            |
     |                                         |-- processSendSnapshotRequest()
     |-- 分批发送数据 ----------------------->|-- 验证 Leader 身份
     |    (SendSnapshotRequest)                |-- reset() 清理状态
     |                                         |-- status_ = WAITING_SNAPSHOT
     |                                         |
     |-- 发送 vector 数据批次 ----------------->|-- commitSnapshot() 接收数据
     |    key: vector key                     |    直接写入 vector column family
     |    value: vector 向量数据               |    ⚠️ 不触发 VectorIndexWal
     |                                         |
     |-- 发送完成标记 ----------------------->|-- finished=true 完成 snapshot
     |    (finished=true)                     |-- 更新 committedLogId/Term
     |                                         |-- status_ = RUNNING
     |                                         |
     |==== 增量日志复制阶段 ====               |==== 增量日志接收阶段 ====
     |                                         |
     |-- appendLogs() 发送新日志 -------------->|-- processAppendLogRequest()
     |    包含新的 vector 操作                 |-- 接收新 vector 操作日志
     |                                         |
     |                                         |-- commitLogs() 重放日志
     |                                         |    ✅ 触发 VectorIndexWal 写入
     |                                         |    ✅ 写入 vector column family
```

### 代码实现详解

#### 1. Leader 端：数据扫描和发送

##### 1.1 触发 Snapshot 发送

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/Host.cpp:349`

```cpp
nebula::cpp2::ErrorCode Host::startSendSnapshot() {
  CHECK(!lock_.try_lock());
  if (!sendingSnapshot_) {
    VLOG(1) << idStr_ << "Can't find log " << lastLogIdSent_ + 1 << " in wal, send the snapshot"
            << ", logIdToSend = " << logIdToSend_
            << ", firstLogId in wal = " << part_->wal()->firstLogId()
            << ", lastLogId in wal = " << part_->wal()->lastLogId();

    sendingSnapshot_ = true;
    stats::StatsManager::addValue(kNumSendSnapshot);

    // 调用 SnapshotManager 发送 snapshot
    part_->snapshot_->sendSnapshot(part_, addr_)
        .thenValue([self = shared_from_this()](auto&& status) {
          // 处理发送结果
        });
  }
}
```

##### 1.2 扫描所有数据（包括 vector 数据）

**文件位置**: `/home/lzy/my-nebula/src/kvstore/NebulaSnapshotManager.cpp:32`

```cpp
void NebulaSnapshotManager::accessAllRowsInSnapshot(GraphSpaceID spaceId,
                                                    PartitionID partId,
                                                    raftex::SnapshotCallback cb) {
  // 获取分区和 RocksDB snapshot
  auto partRet = store_->part(spaceId, partId);
  auto snapshot = store_->GetSnapshot(spaceId, partId);

  // 获取 committed log 信息
  std::string val;
  auto commitRet = part->engine()->get(NebulaKeyUtils::systemCommitKey(partId), &val, snapshot);
  LogID commitLogId;
  TermID commitLogTerm;
  memcpy(&commitLogId, val.data(), sizeof(LogID));
  memcpy(&commitLogTerm, val.data() + sizeof(LogID), sizeof(TermID));

  // 扫描所有前缀的数据，包括 vector 数据
  std::vector<std::string> prefixes{
    NebulaKeyUtils::systemPrefix(),       // 系统数据
    NebulaKeyUtils::tagPrefix(partId),    // Tag 数据
    NebulaKeyUtils::edgePrefix(partId),   // Edge 数据
    NebulaKeyUtils::vectorTagPrefix(partId),  // ⭐ Vector Tag 数据
    NebulaKeyUtils::vectorEdgePrefix(partId), // ⭐ Vector Edge 数据
    NebulaKeyUtils::indexPrefix(partId),  // 索引数据
    // ... 其他前缀
  };

  for (const auto& prefix : prefixes) {
    bool hasData = accessTable(spaceId, partId, snapshot, prefix, cb,
                              commitLogId, commitLogTerm, data, totalCount, totalSize, rateLimiter);
    if (!hasData) break;
  }
}
```

##### 1.3 分批发送数据

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/SnapshotManager.cpp:121`

```cpp
folly::Future<raftex::cpp2::SendSnapshotResponse> SnapshotManager::send(
    GraphSpaceID spaceId, PartitionID partId, TermID termId,
    LogID committedLogId, TermID committedLogTerm,
    const HostAddr& localhost, const std::vector<std::string>& data,
    int64_t totalSize, int64_t totalCount,
    const HostAddr& addr, bool finished) {

  // 构造发送请求
  raftex::cpp2::SendSnapshotRequest req;
  req.space_ref() = spaceId;
  req.part_ref() = partId;
  req.current_term_ref() = termId;
  req.committed_log_id_ref() = committedLogId;
  req.committed_log_term_ref() = committedLogTerm;
  req.leader_addr_ref() = localhost.host;
  req.leader_port_ref() = localhost.port;
  req.rows_ref() = data;  // ⭐ 包含 vector 数据的行
  req.total_size_ref() = totalSize;
  req.total_count_ref() = totalCount;
  req.done_ref() = finished;

  // 发送到目标节点
  return client->future_sendSnapshot(req);
}
```

#### 2. Follower 端：接收和安装 Snapshot

##### 2.1 接收 Snapshot 请求

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/RaftPart.cpp:1956`

```cpp
void RaftPart::processSendSnapshotRequest(const cpp2::SendSnapshotRequest& req,
                                          cpp2::SendSnapshotResponse& resp) {
  std::lock_guard<std::mutex> g(raftLock_);

  // 检查状态和验证 Leader 身份
  if (status_ != Status::WAITING_SNAPSHOT) {
    VLOG(2) << idStr_ << "Begin to receive the snapshot";

    // 验证 Leader 身份
    auto err = verifyLeader<cpp2::SendSnapshotRequest>(req);
    if (err != nebula::cpp2::ErrorCode::SUCCEEDED) {
      resp.error_code_ref() = err;
      return;
    }

    // ⭐ 重置状态，准备接收 snapshot
    reset();  // 清理所有本地状态，包括 WAL
    status_ = Status::WAITING_SNAPSHOT;
    lastSnapshotCommitId_ = req.get_committed_log_id();
    lastSnapshotCommitTerm_ = req.get_committed_log_term();
  }

  // ⭐ 提交 snapshot 数据到状态机
  auto ret = commitSnapshot(req.get_rows(),
                           req.get_committed_log_id(),
                           req.get_committed_log_term(),
                           req.get_done());

  if (req.get_done()) {
    // Snapshot 接收完成
    committedLogId_ = req.get_committed_log_id();
    committedLogTerm_ = req.get_committed_log_term();
    lastLogId_ = committedLogId_;
    lastLogTerm_ = committedLogTerm_;
    status_ = Status::RUNNING;  // ⭐ 恢复运行状态

    VLOG(1) << idStr_ << "Receive all snapshot, committedLogId " << committedLogId_
            << ", committedLogTerm " << committedLogTerm_;
  }
}
```

##### 2.2 提交 Snapshot 数据（关键部分）

**文件位置**: `/home/lzy/my-nebula/src/kvstore/Part.cpp:519`

```cpp
std::tuple<nebula::cpp2::ErrorCode, int64_t, int64_t> Part::commitSnapshot(
    const std::vector<std::string>& rows,
    LogID committedLogId,
    TermID committedLogTerm,
    bool finished) {

  auto batch = engine_->startBatchWrite();
  int64_t count = 0;
  int64_t size = 0;

  for (auto& row : rows) {
    count++;
    size += row.size();

    // ⭐ 解码 key-value 对
    auto kv = decodeKV(row);

    // ⚠️ 重要：这里直接调用 batch->put()，不会触发 VectorIndexWal 写入
    // 因为这是 snapshot 安装过程，不是日志重放过程
    auto code = batch->put(kv.first, kv.second);
    if (code != nebula::cpp2::ErrorCode::SUCCEEDED) {
      VLOG(3) << idStr_ << "Failed to call WriteBatch::put()";
      return {code, kNoSnapshotCount, kNoSnapshotSize};
    }
  }

  if (finished) {
    // 记录 committed log 信息
    auto code = putCommitMsg(batch.get(), committedLogId, committedLogTerm);
    if (code != nebula::cpp2::ErrorCode::SUCCEEDED) {
      return {code, kNoSnapshotCount, kNoSnapshotSize};
    }
  }

  // ⭐ 提交到 RocksDB（包括 vector column family）
  auto code = engine_->commitBatchWrite(std::move(batch),
                                       FLAGS_rocksdb_disable_wal,
                                       FLAGS_rocksdb_wal_sync,
                                       true);
  return {code, count, size};
}
```

#### 3. Snapshot 完成后的增量日志复制

##### 3.1 Leader 发送增量日志

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/RaftPart.cpp:925`

```cpp
void RaftPart::replicateLogs(folly::EventBase* eb, AppendLogsIterator iter, /* ... */) {
  // 向所有 Follower 发送日志
  collectNSucceeded(
    gen::from(hosts) |
    gen::map([=](std::shared_ptr<Host> hostPtr) {
      return hostPtr->appendLogs(eb, currTerm, lastLogId, committedId, prevLogTerm, prevLogId);
    }),
    quorum_,  // 需要多数派确认
    [](auto& resp) { return resp.get_error_code() == nebula::cpp2::ErrorCode::SUCCEEDED; }
  );
}
```

##### 3.2 Follower 接收增量日志并重放

**文件位置**: `/home/lzy/my-nebula/src/kvstore/raftex/RaftPart.cpp:1787`

```cpp
void RaftPart::processAppendLogRequest(const cpp2::AppendLogRequest& req,
                                       cpp2::AppendLogResponse& resp) {
  // 追加日志到 WAL
  for (size_t i = 0; i < req.get_log_str_list().size(); ++i) {
    LogID logId = req.get_last_log_id_sent() + 1 + i;
    wal_->appendLog(logId, req.get_current_term(), clusterId_, req.get_log_str_list()[i]);
  }

  // ⭐ 提交可以提交的日志（这里会触发 VectorIndexWal 写入）
  LogID canCommit = std::min(lastLogId_, req.get_committed_log_id());
  if (canCommit > committedLogId_) {
    auto walIt = wal_->iterator(committedLogId_ + 1, canCommit);
    auto result = commitLogs(std::move(walIt), false, true);  // ⭐ 调用我们修改的 commitLogs
  }
}
```

##### 3.3 commitLogs 处理 Vector 数据（我们的修改）

**文件位置**: `/home/lzy/my-nebula/src/kvstore/Part.cpp:234`

```cpp
std::tuple<nebula::cpp2::ErrorCode, LogID, TermID> Part::commitLogs(
    std::unique_ptr<LogIterator> iter, bool wait, bool needLock) {

  auto batch = engine_->startBatchWrite();

  while (iter->valid()) {
    auto log = iter->logMsg();

    switch (log[sizeof(int64_t)]) {
      case OP_PUT: {
        auto pieces = decodeMultiValues(log);

        if (NebulaKeyUtils::isVector(pieces[0].str())) {
          // ⭐ Vector 数据的特殊处理：先写 VectorIndexWal，再写 RocksDB
          if (vectorIndexWal_) {
            auto vectorId = static_cast<VectorID>(std::hash<std::string>{}(pieces[0].str()));
            auto appendResult = vectorIndexWal_->appendEntry(
                vector::VectorWalOp::INSERT, vectorId, pieces[1]);

            if (!appendResult.ok()) {
              LOG(ERROR) << idStr_ << "Failed to append vector entry to VectorIndexWal: "
                         << appendResult.status().toString();
              return {nebula::cpp2::ErrorCode::E_UNKNOWN, kNoCommitLogId, kNoCommitLogTerm};
            }

            VLOG(3) << idStr_ << "Vector entry written to VectorIndexWal with LogID: "
                    << appendResult.value();
          }

          // 然后写入 vector column family
          code = batch->put(NebulaKeyUtils::kVectorColumnFamilyName, pieces[0], pieces[1]);
        } else {
          code = batch->put(pieces[0], pieces[1]);
        }
        break;
      }
      // ... 其他操作类型的处理
    }
    ++(*iter);
  }

  // 批量提交到 RocksDB
  auto code = engine_->commitBatchWrite(std::move(batch));
  return {code, lastId, lastTerm};
}
```

### 关键差异对比

#### Snapshot 阶段 vs 增量日志阶段

| 阶段              | Vector 数据处理方式                     | VectorIndexWal 写入 | 代码位置                 |
| ----------------- | --------------------------------------- | ------------------- | ------------------------ |
| **Snapshot 传输** | 直接写入 vector column family           | ❌ **不写入**       | `Part::commitSnapshot()` |
| **增量日志复制**  | 先写 VectorIndexWal，再写 column family | ✅ **写入**         | `Part::commitLogs()`     |

#### 为什么 Snapshot 阶段不写入 VectorIndexWal？

1. **性能考虑**: Snapshot 是全量数据传输，如果每个 vector 都写 VectorIndexWal，会产生大量额外 I/O
2. **一致性保证**: Snapshot 代表的是一个一致性状态点，不需要额外的 WAL 记录
3. **重建机制**: 后续的增量日志会确保 VectorIndexWal 的完整性

### Vector 索引状态的最终一致性

#### 1. Snapshot 完成后的状态

```
新节点状态：
- RocksDB vector column family: ✅ 包含所有 vector 数据
- VectorIndexWal: ❌ 为空（没有历史记录）
- Vector Index: ❌ 需要重建
```

#### 2. 增量日志复制后的状态

```
新节点状态：
- RocksDB vector column family: ✅ 包含最新 vector 数据
- VectorIndexWal: ✅ 包含 snapshot 后的所有 vector 操作
- Vector Index: ✅ 可以从 VectorIndexWal 重建 + 增量更新
```

#### 3. Vector Index 重建策略

```cpp
// 建议的重建逻辑
void rebuildVectorIndexAfterSnapshot() {
  // 方案 1: 从 RocksDB 重建整个索引
  rebuildVectorIndexFromRocksDB();

  // 方案 2: 从 VectorIndexWal 重建增量部分
  rebuildVectorIndexFromWal();

  // 方案 3: 混合方案
  if (vectorIndexWal_->getStats().totalEntries < threshold) {
    // WAL 记录较少，从 RocksDB 全量重建
    rebuildVectorIndexFromRocksDB();
  } else {
    // WAL 记录较多，增量重建
    rebuildVectorIndexFromWal();
  }
}
```
