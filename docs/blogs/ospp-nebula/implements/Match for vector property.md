# Match for Vector Property

## Simplest match case

```cypher
MATCH (v)
RETURN v
LIMIT 3;
```

### Process

1. MatchValidator 验证阶段
   在验证阶段，节点 (v) 被识别为：

   - 没有指定标签的节点模式
   - 没有属性过滤条件
   - 别名为 v，类型为 AliasType::kNode

2. MatchPathPlanner 路径规划阶段
   StartVidFinder 寻找起始点
   在 MatchPathPlanner::findStarts() 中

```cpp
// 遍历所有的 StartVidFinder
for (auto& finder : startVidFinders) {
    for (size_t i = 0; i < nodeInfos.size() && !foundStart; ++i) {
        NodeContext nodeCtx(qctx, bindWhereClause, spaceId, &nodeInfos[i]);
        nodeCtx.aliasesAvailable = &nodeAliasesSeen;
        auto nodeFinder = finder();
        if (nodeFinder->match(&nodeCtx)) {
            // 找到匹配的 finder，生成初始计划
        }
    }
}
```

对于 MATCH (v) 模式，会使用 ScanSeek：

ScanSeek::matchNode() 匹配

```cpp
bool ScanSeek::matchNode(NodeContext *nodeCtx) {
    auto &node = *nodeCtx->info;
    // 没有指定标签，获取所有标签
    if (node.tids.empty()) {
        // empty labels means all labels
        auto allLabels = qctx->schemaMng()->getAllTags(nodeCtx->spaceId);
        for (const auto &label : allLabels.value()) {
            nodeCtx->scanInfo.schemaIds.emplace_back(label.first);
            nodeCtx->scanInfo.schemaNames.emplace_back(label.second);
        }
        nodeCtx->scanInfo.anyLabel = true;
    }
    return true;
}
```

ScanSeek::transformNode() 生成 ScanVertices

```cpp
StatusOr<SubPlan> ScanSeek::transformNode(NodeContext *nodeCtx) {
    SubPlan plan;

    // 创建 ScanVertices 节点
    auto *scanVertices = ScanVertices::make(qctx, nullptr, nodeCtx->spaceId, std::move(vProps));
    plan.root = scanVertices;
    plan.tail = scanVertices;

    // 如果需要，添加标签过滤器
    if (prev != nullptr) {
        auto *filter = Filter::make(qctx, scanVertices, prev);
        plan.root = filter;
    }

    nodeCtx->initialExpr = InputPropertyExpression::make(pool, kVid);
    return plan;
}
```

1. 路径扩展阶段
   由于 MATCH (v) 只有一个节点，没有边，在 MatchPathPlanner::expandFromNode() 中:

```cpp
Status MatchPathPlanner::expandFromNode(size_t startIndex, SubPlan& subplan) {
    const auto& nodeInfos = path_.nodeInfos;
    DCHECK_LT(startIndex, nodeInfos.size());
    // 只有一个节点的情况，startIndex = 0 且 nodeInfos.size() = 1
    nodeAliasesSeenInPattern_.emplace(nodeInfos[startIndex].alias);

    if (startIndex == 0) {
        // Pattern: (start) - 没有后续边
        return rightExpandFromNode(startIndex, subplan);
    }
}
```

在 rightExpandFromNode() 中，由于没有边信息（edgeInfos.size() = 0），会直接跳过循环，执行最后的 AppendVertices：

```cpp
Status MatchPathPlanner::rightExpandFromNode(size_t startIndex, SubPlan& subplan) {
    const auto& nodeInfos = path_.nodeInfos;
    const auto& edgeInfos = path_.edgeInfos;
    // edgeInfos.size() = 0，所以 for 循环不执行

    auto& lastNode = nodeInfos.back(); // 即节点 v

    // 创建 AppendVertices 节点
    auto appendV = AppendVertices::make(qctx, subplan.root, spaceId);
    auto vertexProps = SchemaUtil::getAllVertexProp(qctx, spaceId, true);
    appendV->setVertexProps(std::move(vertexProps).value());
    appendV->setSrc(nextTraverseStart); // 来自 ScanVertices 的输出
    appendV->setVertexFilter(genVertexFilter(lastNode));
    appendV->setDedup();
    appendV->setTrackPrevPath(!edgeInfos.empty()); // false，因为没有边
    appendV->setColNames(genAppendVColNames(subplan.root->colNames(), lastNode, !edgeInfos.empty()));
    subplan.root = appendV;

    return Status::OK();
}
```

4. RETURN 和 LIMIT 处理
   YieldClausePlanner 处理 RETURN v
   处理 RETURN v 部分，生成 Project 节点来投影列 v。

PaginationPlanner 处理 LIMIT 3 5. 最终执行计划结构

生成 Limit -> AppendVertices -> ScanVertices 这样的三层执行计划。

- ScanVertices 只扫描顶点 ID 和标签信息，不包含顶点的完整属性
- AppendVertices 根据 ScanVertices 输出的顶点 ID，去获取完整的顶点属性数据

这种执行计划设计允许：

- 延迟加载：只有真正需要的顶点才会获取完整属性
- 内存效率：扫描阶段只处理 ID 和基本过滤，减少内存使用
- 可扩展性：支持复杂的图遍历模式

### Vector Property Support

- AppendVertices 支持向量属性的处理
  - 在 SchemaUtil::getAllVertexProp()中获取所有 tag 的 schema 信息，需要获取 vector property 的信息。
