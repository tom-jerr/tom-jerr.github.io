---

title: 12 Query Planning and Optimization
created: 2024-10-31
tags:

- Database

---

# Query Planning and Optimization

> catalog 是一个记录元数据信息的文件

![](https://github.com/tom-jerr/MyblogImg/raw/15445/simple_query_plan.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/join_query.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/sort_merge_query.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/pipeline_query.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/push_down_query.png)

## The Physical Plan

![](https://github.com/tom-jerr/MyblogImg/raw/15445/physical_plan.png)

## Query Optimization(QO)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/query_optimization.png)

### Heuristics/Rules

- 重写查询来去除那些无效的条件
- 这些技巧需要访问 catalog，但是它们不需要访问数据

#### Predicate Pushdown

![](https://github.com/tom-jerr/MyblogImg/raw/15445/pushdown.png)

#### Replace Cartesian Product

![](https://github.com/tom-jerr/MyblogImg/raw/15445/replace_product.png)

#### Projection Pushdown

![](https://github.com/tom-jerr/MyblogImg/raw/15445/projection_pushdown.png)

#### Equivalence

![](https://github.com/tom-jerr/MyblogImg/raw/15445/equivalence.png)

### Architecture Overview

![](https://github.com/tom-jerr/MyblogImg/raw/15445/architecture_overview.png)

### Cost-based Search

- 使用一个模型来预测执行一个计划的成本
- 遍历多种计划，选择一个成本最小的计划来执行

#### Bottom-up Optimization

> 使用动态规划，自底向上构建查询计划成本最低的计划

![](https://github.com/tom-jerr/MyblogImg/raw/15445/botton-up-optimization.png)

- single-relation query palnning

![](https://github.com/tom-jerr/MyblogImg/raw/15445/single_relation_query.png)

> system R optimization 将逻辑计划构建为左深树

![](https://github.com/tom-jerr/MyblogImg/raw/15445/systemRoptimization.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/systemRoptimization1.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/systemRoptimization2.png)

#### Top-Down Optimization

> 自顶向下的优化控制权更多，我们可以从一个计划开始逐步细化过程

![](https://github.com/tom-jerr/MyblogImg/raw/15445/top-down-optimization.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/top-down-optimization1.png)

#### Nested Sub-queries

> 相关子查询很容易被扁平化为 join 的查询\
> 不相关的子查可以拆分成不同的语句进行执行

![](https://github.com/tom-jerr/MyblogImg/raw/15445/nested_sub_query.png)

##### Rewrite

![](https://github.com/tom-jerr/MyblogImg/raw/15445/sub_query_rewrite.png)

##### Decomposing Queries

![](https://github.com/tom-jerr/MyblogImg/raw/15445/decomposing_queries.png)

#### Expression rerwriteing

![](https://github.com/tom-jerr/MyblogImg/raw/15445/expression_rewriting.png)

#### Cost Estimation

- Physical Costs

  > predict CPU-cycles, I/O, cache misses, RAM consumption, network messages...\
  > Depends heavily on hardware

- Logical Costs

  > estimate output size per operator\
  > independent of the operator algorithm\
  > need estimations for operator result sizes

![](https://github.com/tom-jerr/MyblogImg/raw/15445/postgres_cost_model.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/statistics.png)

##### selection cardinality

> 可以使用 selection cardinality 来推测输出的大小

![](https://github.com/tom-jerr/MyblogImg/raw/15445/select_cardinality.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/select_cardinality1.png)

##### Statistics

![](https://github.com/tom-jerr/MyblogImg/raw/15445/statistics1.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/histogram1.png)
![](https://github.com/tom-jerr/MyblogImg/raw/15445/histogram2.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/sketches.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/sampling.png)
