# Active Slam

## Overview

Active Slam 的工作流程:

- Perception: 传感器数据获取与预处理
- SLAM(Localization + Mapping + Loop Closure)
- Candidate Generation: 生成候选观测点(frontiers/viewpoints)
- Evaluator: 对每个候选估算：预期信息增益（raycast / mutual information / belief propagation）、执行代价（路径长度、能耗）、定位风险（协方差增长）、碰撞风险。
- Planner: 短期执行目标或控制序列（通常只执行第一步或首段，随后重新评估）
- Controller: 低级运动控制与避障
- Monitor & Replan: 监控执行状态，若出现意外（新障碍、定位不确定性过大等）则重新规划

```python
while not exploration_done: # we have stop standards
  obs = read_sensors() # RGB_D, Lidar, IMU, etc.
  pose, map, cov = SLAM.update(obs) # state estimation + loop closure + mapping
  candidates = CandidateGenerator(map, pose) # viewpoints
  scores = Evaluator(candidates, map, cov) # info gain - lambda * cost
  plan = Planner.select_and_plan(candidates, scores) # A*(global) + local planner
  execute_first_phase(plan) # Controller + obstacle replan + Monitor
```

**:warning: 何时停止探索？**

- 无新前沿 (No New Frontiers)： 最常见的终止条件是当探索算法再也无法在地图中找到任何已知空间与未知空间的边界（即“前沿”）时，系统会判定探索已完成并停止 。

- 覆盖率阈值 (Coverage Threshold)： 规划器可以被设定一个目标，例如当已建图区域达到预定环境总体积的 95%时，任务终止 。

- 预算耗尽 (Budget Depletion)： 可以给机器人设置明确的“预算”（如路径长度或续航时间）。当这个预算被完全消耗时，任务自然终止 。

- 人工干预 (Manual Intervention)： 在实际部署中，操作员可以随时通过远程指令终止任务，例如当他们对地图的完整性和精度感到满意时

## Active Planning

选择最优动作/路径，平衡信息增益与代价。

- 基于边界的策略
- 基于采样或优化的策略
- 基于 Reinforcement Learning 的策略

### Frontier-based Exploration

- 获取已知-未知的交界点作为候选目标
- raycast:
  从候选观测点（viewpoint）出发，向各个方向发射射线（rays），模拟传感器的视线；射线会穿过占据栅格地图（occupancy grid），直到：

  1. 碰到障碍物（occupied cell）

  2. 达到传感器最大探测距离（sensor range）

  在射线穿过的栅格中，统计那些当前地图中标记为 unknown（未探索） 的单元格数量；

  > 这些未知格子的数量就代表：从该 viewpoint 出发可以看到多少未知区域

```python
loop:
  update_map_with_SLAM() # state estimation + loop closure + mapping
  frontiers = detect_frontiers(map) # detect frontiers
  clusters = cluster_frontiers(frontiers) # group nearby frontiers
  for each cluster:
    I = estimate_information_gain(cluster)   # raycast / unknown cell count
    C = travel_cost(robot_pose, cluster.centroid) # cost function
    score = I - lambda * C # get low cost and high info gain
  goal = argmax(score)
  plan_path = global_planner(robot_pose, goal) # A* / Dijkstra / etc.
  execute_path(plan_path)  # local controller + obstacle replan
  if new_info or localization_uncertain: continue loop
```

- 优点

  - 简单高效：实现简单、实时性强，广泛应用于 2D 室内环境。

  - 无需全局模型：直接基于局部地图和边界信息，计算代价较低。

  - 渐进式探索：随着地图扩展自然进行，适合增量式 SLAM 框架。

- 局限性

  - 短视（Myopic）：每次只关注当前最有信息增益的边界，而不考虑全局探索效率。

  - 容易陷入局部最优：可能反复探索边缘区域或漏掉隐蔽空间。

  - 路径冗余：频繁的来回移动（“之”字形路径）增加能耗与时间。

  - 信息增益估计粗糙：仅基于地图几何，不考虑感知噪声、视野遮挡等因素。

### Sampling-based Exploration

采样型方法不依赖显式 frontier，而是在已知自由空间内采样若干视点（pose 或 pose+yaw），评估每个视点的 信息增益$I(v)$ 与 代价 $C(v)$，并选择效用函数 $U(v)=I(v)−λ⋅C(v)$ 最大的视点作为下一个目标。

- 采样观测点(viewpionts)

```python
loop:
  update_map_with_SLAM() # state estimation + loop closure + mapping
  candidates = sample_actions_or_goals() # get viewpoints
  for each candidate:
    I = estimate_information_gain(cluster)   # raycast / unknown cell count
    C = travel_cost(robot_pose, cluster.centroid) # cost function
    score = I - lambda * C # get low cost and high info gain
  goal = argmax(score)
  plan_path = global_planner(robot_pose, goal) # A* / Dijkstra / etc.
  execute_path(plan_path)  # local controller + obstacle replan
  if new_info or localization_uncertain: continue loop
```

- 优点

  - 更灵活：能探索复杂空间，不依赖规则边界。

  - 可扩展性强：能自然结合概率模型或学习策略。

  - 更易与信息理论框架结合：支持熵、互信息等信息量指标。

- 局限性

  - 计算代价高：每轮都需大量采样与信息增益评估。

  - 缺乏全局策略：与 frontier-based 类似，依旧是“greedy”式局部决策。

  - 短视问题依旧：仅依据当前地图和即时信息增益选点，未考虑长远路径收益。

  - 重复探索现象：容易绕远路或回头探索已部分观测区域。
