# KR 3D Active SLAM 系统架构详解

## 分层

### 传感器与输入层

实际系统使用以下输入源：

- VIO 里程计：/quadrotor/vio/odom
- 语义分割点云（同步的）：/sem_detection/sync_pc_odom
- 原始控制里程计：/dragonfly67/quadrotor_ukf/control_odom

### 感知与建图层（SLOAM 核心）

#### process_cloud_node (Python 节点)

**主要职责：**

- 接收语义分割点云，提取树木、车辆、地面等语义目标
  = 进行 DBSCAN 聚类识别车辆实例
  = 拟合 3D cuboid 包围盒
- 目标跟踪与数据关联
- 发布语义测量（range-bearing measurements）

**关键 Topic：**

订阅：/sem_detection/sync_pc_odom (语义分割点云+里程计)
发布：

- /semantic_range_bearing_measurements (ROSRangeBearing 消息)
- /tree_cloud (树木点云)
- /ground_cloud (地面点云)
- /car_cuboids (车辆包围盒 MarkerArray)
- /quadrotor/lidar_odom (里程计)

#### ActiveSlamInputNode (C++核心 SLAM 节点)

**主要职责：**

- 接收 VIO 里程计和语义测量
- 维护双因子图（主图 isam + 信息增益评估图 isam_loop）
- 增量式 SLAM 优化（GTSAM/ISAM2）
- 主动回环闭合管理
- 发布高频 SLOAM 里程计

**关键 Topic：**

订阅：

- /quadrotor/vio/odom (VIO 里程计)
- /semantic_range_bearing_measurements (语义测量)
- /loop_closure/odom (回环检测结果)
  发布：
- /quadrotor/high_freq_pose (高频位姿)
- /quadrotor/high_freq_odom (高频里程计)
- /optimized_trajectory (优化后轨迹 MarkerArray)
- /optimized_point_landmarks (优化后地标 MarkerArray)
- /quadrotor/sloam_to_vio_odom (漂移补偿变换)
  **Action Server：**

- /loop_closure/active_loop_closure_server (主动回环闭合) activeSlamInputNode.cpp:123-124
  Service Server：

- /estimate_info_gain_server (信息增益估计) activeSlamInputNode.cpp:145-147

### 探索规划与控制层

#### msExplorationFSM (探索有限状态机)

**主要职责：**

- 管理探索状态，协调探索规划器和轨迹执行
- 触发主动回环闭合
- 系统状态包括：INIT, WAIT_TRIGGER, INIT_ROTATE, PLAN_BEHAVIOR, REPLAN_TRAJ, EXEC_TRAJ, FINISH, EMERGENCY_STOP, CLOSURE_PANO ms_exploration_fsm.cpp:38

**关键 Topic：**

订阅：

- /odom_slam (SLAM 里程计) ms_exploration_fsm.cpp:49
- /odom_vio (VIO 里程计) ms_exploration_fsm.cpp:50-51
- /waypoints (触发器) ms_exploration_fsm.cpp:48
  发布：
- /tracker_cmd (轨迹跟踪指令) ms_exploration_fsm.cpp:52

**Action Client：**

连接到 /loop_closure/active_loop_closure_server

## 数据流

![](img/data_flow.png)

## Topic 总结

### Ground Truth and Odometry Topics

/ddk/ground_truth/odom
发布者 ：模拟模式下的 Gazebo 模拟器
订阅者 ：控制器和传感器模拟节点
功能 ：提供来自物理模拟器的地面真实里程表，用于模拟测试。gazebo_sim.launch :26

/gt_odom and /gt_odom_perturbed
功能 ：该脚本将高斯噪声（位置和方向扰动）添加到地面真实里程计中，以模拟 VIO 漂移并测试 SLAM 鲁棒性。扰动会随着行驶距离的增加而累积。add_noise_to_ground_truth_odom.py ：35-36

/quadrotor/high_freq_odom and /quadrotor/high_freq_pose
发布者 ： activeSlamInputNode.cpp 中的 ActiveSlamInputNode
功能 ：通过将最新优化的 SLAM 关键姿态与相对 VIO 运动相结合，发布高频 (20 Hz) SLAM 校正里程/姿态。这可在 SLAM 优化更新之间提供平滑的、漂移校正的状态估计。activeSlamInputNode.cpp :53-67

/quadrotor/lidar_odom
发布者 ： process_cloud_node.py
功能 ：发布基于激光雷达的里程计以进行 SLOAM 处理。process_cloud_node.py ：231-232

/quadrotor/vio/odom
功能 ：原始视觉惯性里程计的输入主题，在应用 SLAM 校正之前由 SLOAM 节点订阅。

/quadrotor/sloam_to_vio_odom
发布者 ： ActiveSlamInputNode
功能 ：发布从 SLOAM 框架到 VIO 框架的转换，计算结果为 vio_odom \* sloam_odom.inverse() 。探索规划器使用它来补偿坐标系之间的漂移。

### Sensor Simulation Topics (pcl_render_node)

/pcl_render_node/depth, /pcl_render_node/colordepth, /pcl_render_node/sensor_pose
发布者 : pcl_render_node
订阅者 ：勘探管理器和测绘节点

/depth ：使用 GPU 加速 CUDA 从点云渲染的原始深度图像（CV_32FC1）
/colordepth: Colorized depth visualization with rainbow colormap (BGR8)
/sensor_pose: Camera pose in world frame pcl_render_node.cpp:392-396
/pcl_render_node/rendered_pcl and /pcl_render_node/cloud
Publisher: pcl_render_node
发布者 : pcl_render_node
功能 ：通过将深度像素反向投影到感知范围内的 3D 世界坐标来发布渲染的点云。

/pcl_render_node/camera_info
发布者 : pcl_render_node
功能 ：将相机固有参数（fx、fy、cx、cy）发布为 CameraInfo 消息，用于深度图像处理。

/ddk/rgbd/depth/image_raw
Function: In Gazebo simulation, this is the camera depth topic parameter used by the exploration manager for occupancy mapping. gazebo_sim.launch:64
功能 ：在 Gazebo 模拟中，这是探索管理器用于占用映射的相机深度主题参数。gazebo_sim.launch :64

### Semantic Detection Topics

/sem_detection/sync_pc_odom and /sem_detection/point_cloud
发布者 ： detect.py （使用 YOLOv8 的语义检测节点）
订阅者 ： process_cloud_node.py
功能 ： /sync_pc_odom 在 RGB-D 图像上进行 YOLOv8 实例分割后，发布带有里程计的同步语义点云 /point_cloud 单独发布带语义标记的点云。

/semantic_range_bearing_measurements
发布者 ： process_cloud_node.py
订阅者 ： activeSlamInputNode.cpp
功能 ：将检测到的语义对象（椅子、桌子、电视）的方位角测量结果以 ROSRangeBearing 消息的形式发布。这些测量结果包括方位角矢量、范围、测量 ID 以及身体框架中的地标位置，并作为约束添加到 GTSAM 因子图中。

/process_cloud_node/filtered_semantic_segmentation
发布者 ： process_cloud_node.py
功能 ：处理后发布过滤后的语义分割点云。process_cloud_node.py : 202

/tree_cloud, /ground_cloud, /car_cuboids, /car_cuboids_body
发布者 ： process_cloud_node.py
订阅者 ： inputNode.cpp （SLOAM 输入管理器）
功能 ：这些主题按类别分离语义点云：

/tree_cloud ：树干和杆点
/ground_cloud ：地平面点
/car_cuboids: Fitted cuboid markers for cars in world frame
/car_cuboids_body: Car cuboids in body frame

输入管理器使用树木的网格图分割和地面的平面拟合来同步这些主题以进行 SLOAM 处理。

/cuboid_centers and /car_instance_segmentation_accumulated
发布者 ： process_cloud_node.py

/cuboid_centers: Publishes MarkerArray of cuboid center positions
/car_instance_segmentation_accumulated ：发布汽车实例的累积点云以实现可视化

### SLAM/Factor Graph Topics

/optimized_trajectory
发布者 ： activeSlamInputNode.cpp
功能 ：将经过 ISAM2 因子图优化后的机器人轨迹发布为 MarkerArray。包含点标记（绿色球体）和线带（蓝色线条），用于显示机器人的路径。activeSlamInputNode.cpp :59-60

/factor_graph_atl/optimized_trajectory_with_pose_inds
发布者 ： activeSlamInputNode.cpp
订阅者 ： loop_closure_submap_node.py
功能 ：发布优化后的轨迹，每个标记点的 ID 与因子图中的关键姿势索引相对应。用于生成回环子图。activeSlamInputNode.cpp :61-62 loop_closure_submap_node.py:68-69

/factor_graph_atl/optimized_point_landmarks
发布者 ： activeSlamInputNode.cpp
订阅者 ： loop_closure_submap_node.py
功能 ：将所有优化的语义地标位置（树木、汽车等）从因子图发布为 MarkerArray。

/factor_graph_atl/loop_closure_submap
发布者 ： loop_closure_submap_node.py
功能 ：发布环路闭合子图可视化，显示检测到的环路闭合，并根据重新访问的历史轨迹段使用颜色标记。

### Map Topics

/map_ros/depth, /map_ros/cloud, /map_ros/pose
/map_ros/depth 、 /map_ros/cloud 、 /map_ros/pose
Subscribers: MapROS class in map_ros.cpp
订阅者 ： map_ros.cpp 中的 MapROS 类
Function: These are the input topics for the mapping pipeline. They are remapped from sensor-specific topics (like /pcl_render_node/depth) and synchronized using message_filters for occupancy map updates. map_ros.cpp:241-245 algorithm.xml:37-39
功能 ：这些是映射管道的输入主题。它们从特定于传感器的主题（例如 /pcl_render_node/depth ）重新映射，并使用 message_filters 进行同步，以更新占用图。map_ros.cpp :241-245 algorithm.xml:37-39

/sdf_map/occupancy_global and /sdf_map/occupancy_inflate_local
/sdf_map/occupancy_global 和 /sdf_map/occupancy_inflate_local
Publisher: MapROS class
发布者 ： MapROS 类
Function: Publishes occupancy grid maps as PointCloud2 messages. The global map covers the entire explored space, while the inflated local map includes obstacle inflation for collision checking during path planning. map_ros.cpp:227-234
功能 ：将占用网格地图发布为 PointCloud2 消息。全局地图覆盖整个探索空间，而膨胀的局部地图则包含障碍物膨胀，以便在路径规划过程中进行碰撞检查。map_ros.cpp :227-234

/sdf_map/esdf
Publisher: MapROS class
发布者 ： MapROS 类
Function: Publishes the Euclidean Signed Distance Field computed via distance transform from the occupancy buffer. Used by the trajectory optimizer for smooth collision-free planning. map_ros.cpp:237
功能 ：发布通过距离变换从占用缓冲区计算出的欧氏有向距离场。轨迹优化器使用它来实现平滑的无碰撞规划。map_ros.cpp :237

### Navigation/Planning Topics

/move_base_simple/goal
订阅者 ： waypoint_generator 节点
功能 ：标准 RViz 二维导航目标主题。当用户在 RViz 中点击设置目标时，waypoint_generator 会接收该目标，并根据配置的航点类型（点、圆、八、系列）进行处理。w

/ddk/waypoint_generator/waypoints
发布者 ： waypoint_generator 节点
功能 ：将用户目标中已处理的航点序列发布为 nav_msgs::Path 消息。waypoint_generator.cpp :86-95

/ddk/trackers_manager/line_tracker_min_jerk/LineTracker
功能 ：线路跟踪器的动作服务器端点。规划管理器向此动作服务器发送 LineTrackerGoal 消息，以执行最小加加速度轨迹来跟踪航点。

## 系统模块详解

### Metric-Semantic SLAM 模块

#### 语义检测模块

语义检测模块使用 YOLOv8 对 RGB-D 图像进行实例分割

1. 系统订阅同步 RGB 图像、对齐深度图像和里程计消息
2. YOLOv8 对 RGB 图像进行实例分割，检测椅子、餐桌、电视等物体
3. 使用对齐的深度图和相机内部函数将每个检测到的实例投影到 3D 中，从而创建具有类标签、实例 ID 和置信度分数的语义点云
4. 输出以 syncPcOdom 消息的形式发布，其中包含与里程计检测同步的语义点云

#### 语义处理模块

处理模块接收语义点云并提取长方体地标

1. 点云过滤 ：该模块通过范围、置信度阈值和深度百分位数过滤语义点
2. 初始长方体拟合 ：对于每个实例，系统执行初始边界框拟合
3. 对象跟踪 ：使用匈牙利分配法跨帧跟踪对象，以保持一致的 ID
4. 点云累积 ：每个跟踪实例的点云被累积并下采样
5. 最终长方体检测 ：PCA 用于将定向 3D 边界框（长方体）拟合到累积点云
6. 测距测量生成 ：长方体在机器人身体框架中转换为测距测量 ​
7. 输出包括方位矢量（单位方向）、范围（距离）、身体框架中的地标位置以及作为 ROSRangeBearing 消息发布的测量 ID

#### SLAM 优化模块

SLAM 模块将语义观察整合到因子图中以进行姿势优化：对于通用 SLOAM（包含圆柱体和长方体）：runSLOAMNode 函数负责协调 SLAM 过程

1. 使用网格图检测对圆柱体（树）进行模型估计
2. 当前观测结果与地图地标之间的数据关联
3. 当前扫描和子图之间的长方体模型匹配
4. 长方体匹配使用基于距离的关联
5. addSLOAMObservation 函数将圆柱体和立方体因子添加到 GTSAM 因子图中

##### ALC (Range-Bearing Mode):

1. 主动 SLAM 输入节点接收测距测量
2. 这些测量值在因子图更新中进行处理
3. addOdomBearingObservation 函数使用最近邻搜索执行数据关联，并将方位和范围因子添加到图中

## Active Loop Closure 与 Metric-Semantic SLAM 结合

这两个进程通过 ROS 操作和服务进行通信：

Active Loop Closure Action Server/Client: The Semantic SLAM system (ActiveSlamInputNode) runs an action server, while the exploration planner runs an action client to request loop closures.

信息增益估计服务 ：语义 SLAM 系统提供一项服务，探索规划器在执行假设的循环闭包之前会进行查询以对其进行评估。

语义 SLAM 过程维护一个包含语义地标（树木的圆柱体、汽车/箱子的立方体）和机器人姿势的因子图 ，使用 GTSAM 的 ISAM2 执行增量优化

系统处理 VIO 里程计和语义测量，并以可配置的速率（默认 5 Hz）将它们添加到因子图中

### 主动 SLAM 决策

探索规划器 ( msExplorationManager ) 使用主动感知来决定是探索新边界还是重新访问已知位置以进行回环闭合。它会查询信息增益服务来评估候选回环闭合

规划器比较了访问边界与执行回环的效用，当回环目标相对于其成本提供更好的信息增益时，插入回环目标

### Loop Closure Request Flow

当探索规划器决定执行循环闭合时，它会：

将目标与目标关键姿势索引和子图（地标位置）一起发送到主动循环闭合动作服务器

Semantic SLAM 接收这个目标并接受，存储关闭请求参数

当机器人足够接近目标姿势时，语义 SLAM 系统会触发自己的回环检测客户端

当检测到回环时，语义 SLAM 过程通过调用 addLoopClosureObservation 而不是常规的 addOdomBearingObservation 来更新因子图

系统在处理环路闭合时暂时禁用添加其他因素
