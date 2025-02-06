---
title: 7 Pipeline
tags: [CS61C, hardware, pipeline, RISC-V]
math: "false"
modified: 星期四, 二月 6日 2025, 12:18:02 中午
---

## Implement in RISC-V

使用 pipeline 虽然不能提升单个任务的效率，但是可以从整体上提升工作效率。如下图

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250205151130846.png)

可以在 [6-processor-design](6-processor-design.md) 的 datapath 中进行修改，实现 pipeline

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250206095327322.png)

**Pipeline Control**：control signal 从 instructions 中推出，并存储在 pipeline registers 中

## Structure Hazards

- 问题：多条指令竞争获取同一个硬件资源（如同一寄存器端口、同一内存访问） 
- 解决方案：
    - 指令轮流访问，需要暂时存储一些指令
    - 增加更多硬件资源

更具体的，对于**两条指令同时读写同一寄存器端口**的问题，使用多端口寄存器文件（如 2 读端口 + 1 写端口）；对于**取指和 Load/Store 同时访问内存**的问题，分离指令存储器（IMEM）和数据存储器（DMEM），或使用独立缓存，如下

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250206102539404.png)

总结：

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250206121745271.png)

## Data Hazard
