---
title: Processor Design
tags: [CS61C, hardware, processor]
math: "true"
modified: 星期四, 二月 6日 2025, 9:54:57 上午
---

Processor 由 datapath 和 controller 组成，如图

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250131100849441.png)

## Datapath

> 这部分图片较多，截图也会更多一些

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250205094044289.png)

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250205094229738.png)

一条指令的五个执行阶段：取指、译码、执行、访存、写回

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250205094836582.png)

 例如，对 `add`/`sub` 命令，其 datapath 为

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250205095751216.png)

其中 `inst[]` 的位数是由 `add`/`sub` 命令的字节实现决定的（见 [4-RISC-V](4-RISC-V.md) 中的 R-format）

还可以在此基础上加入 `addi` 等 I-format 的指令实现

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250205100327849.png)

最后可以得到一个通用的线路图，使得能够在一个 cycle 中能够运行任何一个指令

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250205102855891.png)

## Controller

control logic 的真值表为

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250205103025521.png)

实现：

- **ROM**(Read-Only Memory): 易于重新编程；当需要人工设计 control logic 时常用
- **Combinatorial Logic**

## Instruction Timing

在一个整的 datapath 中，取耗时最长的指令为一个 cycle 的用时，例如

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250206093346661.png)

## Performance Measures

**Iron low** of processor performance

$$
\frac{\text{time}}{\text{program}} = \frac{\text{instructions}}{\text{program}} \cdot \frac{\text{cycles}}{\text{instruction}} \cdot \frac{\text{time}}{\text{cycle}}
$$

其中

- $\frac{\text{instructions}}{\text{program}}$ 由 task、algorithm、编程语言、compiler、ISA 决定
- $\frac{\text{cycles}}{\text{instruction}}$ 由 ISA、processor 的实现、pipelined processors、superscalar processors 决定
- $\frac{\text{time}}{\text{cycle}}$ 由 processor microarchitecture、technology、supply voltage 决定
 

当然也可以从 response time、energy per task 等角度来衡量
