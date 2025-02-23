---
title: Introduction
tags:
  - os
  - MIT-s081
math:
---
## 课程内容

**Aim**: 

1. 理解操作系统的设计和实现
2. 为了深入了解具体的工作原理，可以通过一个小的叫做XV6的操作系统，获得实际动手经验

**OS Purpose**:

1. Abstract hardware：为程序提供抽象后的高级接口，方便移植和开发
2. Multiplex：多个应用程序之间共用硬件资源
3. Isolation：不同活动之间不能互相干扰
4. Sharing：不同的活动之间有时又想要相互影响，比如说数据交互，协同完成任务等，就希望实现共享
5. Security：比如用户文件不共享
6. Performance：不能阻止程序获得高性能，甚至应该帮助程序获得高性能
7. Range of uses：支持大量应用场景

## OS Organization
