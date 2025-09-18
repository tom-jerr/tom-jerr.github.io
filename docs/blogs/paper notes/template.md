---
title: PL-VINS: Real-Time Monocular Visual-Inertial SLAM with Point and  Line Features 论文阅读笔记
date: 2025/9/9
update:
comments: true
description: PL-VINS: Real-Time Monocular Visual-Inertial SLAM with Point and  Line Features：
    - PL-VINS is the first real-time optimization-based monocular VINS method with point and line features.
    - A modified LSD algorithm is presented for the pose estimation problem by studying a hidden parameter tuning and length rejection strategy.    

katex: true
tags:
  - Paper Notes
  - VINS
---

# PL-VINS: Real-Time Monocular Visual-Inertial SLAM with Point and Line Features 论文阅读笔记

## Abstract

浏览文章的图片，可以稍微总结一下图片传递的信息。

## Intro 总体介绍

看懂原文内容后，翻译或者用自己的话阐述，这部分要把几个问题说清楚，并且要有逻辑。

- 文章大概的背景，之后会引申出，目前的问题是什么？
- 目前有哪些解决方案，有哪些工作，之后会阐述目前的工作的问题
- 介绍他们的解决方案（他们的 paper 做了什么，contribution 是什么）

## 背景知识的相关工作

这部分会根据文章背景，把一些需要了解的概念稍微展开讲一下。

可能会把相关工作归类之后再讲一遍，如果是第一次读这个子领域的顶会，那么这部分内容涉及到的工作可以都浏览一下，了解大概的研究情况。

## Method

这部分占据了主要的篇幅，一篇 12 页的文章可能有一半篇幅。我认为核心的地方是看懂每一张图片，暂时先不要过分沉浸于细节（比如各种参数是怎么设置的，除非图片有清晰的说明，否则这个需求应该是为复现服务的）。

总结一下，可能会有以下几个部分。

- 展示一些实验结果，主要是为了说明： XX 参数我们为什么要设置为 YY；根据这个观察（例：前几层 layer 不需要高精度）所以我们有了 XX insight；主要是阐述为什么这个系统/架构中的一些参数/模块为什么要这么设计。
- 架构/系统的设计细节，能够完整地介绍工作的 flow

## 实验/评估/结果

介绍实验是怎么做的，用的什么模型/benchmark，在什么框架/系统/模拟器上实现的。

结果怎么样。一般很多人浏览时会在看完 intro 之后直接看结果如何。所以结果一定是直观的，一目了然的。

## 结论

总结一下作者结论部分的内容。

## 总结

⭐ 这部分是用自己的话总结，也是对自己的锻炼。阐述这篇文章解决的问题，他们的效果如何，他们和之前看的一些 文章/方法/解决方案 相比如何。如果对小领域有了一些了解，还可以写下自己的 “主观” 评价，他们对于实际的 模型/硬件/框架 是否有作用，文章图片画的如何，文章结构如何，写作怎么样，有哪些可以学习的地方，工作量怎么样，有哪些地方比较 confuse，有哪些可以借鉴的地方，如果是你在设计实验，你会怎么做。
