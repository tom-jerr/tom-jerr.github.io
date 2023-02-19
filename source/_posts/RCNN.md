---
title: RCNN
index_img: /img/DP.png
categories:
- 机器学习
tags:
- 深度学习
- 目标检测
- pytorch
comment: valine
math: true
---

# 前言

本文是对RCNN论文复现过程中的一些理解
<!-- more -->

## 1. RCNN

### 1.0 RCNN结构

两过程模型（two stage）：1.区域检测	2.目标分类

使用候选区域方法（region proposal method）
![](https://github.com/tom-jerr/MyblogImg/raw/main/src/rcnn1.jpg)



### 1.1 选择型搜索（Selective Search, SS）

将像素作为一组，计算每一组纹理，将两个最接近的组合起来；搜索2000个左右的候选框（大小相同，CNN需要大小相同的输入）

1.各向异性缩放	2.同性缩放，延伸正方形进行裁剪；扩展延伸，裁剪出来有背景固定色填充正方形图片

openCV库中实现了SS算法。

#### 示例图片

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/SS.png)

### 1.2 SVM

RCNN中默认CNN输出了4096特征分量；有2000个候选区域；

分类需要20个SVM；$[2000,20]$：得分矩阵大小



### 1.3 非极大值抑制算法（NMS）

按照搜索局部极大值，抑制非极大值元素的思想来实现的，具体的实现步骤如下：

（1）设定目标框的置信度阈值，常用的阈值是0.5左右

（2）根据置信度降序排列候选框列表

（3）选取置信度最高的框A添加到输出列表，并将其从候选框列表中删除

（4）计算A与候选框列表中的所有框的IoU值，删除大于阈值的候选框

（5）重复上述过程，直到候选框列表为空，返回输出列表

```python
#极大值抑制算法——剔除过多的搜索框
def NMS(dets, thresh):

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2-y1+1)*(x2-x1+1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22-x11+1)
        h = np.maximum(0, y22-y11+1)
        overlaps = w*h
        IOU = overlaps/(areas[i] + areas[index[1:]] - overlaps)

        print(np.where(IOU <= thresh))
        idx = np.where(IOU <= thresh)[0]    #获取保留下来的索引
        index = index[idx + 1]

    return keep

#显示搜索框
def plot_bbox(dets, c='k', title_name="title"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)
    plt.title(title_name)
```

##### 结果

before_nms

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/nms_original.png)



after_nms

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/after_nms.png)



### 1.4 修正候选区域

建立一个boundingbox regressor；使用线性回归，对候选框进行修正



### 1.5 缺点

1. 训练阶段多：微调网络 + SVM + 边框回归
2. 消耗时间
3. 处理速度慢，处理一张图片需要几十秒
4. 图片形状在进行SS候选框搜索时发生改变；在经过crop进行图片固定大小改变时，无法保证图像不失真