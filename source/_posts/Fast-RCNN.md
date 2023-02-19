---
title: Fast-RCNN
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

本文是对Fast-RCNN论文复现过程中的一些理解
<!-- more -->

## 2. Fast-RCNN

### 2.0 Fast-R-CNN结构和原理

将整张图片和候选框（SS算法生成）直接输入到卷积网络Conv中，在得到的特征图上设置一个ROI pooling层，这个层将候选框变成一致大小进入两个全连接层；得到总的损失函数；可以进行反向传播

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/fast-rcnn3.jpg)



#### ROI pooling层的对应关系：

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/fast-rcnn2.jpg)



#### 损失函数

$$
L(p,u,t^{u},v) = L_{cls}(p,u) + \lambda L_{loc}(t^{u},v)
$$

$$
L_{cls} = -logp_{u}
$$

$$
L_{loc} = \sum \limits_i smooth_{L1}(t^{u},v)(i\in(w,x,y,h))
$$



#### 反向传播

$$
\frac{\partial L}{\partial x{i}}=\sum \limits_r \sum \limits_j[i = i^{*}(r,j)] \frac{\partial L}{\partial y_{r,j}}
$$



#### 输入可以使用SVD分解加快速度

$$
Input = W\sum U^{T}
$$

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/fast-rcnn1.jpg)

------

### 2.1 ROI pooling层

将输入向量大小转变为一致大小

```python
class ROI_Pool(nn.Module):

    def __init__(self, size):
        super(ROI_Pool, self).__init__()
        assert len(size) == 2, 'size参数输入(长, 宽)'
        pool_func = nn.AdaptiveMaxPool2d

        self.roi_pool = pool_func(size)

    def forward(self, feature_maps):
        assert feature_maps.dim() == 4, 'Expected 4D input of (N, C, H, W)'
        return self.roi_pool(feature_maps)
```

------

### 2.2 VGG（Backbone）

```python
class VGG16_RoI(nn.Module):

    def __init__(self, num_classes=1000, init_weights=True):
        """
        :param num_classes: 类别数，不包括背景类别
        :param init_weights:
        """
        super(VGG16_RoI, self).__init__()
        # VGG16模型的卷积层设置，取消最后一个最大池化层'M'
        feature_list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

        self.features = models.vgg.make_layers(feature_list)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.roipool = roi_pool.ROI_Pool((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )
        self.softmax = nn.Linear(4096, num_classes + 1)
        self.bbox = nn.Linear(4096, num_classes * 4)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = self.roipool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        classify = self.softmax(x)
        regression = self.bbox(x)
        return classify, regression

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```