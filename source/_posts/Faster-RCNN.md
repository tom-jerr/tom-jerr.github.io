---
title: Faster-RCNN
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

本文是对Faster-RCNN论文复现过程中的一些理解
<!-- more -->

## 3. Faster-RCNN

### 3.0 原理概述

##### faster-rcnn结构：

使用卷积网络提取特征图；将特征图和anchors放入RPN网络中进行训练，同时ROI pooling即负责图片的分类又负责预选框的回归；二者都使用卷积网络，可以实现反向传播更改参数

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/faster-rcnn-3.jpg)



##### RPN过程

通过softmax判断anchors的正负性；利用bounding box regression修正anchors，获得精确的proposals

在特征图的每一个点上（对应原图$16*16$大小），都放置base anchors

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/faster-rcnn-2.jpg)

##### ROI pooling层

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/faster-rcnn-structure.jpg)

##### 目标框回归方程

positive anchor与ground truth的偏移量$(t_{x},t_{y})$和尺度因子$(t_{w},t_{h})$
$$
t_{x} = (x-x_{a})/w_{a},t_{y} = (y-y_{a})/h_{a}
$$

$$
t_{w} = log(w/w_{a}),t_{h} = log(h/h_{a})
$$



##### 损失函数

`cls`为分类损失
`reg`为目标框误差损失
$$
L({p_{i}},{t_{i}}) = \dfrac{1}{N_{cls}}\sum\limits_{i} L_{cls}(p_{i},p_{i}^{'}) +
$$
$$
\lambda \dfrac{1}{N_{reg}} \sum \limits_{i} p_{i}^{'} L_{reg}(t_{i},t_{i}^{'})
$$

------

### 3.1 RPN网络

#### 3.1.0 目标框的绘制（在feature map上）

返回的是base_anchor，其中每个anchor是$[y_{min},x_{min},y_{max},x_{max}]$；

```python
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
   """"
    Returns:
        An array of shape :math:`(R, 4)`.
        Each element is a set of coordinates of a bounding box.
        The second axis corresponds to
        :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.

    """
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
```

##### 目标框 `(x, y, w, h) `的回归方程：

$$
t_{x} = (x-x_{a})/w_{a}, t_{y} = (y-y_{a})/h_{a}
$$

$$
t_{w} = log(w/w_{a}),t_{h} = log(h/h_{a})
$$

##### 损失函数:

`cls`为分类损失
`reg`为目标框误差损失
$$
L({p_{i}},{t_{i}}) = \dfrac{1}{N_{cls}}\sum\limits_{i} L_{cls}(p_{i},p_{i}^{'}) +
$$
$$
\lambda \dfrac{1}{N_{reg}} \sum \limits_{i} p_{i}^{'} L_{reg}(t_{i},t_{i}^{'})
$$

#### 3.1.1 _enumerate_shifted_anchor(产生整个特征图的所有anchor)

feature map的每一个点产生anchor的思想，首先是将特征图放大16倍对应回原图;

原图是经过4次pooling得到的特征图，所以缩小了16倍，

`shift_y /shift_x = xp.arange(0, height * feat_stride, feat_stride) `

这个函数将原来的特征图纵横向都扩大了16倍对应回原图大小；

`shift_x,shift_y = xp.meshgrid(shift_x,shift_y)`就是形成了一个纵横向偏移量的矩阵，也就是特征图的每一点都能够通过这个矩阵找到映射在原图中的具体位置.

```python
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # return (K*A, 4)

    import numpy as xp
    shift_y = xp.arange(0, height * feat_stride, feat_stride)
    shift_x = xp.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = xp.meshgrid(shift_x, shift_y)    #shift_x, shift_y变成大小相同的矩阵，大小为像素点的总个数
    shift = xp.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)    #shift变量变成了以特征图像素总个数为行，4列的这样的数据格式

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))  #用基础的9个anchor的坐标分别和偏移量相加，最后得出了所有的anchor的坐标
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)  #得到原图上所有的anchor
    return anchor
```

------

#### 3.1.2 AnchorTargetCreator

_enumerate_shifted_anchor函数在一幅图上产生了20000多个anchor，而AnchorTargetCreator就是要从20000多个Anchor选出256个用于二分类和所有的位置回归；为预测值提供对应的真实值，选取的规则是：

1. 对于每一个`Ground_truth bounding_box `从anchor中选取和它重叠度最高的一个anchor作为样本！
2. 从剩下的anchor中选取和`Ground_truth bounding_box`重叠度超过0.7的anchor作为样本，注意正样本的数目不能超过128
3. 随机的从剩下的样本中选取和gt_bbox重叠度小于0.3的anchor作为负样本，正负样本之和为256

需要注意的是对于每一个anchor，gt_label要么为1,要么为0,所以这样实现二分类，而计算回归损失时，只有正样本计算损失，负样本不参与计算。



**`__call__`函数允许该对象像函数一样被调用**

```python
    def __call__(self, bbox, anchor, img_size):

        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W) #排除在图片外面的那些目标选择框
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious])   #目标框与每个预测框进行位置回归

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label
```

------

#### 3.1.3 ProposalCreator

利用feature map,计算$\dfrac{H}{16}*\dfrac{W}{16}*9$大约20000个anchor属于前景的概率和其对应的位置参数，这个就是RPN网络正向作用的过程，然后从中选取概率较大的12000张，利用位置回归参数，修正这12000个anchor的位置;

利用非极大值抑制，选出2000个rois

```python
    def __call__(self, loc, score,
                 anchor, img_size, scale=1.):
 
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)
        roi = loc2bbox(anchor, loc)

        # Clip predicted boxes to image.
        # 将roi大小限制在图片范围内
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        # Remove predicted boxes with either height or width < threshold.
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1]   #只取前n_pre_nms个
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]
		
        #使用nms找到最符合的一个标准框
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi
```

------

#### 3.1.4 Proposal_TargetCreator

ProposalCreator产生2000个rois，但是这些rois并不都用于训练，经过本ProposalTargetCreator的筛选产生128个用于自身的训练，规则如下:

1. rois和GroundTruth_bbox的IOU大于0.5,选取一些(比如说本实验的32个)作为正样本
2. 选取ROIS和GroundTruth_bbox的IOUS小于等于0的选取一些比如说选取128-32=96个作为负样本
3. 然后分别对ROI_Headers进行训练

```python
    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        
        n_bbox, _ = bbox.shape	#找到预测框的数量

        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index]

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32)) #去归一化操作

        return sample_roi, gt_roi_loc, gt_roi_label
```

### 3.2 ROIHead网络

由RPN网络已经得到了rois，接下来只需要对rois进行边框位置回归以及框内类别的回归了，分为以下几个步骤：

1. 按照缩放比例，将rois映射到feature map上，然后进行RoiPooling，从而达到相同大小的特征输出，最后输出为pool，shape = (R', 7, 7, 512)。
2. 进行flat操作，pool，shape = (R', `7*7*512`) = (R', 25088)
3. 将pool送入坐标回归的全连接网络，输出为roi_cls_locs，shape = (R', n_class*4)
4. 将pool送入类别回归的全连接网络，输出为roi_scores，shape = (R', n_class)

```python
class VGG16RoIHead(nn.Module):

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
      
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]	 # (index, x_min, y_min, x_max, y_max)
        indices_and_rois =  xy_indices_and_rois.contiguous()	#让矩阵在内存中连续分布

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores
```

------

### 3.3 FasterRCNN(_suppress()函数)

1. 将roi_scores输入到softmax层中，得到prob置信概率
2. 然后对于第l类的框cls_bbox_l以及相应的置信概率prob_l，首先对其筛选出prob_l大于阈值的，然后再对框进行nms处理，这样就得到了最后的结果，将相应的位置、标签以及置信概率记录下来即可

```python
	def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]   # 所有的第L类的框
            prob_l = raw_prob[:, l]     # 第L类的概率
            mask = prob_l > self.score_thresh   # 如果概率大于阈值，对应mask相应的位置就会被选取
            cls_bbox_l = cls_bbox_l[mask]   # 得到概率大于阈值的框
            prob_l = prob_l[mask]   # 得到概率大于阈值的框
            keep = nms(cls_bbox_l, prob_l,self.nms_thresh)      # 对这些框进行nms处理
            # import ipdb;ipdb.set_trace()
            # keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep].cpu().numpy())     # 将L类的框保存
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))   # 保存相应标签
            score.append(prob_l[keep].cpu().numpy())        # 保存相应置信概率
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)      # 将所有的数据链接成一个矩阵，下同
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score
```

------

### 3.4 Trainer

然后在计算过程中**只计算非背景类的回归损失**

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/1.png)

##### 计算`smooth_l1`损失

```python
def _smooth_l1_loss(x, t, in_weight, sigma):  # x代表预测,t代表真值，in_weight代表权重
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)  # 被标定为背景的类的权重设置为0（忽略背景类），这也就是为什么计算位置的损失函数还要传入真实label作为参数的原因
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()
```



##### 计算位置损失

```python
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):   
    in_weight = t.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1  # 将不是背景的anchor或ROIs的位置加入到损失函数的计算中来，方法是只给不是背景的anchor/ROIs的in_weight设置为1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)  # sigma=1
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float())  # 只计算前景类
    return loc_loss
```

------

### 3.5 Train

#### eval()函数

eval()就是一个评估预测结果好坏的函数，首先pred_bboxes,pred_labels,pred_scores ,gt_bboxes,gt_labels,gt_difficults 分别是预测框的位置，预测框的类别和分数以及相应的真实值的类别分数等等，全为`list`列表

```python
def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()    # 定义了预测和真实的框的位置，类别和分数的列表
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader))   #tqdm进行封装实现可视化进度
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])  # 利用faster_rcnn.predict预测bbox的位置，label以及分数。
        gt_bboxes += list(gt_bboxes_.numpy())  # 添加预测值和真实值到列表中
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break  # 迭代次=test_num就跳出循环

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)   # 接收列表参数，得到预测结果
    return result
```



#### train()函数

1. 从训练数据中枚举dataloader,设置好缩放范围，将img,bbox,label,scale全部设置为可gpu加速
2. 调用trainer.py中的函数trainer.train_step(img,bbox,label,scale)进行一次参数迭代优化过程
3. 调用 _bboxes,_labels,_socres = trainer.faster_rcnn.predict([ori_img_],visualize=True)调用faster_rcnn的predict函数进行预测，预测的结果保留在以_下划线开头的对象里面
4. 调用`eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)`将测试数据调用eval()函数进行评价，存储在eval_result中
5. `lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr'] `设置学习的learning rate
6. `log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),*str(eval_result['map']),**str(trainer.get_meter_data())) +* *trainer.vis.log(log_info) `将损失学习率以及map等信息及时显示更新

```python
def train(**kwargs):
    opt._parse(kwargs)  

    dataset = Dataset(opt) # 获取数据存储的路径
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \ # 允许数据打乱排序
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)  # 设置数据几次处理完
    testset = TestDataset(opt)  # 与上边进行相同操作
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()  # 定义模型
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()  # 使用VGG16模型
    if opt.load_path:  # 判断opt.load_path是否存在
        trainer.load(opt.load_path) # 读取pre_train模型
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')  # 可视化操作
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):  # 训练
        trainer.reset_meters()  # 在可视化界面reset所有数据
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):  
            scale = at.scalar(scale)                                       # 设置缩放范围
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()  # gpu加速
            trainer.train_step(img, bbox, label, scale)  # 参数优化

            if (ii + 1) % opt.plot_every == 0:  # 读取次数是否达到了画图次数
                if os.path.exists(opt.debug_file):  
                    ipdb.set_trace()  # 设置断点

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())  # 读取训练数据并上传进行可视化

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))  # 预处理迭代读取的图片
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0])) 
                trainer.vis.img('gt_img', gt_img)  # 读取原始数据中的原图，边界框，标签，并显示在visdom界面

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)  # 预测并保存结果
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)  # 同理将原始图片以及预测边框和预测类别显示在visdom界面

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')  # 在visdom界面显示混淆矩阵
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())  # 在visdom界面以图片的形式显示可视化矩阵
        # 接下来是测试阶段的代码
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)  # 将测试数据进行评价
        trainer.vis.plot('test_map', eval_result['map'])  # 在visdom界面显示map
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']   # 设置学习率
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),  # 更新损失学习率以及map
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:  # 保存效果最好的map
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:  # 学习9次就将学习率变成原来的十分之一
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13:   # 结束训练验证过程
            break
```

