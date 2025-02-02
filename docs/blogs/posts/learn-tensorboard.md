---
tags:
  - pytorch
title: Tensorboard 浅尝
---

> `pytorch` 中集成的 tensorboard 的功能，对于查看训练进度有较大帮助

## 基本使用

安装 `Tensorboard`:

```bash
pip install tensorboard
```

调入 `SummaryWriter` 类：

```python
from torch.utils.tensorboard import SummaryWriter
```

创建实例：默认会存储在 `./runs/<CURRENT_DATETIME>_<HOSTNAME>` 文件夹内，可以通过 `log_dir` 参数指定，还可以用 `comment` 参数在文件名中添加注释

```python
writer = SummaryWriter()
writer = SummaryWriter(log_dir="logs")
writer = SummaryWriter(log_dir="logs", comment="LR_0.1_BATCH_16")
```

有一个优雅的方式是

```python
import time
now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
model_name = f'transformer_{now}'
writer = SummaryWriter(f'logs/{model_name}')
```

以 `writer.add_scalar()` 为例：

```python
for i in range(100):
    writer.add_scalar(tag="y=x", scalar_value=i, global_step=i)
```

然后使用

```python
writer.close()
```

关闭 tensorboard。然后在命令行中使用 `tensorboard --logdir="logs"` 进行查看。可以使用 `--port=<port>` 来更改端口。效果如下：

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241215141431539.png)

另外，如果两次使用一个 `tag` 参数，则第二次的数据会覆盖第一次的数据：

> [!note]- 示例
> ```python
> from torch.utils.tensorboard import SummaryWriter
>
> writer = SummaryWriter("logs")
>
> for i in range(100):
>     writer.add_scalar("y=x", i, i)
>
> writer.close()
>
> # write again
> writer = SummaryWriter("logs")
>
> for i in range(100):
>     writer.add_scalar("y=x", 3*i, i)
>
> writer.close()
> ```
>
> ![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241215141915997.png)

还可以使用 `train/loss` `train/acc` 这类 tag 来让不同的曲线展示在一个标签下：

> [!note]- 示例
> ```python
> from torch.utils.tensorboard import SummaryWriter
>
> writer = SummaryWriter("logs")
>
> for i in range(100):
>     writer.add_scalar("test/y=x", i, i)
>     writer.add_scalar("test/y=x^2", i * i, i)
>
> writer.close()
> ```
>
> ![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241215150722211.png)

而为了使不同曲线在同一个图中，可以使用 `add_scalars` 方法：

> [!note]- 示例
> ```python
> from torch.utils.tensorboard import SummaryWriter
> writer = SummaryWriter()
> r = 5
> for i in range(100):
>     writer.add_scalars('run_14h', {'xsinx':i*np.sin(i/r),
>                                     'xcosx':i*np.cos(i/r),
>                                     'tanx': np.tan(i/r)}, i)
> writer.close()
> ```
>
> ![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20241215151248415.png)

在训练模型中，可以参考如下代码：

```python
def train(model, optimizer, epoch, dataloader, writer):
    # ...
    for batch_index, (data, target) in enumerate(pbar):
        # ...
        output = model(data)
        loss = criterion(output, target)
        acc = calc_acc(target, output)
        # ...
        iteration = (epoch - 1) * len(dataloader) + batch_index
        writer.add_scalar('train/loss', loss, iteration)
        writer.add_scalar('train/acc', acc, iteration)
        writer.add_scalar('train/error_rate', 1 - acc, iteration)
```

## 其他方法

> 还有 `add_image` `add_audio` `add_histogram` `add_mesh` 等方法可以使用，具体细节需要时查找[官方文档](https://tensorboardx.readthedocs.io/en/latest/tensorboard.html)即可。

### `writer.add_graph()`

用来可视化模型结构，用法：

```python
add_graph(model, input_to_model=None, verbose=False, use_strict_trace=True)
```

- `model`：需要可视化的模型，类型为 `torch.nn.Module`
- `input_to_model`：输入模型的向量，可以展示数据流动过程

## 参考资料

1. [PyTorch 深度学习快速入门教程（绝对通俗易懂！）【小土堆】](https://www.bilibili.com/video/BV1hE411t7RN/?p=8&share_source=copy_web&vd_source=c9e11661823ca4062db1ef99f7e0eee1)
2. [深度学习工程师生存指南-如何使用 TensorBoard](https://dl.ypw.io/how-to-use-tensorboard/)
