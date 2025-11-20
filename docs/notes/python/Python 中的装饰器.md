
# Python 中的装饰器

它是 Python 3.7 引入的一个标准库工具（`from dataclasses import dataclass`）。它的主要作用是 **自动生成样板代码**。

装饰器本质上就是一个 **“接收一个函数/类，并返回一个新的函数/类”的高阶函数**

## 工作流程

`@dataclass` 的核心工作流程是利用 Python 的 **元编程 (Metaprogramming)** 能力。具体步骤如下：

1. **读取类型注解 (`__annotations__`)**： Python 的类会把变量定义的类型保存在 `__annotations__` 字典中。 `dataclass` 函数会去读取 `Student` 类中的 `name: str`, `age: int` 等信息。
    
2. **动态生成代码字符串**： 根据读取到的字段，`dataclass` 会在内存中动态拼接出方法的源代码字符串。例如，它会拼出类似这样的字符串：
    
    ```python
    "def __init__(self, name: str, age: int, score: float):\n self.name=name\n self.age=age..."
    ```
    
3. **编译并挂载方法**： 它使用 Python 内置的 `exec()` 函数将这些字符串代码编译成真正的函数对象，然后通过 `setattr` 把这些函数挂载到 `Student` 类上。
    
    - `setattr(Student, '__init__', generated_init_func)`
        
    - `setattr(Student, '__repr__', generated_repr_func)`

## 手写装饰器示例

```python
def add_str_method(cls):
    # 1. 定义一个新方法
    def new_str(self):
        return f"我是类 {cls.__name__} 的实例"
    
    # 2. 把这个方法强行塞给传入的类
    cls.__str__ = new_str
    
    # 3. 返回修改后的类
    return cls

# 使用装饰器
@add_str_method
class MyConfig:
    pass

# 测试
c = MyConfig()
print(c)  # 输出: "我是类 MyConfig 的实例"
```

## pytorch 中常用的装饰器

### 1. 梯度与计算图控制（最常用）

这几个装饰器既可以用作函数装饰器，也可以用作上下文管理器（`with ...:`）。

#### **`@torch.no_grad()`**

- **作用**：**禁用梯度计算**。
    
- **原理**：在被装饰的函数执行期间，PyTorch 的 Autograd 引擎会停止记录操作历史。这会减少内存消耗（不存中间激活值）并加快计算速度。
    
- **场景**：模型**验证（Validation）**或**推理（Inference）**阶段。
    
- **代码示例**：
    
    Python
    
    ```
    @torch.no_grad()
    def evaluate(model, x):
        return model(x)  # 这里的计算不会产生梯度，显存占用更低
    ```
    

#### **`@torch.inference_mode()`** (推荐用于推理)

- **作用**：**更极致的推理模式**。
    
- **原理**：它是 `@torch.no_grad()` 的升级版。除了禁用梯度，它还禁用了一些在推理时不需要的运行时检查（view tracking 等）。
    
- **场景**：**生产环境推理**、部署。比 `no_grad` 更快，是 PyTorch 官方推荐用于纯推理的模式。
    
- **注意**：在这个模式下生成的 Tensor 无法被用于后续的训练（无法反向传播）。
    

#### **`@torch.enable_grad()`**

- **作用**：强制开启梯度计算。
    
- **场景**：比较少用。通常用于在 `no_grad` 的大环境下，临时需要对某原本冻结的模型部分求导（例如：Freeze 骨干网络，但在推理时想用梯度做一些可视化或对抗攻击）。
---
### 2. 编译与加速 (PyTorch 2.x)

这一类装饰器涉及 PyTorch 的编译器前端，将 Python 代码转换为更高效的中间表示（IR）。
#### **`@torch.compile`** (PyTorch 2.0+ 核心)

- **作用**：**一行代码加速模型**。
- **原理**：使用 TorchDynamo 捕获计算图，并使用 Triton 等后端将一系列算子融合（Kernel Fusion），生成优化后的二进制代码。
- **场景**：大模型训练和推理加速。
- **代码示例**：

    ```python
    import torch
    
    @torch.compile  # 自动优化这个函数/模型
    def fast_inference(x, y):
        return torch.sin(x) + torch.cos(y)
    ```