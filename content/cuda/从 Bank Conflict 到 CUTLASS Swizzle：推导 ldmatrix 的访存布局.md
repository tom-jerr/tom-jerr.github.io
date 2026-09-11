---
title: 从 Bank Conflict 到 CUTLASS Swizzle：推导 ldmatrix 的访存布局
created: 2026-09-08
updated: 2026-09-08
tags:
  - CUDA
description: 从 shared memory 的 32 个 bank 出发，逐地址推导 ldmatrix 的行地址、寄存器分配与 bank conflict，再解释 CUTLASS 的 XOR permutation、CuTe Swizzle 参数和 producer/consumer 布局约束，附交互图与可运行验证程序。
katex: true
---

# 从 Bank Conflict 到 CUTLASS Swizzle：推导 ldmatrix 的访存布局

读 CUTLASS 或 FlashAttention Kernel 时，经常会遇到这样一段定义：

```cpp
composition(Swizzle<3, 3, 3>{},
            Layout<Shape<_8, _64>, Stride<_64, _1>>{})
```

“通过 XOR swizzle 避免 bank conflict”解释了目的，却没有解释地址为什么这样排。要读懂三个 `3`，需要把矩阵坐标、bank 分布，以及 `ldmatrix` 的线程分工放在一起看。

本文先建立 bank 与 `ldmatrix` 的最小模型，再用一张可切换的交互图对照 plain / swizzled 布局，最后把地址公式还原成 CuTe 的 `Swizzle<3,3,3>` 和 CUTLASS 的向量存储映射。沿途始终追踪同一件事：**逻辑元素放在哪里，由谁提供地址，最终进入谁的寄存器。**

这篇文章是 [CuTe 详解：以 FlashAttention-2 拆解 Layout、TiledCopy 与 TiledMMA](CuTe%20初探：以%20FlashAttention-2%20拆解%20Layout、TiledCopy%20与%20TiledMMA.md) 中 Swizzle 一节的展开；普通 GEMM 优化的上下文可以参考 [从 GEMM 实践 CUDA 优化](从GEMM实践CUDA优化.md)。

> [!NOTE]
> 主例固定为 row-major、每行 64 个 FP16、行跨度 128 B，shared-memory 基址按 128 B 对齐。指令讨论限定在 `ldmatrix.m8n8.*.b16`，CuTe 源码固定为 CUTLASS v4.6.1。文中的 bank 分布是地址推导；后文单独交代 GPU 正确性验证与未取得的 profiler 数据。


## 1. 从 word 地址判断 bank conflict


### 从 byte address 算 bank

在本文使用的现代 NVIDIA GPU shared-memory 模型中，有 32 个 bank，连续的 32-bit word 轮流映射到连续 bank。官方 Best Practices Guide 给出的带宽单位也是每 bank 每周期 32 bit。[^banks]

令共享内存中的字节地址为 $a$，则：

$$
w(a)=\left\lfloor\frac{a}{4}\right\rfloor,
\qquad
b(a)=w(a)\bmod32.
$$

其中 $w$ 是 32-bit word 编号，$b$ 才是 bank 编号。例如：

| byte address | word 编号 | bank | 与地址 0 的关系                  |
| ------------ | --------: | ---: | -------------------------------- |
| 0、1、2、3   |         0 |    0 | 同一个 32-bit word               |
| 4、5、6、7   |         1 |    1 | 下一个 word、下一个 bank         |
| 124–127      |        31 |   31 | 第 32 个 word                    |
| 128–131      |        32 |    0 | 另一个 word，重新落到 bank 0     |
| 256–259      |        64 |    0 | 又一个不同 word，同样落到 bank 0 |

因此，“一个 bank 是 4 B”只能作为**一次服务宽度**的简写。每个 bank 保存许多 word，shared memory 也远不止 `32×4=128 B`。128 B 是一个完整 bank 轮转所对应的地址跨度。

若 FP16 数组的元素偏移是 $x$，基址为 $a_0$，就必须先乘元素宽度：

$$
b(x)=\left\lfloor\frac{a_0+2x}{4}\right\rfloor\bmod32.
$$

在本文对齐条件下，基址项对 bank 取模为零，可以简化成 $(x\mathbin{//}2)\bmod32$。直接写 `x % 32` 会把 FP16 元素编号误当成 32-bit word 编号。


### 同一个 bank，不一定是冲突

分析一次 shared load 时，应把“相同 bank”继续拆开：

- 多个 lane 读取同一个 32-bit word，可以利用广播机制；读取这个 word 内不同的 FP16 半字，也不能仅凭“bank 相同”判成冲突。
- 多个 lane 请求同一个 bank 中的不同 word，才需要额外串行服务。
- 访问多个不同 bank，可以由不同 bank 并行处理。

一个方便的分析量，是在**同一个服务分组内**统计每个 bank 收到的不同 word 数：

$$
n_b=\left|\{w:\ w\bmod32=b,\ w\text{ 被本组请求}\}\right|.
$$

对最简单的 32-lane、每 lane 一个 32-bit word 的 load，$\max_b n_b$ 就给出地址模型下的冲突路数。这里对 word 去重，是为了正确处理广播。写入则还涉及数据竞争：不能把读广播解释成“多个线程写同一个地址也安全”。

还要区分 shared-memory bank conflict 与 global-memory coalescing：前者关心 bank 上的 word 竞争，后者关心 warp 的地址如何组成内存事务。global load 连续，并不自动保证写到 shared 后的读取没有冲突。


### 先确定一次指令的服务分组

如果每个 lane 只读 4 B，一个完整 warp 的数据量是 128 B；但如果每个 lane 读 16 B，总量就是 512 B。即使所有地址都很理想，也不可能把这 512 B 当成“一次每 bank 只读一个 word”的访问来分析。

所以判断冲突要区分两种工作量：

1. **数据宽度本来就需要的服务工作量**。
2. **地址碰撞引入的额外串行工作量**。

对于常见的每线程 16 B shared-memory 指令，可以按对应硬件的服务分组检查，例如连续 8 个 lane 的 128 B 访问；实际分解仍应以编译后的 SASS 和目标 GPU 为准。不能把 32 个 lane 的全部 512 B 请求摊开，发现每个 bank 出现四次，就直接说有 4 路冲突。

Nsight Compute 用 request、wavefront 等指标描述这些工作：一个 request 可以需要多个 wavefront，wavefront 数量大于一并不直接等于存在额外冲突。[^ncu] 对 `ldmatrix`，我们接下来用每个 `8×8` 矩阵的地址集合建立分析单元，再讨论 `.x2/.x4`。

还有一个容易混淆的地方：**shared memory 中分散在多个 128 B 对齐区间的地址，仍然可以没有 bank conflict。** bank 模型看的是模 32 后的 word 分布；不能套用 global memory 的连续 segment 直觉。


## 2. ldmatrix：谁提供地址，谁接收数据


### 一条 .x1 搬了多少数据

考虑下面的 PTX：

```ptx
ldmatrix.sync.aligned.m8n8.x1.shared.b16 {d}, [addr];
```

它让一个 warp 协作读取一个 `8×8`、16-bit 元素矩阵，共 `8×8×2=128 B`。每个 lane 获得一个 32-bit 寄存器 `d`，保存两个 16-bit 元素，合计 `32×4=128 B`。`.b16` 描述元素位宽，本身不做 FP16 数值转换；后面的验证程序会故意用整数标签识别数据来源。

PTX 规定 `.x1` 的 8 个行首地址由 T0–T7 提供，但结果分散到全部 32 个 lane。地址提供者与数据接收者是两套角色。[^ldmatrix]

对于非 `.trans` 形式，令接收者 `lane = 4r + k`，其中 $r\in[0,7]$、$k\in[0,3]$，则：

$$
T_{4r+k}.d=\operatorname{pack}_{16}\big(A[r,2k],A[r,2k+1]\big).
$$

这里 `pack16(lo, hi)` 表示两个 16-bit 值分别放在低半部和高半部。这是从指令 fragment 布局直接展开的索引关系。

| 地址提供者 | 提供的地址  | 数据接收者 | 每个接收者获得的列 |
| ---------- | ----------- | ---------- | ------------------ |
| T0         | 第 0 行起点 | T0–T3      | 0–1、2–3、4–5、6–7 |
| T1         | 第 1 行起点 | T4–T7      | 0–1、2–3、4–5、6–7 |
| T5         | 第 5 行起点 | T20–T23    | 0–1、2–3、4–5、6–7 |
| T7         | 第 7 行起点 | T28–T31    | 0–1、2–3、4–5、6–7 |

因此，T5 提供第 5 行地址，**不意味着第 5 行都被装进 T5 的寄存器**。看图或写 inline PTX 时，必须持续区分这两种线程编号。


### .x2 与 .x4 增加的是矩阵和每 lane 寄存器数

| 形式  | 矩阵数量 | 提供行地址的线程组              | 每 lane 的 32-bit 寄存器数 | warp 数据量 |
| ----- | -------: | ------------------------------- | -------------------------: | ----------: |
| `.x1` |        1 | T0–T7                           |                          1 |       128 B |
| `.x2` |        2 | T0–T7；T8–T15                   |                          2 |       256 B |
| `.x4` |        4 | T0–T7；T8–T15；T16–T23；T24–T31 |                          4 |       512 B |

对于 `.x4`，T0–T7 提供的矩阵进入每个 lane 的 `d0`，T8–T15 提供的矩阵进入每个 lane 的 `d1`，依次类推。具体把哪四块逻辑矩阵放进去，由软件提供的行首地址决定。

**PTX 规定的是 collective 行为与 fragment 布局，并不在这里承诺微架构周期数。**

`.sync.aligned` 要求整个 warp 一致执行这条指令；其中 `.aligned` 的语义是执行一致性，不能用它代替对数据地址对齐的检查。`m8n8.b16` 的每行 16 B 还要自然对齐。基础 `ldmatrix` 从 PTX 6.5、`sm_75` 开始支持；不是到 Ampere 才引入。面向 `sm_75` 时，未提供有效矩阵行的高位 lane 也应持有有效地址，必要时复制低位 lane 的地址。[^ldmatrix]

### .trans 改变寄存器排布


`ldmatrix` 的 `.trans` 会按转置形式组织寄存器 fragment。对本例一个 `8×8` 矩阵，令 $g=lane\mathbin{//}4$、$t=lane\bmod4$：

```text
普通 .x1：       d = pack16(A[g, 2t], A[g, 2t+1])
.x1.trans：      d = pack16(A[2t, g], A[2t+1, g])
```

两者使用相同的一组合法行地址，返回的 lane/register 排布不同。下面的 GPU 程序同时验证这两种结果。

因此 `.trans` 不能替代 shared-memory swizzle：它解决 fragment 的组织方向，swizzle 解决数据所在的物理地址。若传入的行段地址仍然互相竞争同一组 bank，不能仅凭 `.trans` 这个名字声称冲突消失。


## 3. 8 路冲突怎样消失

现在把 shared memory 中的 `A[8][64]` FP16 tile 作为完整例子。一次 `.x1` 取八行、每行连续八个元素；用 $q\in[0,7]$ 选择列段，段内位置记为 $j$：

$$
c=8q+j,\qquad j=0,\ldots,7.
$$

图中每个着色格代表一个 16 B 行段。横向 G0–G7 是 **bank group**，每组包含四个连续的 4 B bank：$G_g=\{4g,4g+1,4g+2,4g+3\}$。左边的 T0–T7 提供行地址，下方展开的是当前行的数据接收者；“每 bank 的 word 数”用于判断冲突。

<style>
.bank-swizzle-figure { display: block; width: 100%; max-width: 100%; height: 880px; border: 0; border-radius: 8px; margin: 1rem 0; }
@media (max-width: 600px) { .bank-swizzle-figure { height: 920px; } }
@media (max-width: 380px) { .bank-swizzle-figure { height: 960px; } }
</style>

<iframe class="bank-swizzle-figure" src="img/bank-conflict-swizzle/ldmatrix-swizzle.htm" title="ldmatrix 与 XOR swizzle：切换布局、列段和展开行" width="100%" height="880" loading="lazy"></iframe>

可以按以下顺序操作，再对照后面的地址推导：

| 操作                         | 重点观察                                            |
| ---------------------------- | --------------------------------------------------- |
| 关闭 XOR，选择 `q=0、r=5`    | 八行挤在 G0；第 5 行地址为 640 B，交给 T20–T23 接收 |
| 打开 XOR，保持 `q=0、r=5`    | 八行分散到 G0–G7；第 5 行移到 720 B，接收线程不变   |
| 保持 XOR 打开，把 `q` 改为 3 | group 顺序改变，但每个 bank 仍只读取一个 word       |


### 关闭 XOR：行跨度怎样造成 8 路冲突

不做 swizzle 时，元素偏移为 $x=64r+c$，第 $r$ 行被读取的 16 B 行段起点是：

$$
a(r,q)=128r+16q.
$$

这里及后文的 $a$ 都是相对基址的 byte offset。该行的四个 word，编号和 bank 分别为：

$$
w(r,q,k)=32r+4q+k,
$$

$$
b(r,q,k)=(4q+k)\bmod32,\qquad k=0,1,2,3.
$$

**bank 表达式中的 $r$ 消失了。** 这就是冲突的根源：八行的同一个列段都落到同一组四个 bank，实际 word 却因 $32r$ 不同而各不相同，无法广播。

在交互图中关闭 XOR，并选择 `q=0`，八行都会落到 G0。bank 0–3 各收到八个不同 word，因此 G0 下方的计数为 8，其余组为 0。这是 **8 路冲突**：一次矩阵加载的 32 个 word 分布在四个 bank 上，每个 bank 服务八个，而不是把 32 个接收线程全部算成 32 路冲突。

对展开的 `r=5`，行段地址是 $128\times5=640$ B，覆盖 bank 0–3。T5 提供这个地址，实际接收 `A[5,0..7]` 的依然是 T20–T23。



### 打开 XOR：为什么八行能覆盖全部 bank

继续使用 $c=8q+j$，现在只改变列段编号：

$$
q'=q\oplus(r\bmod8),\qquad c'=8q'+j.
$$

物理元素偏移改为：

$$
x'=64r+\big(c\oplus((r\bmod8)\ll3)\big).
$$

对应的 16 B 行段起点变成：

$$
a'(r,q)=128r+16\big(q\oplus(r\bmod8)\big).
$$

一个行段内部的 $j$ 没有改变，所以八个 FP16 仍然连续，并保持 16 B 对齐。改变的是整个行段在当前物理行中的位置。

此时 bank 编号为：

$$
b'(r,q,k)=4\big(q\oplus(r\bmod8)\big)+k.
$$

对于固定 $q$，当 $r=0,\ldots,7$ 时，$q\oplus r$ 恰好遍历 `0..7` 的一个排列。因此八行各占一个 bank group，32 个 word 覆盖 32 个不同 bank。这不是依赖某个特例的观察，而是 XOR 的双射性质带来的结果。

回到图中，保持 `q=0、r=5` 并打开 XOR。第 5 行从 640 B 移到 $128\times5+16\times5=720$ B，覆盖 bank 20–23；接收者仍然是 T20–T23，逻辑数据仍然是 `A[5,0..7]`。发生变化的是物理位置。

再把 `q` 切到 3，八行的 group 次序变成 `3、2、1、0、7、6、5、4`，仍然每组一行。第 5 行此时落在 G6，由 T20–T23 接收。这也说明，bank 编号与接收 lane 编号并没有绑定，刚才的编号相同只是 `q=0` 下的巧合。


### 为什么逻辑矩阵没有变

对固定 $r$，映射 $q\mapsto q\oplus r$ 可逆，而且做两次就回到原值：

$$
(q\oplus r)\oplus r=q.
$$

因此完整一行的八个列段只是重新排列，没有覆盖或丢失任何数据，也没有增加存储容量。但这一性质要求 producer 和 consumer 使用相同的地址函数：

```cpp
// 本例专用：每行 64 个 16-bit 元素。
__host__ __device__ constexpr int smem_offset(int r, int c) {
  return 64 * r + (c ^ ((r & 7) << 3));
}

// producer: 把逻辑 A[r,c] 写到其物理位置。
smem[smem_offset(r, c)] = global_A[r * ld + c];

// consumer: 仍然访问逻辑 A[r,c]，沿同一函数找到它。
value = smem[smem_offset(r, c)];
```

如果已经按 plain row-major 写好 tile，读取时才突然给地址加 XOR，得到的是错误数据。

## 4. 把地址公式还原成 CuTe Swizzle


### 先看源码里的输入单位

CUTLASS v4.6.1 的 `include/cute/swizzle.hpp` 把 Swizzle 实现成一个整数 offset 变换。下面是针对 $S\ge B\ge0$ 的等价简化，省去了类型包装；负 shift 的完整处理以源码为准。[^swizzle]

```cpp
// S >= B >= 0，M >= 0。
int swizzle(int x, int B, int M, int S) {
  int mask = ((1 << B) - 1) << (M + S);
  return x ^ ((x & mask) >> S);
}
```

三个参数在这个正 shift 情况下分别表示：

- `B`：参与 XOR 的 bit 数。
- `M`：保留的最低 bit 数，目标字段从 bit `M` 开始。
- `S`：源字段向右移动的距离，源字段从 bit `M+S` 开始。

源码要求 `M >= 0`、`B >= 0` 和 `abs(S) >= B`，从而让两个字段不重叠；若 `S < 0`，源字段在较低位置，向左 XOR 到较高的目标字段，不能照搬上面的右移代码。

**Swizzle 本身不知道输入是 byte、FP16 元素还是 FP32 元素。** 单位来自它所复合的 Layout。对以 `half_t*` 为引擎、底层 Layout 返回元素偏移的 Tensor，输入通常是 half 元素编号。


### 展开 `Swizzle<3,3,3>`

本例 row-major layout 的 offset 是 $x=64r+c$。按 bit 切开：

```text
half 元素 offset x：

  [ 更高位 ] [ bit 8..6 ] [ bit 5..3 ] [ bit 2..0 ]
                r % 8          q             j

Swizzle<3,3,3>：

  [ 更高位 ] [   r % 8  ] [ q XOR (r%8) ] [    j    ]
```

代入简化实现：

```cpp
x_prime = x ^ ((x & 0x1c0) >> 3);
```

因为 $x$ 的 bit 6–8 恰好保存行号的低三位，bit 3–5 恰好保存列段编号，所以得到前面的矩阵坐标表达式：

```cpp
x_prime = 64 * r + (c ^ ((r & 7) << 3));
```

这里的三个 `3` 都有具体来历：8 个列段需要 3 bit；每个列段有 8 个 half，需要保留最低 3 bit；行号从 bit 6 开始，向目标 bit 3 移动 3 位。

如果换成相对 **byte offset** $a=2x$，同一物理变换写成：

```cpp
a_prime = a ^ ((a & 0x380) >> 3);  // Swizzle<3,4,3> 作用于 byte offset
```

也就是：

$$
2S_{3,3,3}(x)=S_{3,4,3}(2x).
$$

多出来的一个低位是 byte-in-half。这个等价关系只是在比较同一个 offset 变换的不同计量单位；把某个相对 offset 公式直接作用到绝对指针，还必须检查基址对 bit 字段的影响。


### Composition 的方向决定含义

把地址函数直接写成 CuTe layout：

```cpp
using Plain = Layout<Shape<_8, _64>, Stride<_64, _1>>;
auto smem_layout = composition(Swizzle<3, 3, 3>{}, Plain{});
auto sA = make_tensor(make_smem_ptr(smem), smem_layout);
```

它对应：

$$
L_{smem}(r,c)=\operatorname{Swizzle}(L_{plain}(r,c)).
$$

也就是先把二维坐标变成线性元素偏移，再 swizzle。`sA(r,c)` 仍然使用原来的逻辑坐标；创建 layout 并不执行 copy。真正写入和读取时，CuTe 才用这个映射找到物理位置。Layout 作为坐标到索引的函数，以及 composition 的顺序，可对照 CuTe Layout Algebra 文档。[^layout]

为什么不能把 `Swizzle<3,3,3>` 复制到任何 tile？因为 **bit 6–8 能解释成行号，依赖本例每行 64 个元素**。换成 32 列、128 列、另一种元素位宽，或者把 mode 顺序交换，字段代表的坐标就可能改变。XOR 仍然可逆，但可逆不等于当前指令没有冲突。


## 5. CUTLASS 怎样连接 producer 与 consumer

CUTLASS 的 Turing implicit GEMM 文档给出了另一个常见公式：

```cpp
int store_column = (lane_id % 8) ^ (lane_id / 8);
```

它的图把 shared memory 分成八列，文档称作八个“128-bit banks”。理解这个图时，应把每一列还原为四个 32-bit bank 组成的向量访问组，而不是认为底层硬件切换成了八个独立的 16 B bank。[^cutlass-layout]

令：

$$
v=lane\bmod8,\qquad u=\lfloor lane/8\rfloor.
$$

公式就是 $v'=v\oplus u$。对于固定 $u$ 的八个 lane，$v$ 遍历 `0..7`，物理列仍是一个排列：

| producer 线程组 |  $u$ | 八个 16 B 向量的物理列顺序 |
| --------------- | ---: | -------------------------- |
| T0–T7           |    0 | 0、1、2、3、4、5、6、7     |
| T8–T15          |    1 | 1、0、3、2、5、4、7、6     |
| T16–T23         |    2 | 2、3、0、1、6、7、4、5     |
| T24–T31         |    3 | 3、2、1、0、7、6、5、4     |

它与主例共享向量内低位不动、向量列编号与另一维 XOR的结构。不过，这里的 `lane_id` 描述 **producer 的向量存储映射**；主例中的 T0–T7 描述 **consumer 的矩阵行地址提供者**。同一个 T5，在两张不同指令的线程图里可以承担完全不同的工作。

因此不能只把两个公式里都出现的 `%8`、`/8` 对上，就认定它们是同一套全局 tile 坐标。必须把完整 thread map、tile shape 和迭代偏移一起展开。CUTLASS 各种 TensorOp layout、CuTe composition 都服务于这个目标，但不共享一条适用于所有配置的坐标公式。


### 为什么读与写需要一起设计

从 GEMM 的数据通路看，布局同时面对两端：

```text
global memory
  -- producer: 连续、对齐的向量拷贝 -->
shared memory 的 permuted layout
  -- consumer: ldmatrix 行地址 + 固定 fragment 分配 -->
registers
  -- mma.sync --> accumulator
```

本例保留段内 8 个 half，就保留了 producer 每次搬 16 B 的机会。例如 Ampere 的 `cp.async` 可以从连续的 global 地址写到计算好的 shared 目的地址；源布局和目的布局不必相同。[^cpasync] 但“每个线程的向量连续”仍不是完整证明：还要验证线程之间的 global coalescing，以及 shared store/copy 的实际分组。

另一方面，consumer 可以由各 lane 提供独立的行首地址，而每行内部保持连续。这个接口刚好允许软件重排行段的位置，又不破坏 `ldmatrix` 的行内加载要求。

这里存在一个比“加 XOR”更有用的工程约束：**producer 与 consumer 可以有不同的线程分工，但必须通过同一份逻辑坐标到 shared 物理地址的映射相遇。** CuTe 的 `partition_S/D` 可以帮助表达这个对应关系；它不会自动证明任意自定义 layout 都适合硬件。

### 布局之外，还要满足同步与架构约束

同步也一样是独立问题：`.sync` 表达 warp 对矩阵加载的协作，不能代替 producer/consumer 之间所需的内存可见性与完成等待。普通跨线程 shared store 之后，应使用适当的 warp/CTA 同步；如果由 `cp.async` 或其他异步路径填充，还要按该路径完成等待。附带程序用 `__syncthreads()` 连接普通 shared store 和 `ldmatrix`。

本文也不把软件 XOR 公式直接推广到 Hopper TMA/WGMMA 等通路。那些接口还受硬件 swizzle 模式、descriptor、对齐和同步规则约束，应按相应架构的契约重新建立布局。[^tma]


## 6. Padding、转置与 Swizzle 的取舍


### 从标量 stride 理解 padding

假设 32 个 lane 都有效，每个 lane 读取一个不同的 32-bit word：

$$
w_t=w_0+s t,\qquad t=0,\ldots,31.
$$

bank 序列为 $(w_0+st)\bmod32$，不同 bank 的数量为 $32/\gcd(s,32)$，每个被使用的 bank 收到 $\gcd(s,32)$ 个不同 word。因此：

| stride，单位为 32-bit word | 冲突路数 | 说明                        |
| -------------------------: | -------: | --------------------------- |
|                          1 |        1 | 连续读，32 个 bank 全部覆盖 |
|                          2 |        2 | 只覆盖 16 个 bank           |
|                          4 |        4 | 只覆盖 8 个 bank            |
|                         32 |       32 | 所有请求落到同一个 bank     |
|                         33 |        1 | 模 32 后等效 stride 1       |

这也解释了经典的 `float tile[32][33]` padding：按列访问时，行跨度从 32 word 变成 33 word，消除了原先的 32 路冲突。这里真正要求的是互素；“stride 不是 32 的倍数”只是必要性很弱的判断，stride 2、4、8 仍有冲突。

这个公式有明确边界：stride 0 是广播，不能套成 32 路冲突；FP16、向量 load 和 `ldmatrix` 也要先还原 word 与指令分组，不能直接把元素 stride 代入。

### FP16 矩阵加载还要保留 16 B 对齐


对于 `A[8][64]` FP16，不能机械地把经典 padding 改成 `[8][65]`。这样每行跨度变为 130 B，后续行的起点不再全部满足 `ldmatrix` 的 16 B 自然对齐要求。

一个更合适的对照是 padding 到 72 个 half，每行 144 B：

$$
a_{pad}(r,q)=144r+16q,
$$

$$
b_{pad}(r,q,k)=\big(4(r+q)+k\big)\bmod32.
$$

对固定 $q$，八行仍覆盖八组不同 bank，且每行都保持 16 B 对齐。本例中它也能解决 `.x1` 的访问冲突，只是使用更多 shared memory。

| 方案                    | 8 行的存储量 | 行首对齐               | 本例固定列段的 .x1 读取 |
| ----------------------- | -----------: | ---------------------- | ----------------------- |
| plain `[8][64]`         |       1024 B | 16 B 对齐              | 8 路冲突                |
| padding `[8][65]`       |       1040 B | 部分行不满足 16 B 对齐 | 不能直接按本例行首加载  |
| padding `[8][72]`       |       1152 B | 16 B 对齐              | 无冲突                  |
| XOR swizzle，64 half/行 |       1024 B | 行段保持 16 B 对齐     | 无冲突                  |

上述结果来自地址推导，不是 kernel 时间对比。更大 tile、更深 pipeline 会放大 padding 的容量成本；但这是否跨过 occupancy 的资源阈值，需要代入完整 CTA 的 shared-memory 分配。Swizzle 则把成本转移到布局约束与地址计算上，编译器能消去多少计算同样要检查 SASS。

转置也可以改变访问模式，但要区分“交换逻辑坐标的视图”和“实际把数据转置存储”。视图变化本身不搬数据，也不会自动消除原物理地址集合的冲突。


### 一个反例：紧凑的 8×8 原本就没有冲突

如果只存一个紧凑的 `A[8][8]` FP16 矩阵，行跨度本来就是 16 B。`.x1` 的八个行地址是 `0、16、32、48、64、80、96、112 B`，已经覆盖 G0–G7，不需要本文的 swizzle。

对这 64 个 half 的元素偏移直接使用 `Swizzle<3,3,3>`，源字段 bit 6–8 全为零，实际上也不会发生任何变化。这说明：**产生冲突的不是“8×8 矩阵”这个 shape，而是它被嵌入大 tile 后的行跨度与读取方式。**

另一个边界是改变 consumer：如果之后改成普通标量访问、不同的 `ldmatrix` 切片，或跨越不同 tile 的行集合，必须重新枚举地址。某次读取无冲突，不能推广成整个 shared layout 对所有读取都无冲突。

## 7. 从具体地址推导布局

回到开头的三个 `3`：它们表达的是本例 **8 个 half 构成一个 16 B 向量，8 个向量构成一行，再用行号低三位改变向量所在的 bank group**。这个结构同时保留了行段连续性，并让 `.x1` 所读取的八个行段分散到全部 32 个 bank。

下次遇到另一个 shared-memory layout，可以按下面的顺序推导：

1. 写出逻辑坐标到物理 offset 的函数，注明 offset 单位。
2. 确认目标指令每个 lane 提供什么地址、获得什么数据，以及参与访问的分组。
3. 将每个请求展开成 byte address、word 编号与 bank，分别处理广播和不同 word 竞争。
4. 检查 swizzle 保留的低位是否足够支持向量宽度，并确认 tile 边界、容量、行地址对齐。
5. 用同一映射连接 producer 和 consumer，先验证数据，再比较硬件指标与端到端时间。

Bank conflict 的判断最终落在具体地址上，CUTLASS/CuTe 的价值则是把这些地址关系组织成可以复用和组合的布局。掌握这条推导链，比背下 `column ^ row` 更容易迁移到新的 Kernel。


## 参考资料

[^banks]: NVIDIA, [CUDA C++ Best Practices Guide：Shared Memory and Memory Banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-and-memory-banks)。支持 bank 映射、广播和请求串行化的基本模型；本文的 stride 与矩阵实例由该模型独立推导。
[^ldmatrix]: NVIDIA, [PTX ISA：Warp-level matrix load instruction `ldmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix)。本文只使用 `m8n8.b16` 的地址提供者、寄存器 fragment、对齐与 target 规则，不涉及新增低位宽格式。
[^swizzle]: NVIDIA CUTLASS v4.6.1, [`include/cute/swizzle.hpp`](https://github.com/NVIDIA/cutlass/blob/v4.6.1/include/cute/swizzle.hpp)。`Swizzle::apply`、mask 定义与参数约束。
[^layout]: NVIDIA CUTLASS, [CuTe Layout Algebra：Composition](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#composition)。
[^cutlass-layout]: NVIDIA CUTLASS, [Implicit GEMM Convolution：Shared Memory Layouts](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html#shared-memory-layouts)。Turing TensorOp 的 permutation 示例与 `store_column` 公式。
[^cpasync]: NVIDIA, [PTX ISA：`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)。
[^ncu]: NVIDIA, [Nsight Compute Profiling Guide：Shared Memory](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#shared-memory) 与 [Warp Stall Reasons](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#statistical-sampler)。文档指标解释与本机 CLI 版本分别记录，实际可用指标以目标设备和工具查询结果为准。
[^tma]: NVIDIA CUTLASS, [CuTe TMA Tensors](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html)。用于继续研究 TMA swizzle 的布局与接口约束。
