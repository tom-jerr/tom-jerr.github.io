---
modified: 星期日, 二月 2日 2025, 8:01:45 晚上
title: RISC-V Basic
tags:
  - RISC-V
  - assembly
  - CS61C
math: "true"
---

> 对应 lecture 6 - 8

## Basic

- **CPU**: 
    - Basic Job: 
        - Execute instructions one after another in sequence
        - Each instruction does a small amount of work (a tiny part of a larger program)
    - 不同 CPU implement different sets of instructions
- **ISA**: Instruction Set Architecture
- **RISC**: Reduced Instruction Set Computer
    - 一个指令对应一个操作
- **RISC-V**: RISC 的一个后续版本，简洁 [About RISC-V International](https://riscv.org/about/history/)
- **Register**: 在 processor 中的小存储单元
    - Operations 在 register 中完成，非常快
        - RISC-V 中共 32 个 registers，每个 register 大小为 32 bits (4 bytes / 1 word)
    - 从 0 编号到 31: `x0, x1, …, x31`
        - `x0` 始终存储 0
        - 不能改变名称
    - **Register file**: 在 processor 中的 general purpose registers 

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250131100849441.png)

## Operations

```asm
# x1 = x2 + x3

add x1, x2, x3 

# x1 = x2 - x3

sub x1, x2, x3 

# 取出数组第三位的值，加一后放回

lw x10, 12(x15)
addi x10, x10, 1
sw x10, 12(x15)

# 读取 byte

lb x10, 3(x15)
addi x10, x10, -1
sb x10 3(x15)
```

- `add` 和 `x0` 可用于移动值：`add x3, x4, x0`

### Immediates

提供常数，如 `addi x3, x4, -10`

- `sub` 命令没有 immediates，因为可以用加法实现，就没必要引入额外的指令
- Addi immediates *限制在 12 bits*，在进行运算时，sign extended to 32 bits 

### Load and Store

- `lw`(load word): load data from memory to processor registers
    -  `lw x10, 12(x15)` 将 `x15` 中存储的*地址*偏置（offset） *12 bytes* 之后，取得的值存储进 `x10` （可以理解为取 int 数组第 3 个元素的地址）
    - offset: 必须为 constant，不能为 register
- `sw`(store word): save data from processor registers to memory
    - `sw x10 12(x15)` 将 `x10` 中的值存储进地址 `x15` 偏置 12 bytes 得到的地址
- 类似的还有 `lb`(load byte)、`sb`(store byte)、`lh`(load half-word)、`lhu`
    - `lb` 中，使用 sign extend；可以使用 `lbu`(load byte unsigned) 来进行 zero extend
    - `sb` 中，将最低 8 位存储进去，所以没有 extend
- load instructions 是 I-format 的

## Logical Instructions

| C Operator | RISC-V Instructions                                    |
| ---------- | ------------------------------------------------------ |
| &          | and                                                    |
| \|         | or                                                     |
| ^          | xor                                                    |
| <<         | sll(shift left logical)                                |
| >>         | srl(shift right logical) / sra(shift right arithmetic) |

```asm
# x10 = x11 << x12

sll x10, x11, x12

# x10 = x11 << 2

slli x10, x11, 2
```

## Branches

### Conditional

```asm
beq reg1, reg2, L1
```

- 当 `reg1 == reg2` 时，跳转至 `L1` 标签处；否则继续执行下一个语句

```asm
bne reg1, reg2, L1
```

- 举例

```c
if (a == b)
    e = c + d
```

```asm
      bne x10, x11, Exit
      add x14, x12, x13
Exit:
```

- 类似的还有 `blt`(branch less than), `bge`(branch greater or equal to)
    - 对于无符号的数，加上 `u` 后缀 `bgeu` `bltu`
    - 只有 `blt` `bge`，因为这两个足以应对所有比较；但是存在伪指令 `bgt`，会被编译器转换为真实 RISC-V 指令 `bgt x2, x3, L1 -> blt x3, x2, L1`

### Unconditional

- jump to label: `j label`
- 举例：实现 if-else 语句

```c
if (a == b)
    e = c + d;
else
    e = c - d
```

```asm
      bne x10, x11, else
      add x14, x12, x13
      j done
else: sub x14, x12, x13
done:
```

- 实现 for 循环：

```c
int A[20];
int sum = 0;
for (int i = 0; i < 20; i++)
    sum += A[i]
```

- 假设 `x8` 存储了 `A` 的地址

```asm
      add x9, x8, x0          # x9 = &A[0]
      add x10, x0, x0         # sum = 0
      add x11, x0, x0         # i = 0
      addi x13, x0, 20        # x13 = 20
Loop: bge x11, x13, Done
      lw x12, 0(x9)           # x12 = A[i]
      add x10, x10, x12       # sum += A[i]
      addi x9, x9, 4          # x9 =  &A[i] + 1
      addi x11, x11, 1        # i += 1
      j Loop
Done: 
```

## Jump Function

- jump function 需要执行两个功能：
    - 存储跳转位置（在函数中需要，而 if-else、for 语句中不需要）
    - 更新 PC 值
- `jal rd, label`(jump and link)
    - `rd`(register destination): 用于存储 return address 的寄存器，即 `PC + 4`
    - `label`: 标签。被 assembler 解释为有 20 bits 的 offset
    - 实际上是命令 `jal rd, offset` 的更高级的抽象形式，将 `label` 转换为对应的 `offset` 之后进行操作
    - `jar rd, offset`: 将 `PC + 4` 的值存入 `rd`，然后进行跳转 `PC = PC + offset`
    - 可以使用 `x0` 来避免储存返回地址：`jal x0, label`
        - `j label` 实际上是 `jal x0, label` 的伪指令
- `jalr rd, offset(rs1)`(jump and link register): 用于实现任意跳转
    - 先将 `PC + 4` 存储到 `rd`，然后跳转 `PC = rs1 + offset`
    - 在本课中，写作 `jalr rd, rs, imm`
    - 同样有伪指令 `jalr x0, ra, 0 -> jr ra -> ret`（`ra` 是一个 calling convention）
    - 举例

```asm
caller:
    # do some thing
    jal ra, callee
    # do more thing
callee:
    # do some thing
    ret
```

## Saving Registers

- 在多重调用函数的情况下，需要用 stack 来存储中间变量（registers）
    - 压栈：`addi sp, sp, -x`
    - 出栈：`addi sp, sp, x`
- 保存 registers 的方式需要遵循 *calling convention*

### Calling Convention

- Temporary registers: 由 caller 保存
- Saved registers: 由 callee 保存

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250131203313591.png)

### Calling Functions

1. Put **parameters** in a place where function can access them (`a0` - `a7`)
2. **Transfer control** to function (`jal, j, jalr`)
3. Acquire (local) **storage resources** needed for function (`sp` move)
4. **Perform** desired task of the function
5. Put **result value** in a place where calling code can access it (`a0` - `a1`)
6. **Return control** to point of origin (`ret`)

## Instruction Formats

对于 instructions，有

- 每一个 instruction 都是 32 bits 大小
- 都被划分为不同的 fields
- 有不同的方式划分一个 instruction 的不同部分

### R-format

Deepseek 的讲解：[deepseek-R-format](deepseek-R-format.md)

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250131213646707.png)

### I-format

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250131214059836.png)

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250131214428117.png)

### S(tore)-Format

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250131215244581.png)

如 store instructions `sw`，`sb`，`sh`

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250131215501342.png)

### B(ranch)-format

由于在 branch 中，有 12 bits 用于存储 offset immediates，且又由于 PC 的值为 4 bytes 的，所以可以忽略最低的两位（总为 0），则可以跳转到相对位置 $\pm 2^{11}$ 个指令（$\pm 2^{13}$ 个 bytes）

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250202181547117.png)

实际上，immediates 的有效值为 `imm[12:1] << 1`

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250202191212669.png)

### J(al)-format

在 `jal rd, label` 指令中，assembler 解释为有 20 bits 的偏置 immediate

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250202191640015.png)

- 实际偏移量为 `imm[20:1] << 1`，因为最后一位 0 被省略了
- 这种不自然的划分是为了简化硬件上的实现
- 目标地址的计算：`target_address = PC + sign_extend(imm[20:1] << 1)`

在 `jalr rd, rs, imm` 指令中，使用 I-format，故最后一位 0 不可省略

### U-format

- `lui x10, imm`(load upper immediate): 将 `imm` 存储到目标寄存器中的前 20 bits 中
- 由于 `addi` 中的 `imm` 限制为 12 bits，故可以结合 `lui` 和 `addi` 用来存储任意大小的 32 bits 常数

```asm
lui x10, 0x87654
addi x10, x10, 0x321
```

- 可以用伪指令 `li` 来简化上述过程：`li x10, 0x87654321`
- `auipc x10, imm`(add upper immediate pc): `x10 = pc + imm << 12`

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250202200112986.png)

### Summary

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250202200158345.png)
