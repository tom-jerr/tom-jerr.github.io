---
title: CALL
tags: [CS61C]
math: "true"
modified: 星期二, 二月 4日 2025, 9:09:22 晚上
---
    
## Interpretation and Compilation

 当效率不重要时，**interpret** high-level language; 如果需要提升吧表现，则 **translate** 为 lower-level language

- **Interpreter**: 直接运行源代码；能给出更好的报错信息；更慢，但代码更少；平台无关，可以在任何机器上运行
- **translator(compiler)**: 将高级语言翻译为低级语言；可以在转换时加入额外的信息来 帮助 debug，如 `gcc -g`；效率更高，表现更好

## Compiler

- 输入：高级语言 (`foo.c`)
- 输出：汇编语言 (`foo.s` for RISC-V)，可能包含伪指令（assembler 可以处理但不包含在机器中的指令）
- 中间步骤：
    - Lexer: Turns the input into "tokens", recognizes problems with the tokens
    - Parser: Turns the tokens into an "Abstract Syntax Tree", recognizes problems in the program structure
    - Semantic Analysis and Optimization: Checks for semantic errors, may reorganize the code to make it better
    - Code generation: Output the assembly code

## Assembler

- 输入：汇编代码 (`foo.s`)
- 输出：object file (`foo.o`)，包含
    - 机器码（machine code）：CPU 直接执行的二进制指令
    - 符号表（symbol table）：记录函数名（labels）、全局变量（`.data` 后的变量、可能在多个文件中访问的变量）等符号的地址信息
    - 重定位表（relocation table）：标记需要链接器处理的地址（例如未解析的外部函数），以及 static section 中的 data（如在 `la` 指令中使用的）
    - 可以参考 [deepseek 的讲解](deepseek-relocation-table-and-symbol-table.md)
- 读入并利用 directives、替换 pseudo-instructions

### Directives

Give directions to assembler, but not produce machine instructions

- `.text`: Subsequent items put in user text segment (machine code)
- `.data`: Subsequent items put in user data segment (binary rep of data in source file)
- `.global sym`: declares `sym` global and can be referenced from other files
- `.string str`: Store the string `str` in memory and null-terminate it
- `.word w1…wn`: Store the `n` 32-bit quantities in successive memory words

### Pseudo-instructions

| Pseudo-instructions | Real instructions                                        |
| ------------------- | -------------------------------------------------------- |
| `not rd, rs`        | `xori rd, rs, -1`                                        |
| `beqz rs, offset`   | `beq rs, x0, offset`                                     |
| `j offset`          | `jal x0, offset`                                         |
| `ret`               | `jalr x0, rd, offset`                                    |
| `call offset`       | `auipc x6, offset[31:12]`<br>`jalr x1, x6, offset[11:0]` |
| `tail offset`       | `auipc x6, offset[31:12]`<br>`jalr x0, x6, offset[11:0]` |

其中 `tail` 可用于尾递归优化（在 CS61a 中有讲）

### Object File Format

1. **object file header**: size and position of the other pieces of the object file
2. **text segment**: the machine code
3. **data segment**: binary representations of the static data in the source file
4. **relocation information**: identifies lines of the code the need to be fixed up later
5. **symbol table**: list of this file's labels and static data that can be referenced
6. **debugging information**

## Linker

- 输入：object files (`foo.o` `libc.o`)
- 输出：可执行代码 (`foo.out`)
- 步骤：
    - 将每个文件中的 text segment 提出并整合
    - 将每个文件中的 data segment 提出并整合，然后拼接到上一步的 text segment 后
    - Resolve references: 遍历 relocation table 并处理每一个 entry（fill in all absolute addresses）
        - 首先在所有 symbol tables 中搜索 reference
        - 如果找不到，则在标准库中搜索（如 `printf`）
        - 找到之后，就填入对应的地址
- linker 假设 text segment 的第一个 word 的地址为 `0x04000000`
- linker 能知道每个 text 和 data segments 的长度，以及它们间的位置关系
- linker 需要计算每个被跳转的 label 和每份被引用的 data 的 **absolute addresses**

## Loader Basics

- 输入：executable code
- 输出：(program is run)
- 可执行文件存储在磁盘上
- 当指定一个可执行文件运行时，loader 负责将它放入内存并开始运行
- 在现实中，loader 就是操作系统，也会进行许多 linking 的工作
- loader 的工作
    - 读取可执行文件文件头，以决定分配给 text 和 data 的大小
    - 创建新的 **address space**
    - 将 instructions 和 data 复制进这个 address space
    - 将参数复制进 stack
    - 初始化 machine registers（大部分寄存器被清空，但 stack pointer `sp` 被初始化为第一个空闲的 stack location）
    - 跳转至 start-up routine that copies program's arguments from stack to registers and set the PC
        - 当 main routine 返回时，start-up routine terminates the program with the system call

## Static vs. Dynamic Linked Libraries

上述过程是 statically-linked 方式，

- Libraries 是 executable 的一部分，当 libraries 更新时，executable 只能通过重新编译更新
- executable 会包含整个 library，即使只使用了其中一部分代码

与之相对的是 dynamically linked libraries