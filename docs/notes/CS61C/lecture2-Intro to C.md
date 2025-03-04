---
title: lecture2_Intro_to_C
date: 2023-09-11 22:17:49
tags:
  - CS61C
---

# Intro to C

## Great idea in Computer Architecture

### 1. Abstraction

- (Layers of Representation/Interpretation)

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/abstraction.png)

### 2. Moore's Law（摩尔定律）

### 3. Principle of Locality/Memory Hierarchy（局部性和内存层次性原则）

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/memory.png)

### 4. Parallelism（并行）

- 加速频繁事件（Amdahl's law）

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/amdahl.png)

### 5. Performance Measurement & Improvement

### 6. Dependability via redundancy（冗余实现可靠性）

- RAID...

- 失去一个数据中心，但是整个网络不会宕机

## Compiled & Interpreted（编译和解释）

- Translation happens in two ways
  ○ Compilation
  ○ Interpretation
  ○ Some languages use both!

- C compilers map C programs directly into architecture-specific machinecode (string of 1s and 0s)

- Java converts to architecture-independent bytecode which is then compiled by a just-in-time (JIT) compiler.
- Python environments converts to Python bytecode at runtime instead of at compile-time.

  - Runtime versus JIT compilation differ in when the program is converted to low-level assembly language that is eventually translated into machine code.

- With C, there is generally a 3-part process in handling a .c file

  - .c files are compiled into .s files ⇒ compilation by the compiler
  - .s files are assembled into .o files ⇒ assembly by the assembler (this step is generally hidden, so most of the time we directly convert .c files into .o files)
  - .o files are linked together to create an executable ⇒ linking by the linker

### 编译的优点和缺点

- 优点

  - 很好的运行时表现(because it optimizes for a given architecture)

- 缺点
  - 依赖特定的硬件平台
  - 在新系统上必须重新 build
  - “Change → Compile → Run [repeat]” iteration cycle can be slow during development
    - but make only rebuilds changed pieces, and can compile in parallel: make -j
    - **linker is sequential though** → Amdahl’s Law
