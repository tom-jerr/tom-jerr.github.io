---
title: lecture14_DataPath_Hazards
date: 2023-09-11 22:17:49
tags:
  - CS61C
---

# Hazards

## Structural Hazards

- 两条或更多指令在流水线需要访问同一个物理单元

### Solutions

- 指令轮流使用物理资源
- 增加额外的硬件资源
- 设计指令集避免结构冒险

## Data Hazards

- Some regfiles support writing a new value to a register, then reading the new value, in the same cycle.
- a register being written to is read from later.

### Solutions

- Stalling

  - Wait for the first instruction to write its result before the second instruction reads the
    value

  ![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/stallIng.png)

- Forwarding

  - Add hardware to send the result back to earlier stages before the result is written
  - Requires extra connections in the datapath

  ![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/forward.png)

- Code Scheduling

  - Rearrange instructions to avoid data hazards
  - Compiler can put an unrelated instruction in the nop slot

  ![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/code_schedule.png)

## Control Hazards

- If branch is not taken:

  - Instructions fetched sequentially after branch are correct

  - No control hazard

- If branch is taken, or there’s a jump:
  - The next two instructions still in the pipeline are incorrect
  - Need to convert incorrect instructions in the pipeline to nops
  - Called “flushing” the pipeline

### Solutions

- Branch Prediction to Reduce Penalties（分支预测）

  - Early in the pipeline, guess which way branches will go.
  - Flush pipeline if the guess was incorrect.

- Naive branch prediction: just predict branch “not taken”, which always fetches PC+4

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/superscalar.png)
