---
title: lecture11_FSMs
date: 2023-09-11 22:17:49
tags:
  - CS61C
---

# FSMs, Synchronous Digital Systems

## FSM: Finite State Machine

### state transition diagram

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/state_transition_diagram.png)

- s0 初始状态；input/output；箭头指向是 next state

## clock and registers

### Flip-Flop

- 时钟上升沿；Q = D，其余时刻什么也不做

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/flip-flop.png)

### register delay

- clk-to-q delay
  - Registers can’t transfer the D input to Q output instantly
  - clk-to-q delay: Time it takes after the rising edge for the Q output to change

## register constraints

### setup time

- Setup time: Time before rising edge when the D input must be stable

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/setuptime.png)

### hold time

- D input cannot change before this, so the register can read a stable value.

![](https://github.com/tom-jerr/MyblogImg/raw/main/architecture/holdtime.png)

### maximizing clock frequency

- `Clock period ≥ clk-to-q delay + longest combinational delay + setup time`

- `Hold time ≤ clk-to-q delay + shortest combinational delay`
