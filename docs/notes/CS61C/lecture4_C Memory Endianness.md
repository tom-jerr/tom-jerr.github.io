---
title: lecture4_C_Memory
date: 2023-09-11 22:17:49
tags:
  - System Arch
---

# C Memory

## memory alignment（内存对齐）

- Memory alignment: a system-dependent rule that tells us where data
  can be stored (usually not at every byte)

- Can be accomplished with the "aligned" attribute
  `__attribute__((aligned(n)));`

## Strings

- Example: The string “Hi” is stored in memory as an array of characters. This array would look like: {‘H’, ‘i’, ‘\0’}

- If the null terminator is missing, C will continue to read memory as if
  it was part of the string until it comes across a byte that happens to **have value 0**.
- This is a very common source of bugs.
