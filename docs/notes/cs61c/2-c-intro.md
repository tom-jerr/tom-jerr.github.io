---
math: false
tags:
  - CS61C
  - C
title: C Intro
---

- C compilation simplified overview: `foo.c -> compiler -> foo.o -> linker -> foo.out` 
- **CPP**: C Pre-Precessor. `foo.c -> CPP -> foo.i -> compiler`
    - Commands begin with `#`
    - 将注释转换成空格
- [C vs. Java](https://introcs.cs.princeton.edu/java/faq/c2java.html)
- `{u|}int{#}_t`: 指定存储方式，如 `uint8_t`，`int64_t`
- structure:

```c
typedef struct {
    int a;
    char c;
    float* b;
} test_t;

// equivalently
struct test_t {
    ...
};
```

- 一个结构体中的字节被对齐，故不能简单的进行加法来得到整个结构体的大小
    - `char`: 一个 byte，不需对齐
    - `short`: 两个 bytes，需要对齐到 `0, 2, 4, ...`处
    - `int` 和 `void *`: 四个 bytes，不需要对齐
    - 如果最终大小不为 4 的倍数，则扩展到 4 的倍数（如 `13 -> 16`）
- 使用 `union` 可以指定用最大的元素的大小进行对齐：

```c
union foo {
    int a;
    char b;
    union foo* c;
};
```

- 获取 main 函数的参数：

```c
int main(int argc, char* argv[]) {}
```

- 大端、小端表示的不同

## Memory Management

![](https://cdn.jsdelivr.net/gh/KinnariyaMamaTanha/Images@images/20250122083503028.png)


- **Stack**:
    - 包括若干 **frames**，其中有 return address、arguments、space for local variables
    - 当一个 **frame** 执行完毕后，stack pointer 上移，释放空间
- **Heap**:
    - `malloc(size_t n)`: 分配未初始化的内存，返回类型 `void *`
    - `calloc(size_t nmem, size_t size)`: 分配 0 初始化的内存，大小为 `nmem * size`
    - `free(void *)`: 释放内存，只能释放由 `malloc` 分配的
        - 如 `int *a = malloc(sizeof(int) * 2); free(a+1)` 是不可行的
    - `realloc(void *p ,size_t size)`: 改变之前分配内存的大小，返回类型 `void *`
        - 可能会导致内存地址移动。其他指针不会随之一同改变！
        - 当 `p == NULL` 时，和 `malloc` 行为相似
        - 当 `size == 0` 时，和 `free` 行为相似

```c
int *ip = (int *) malloc(sizeof(int))
```

- [Buddy memory allocation - Wikipedia](https://en.wikipedia.org/wiki/Buddy_memory_allocation)

## String

- 拷贝字符串：

```c
char *a;
char *b = "Hello";

a = malloc(sizeof(char) * (strlen(b) + 1))

strcpy(a, b);
// or
strncpy(a, b, strlen(b) + 1);
```
