---
title: C++-IO库
tags:
  - C++
---
# IO库

## IO class

![](https://github.com/tom-jerr/MyblogImg/raw/main/C++/IO/IOKU.png)

- **IO对象无拷贝，无赋值：不能作为参数和返回值**
- 读写一个IO对象会改变其状态，传递和返回的引用不能是const

### 条件状态

![](https://github.com/tom-jerr/MyblogImg/raw/main/C++/IO/IOstate.png)

### 管理输出缓冲

- **程序崩溃，输出缓冲区不会被刷新**

- 每个输出流都管理一个缓冲区

- 缓冲刷新

  - 程序正常结束，作为main函数的return操作一部分，缓冲被执行
  - 缓冲区满时，需要刷新缓冲
  - endl等操纵符，显示刷新缓冲区
  - 设置unitbuf来清空缓冲区
  - 一个输出流可能被关联到另一个流。这种情况下，读写被关联的流时，关联到的流的缓冲区会被刷新

- endl：输出换行符，刷新缓冲区

- flush：刷新缓冲区，不附加任何额外字符

- ends：输出一个空字符，刷新缓冲区

  ~~~c++
  cout << unitbuf;	// 所有输出操作立即刷新缓冲区
  cout << nounitbuf;	// 回到正常的缓冲方式
  ~~~

## 文件输入输出

![](https://github.com/tom-jerr/MyblogImg/raw/main/C++/IO/fstream.png)

- 自动构造和析构：出作用域，立即销毁
- 当一个fstream对象被销毁时，close自动调用

### 文件模式

![](https://github.com/tom-jerr/MyblogImg/raw/main/C++/IO/filemode.png)

- 以out模式打开文件丢弃已有数据；避免清空同时指定app
- 每次调用open确定文件模式（显示或隐式（out | trunc））

## string流

![](https://github.com/tom-jerr/MyblogImg/raw/main/C++/IO/stringstream.png)