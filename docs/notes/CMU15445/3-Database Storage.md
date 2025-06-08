---
title: 3_Database_Storage I
date: 2023-09-11 22:17:49
tags:
  - Database
---

# 3-Database Storage I

## 1 DISK-BASED ARCHITECTURE

- 易失性存储和非易失性存储相结合
- Volatile：Random Access Byte-Addressable （DRAM 之上）
- Non-Volatile：Sequential Access Block-Addressable（SSD 之下 ）
- SSD 以下只能按块来存取

![](https://github.com/tom-jerr/MyblogImg/raw/15445/storage.png)

### 顺序访问和随机访问

- 非易失性存储中顺序访问比随机访问快得多
- 一般数据存储在连续的块中，同时分配多个物理页叫区

## 2 DBMS 设计目标

- 允许管理比可用内存大的数据库
- 读写磁盘代价高昂，尽可能避免大的停顿和性能下降
- DBMS 希望最大化顺序读写

![](https://github.com/tom-jerr/MyblogImg/raw/15445/DBMS_DISK.png)

## 3 为什么不使用 OS

- The DBMS can use memory mapping (**mmap**) to store the contents of a file into the address space of a program.

- The OS is responsible for moving the pages of the file in and out of memory, so the DBMS doesn’t need to worry about it.

![](https://github.com/tom-jerr/MyblogImg/raw/15445/mmap.png)

### 使用 OS 可能出现问题

- 如果此时物理内存已经占满，需要淘汰一个物理页，造成阻塞
- 多线程并发读写可能出现问题
- 错误处理
- 性能问题：OS 的结构成为性能瓶颈

![](https://github.com/tom-jerr/MyblogImg/raw/15445/MMAP_PRO.png)

### OS 解决方案

- 按正确的顺序将脏页刷新到磁盘中
- 特殊的预缓存情况，这样执行器就不用等待特定的页加载到缓冲池中
- 缓冲替换策略
- 线程、进程调度

## 4 File Storage

- Table 可以看为是 tuple 的集合，在磁盘中与之对应的单位是文件。一个文件中会保存多个 page，而 page 中会存储多个 tuple。

- DBMS 在磁盘上，用一个或多个文件，以特定的格式存储数据库

  - OS 不关心也不知道这些文件的内容

- 1980 年代的 一些 DBMS 在原始存储（Raw Storage）上使用自定义的文件系统
  - 现在的一些商业 DBMS 还在这么干
  - 大多数新出来的 DBMS 不这么做了

### 4.1 storage manager

- 调度读写操作提高空间和时间局部性
- 将文件组织成页的集合
  - 跟踪已经读和写的页
  - 跟踪空闲页

### 4.2 Database Pages

因为数据库文件可能很大，需要切分成多个块

- 一页是很多个 block 组成的集合

  - 页里可能存着元组、元数据、索引、日志记录...
  - 大多数系统对于页的使用都是固定的，比如存储索引的页不会用来存数据
  - 大多数系统需要页是自组织（self-contained）的，也就是说页的 Header 里标识了这个页的类型

- 每个页都有一个独特的标识符

  - DBMS 使用非直接的层来把 page id 映射到物理位置硬盘页
  - 最小能保证原子操作的单位

#### 不同的页概念

- 硬件页（如坏块）：4KB

- 操作系统页（一般为 4kb）操作系统读写硬盘的最小单位

- 数据库页（512b ~ 16kb）

### 4.3 Page Storage Architecture

- heap file organization
- tree file organization
- sequential/sorted file organization (ISAM)
- hashing file organization

#### heap file

- 无序存放页，容易寻找单个文件
- 可以 create, get, write, delete 页
- 必须支持迭代所有的页

#### 实现方式

- **page directory** (设置一页为 directory)

  - pagedirectory 和 data page 要同步更新
  - The directory also records meta-data about available space:
    → The number of free slots per page.
    → List of free / empty pages

  ![](https://github.com/tom-jerr/MyblogImg/raw/15445/page-dic.png)

- **linked list**

  - 一个 header page，维护两个链表头节点；free page list, data page list

  ![](https://github.com/tom-jerr/MyblogImg/raw/15445/linked-list.png)

### 4.4 page layout

#### page header

- page 内容的元信息
  - page size
  - checknum（页的校验和）
  - DBMS version
  - Transaction Visibility（并发是否锁住该页）
  - Compression Information（压缩信息）

#### data inside of the page

- tuple-oriented
- Log-strctured

##### 1 tuple storage in page

- Keep track of the number of tuples in a page and then just append a new tuple to the end.

![](https://github.com/tom-jerr/MyblogImg/raw/15445/tuple.png)

- 上图所示的是一种朴素的做法，类似数组的方式去存储，header 维护一个数组的信息。
- 主要问题：删除时产生内存碎片，文件的不连续等等（如果是**变长的 tuple** 将会有更多问题），tuple 的查找也是一个很大的开销

##### 2 slotted pages（页槽）

- 将特定 slot 映射到 page 某个特定偏移量上的数据结构，这样一个元组就是由一个 page id 和 slot id 来唯一定位，并且 tuple 是倒序存储的。这样当然可能会在中间有部分数据的浪费，但是为了支持变长的元素我们不得不这么做。当然，有办法去应对，**可以去整理或者压缩**
- header keeps track of
  - used slots
  - the offset of the starting location of the last slot used

![](https://github.com/tom-jerr/MyblogImg/raw/15445/slotted_pages.png)

- 每个 tuple 需要一个独一无二的记录号：page_id + offset/slot
- 用户不能使用 record ID 来使用

### 4.5 tuple layout

- tuple 是磁盘上的二进制流数据

![](https://github.com/tom-jerr/MyblogImg/raw/15445/tuple2.png)

- header contaions meta-data about tuple
  - visibility info (concurrency control)
  - Bit Map for **NULL** values
- 存储数据无分隔符，需要记录 NULL 信息

![](https://github.com/tom-jerr/MyblogImg/raw/15445/tuple3.png)

#### denormalized tuple data

- 物理存储上反规范化，将有关联的元组存储在同一页上；相当于提前将表关联 join
- 减少 IO 数量
- 更新的代价巨大

![](https://github.com/tom-jerr/MyblogImg/raw/15445/denormalize.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/denormalize1.png)

![](https://github.com/tom-jerr/MyblogImg/raw/15445/denormalize2.png)
