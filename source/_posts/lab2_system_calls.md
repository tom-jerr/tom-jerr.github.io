---
title: MIT6.S081-System Calls
index_img: /img/mit.png
categories:
- OS
tags:
- labs
- C&C++
- linux
comment: valine
math: true
---

# 前言

本文是对MIT6.S081中lab2 system calls的实现
<!-- more -->

# System Calls

## 系统调用

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/system_call.png)

initcode.S调用exec系统调用结束（user/initcode.S:11）。让我们来看看用户调用是如何在内核中实现exec系统调用的。

用户代码将**exec**的参数放在寄存器**a0**和**a1**中，并将系统调用号放在**a7**中。系统调用号与函数指针表**syscalls**数组(kernel/syscall.c:108)中的项匹配。

```C
... 
  3 .global fork
  4 fork:
  5 	li a7, SYS_fork
  6 	ecall
  7 	ret
  8 .global exit
  9 exit:
 10 	li a7, SYS_exit
 11 	ecall
 12 	ret
...
```

1. 陷入陷阱（trap）时，`uservec(kernel/trampoline.S:16)` 被调用，用户寄存器被保存至`trapframe`中，由于在函数调用时，调用参数被存放在在寄存器中，因此使得内核对用户系统调用的参数是可见的。
2. 在`uservec`的结尾处，`usertrap (kernel/trap.c:37)`被调用。在该函数中，内核检测产生陷阱的原因（硬件中断、系统调用或异常），当陷阱原因为系统调用时，`syscall (kernel/syscall.c:133)`被调用以处理系统调用，最后内核通过`usertrapret (kernel/trap.c:90)`返回至用户态。

在`syscall`中，系统调用号通过存放在`trapframe`中的寄存器`a7`被取出，并被作为系统调用表`syscalls`的索引，以寻找所需的系统调用函数，并对其进行调用。其中，系统调用表`syscalls`是一个函数指针数组，其定义如下：
```

​```C
110 static uint64 (*syscalls[])(void) = {
111 [SYS_fork]    sys_fork,
112 [SYS_exit]    sys_exit,
113 [SYS_wait]    sys_wait,
114 [SYS_pipe]    sys_pipe,
115 [SYS_read]    sys_read,
116 [SYS_kill]    sys_kill,
117 [SYS_exec]    sys_exec,
118 [SYS_fstat]   sys_fstat,
119 [SYS_chdir]   sys_chdir,
120 [SYS_dup]     sys_dup,
121 [SYS_getpid]  sys_getpid,
122 [SYS_sbrk]    sys_sbrk,
123 [SYS_sleep]   sys_sleep,
...
134}
```

在`sysproc.c`中，实际的系统调用行为被完成，值得注意的是，用户传入系统调用函数的参数通过 `argraw` 被从 `trapframe` 中提取：

```C
 34 static uint64
 35 argraw(int n)
 36 {
 37   struct proc *p = myproc();
 38   switch (n) {
 39   case 0:
 40     return p->trapframe->a0;
 41   case 1:
 42     return p->trapframe->a1;
 43   case 2:
 44     return p->trapframe->a2;
 45   case 3:
 46     return p->trapframe->a3;
 47   case 4:
 48     return p->trapframe->a4;
 49   case 5:
 50     return p->trapframe->a5;
 51   }
 52   panic("argraw");
 53   return -1;
 54 }
```

**syscall** (kernel/syscall.c:133)从**trapframe**中的a7中得到系统调用号，并其作为索引在**syscalls**查找相应函数。对于第一个系统调用**exec**，a7将为**SYS_exec(kernel/syscall.h:8)**，这会让**syscall**调用**exec**的实现函数**sys_exec**。

当系统调用函数返回时，**syscall**将其返回值记录在**p->trapframe->a0**中。用户空间的**exec()**将会返回该值，因为RISC-V上的C调用通常将返回值放在**a0**中。系统调用返回负数表示错误，0或正数表示成功。如果系统调用号无效，**syscall**会打印错误并返回1。

~~~C
void
syscall(void)
{
  int num;
  struct proc *p = myproc();

  num = p->trapframe->a7;
  if(num > 0 && num < NELEM(syscalls) && syscalls[num]) {
    // Use num to lookup the system call function for num, call it,
    // and store its return value in p->trapframe->a0
    p->trapframe->a0 = syscalls[num]();
  } else {
    printf("%d %s: unknown sys call %d\n",
            p->pid, p->name, num);
    p->trapframe->a0 = -1;
  }
}
~~~



***

## 系统调用传递参数

内核的系统调用实现需要找到用户代码传递的参数。因为用户代码调用系统调用的包装函数，参数首先会存放在寄存器中，这是C语言存放参数的惯例位置。内核trap代码将用户寄存器保存到当前进程的**trapframe**中，内核代码可以在那里找到它们。函数**argint**、**argaddr**和**argfd**从**trapframe**中以整数、指针或文件描述符的形式检索第n个系统调用参数。它们都调用**argraw**在**trapframe**中检索相应的数据(kernel/syscall.c:35)。

~~~C
static uint64
argraw(int n)
{
  struct proc *p = myproc();
  switch (n) {
  case 0:
    return p->trapframe->a0;
  case 1:
    return p->trapframe->a1;
  case 2:
    return p->trapframe->a2;
  case 3:
    return p->trapframe->a3;
  case 4:
    return p->trapframe->a4;
  case 5:
    return p->trapframe->a5;
  }
  panic("argraw");
  return -1;
}
~~~



一些系统调用传递指针作为参数，而内核必须使用这些指针来读取或写入用户内存。例如，**exec**系统调用会向内核传递一个指向用户空间中的字符串的指针数组。这些指针带来了两个挑战。首先，用户程序可能是错误的或恶意的，可能会传递给内核一个无效的指针或一个旨在欺骗内核访问内核内存而不是用户内存的指针。第二，xv6内核页表映射与用户页表映射不一样，所以内核不能使用普通指令从用户提供的地址加载或存储。

内核实现了安全地将**数据复制到用户提供的地址或从用户提供的地址复制数据**的函数。例如**fetchstr(kernel/syscall.c:25)**。文件系统调用，如exec，使用**fetchstr**从用户空间中检索字符串文件名参数，**fetchstr**调用**copyinstr**来做这些困难的工作。

~~~C
// Fetch the nul-terminated string at addr from the current process.
// Returns length of string, not including nul, or -1 for error.
int
fetchstr(uint64 addr, char *buf, int max)
{
  struct proc *p = myproc();
  if(copyinstr(p->pagetable, buf, addr, max) < 0)
    return -1;
  return strlen(buf);
}
~~~



**copyinstr** (kernel/vm.c:406) 将用户页表 **pagetable** 中的虚拟地址 **srcva** 复制到 **dst**，需指定最大复制字节数。它使用**walkaddr**（调用**walk**函数）在软件中模拟分页硬件的操作，以确定**srcva**的物理地址**pa0**。**walkaddr** (kernel/vm.c:95)检查用户提供的虚拟地址是否是进程用户地址空间的一部分，所以程序不能欺骗内核读取其他内存。类似的函数**copyout**，可以将数据从内核复制到用户提供的地址。

~~~C
// Copy a null-terminated string from user to kernel.
// Copy bytes to dst from virtual address srcva in a given page table,
// until a '\0', or max.
// Return 0 on success, -1 on error.
int
copyinstr(pagetable_t pagetable, char *dst, uint64 srcva, uint64 max)
{
  uint64 n, va0, pa0;
  int got_null = 0;

  while(got_null == 0 && max > 0){
    va0 = PGROUNDDOWN(srcva);
    pa0 = walkaddr(pagetable, va0);
    if(pa0 == 0)
      return -1;
    n = PGSIZE - (srcva - va0);
    if(n > max)
      n = max;

    char *p = (char *) (pa0 + (srcva - va0));
    while(n > 0){
      if(*p == '\0'){
        *dst = '\0';
        got_null = 1;
        break;
      } else {
        *dst = *p;
      }
      --n;
      --max;
      p++;
      dst++;
    }

    srcva = va0 + PGSIZE;
  }
  if(got_null){
    return 0;
  } else {
    return -1;
  }
}


// Return the address of the PTE in page table pagetable
// that corresponds to virtual address va.  If alloc!=0,
// create any required page-table pages.
//
// The risc-v Sv39 scheme has three levels of page-table
// pages. A page-table page contains 512 64-bit PTEs.
// A 64-bit virtual address is split into five fields:
//   39..63 -- must be zero.
//   30..38 -- 9 bits of level-2 index.
//   21..29 -- 9 bits of level-1 index.
//   12..20 -- 9 bits of level-0 index.
//    0..11 -- 12 bits of byte offset within the page.
pte_t *
walk(pagetable_t pagetable, uint64 va, int alloc)
{
  if(va >= MAXVA)
    panic("walk");

  for(int level = 2; level > 0; level--) {
    pte_t *pte = &pagetable[PX(level, va)];
    if(*pte & PTE_V) {
      pagetable = (pagetable_t)PTE2PA(*pte);
    } else {
      if(!alloc || (pagetable = (pde_t*)kalloc()) == 0)
        return 0;
      memset(pagetable, 0, PGSIZE);
      *pte = PA2PTE(pagetable) | PTE_V;
    }
  }
  return &pagetable[PX(0, va)];
}
~~~

***

## xv6系统调用函数

### fstat

**int** fstat(**int** fd, **struct** stat*****)：通过文件描述符获取文件状态；

### sbrk

**char\*** sbrk(**int**)：实现虚拟内存到内存的映射；

### dup

**int** dup(**int**)：复制一个文件描述符，二者指向同一个文件；

### uptime

**int** uptime(**void**)：显示系统总共运行了多长时间和系统的平均负载

***

## System call tracing 

`sysproc`中函数的外来值均是标准输入；命令传入



## Sysinfo

从标准输入传入一个`sysinfo`类型的指针；

为了提取系统进程数和系统空闲内存数，我们需要在负责遍历进程列表的 `proc.c` 和空闲内存页链表的 `kalloc.c` 文件中增加新的功能函数。

在xv6中，进程的信息是被`proc.c`中的进程列表所保存的：

```C
 11 struct proc proc[NPROC];
```

为了提取系统进程数，我们仅需遍历该进程列表，寻找状态 `state` 不为 `UNUSERD` 的进程即可：

```C
701 // lab2 syscall : get the nproc
702 uint64
703 getnproc(void){
704     struct proc* p;
705     uint64 ret = 0;
706     for(p = proc; p < &proc[NPROC]; p++) 
707         if(p->state != UNUSED)
708             ret++;
709     return ret;
710 }
```

在xv6中，空闲的内存块被组织成空闲内存页链表，其中的每个内存块的大小均为`PGSIZE`字节，即为页大小。在这里，我们遍历链表，记录结点个数即可，需要注意遍历前后对链表加删锁：

```C
 84 //lab2 syscall : get the freemem
 85 uint64
 86 getfreemem(void){
 87     struct run *r;
 88     uint64 ret = 0;
 89     acquire(&kmem.lock);
 90     r = kmem.freelist;
 91     while(r){
 92         ret += PGSIZE;
 93         r = r->next;
 94     }
 95     release(&kmem.lock);
 96     return ret;
 97 }
```

有了这两个功能函数，我们仅需实现 `sysproc.c` 中的实际系统调用函数即可，**注意需要导入 `sysinfo.h` 头文件**，并且在`kernel/def.h` 中声明`struct sysinfo` 以及`getnproc`、`getfreemem`：

***

## 调试

`gdb-multiarch`进行调试；在kernel.asm处寻找需要打断点的函数；一般为函数名；执行程序(continue)可以进行调试。