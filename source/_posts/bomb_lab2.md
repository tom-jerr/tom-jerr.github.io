---
title: CSAPP--bomblab
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

本文是对CSAPP中bomb lab的实现
<!-- more -->

# bomb_lab2

## 一些基本操作

### 反汇编文件生成

文件中只有bomb.c和bomb可执行文件，所以需要查看汇编代码，需要生成反汇编文件；

~~~bash
$ objdump -d bomb > bomb.asm
~~~

### gdb查看寄存器值

~~~shell
x/s $esi
~~~

## phase_1

- 输入一个字符串

~~~C
400ee0:	48 83 ec 08          	sub    $0x8,%rsp	/*设置栈空间*/
400ee4:	be 00 24 40 00       	mov    $0x402400,%esi
400ee9:	e8 4a 04 00 00       	callq  401338 <strings_not_equal>
400eee:	85 c0                	test   %eax,%eax	 /*test指令同逻辑与and运算，但只设置条件码													寄存器，不改变目的寄存器的值，test %eax,%eax												   用于测试寄存器%eax是否为空，由于寄存器%rax一般													存放函数的返回值，此处应该存放的是函数 													  strings_not_equal的值，而%eax是%rax的低32												  位表示，所以不难分析出，当%eax值为0时，test的												   两个操作数相同且都为0，条件码ZF置位为1，即可满												  足下一行代码的跳转指令*/
400ef0:	74 05                	je     400ef7 <phase_1+0x17>
400ef2:	e8 43 05 00 00       	callq  40143a <explode_bomb>
400ef7:	48 83 c4 08          	add    $0x8,%rsp	/*回收栈空间*/
400efb:	c3                   	retq   
~~~

### Input

Border relations with Canada have never been better.

***

## phase_2

- 输入6个数字比较是否相等

~~~c
  400efc:	55                   	push   %rbp	//把数据压入栈中
  400efd:	53                   	push   %rbx
  400efe:	48 83 ec 28          	sub    $0x28,%rsp
  400f02:	48 89 e6             	mov    %rsp,%rsi
  400f05:	e8 52 05 00 00       	callq  40145c <read_six_numbers>	/*调用																	read_six_number程序，读取6个数字*/
  400f0a:	83 3c 24 01          	cmpl   $0x1,(%rsp)	//将第一个输入数字与立即数1比较
  400f0e:	74 20                	je     400f30 <phase_2+0x34>
  400f10:	e8 25 05 00 00       	callq  40143a <explode_bomb>
  400f15:	eb 19                	jmp    400f30 <phase_2+0x34>
  400f17:	8b 43 fc             	mov    -0x4(%rbx),%eax	//把此时的%rsp值传递给%eax（循环															 //开始）
  400f1a:	01 c0                	add    %eax,%eax	//%eax = %eax * 2
  400f1c:	39 03                	cmp    %eax,(%rbx)	//比较新的输入数据与%eax中的数据
  400f1e:	74 05                	je     400f25 <phase_2+0x29>
  400f20:	e8 15 05 00 00       	callq  40143a <explode_bomb>
  400f25:	48 83 c3 04          	add    $0x4,%rbx
  400f29:	48 39 eb             	cmp    %rbp,%rbx	//判断是否已经判定6个数字
  400f2c:	75 e9                	jne    400f17 <phase_2+0x1b>
  400f2e:	eb 0c                	jmp    400f3c <phase_2+0x40>
  400f30:	48 8d 5c 24 04       	lea    0x4(%rsp),%rbx	//当前指针加4
  400f35:	48 8d 6c 24 18       	lea    0x18(%rsp),%rbp	//当前指针加24
  400f3a:	eb db                	jmp    400f17 <phase_2+0x1b>
  400f3c:	48 83 c4 28          	add    $0x28,%rsp
  400f40:	5b                   	pop    %rbx
  400f41:	5d                   	pop    %rbp
  400f42:	c3                   	retq   
~~~

### Input

1 2 4 8 16 32

***

## phase_3

- 条件分支switch

~~~c
  400f43:	48 83 ec 18          	sub    $0x18,%rsp
  400f47:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx	//存储第二个数
  400f4c:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx	//存储第一个数
  400f51:	be cf 25 40 00       	mov    $0x4025cf,%esi	//存放输入数据（x/s 0x4025cf）
  400f56:	b8 00 00 00 00       	mov    $0x0,%eax
  400f5b:	e8 90 fc ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  400f60:	83 f8 01             	cmp    $0x1,%eax	//此时eax中存放scanf返回值，输入数据的														//个数
  400f63:	7f 05                	jg     400f6a <phase_3+0x27>	//大于1则跳转
  400f65:	e8 d0 04 00 00       	callq  40143a <explode_bomb>
  400f6a:	83 7c 24 08 07       	cmpl   $0x7,0x8(%rsp)	//判断num1是否大于7
  400f6f:	77 3c                	ja     400fad <phase_3+0x6a>	//大于则跳转执行																			//<explode_bomb>
  400f71:	8b 44 24 08          	mov    0x8(%rsp),%eax	//eax中存放第一个数
  400f75:	ff 24 c5 70 24 40 00 	jmpq   *0x402470(,%rax,8)	//switch判断(x/8xg 																		//0x402470查看case对应的内																//存)
  400f7c:	b8 cf 00 00 00       	mov    $0xcf,%eax	//case 0 
  400f81:	eb 3b                	jmp    400fbe <phase_3+0x7b>
  400f83:	b8 c3 02 00 00       	mov    $0x2c3,%eax	//case 2
  400f88:	eb 34                	jmp    400fbe <phase_3+0x7b>
  400f8a:	b8 00 01 00 00       	mov    $0x100,%eax	//case 3
  400f8f:	eb 2d                	jmp    400fbe <phase_3+0x7b>
  400f91:	b8 85 01 00 00       	mov    $0x185,%eax	//case 4
  400f96:	eb 26                	jmp    400fbe <phase_3+0x7b>
  400f98:	b8 ce 00 00 00       	mov    $0xce,%eax	//case 5
  400f9d:	eb 1f                	jmp    400fbe <phase_3+0x7b>
  400f9f:	b8 aa 02 00 00       	mov    $0x2aa,%eax	//case 6
  400fa4:	eb 18                	jmp    400fbe <phase_3+0x7b>
  400fa6:	b8 47 01 00 00       	mov    $0x147,%eax	//case 7
  400fab:	eb 11                	jmp    400fbe <phase_3+0x7b>
  400fad:	e8 88 04 00 00       	callq  40143a <explode_bomb>
  400fb2:	b8 00 00 00 00       	mov    $0x0,%eax
  400fb7:	eb 05                	jmp    400fbe <phase_3+0x7b>
  400fb9:	b8 37 01 00 00       	mov    $0x137,%eax	//case 1
  400fbe:	3b 44 24 0c          	cmp    0xc(%rsp),%eax	//判断num2与eax中的值是否相等
  400fc2:	74 05                	je     400fc9 <phase_3+0x86>
  400fc4:	e8 71 04 00 00       	callq  40143a <explode_bomb>
  400fc9:	48 83 c4 18          	add    $0x18,%rsp
  400fcd:	c3                   	retq   
~~~



x表明以十六进制的形式显示地址，g表示每8个字节的内存，因为这是x64平台，所以地址占8个字节

~~~shell
x/8xg 0x402470
~~~

### Input

**0 207	1 311	2 707	3 256	4 389	5 206	6 682	7 327**

***

## phase_4

- 函数递归func4

~~~C
0000000000400fce <func4>:
  400fce:	48 83 ec 08          	sub    $0x8,%rsp	
  400fd2:	89 d0                	mov    %edx,%eax	//eax = 14
  400fd4:	29 f0                	sub    %esi,%eax	//esi = 14 - 0 = 14
  400fd6:	89 c1                	mov    %eax,%ecx	//ecx = 14
  400fd8:	c1 e9 1f             	shr    $0x1f,%ecx	//逻辑右移31位ecx = 0
  400fdb:	01 c8                	add    %ecx,%eax	//eax = 14 + 0 = 14
  400fdd:	d1 f8                	sar    %eax		//算数右移1位 eax = 7
  400fdf:	8d 0c 30             	lea    (%rax,%rsi,1),%ecx	//%eax为%rax的第32位表示，											//%esi为%rsi的低32位表示，初始时%rax=%eax，%rsi=%esi
                                        //加载有效地址，ecx = eax + esi = 7 + 0 = 7
  400fe2:	39 f9                	cmp    %edi,%ecx  
  400fe4:	7e 0c                	jle    400ff2 <func4+0x24>	//有符号小于等于则跳转,						//若%ecx（初始为7）小于等于num1,则跳转至400ff2,%eax=0,说明num1=7,为其中一个解
  400fe6:	8d 51 ff             	lea    -0x1(%rcx),%edx	//edx = ecx - 1 = 6
  400fe9:	e8 e0 ff ff ff       	callq  400fce <func4>
  400fee:	01 c0                	add    %eax,%eax	//eax = eax * 2
  400ff0:	eb 15                	jmp    401007 <func4+0x39>	//递归退出
  400ff2:	b8 00 00 00 00       	mov    $0x0,%eax	//eax = 0
  400ff7:	39 f9                	cmp    %edi,%ecx	
  400ff9:	7d 0c                	jge    401007 <func4+0x39>	//ecx > num1递归出去
  400ffb:	8d 71 01             	lea    0x1(%rcx),%esi	//esi = ecx + 1
  400ffe:	e8 cb ff ff ff       	callq  400fce <func4>
  401003:	8d 44 00 01          	lea    0x1(%rax,%rax,1),%eax	//eax = eax * 2 + 1
  401007:	48 83 c4 08          	add    $0x8,%rsp
  40100b:	c3                   	retq   

000000000040100c <phase_4>:
  40100c:	48 83 ec 18          	sub    $0x18,%rsp
  401010:	48 8d 4c 24 0c       	lea    0xc(%rsp),%rcx	//num2
  401015:	48 8d 54 24 08       	lea    0x8(%rsp),%rdx	//num1
  40101a:	be cf 25 40 00       	mov    $0x4025cf,%esi
  40101f:	b8 00 00 00 00       	mov    $0x0,%eax	//eax = 0
  401024:	e8 c7 fb ff ff       	callq  400bf0 <__isoc99_sscanf@plt>
  401029:	83 f8 02             	cmp    $0x2,%eax	//判断输入的数字是否为2
  40102c:	75 07                	jne    401035 <phase_4+0x29>
  40102e:	83 7c 24 08 0e       	cmpl   $0xe,0x8(%rsp)	//num1 = 14
  401033:	76 05                	jbe    40103a <phase_4+0x2e>	//jdb：无符号小于等于跳						//转，当num1小于等于14时，跳转至40103a，否则爆炸，所以num1的限制条件为[0,14]
  401035:	e8 00 04 00 00       	callq  40143a <explode_bomb>
  40103a:	ba 0e 00 00 00       	mov    $0xe,%edx	//edx = 14
  40103f:	be 00 00 00 00       	mov    $0x0,%esi	//esi = 0
  401044:	8b 7c 24 08          	mov    0x8(%rsp),%edi	//edi = num1
  401048:	e8 81 ff ff ff       	callq  400fce <func4>
  40104d:	85 c0                	test   %eax,%eax	
  40104f:	75 07                	jne    401058 <phase_4+0x4c	//返回的eax不为0，跳转到爆炸
  401051:	83 7c 24 0c 00       	cmpl   $0x0,0xc(%rsp)	// num2 = 0
  401056:	74 05                	je     40105d <phase_4+0x51>
  401058:	e8 dd 03 00 00       	callq  40143a <explode_bomb>
  40105d:	48 83 c4 18          	add    $0x18,%rsp
  401061:	c3                   	retq   
~~~

### Input

7 0	0 0	1 0	3 0

***

## phase_5

- 类似字符串指针，ch[rax]

~~~C
0000000000401062 <phase_5>:
  401062:	53                   	push   %rbx	//压入输入字符串的地址
  401063:	48 83 ec 20          	sub    $0x20,%rsp
  401067:	48 89 fb             	mov    %rdi,%rbx
  40106a:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax	//金丝雀值
  401071:	00 00 
  401073:	48 89 44 24 18       	mov    %rax,0x18(%rsp)	//移到数据末尾
  401078:	31 c0                	xor    %eax,%eax	//初始化eax
  40107a:	e8 9c 02 00 00       	callq  40131b <string_length>
  40107f:	83 f8 06             	cmp    $0x6,%eax
  401082:	74 4e                	je     4010d2 <phase_5+0x70>
  401084:	e8 b1 03 00 00       	callq  40143a <explode_bomb>
  401089:	eb 47                	jmp    4010d2 <phase_5+0x70>
  40108b:	0f b6 0c 03          	movzbl (%rbx,%rax,1),%ecx	// ecx = (rbx + eax);ecx 																 //= num[i]
  40108f:	88 0c 24             	mov    %cl,(%rsp)	//(rsp) = ecx的低8位 = num[i]low 8 bit
  401092:	48 8b 14 24          	mov    (%rsp),%rdx	//rdx = ecx的低8位
  401096:	83 e2 0f             	and    $0xf,%edx	//取出低8位
  401099:	0f b6 92 b0 24 40 00 	movzbl 0x4024b0(%rdx),%edx	//edx从0x4024b0按(rdx)索引拷贝一个字节
  4010a0:	88 54 04 10          	mov    %dl,0x10(%rsp,%rax,1)	//栈中存放edx低8位
  4010a4:	48 83 c0 01          	add    $0x1,%rax
  4010a8:	48 83 f8 06          	cmp    $0x6,%rax	//重复取6次
  4010ac:	75 dd                	jne    40108b <phase_5+0x29>
  4010ae:	c6 44 24 16 00       	movb   $0x0,0x16(%rsp)
  4010b3:	be 5e 24 40 00       	mov    $0x40245e,%esi
  4010b8:	48 8d 7c 24 10       	lea    0x10(%rsp),%rdi	//rdi指向最后的字符
  4010bd:	e8 76 02 00 00       	callq  401338 <strings_not_equal>
  4010c2:	85 c0                	test   %eax,%eax
  4010c4:	74 13                	je     4010d9 <phase_5+0x77>
  4010c6:	e8 6f 03 00 00       	callq  40143a <explode_bomb>
  4010cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
  4010d0:	eb 07                	jmp    4010d9 <phase_5+0x77>
  4010d2:	b8 00 00 00 00       	mov    $0x0,%eax	//eax = 0
  4010d7:	eb b2                	jmp    40108b <phase_5+0x29>
  4010d9:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
  4010de:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
  4010e5:	00 00 
  4010e7:	74 05                	je     4010ee <phase_5+0x8c>
  4010e9:	e8 42 fa ff ff       	callq  400b30 <__stack_chk_fail@plt>
  4010ee:	48 83 c4 20          	add    $0x20,%rsp
  4010f2:	5b                   	pop    %rbx
  4010f3:	c3                   	retq   
~~~

- 由输入字符串ASCII码的后4位作为索引；从字符串地址`0x40245e`中得到`flyers`
- 输入位6位字符串

### Input

ionuvw；ionefg；9?>567 … （不唯一）

***

## phase_6

- 结构体链表

![](C:\Github_io\source\img\MyblogImg\src\bomb\phase_6_1.png)

- 输入的每个数字要求不大于6，且互不相同

![](C:\Github_io\source\img\MyblogImg\src\bomb\phase_6_2.png)

- 说明栈中存放的是结构体，后8个字节存放的是结构体指针指向的下一个位置。

~~~C
struct node{
    int value;
    int number;
    node* next;
}
~~~

- 对链表进行重构；使其由大到小排列

~~~C
  4011ab:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx          //将(%rsp+32)的链表节点地址复制到 %rbx
  4011b0:	48 8d 44 24 28       	lea    0x28(%rsp),%rax          //将 %rax 指向栈中下一个链表结点的地址(%rsp+40)
  4011b5:	48 8d 74 24 50       	lea    0x50(%rsp),%rsi          //将 %rsi 指向保存的链表节点地址的末尾(%rsp+80)
  4011ba:	48 89 d9             	mov    %rbx,%rcx
  4011bd:	48 8b 10             	mov    (%rax),%rdx
  4011c0:	48 89 51 08          	mov    %rdx,0x8(%rcx)           //将栈中指向的后一个节点的地址复制到前一个节点的next指针位置
  4011c4:	48 83 c0 08          	add    $0x8,%rax          //移动到下一个节点
  4011c8:	48 39 f0             	cmp    %rsi,%rax           //判断6个节点是否遍历完毕
  4011cb:	74 05                	je     4011d2 <phase_6+0xde>  
  4011cd:	48 89 d1             	mov    %rdx,%rcx          //继续遍历
  4011d0:	eb eb                	jmp    4011bd <phase_6+0xc9>
  4011d2:	48 c7 42 08 00 00 00 	movq   $0x0,0x8(%rdx)     //末尾链表next 为 NULL 则设置为0x0

  //该循环按照7减去输入数据的索引重新调整链表

  4011d9:	00 
  4011da:	bd 05 00 00 00       	mov    $0x5,%ebp
  4011df:	48 8b 43 08          	mov    0x8(%rbx),%rax                //将 %rax 指向 %rbx 下一个链表结点
  4011e3:	8b 00                	mov    (%rax),%eax
  4011e5:	39 03                	cmp    %eax,(%rbx)                    //比较链表结点中第一个字段值的大小,如果前一个节点值大于后一个节点值,跳转
  4011e7:	7d 05                	jge    4011ee <phase_6+0xfa>
  4011e9:	e8 4c 02 00 00       	callq  40143a <explode_bomb>
  4011ee:	48 8b 5b 08          	mov    0x8(%rbx),%rbx               //将 %rbx 向后移动,指向栈中下一个链表节点的地址
  4011f2:	83 ed 01             	sub    $0x1,%ebp                   
  4011f5:	75 e8                	jne    4011df <phase_6+0xeb>         //判断循环是否结束
  //该循环判断栈中重新调整后的链表结点是否按照降序排列

  4011f7:	48 83 c4 50          	add    $0x50,%rsp
  4011fb:	5b                   	pop    %rbx
  4011fc:	5d                   	pop    %rbp
  4011fd:	41 5c                	pop    %r12
  4011ff:	41 5d                	pop    %r13
  401201:	41 5e                	pop    %r13                 //释放空间
  401203:	c3                   	retq   
~~~



### Input

**4 3 2 1 6 5**

***

## secret_phase

- 二叉树结构<phase_defused>

- `phase_4`中输入为“%d, %d, %s”
- esi为系统的变量“%d, %d, %s”；edi为输入的数据；`x/s 0x603870`命令可以知道哪个破解可以进入隐藏关卡
- `%s`中值存放在0x402622

![](C:\Github_io\source\img\MyblogImg\src\bomb\phase_defused.png)

- 0, 0, DrEvil

- 存储的是二叉树结构体

![](C:\Github_io\source\img\MyblogImg\src\bomb\secret_phase.png)

- 需要得到返回值 %eax=2；
- ebx为输入的数据；edi为系统内给定的数据（二叉树结构体）

~~~C
struct tree{
    int val;
    struct tree* left;
    struct tree* right;
}
~~~



![](C:\Github_io\source\img\MyblogImg\src\bomb\secret_phase_2.png)

- 顺推思路：

  > 1. 首先来到二叉树的首地址0x6030f0对应的数据：36，因为36需要大于x，才能得到 %eax = %eax * 2，那么指针值应该为%rdi + 8（加载左结点），指针值为6304016，查看得到值为8
  > 2. 来到8对应的位置，我们想要数据%eax = %eax*2 + 1，则8需要小于等于x，那么指针值应该为0x603110 + 16（加载右结点），指针值为 6304080，查看得到的值为22
  > 3. 最后我们得到了数据22，当我们输入22的时候，因为和指针所处位置对应头部数据的值相等，所以%eax = 0
  > 4. 那么指针值应该为0x603150 + 8（加载左结点），指针值为6304368，查看得到值为20，该位置指针为空，不继续指向下一结点，所以20也为可行解。

~~~C
0000000000401204 <fun7>:
  401204:	48 83 ec 08          	sub    $0x8,%rsp
  401208:	48 85 ff             	test   %rdi,%rdi
  40120b:	74 2b                	je     401238 <fun7+0x34>
  40120d:	8b 17                	mov    (%rdi),%edx
  40120f:	39 f2                	cmp    %esi,%edx	//比较结构体的值和输入的值
  401211:	7e 0d                	jle    401220 <fun7+0x1c>
  401213:	48 8b 7f 08          	mov    0x8(%rdi),%rdi
  401217:	e8 e8 ff ff ff       	callq  401204 <fun7>
  40121c:	01 c0                	add    %eax,%eax
  40121e:	eb 1d                	jmp    40123d <fun7+0x39>
  401220:	b8 00 00 00 00       	mov    $0x0,%eax	//eax = 0
  401225:	39 f2                	cmp    %esi,%edx	//比较结构体和输入的值
  401227:	74 14                	je     40123d <fun7+0x39>
  401229:	48 8b 7f 10          	mov    0x10(%rdi),%rdi	//跳转到下一个结构体（左子树）
  40122d:	e8 d2 ff ff ff       	callq  401204 <fun7>
  401232:	8d 44 00 01          	lea    0x1(%rax,%rax,1),%eax	//跳转到右子树
  401236:	eb 05                	jmp    40123d <fun7+0x39>
  401238:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  40123d:	48 83 c4 08          	add    $0x8,%rsp
  401241:	c3                   	retq   
~~~

### Input

**22 或 20**