---
title: The Senmantics of Data
date: 
update: 2024/5/4 19:49
comments: true
description: 深度探索C++对象chapter 3
katex: true
tags: 
- C++
categories: Knowledge

---



# The Senmantics of Data

## 类大小的计算

- 类大小的计算，遵循结构体的对齐原则；
- 类的大小，与普通数据成员有关，与成员函数和静态成员无关。即普通成员函数、静态成员函数、静态数据成员、静态常量数据成员，均对类的大小无影响；
- 虚函数对类的大小有影响，是因为**虚函数表指针**带来的影响；
- 虚继承对类的大小有影响，是因为**虚基表指针**带来的影响；
- 静态数据成员之所以不计算在类的对象大小内，是因为类的静态数据成员被该类所有的对象所共享，并不属于具体哪个对象，静态数据成员定义在内存的全局区；

### 空类大小

- C++的空类是指这个类不带任何数据，即类中没有非静态(non-static)数据成员变量，没有虚函数(virtual function)，也没有虚基类(virtual base class)。
- C++标准指出，不允许一个对象（当然包括类对象）的大小为0，不同的对象不能具有相同的地址。
  - new需要分配不同的内存地址，不能分配内存大小为0的空间；
  - 避免除以 sizeof(T)时得到除以0错误；
- 计算结果1。
  - 第一种情况，**空类的继承**：当派生类继承空类后，派生类如果有自己的数据成员，而空基类的一个字节并不会加到派生类中去。sizeof(D)为4。
  - 第二种情况，**一个类包含一个空类对象数据成员**：sizeof(HoldsAnInt)为8。在这种情况下，空类的1字节是会被计算进去的。而又由于字节对齐的原则，所以结果为4+4=8。
  - **继承空类的派生类**，如果派生类也为空类，大小也都为1。

### 含有虚函数的类

- **所以可以认为VTABLE是该类的所有对象共有的，在定义该类时被初始化；而VPTR则是每个类对象都有独立一份的，且在该类对象被构造时被初始化。**

### 基类含有虚函数的继承

1）虚函数按照其声明顺序放于表中。

2）基类的虚函数在派生类的虚函数前面。

1）覆盖的f()函数被放到了虚表中原来基类虚函数的位置；

2）没有被覆盖的函数依旧；

3）**派生类的大小仍是基类和派生类的非静态数据成员的大小+一个vptr指针的大小；**

~~~c++
#include<iostream>
using namespace std;
 
 
class A    
{    
};   
 
class B    
{ 
    char ch;    
    virtual void func0()  {  }  
};  
 
class C   
{ 
    char ch1; 
    char ch2; 
    virtual void func()  {  }   
    virtual void func1()  {  }  
}; 
 
class D: public A, public C 
{    
    int d;    
    virtual void func()  {  }  
    virtual void func1()  {  } 
};    
class E: public B, public C 
{    
    int e;    
    virtual void func0()  {  }  
    virtual void func1()  {  } 
}; 
 
int main(void) 
{ 
    cout<<"A="<<sizeof(A)<<endl;    //result=1 
    cout<<"B="<<sizeof(B)<<endl;    //result=16     
    cout<<"C="<<sizeof(C)<<endl;    //result=16 
    cout<<"D="<<sizeof(D)<<endl;    //result=16 
    cout<<"E="<<sizeof(E)<<endl;    //result=32 
    return 0; 
}
~~~

### 虚继承

- 在这里，只说一下在gcc编译器下，虚继承大小的计算。它在gcc下实现比较简单，不管是否虚继承，GCC都是将虚表指针在整个继承关系中共享的，不共享的是指向虚基类的指针。

~~~c++
class A {
 
    int a;　virtual void myfuncA(){}
};
 
class B:virtual public A{
 
    virtual void myfunB(){}
 
};
 
class C:virtual public A{
 
    virtual void myfunC(){}
 
};
 
class D:public B,public C{
 
    virtual void myfunD(){}
 
};
// sizeof(A)=16，sizeof(B)=24，sizeof(C)=24，sizeof(D)=32；
~~~

- B，C中由于是虚继承，**因此大小为A中成员大小加指向虚基类的指针的大小**。

- B,C虽然加入了自己的虚函数，但是虚表指针是和基类共享的，因此不会有自己的虚表指针，**他们两个共用虚基类A的虚表指针**。

- D的大小使用GCC13.2布局为：

  ~~~c++
  Class D
     size=32 align=8
     base size=16 base align=8
  D (0x0x7f3b580201c0) 0
      vptridx=0 vptr=((& D::_ZTV1D) + 24)
  B (0x0x7f3b58066d68) 0 nearly-empty
        primary-for D (0x0x7f3b580201c0)
        subvttidx=8
  A (0x0x7f3b580406c0) 16 virtual
          vptridx=40 vbaseoffset=-24 vptr=((& D::_ZTV1D) + 96)
  C (0x0x7f3b58066dd0) 8 nearly-empty
        subvttidx=24 vptridx=48 vptr=((& D::_ZTV1D) + 64)
  A (0x0x7f3b580406c0) alternative-path
  ~~~

  - clang14布局：

  ~~~c++
  *** Dumping AST Record Layout
           0 | class D
           0 |   class B (primary base)
           0 |     (B vtable pointer)
           8 |   class C (base)
           8 |     (C vtable pointer)
          16 |   class A (virtual base)
          16 |     (A vtable pointer)
          24 |     int a
             | [sizeof=32, dsize=28, align=8,
             |  nvsize=16, nvalign=8]
  ~~~

### 在VC中数据成员的布局顺序为：

1. vptr部分（如果基类有，则继承基类的）
2. vbptr （如果需要）
3. 基类成员（按声明顺序）
4. 自身数据成员
5. 虚基类数据成员（按声明顺序）

### 虚拟成员函数

如果function()是一个虚拟函数，那么用指针或引用进行的调用将发生一点特别的转换——一个中间层被引入进来。例如：

```c
// p->function()   将转化为
(*p->vptr[1])(p);
```

- 其中vptr为指向虚函数表的指针，它由编译器产生。vptr也要进行名字处理，因为一个继承体系可能有多个vptr。 

-  1是虚函数在虚函数表中的索引，通过它关联到虚函数function(). 

何时发生这种转换？**答案是在必需的时候**——一个再熟悉不过的答案。当通过指针调用的时候，要调用的函数实体无法在编译期决定，必需待到执行期才能获得，所以上面引入一个间接层的转换必不可少。但是当我们通过对象（不是引用，也不是指针）来调用的时候，进行上面的转换就显得多余了，**因为在编译器要调用的函数实体已经被决定。此时调用发生的转换，与一个非静态成员函数(Nonstatic Member Functions)调用发生的转换一致。**

### 静态成员函数

- 静态成员函数的一些特性：

1. **不能够直接存取其类中的非静态成员（nostatic members）**，包括不能调用非静态成员函数(Nonstatic Member Functions)。
2. 不能够声明为 const、voliatile或virtual。
3. 它不需经由对象调用，当然，通过对象调用也被允许。

- 除了缺乏一个this指针他与非静态成员函数没有太大的差别。在这里通过对象调用和通过指针或引用调用，将被转化为同样的调用代码。

- 需要注意的是通过一个表达式或函数对静态成员函数进行调用，被C++ Standard要求对表达式进行求值。如：(a+=b).static_fuc();
- 虽然省去对a+b求值对于static_fuc()的调用并没有影响，但是程序员肯定会认为表达式a+=b已经执行，一旦编译器为了效率省去了这一步，很难说会浪费多少程序员多少时间。这无疑是一个明智的规定。