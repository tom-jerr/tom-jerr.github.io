---

title: C++-Makefile语法学习
tags:

- C++

---

# Makefile

## 使用条件判断

```makefile
libs_for_gcc = -lgnu
normal_libs =
foo: $(objects)
ifeq ($(CC),gcc)
	$(CC) -o foo $(objects) $(libs_for_gcc)
else
	$(CC) -o foo $(objects) $(normal_libs)
endif
```

## 使用函数

### 字符串处理函数

```makefile
$(subst <from>,<to>,<text>)
• 名称：字符串替换函数
• 功能：把字串 <text> 中的 <from> 字符串替换成 <to> 。
• 返回：函数返回被替换过后的字符串。

$(patsubst <pattern>,<replacement>,<text>)
• 名称：模式字符串替换函数。
• 功能：查找 <text> 中的单词（单词以“空格”、“Tab”或“回车”“换行”分隔）是否符合模式
<pattern> ，如果匹配的话，则以 <replacement> 替换。这里，<pattern> 可以包括通配符 % ，
表示任意长度的字串。如果 <replacement> 中也包含 % ，那么，<replacement> 中的这个 % 将是
<pattern> 中的那个 % 所代表的字串。（可以用 \ 来转义，以 \% 来表示真实含义的 % 字符）
• 返回：函数返回被替换过后的字符串。
$(objects:.o=.c) 和 $(patsubst %.o,%.c,$(objects)) 是一样的

$(strip <string>)
• 名称：去空格函数。
• 功能：去掉 <string> 字串中开头和结尾的空字符。
• 返回：返回被去掉空格的字符串值。

$(findstring <find>,<in>)
• 名称：查找字符串函数
• 功能：在字串 <in> 中查找 <find> 字串。
• 返回：如果找到，那么返回 <find> ，否则返回空字符串。

$(filter <pattern...>,<text>)
• 名称：过滤函数
• 功能：以 <pattern> 模式过滤 <text> 字符串中的单词，保留符合模式 <pattern> 的单词。可以
有多个模式。
• 返回：返回符合模式 <pattern> 的字串。

$(filter-out <pattern...>,<text>)
• 名称：反过滤函数
• 功能：以 <pattern> 模式过滤 <text> 字符串中的单词，去除符合模式 <pattern> 的单词。可以
有多个模式。
• 返回：返回不符合模式 <pattern> 的字串。

$(sort <list>)
• 名称：排序函数
• 功能：给字符串 <list> 中的单词排序（升序）。
• 返回：返回排序后的字符串。
• 示例：$(sort foo bar lose) 返回 bar foo lose 。
• 备注：sort 函数会去掉 <list> 中相同的单词。

$(word <n>,<text>)
• 名称：取单词函数
• 功能：取字符串 <text> 中第 <n> 个单词。（从一开始）
• 返回：返回字符串 <text> 中第 <n> 个单词。如果 <n> 比 <text> 中的单词数要大，那么返回空字
符串。

$(wordlist <ss>,<e>,<text>)
• 名称：取单词串函数
• 功能：从字符串 <text> 中取从 <ss> 开始到 <e> 的单词串。<ss> 和 <e> 是一个数字。
• 返回：返回字符串 <text> 中从 <ss> 到 <e> 的单词字串。如果 <ss> 比 <text> 中的单词数要大，
那么返回空字符串。如果 <e> 大于 <text> 的单词数，那么返回从 <ss> 开始，到 <text> 结束的
单词串。

$(words <text>)
• 名称：单词个数统计函数
• 功能：统计 <text> 中字符串中的单词个数。
• 返回：返回 <text> 中的单词数。

$(firstword <text>)
• 名称：首单词函数——firstword。
• 功能：取字符串 <text> 中的第一个单词。
• 返回：返回字符串 <text> 的第一个单词。
```

### 文件名操作函数

```makefile
$(dir <names...>)
• 名称：取目录函数——dir。
• 功能：从文件名序列 <names> 中取出目录部分。目录部分是指最后一个反斜杠（/ ）之前的部分。
如果没有反斜杠，那么返回 ./ 。
• 返回：返回文件名序列 <names> 的目录部分。

$(notdir <names...>)
• 名称：取文件函数——notdir。
• 功能：从文件名序列 <names> 中取出非目录部分。非目录部分是指最後一个反斜杠（/ ）之后的部
分。
• 返回：返回文件名序列 <names> 的非目录部分。

$(basename <names...>)
• 名称：取前缀函数——basename。
• 功能：从文件名序列 <names> 中取出各个文件名的前缀部分。
• 返回：返回文件名序列 <names> 的前缀序列，如果文件没有前缀，则返回空字串。

$(addsuffix <suffix>,<names...>)
• 名称：加后缀函数——addsuffix。
• 功能：把后缀 <suffix> 加到 <names> 中的每个单词后面。
• 返回：返回加过后缀的文件名序列。

$(addprefix <prefix>,<names...>)
• 名称：加前缀函数——addprefix。
• 功能：把前缀 <prefix> 加到 <names> 中的每个单词后面。
• 返回：返回加过前缀的文件名序列。

$(join <list1>,<list2>)
• 名称：连接函数——join。
• 功能：把 <list2> 中的单词对应地加到 <list1> 的单词后面。如果 <list1> 的单词个数要比
<list2> 的多，那么，<list1> 中的多出来的单词将保持原样。如果 <list2> 的单词个数要比
<list1> 多，那么，<list2> 多出来的单词将被复制到 <list1> 中。
• 返回：返回连接过后的字符串。


```

### vpath

- vpath %.h include //指定.h类型文件的搜索路径是include

  vpath %.cpp src //指定.cpp类型文件的搜索路径是src

### foreach函数

```makefile
names := a b c d
files := $(foreach n,$(names),$(n).o)
$(files) 的值是 a.o b.o c.o d.o
```

### call函数

```makefile
reverse = $(1) $(2)
foo = $(call reverse,a,b)
```

### origin函数

- origin 函数不像其它的函数，他并不操作变量的值，他只是告诉你你的这个变量是哪里来的？其语法是：`$(origin <variable>)`

### shell函数

```makefile
contents := $(shell cat foo)
files := $(shell echo *.c)
```

### 控制make的函数

```makefile
$(error <text...>)

$(warning <text...>)
```

## 伪目标

```makefile
.PHONY: all
all: prog1 prog2 prog3
```

## 隐含规则

- 隐含规则就会生效。默认的后缀列表是：.out,.a, .ln, .o, .c, .cc, .C, .p, .f, .F, .r, .y, .l, .s, .S, .mod, .sym, .def, .h, .info, .dvi, .tex, .texinfo, .texi, .txinfo, .w, .ch .web, .sh, .elc, .el

### 常用隐含规则

- 编译C程序的隐含规则。
  “<n>.o”的目标的依赖目标会自动推导为“<n>.c”，并且其生成命令是`“$(CC) –c $(CPPFLAGS) $(CFLAGS)”`
- 编译C++程序的隐含规则。
  “<n>.o” 的目标的依赖目标会自动推导为“<n>.cc”或是“<n>.C”，并且其生成命令是`“$(CXX) –c $(CPPFLAGS) $(CFLAGS)”`。（建议使用“.cc”作为C++源文件的后缀，而不是“.C”）
- 汇编和汇编预处理的隐含规则。
  “<n>.o” 的目标的依赖目标会自动推导为“<n>.s”，默认使用编译品“as”，并且其生成命令是：`“$(AS) $(ASFLAGS)”`。“<n>.s” 的目标的依赖目标会自动推导为“<n>.S”，默认使用C预编译器“cpp”，并且其生成命令是：`“$(AS) $(ASFLAGS)”`。
- “<n>” 目标依赖于“<n>.o”，通过运行C的编译器来运行链接程序生成（一般是“ld”），其生成命令是：`“$(CC) $(LDFLAGS) <n>.o $(LOADLIBES) $(LDLIBS)”`。

### 自动化变量

```makefile
• $@ : 表示规则中的目标文件集。在模式规则中，如果有多个目标，那么，$@ 就是匹配于目标中模
式定义的集合。

• $% : 仅当目标是函数库文件中，表示规则中的目标成员名。例如，如果一个目标是 foo.a(bar.o)
，那么，$% 就是 bar.o ，$@ 就是 foo.a 。如果目标不是函数库文件（Unix 下是 .a ，Windows
下是 .lib ），那么，其值为空。

• $< : 依赖目标中的第一个目标名字。如果依赖目标是以模式（即 % ）定义的，那么 $< 将是符合模
式的一系列的文件集。注意，其是一个一个取出来的。

• $? : 所有比目标新的依赖目标的集合。以空格分隔。

• $^ : 所有的依赖目标的集合。以空格分隔。如果在依赖目标中有多个重复的，那么这个变量会去除
重复的依赖目标，只保留一份。

• $+ : 这个变量很像 $^ ，也是所有依赖目标的集合。只是它不去除重复的依赖目标。

• $* : 这个变量表示目标模式中 % 及其之前的部分。如果目标是 dir/a.foo.b ，并且目标的模式是
a.%.b ，那么，$* 的值就是 dir/a.foo 。这个变量对于构造有关联的文件名是比较有较。如果目
标中没有模式的定义，那么 $* 也就不能被推导出，但是，如果目标文件的后缀是 make 所识别的，
那么 $* 就是除了后缀的那一部分。例如：如果目标是 foo.c ，因为 .c 是 make 所能识别的后缀
名，所以，$* 的值就是 foo 。这个特性是 GNU make 的，很有可能不兼容于其它版本的 make，所
以，你应该尽量避免使用 $* ，除非是在隐含规则或是静态模式中。如果目标中的后缀是 make 所
不能识别的，那么 $* 就是空值。

```

## -nostdlib

- 不连接系统标准启动文件和标准库文件，只把指定的文件传递给连接器。这个选项常用于编译内核、bootloader等程序，它们不需要启动文件、标准库文件。

## -fno-builtin

- -fno-builtin这个选项。它的含义即不使用C语言的内建函数

- -fno-builtin-function（其中function为要冲突的函数名）

## ld -e

- 设置入口函数
- \_entry

## 例子

```makefile
CC=gcc
CFLAGS=-Wall -Wformat=0
ODIR=obj
IDIR=../include
LDIR=../lib
PROGRAM=client server

_DEPS = error_functions.o get_num.o inet_sockets.o become_daemon.o
DEPS = $(patsubst %, $(ODIR)/%,$(_DEPS))

all: $(PROGRAM)

# compile
$(ODIR)/%.o: $(LDIR)/%.c
    $(CC) -c -o $@ $<

$(ODIR)/%.o: %.c
    $(CC) -c -o $@ $<

# link
$(PROGRAM): $(patsubst %, $(ODIR)/%,$(addsuffix .o,$(PROGRAM))) $(DEPS)
    $(CC) -o $@ $(ODIR)/$@.o $(DEPS) 

.PHONY: clean
clean: 
    rm $(ODIR)/*.o $(PROGRAM)
```

```makefile
 CC = gcc
 TARGET = prog
 SOURCE = $(wildcard ./src/*.c)      #获取src目录下所有.c文件
 OBJS = $(patsubst %.c, %.o, $(SOURCE))
 INCLUDE = -I./include
 $(TARGET):$(OBJS)  
     $(CC) $(OBJS) -o $(TARGET)     #另一种写法： $(CC) -o $(TARGET) $(OBJS)
 %.o:%.c
     $(CC) $(INCLUDE) -c $^ -o $@   #另一种写法：  $(CC) $(INCLUDE) -c -o $@ $^                                                                       
 .PHONY:clean
 clean:
     rm $(OBJS) $(TARGET) 
```
