---

title: Bison Debug 指北
created: 2025-06-24
tags:

- Database

---

# What is bison?

- Flex (词法分析器)：负责将输入的原始文本流分解成一个个有意义的单元，称为**Tokens**。
- Bison (语法分析器)：接收 Flex 生成的令牌流，并根据你定义的语法规则来检查这些令牌的组合是否合法。检查词语（Tokens）是否组成了合法的句子（Statements/Expressions）。

## 使用指南

1. 编写 Bison 语法文件 (.y)：定义计算器语言的规则。
1. 编写 Flex 词法文件 (.l)：定义如何识别数字和操作符。
1. 编译和链接：使用 Bison 和 Flex 生成 C 代码，然后用 GCC/G++ 编译它们。
1. 运行程序。

# How to view the conflicts in bison?

如果是在 CMake 中可以通过设置 bison_flags 来进行查看：`set(bison_flags "-v -Wcex")`。\
如果我们定义的规则有冲突，bison 会给我们一个最小冲突例子，我们可以通过查看该例子来解决冲突：

```c++
  Example: L_BRACKET DOUBLE • R_BRACKET
  First reduce derivation
    container_expression
    ↳ list_expression
      ↳ L_BRACKET opt_expression_list       R_BRACKET
                  ↳ expression_list
                    ↳ expression_internal
                      ↳ constant_expression
                        ↳ DOUBLE •
  Second reduce derivation
    container_expression
    ↳ vector_expression
      ↳ L_BRACKET vector_item_list            R_BRACKET
                  ↳ vector_element_expression
                    ↳ DOUBLE •
```

# How to debug in bison?

1. 在 parser.yy 文件开头加入设置 bison debug 开启语句
   `%define parse.trace true`
1. 在 C++ 文件中找到 parser 对象，设置 debug level 为 1
   `parser_.set_debug_level(1);`
