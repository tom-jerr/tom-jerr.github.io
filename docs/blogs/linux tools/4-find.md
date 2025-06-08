# 3.4 find
- 从每个指定的起始点 (目录) 开始，搜索以该点为根的目录树，并按照运算符优先级规则从左至右评估给定的表达式，直到结果确定，此时find会继续处理下一个文件名。

## 3.4.1 语法
```shell
find [-H] [-L] [-P] [-D debugopts] [-Olevel] [起始点...] [表达式]
```
- -name pattern：按文件名查找，支持使用通配符 * 和 ?。
- -type type：按文件类型查找，可以是 f（普通文件）、d（目录）、l（符号链接）等。
  > f 普通文件  
l 符号连接  
d 目录  
c 字符设备  
b 块设备  
s 套接字  
p Fifo  
- -size [+-]size[cwbkMG]：按文件大小查找，支持使用 + 或 - 表示大于或小于指定大小，单位可以是 c（字节）、w（字数）、b（块数）、k（KB）、M（MB）或 G（GB）。
- -mtime days：按修改时间查找，支持使用 + 或 - 表示在指定天数前或后，days 是一个整数表示天数。
- -user username：按文件所有者查找。
- -group groupname：按文件所属组查找
- -depth: 让 find 以深度优先的方式遍历目录树，默认情况下 find 以广度优先方式处理目录树
## 3.4.2 示例
```shell
# 当前目录搜索所有文件，且文件内容包含 “140.206.111.111”
find . -type f -name "*" | xargs grep "140.206.111.111"
# 在/home目录下查找以.txt结尾的文件名，忽略大小写
find /home -iname "*.txt"
# 当前目录及子目录下查找所有以.txt和.pdf结尾的文件
find . -name "*.txt" -o -name "*.pdf"
# 基于正则表达式匹配文件路径
find . -regex ".*\(\.txt\|\.pdf\)$"
# 向下最大深度限制为3
find . -maxdepth 3 -type f
```
