# 3.3 awk

- awk会根据空格和制表符，将每一行分成若干字段，依次用$1、$2、$3代表第一个字段、第二个字段、第三个字段等等
- print命令里面，如果原样输出字符，要放在双引号里面。

## 3.3.1 基本用法

```shell
awk action filename
awk '{print $0}' demo.txt
```

## 3.3.2 内置函数

变量NF表示当前行有多少个字段，因此$NF就代表最后一个字段\
变量NR表示当前处理的是第几行\
toupper()用于将字符转为大写\
tolower()：字符转为小写。\
length()：返回字符串长度。\
substr()：返回子字符串。\
sin()：正弦。\
cos()：余弦。\
sqrt()：平方根。\
rand()：随机数。

## 3.3.3 示例

```shell
# 输出奇数行
$ awk -F ':' 'NR % 2 == 1 {print $1}' demo.txt
root
bin
sync

# 输出第三行以后的行
$ awk -F ':' 'NR >3 {print $1}' demo.txt
sys
sync

$ awk -F ':' '$1 == "root" {print $1}' demo.txt
root

$ awk -F ':' '$1 == "root" || $1 == "bin" {print $1}' demo.txt
root
bin
```
