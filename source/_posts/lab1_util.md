---
title: MIT6.S081-util
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

本文是对MIT6.S081中lab1 util的实现
<!-- more -->

# lab1-util

## primes

使用并行流进行素数的筛选

[参考资料](https://swtch.com/~rsc/thread/)

![](https://github.com/tom-jerr/MyblogImg/raw/main/src/Snipaste_2022-11-26_22-20-13.png)

算法伪代码：

~~~c
p = get a number from left neighbor
print p
loop:
    n = get a number from left neighbor
    if (p does not divide n)
        send n to right neighbor
~~~

**代码**

~~~C
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"

void process(int p[]){
    close(p[1]);
    int prime = 0;
    if (read(p[0], &prime, 4) > 0){
        fprintf(1,"prime %d\n", prime);
        int p2[2];
        if (pipe(p2) < 0) {
            printf("kernel pipe err!\n");
            exit(1);
        }
        int pid = fork();
        if (pid < 0){
            printf("fork err!\n");
            exit(1);
        }
        else if (pid == 0){
            // close(p[0]);
            process(p2);
        }
        else{
            close(p2[0]);
            int i;
            while(read(p[0], &i, 4) > 0) {
                if (i % prime != 0) {
                    write(p2[1], &i, 4);
                }
            }
            close(p2[1]);
            wait(0);
        }
    }
}


int main(int argc, char* argv[]){
    int p[2];
    if (pipe(p) < 0) {
        printf("kernel pipe err!\n");
        exit(1);
    }
    int pid = fork();
    if (pid < 0){
        printf("fork err!\n");
        exit(1);
    }
    else if (pid == 0){
        process(p);
    }
    else{
        close(p[0]);
        fprintf(1, "prime 2\n");
        for (int i = 3;i <= 35;i++){
            if (i % 2 != 0) write(p[1], &i, 4);
        }
        close(p[1]);
        wait(0);
    }

    exit(0);
}
~~~



***

## find

可以仿照`ls.c`中的代码，对文件、目录进行递归查找；利用指针和`strcmp`等函数对字符串进行比较，从而实现查找功能

`compare`：对路径和文件的名字进行匹配，匹配成功返回0；

`find`：查找函数，如果第一个参数为文件直接比较argv[1]和argv[2]；如果为目录，递归上述过程；

**注意：**`find`命令需要有三个参数！



**代码**

~~~C
#include "kernel/types.h"
#include "kernel/stat.h"
#include "user/user.h"
#include "kernel/fs.h"

int compare(char* path, char*file){
    char* p,*q;
    for (p = path + strlen(path); p >= path && *p != '/'; p--);
    p++;

    q = file;
    int flag = strcmp(p, q);
    return flag;
}

void find(char* path, char* file){
    char buf[512], *p;
    int fd;
    struct dirent de;
    struct stat st;
    if((fd = open(path, 0)) < 0){
        fprintf(2, "find: cannot open %s\n", path);
        return;
    }

    if(fstat(fd, &st) < 0){
        fprintf(2, "find: cannot stat %s\n", path);
        close(fd);
        return;
    }

    switch(st.type){
    case T_FILE:
        if (!compare(path, file)) printf("%s\n", path);
        break;

    case T_DIR:
        if(strlen(path) + 1 + DIRSIZ + 1 > sizeof buf){
        printf("find: path too long\n");
        break;
        }
        strcpy(buf, path);
        p = buf+strlen(buf);
        *p++ = '/';
        while(read(fd, &de, sizeof(de)) == sizeof(de)){
            if(de.inum == 0)
                continue;
            if(strcmp(de.name, ".") == 0 || strcmp(de.name, "..") == 0)
                continue;
            memmove(p, de.name, DIRSIZ);
            p[DIRSIZ] = 0;
            if(stat(buf, &st) < 0){
                printf("find: cannot stat %s\n", buf);
                continue;
            }
            find(buf, file);
        }
        break;
    default:
        break;
    }
    close(fd);
}

int main(int argc, char* argv[]){
    if (argc != 3){
        fprintf(2, "usage: find [path] [filename]\n");
        exit(1);
    }
    find(argv[1], argv[2]);
    exit(0);
}
~~~

***

## xargs

`xargs`从stdin标准输入中读取（read (0, p , 1)）；读到换行符重新读取新的命令（`buf`字符数组重置（p  = buf））；每读完一行就`fork`一个子进程去执行；主进程等待子进程执行后继续执行。

~~~C
#include "kernel/types.h"
#include "kernel/param.h"
#include "user/user.h"

int main(int argc, char* argv[]){
    char buf[32];
    int n;
    char* args[MAXARG];
    int numArg;
    for (int i = 1; i < argc; i++)
        args[i - 1] = argv[i];
    numArg = argc - 1;
    char* p = buf;
    //从标准输入中读取
    while ((n = read(0, p, 1)) > 0){
        if (*p == '\n'){
            *p = 0;
            int pid = fork();
            if (pid < 0){
                printf("fork err\n");
                exit(1);
            }
            else if (pid == 0){
                args[numArg] = buf;
                exec(args[0], args);
                exit(0);
            }
            else wait(0);
            //重新将指针移回到数组头部
            p = buf;
        }
        else p++;
    }
    exit(0);
}
~~~

