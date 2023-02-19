---
title: DP算法
index_img: /img/algorithm.png
categories:
- 算法
tags:
- 算法
- DP问题
- python
comment: valine
math: true
---

# 前言

本文是选取了一些DP算法的典型问题如背包问题，以及一些变种DP问题和一些蓝桥杯题型进行总结
<!-- more -->

# DP问题


## 复杂DP
### 1. 整数划分问题
$dp[i][j]$：表示总和为 i ，被分成 j 个数的方案数量；
状态转移$f(i,j)$：
	1.方案中最小值为0：$f(i,j-1)$
	2.方案中最小值不为0：将原有方案可以做一个映射，让所有的方案都减去一个1	$f(i-j,j)$
	3.状态转移方程：$f(i,j) = f(i,j-1) + f(i-j,j)$

#### 代码（鸣人的影分身）
```python
import sys
read=lambda:[int(x) for x in sys.stdin.readline().split()]
t,=read()
N=11
f=[[0]*(N+1) for _ in range(N+1)]


while t:
    m,n=read()
    f[0][0]=1
    ##能量可以从0到m
    for i in range(m+1):
        for j in range(1,n+1):
            f[i][j]=f[i][j-1]
            if i>=j:f[i][j]+=f[i-j][j]
    
    print(f[m][n])
    t-=1
```
***
### 2. 糖果
$dp[i][j]$：表示选了 i 件物品，数量总和除以 k 余数为 j 的所有方案
状态转移：
	1.不包含物品i：$f(i-1,j)$
	2.包含物品i：$f(i-1,(j-w[i])\mod k)+w[i]$

**初始化时，数组应先置为负数，仅将$f[0][0]$置为0**

#### 代码
```python
import sys
read=lambda:[int(x) for x in sys.stdin.readline().split()]
N=110
f=[[-1e6]*(N+1) for _ in range(N+1)]

n,k=read()
f[0][0]=0
for i in range(1,n+1):
    w=int(input())
    for j in range(k):
        f[i][j]=max(f[i-1][j],f[i-1][(j-w%k+k)%k]+w)
print(f[n][0])
```
### 3. 密码脱落
**最少加上几个字母使得整个字符串变成一个回文串
等价于
最少删去几个字符使得整个字符串变成一个回文串**

最长回文字串必须连续，最长回文子序列可以不连续

结果就是：输入字符串长度-最长回文子序列长度

$dp[L][R]$：在$S[L,R]$之间回文子序列的集合
状态转移（可重复）：
	1.L、R在区间内
	2.L一定在，R一定不在：$f(L,R-1)包含此状态，但不是此状态
	3.L一定不在，R一定在：$f(L+1,R)$包含此状态，但不是此状态
	4.L、R一定不在：$f(L+1,R-1)$已经在上面的两种状态下被覆盖了，所以不需要再次重复计算

#### 代码
```python
import sys
read=lambda:[int(x) for x in sys.stdin.readline().split()]
s=list(input())
n = len(s)
f=[[0]*(n+1) for _ in range(n+1)]

##将长度作为变量之一length
##右边界 r 由左边界 l 确定
for length in range(1,n+1):
    l=0
    while l+length-1<n:
        r=l+length-1
        if length==1:
            f[l][r]=1
        else:
            if s[l]==s[r]:
                f[l][r]=f[l+1][r-1]+2
            if f[l][r - 1] > f[l][r]:
                f[l][r] = f[l][r - 1]
            if f[l + 1][r] > f[l][r]:
                f[l][r] = f[l + 1][r]
        l+=1
print(n-f[0][n-1])
```
***
### 4. 生命之树（树形DP）
就是树形的最长联通区域
一般使用递归子树

$f(u)$：在以 u 为根的子树中，包含 u 的所有连通块的权值最大值

用单链表模拟邻接表
```python
##手动设置递归深度
import sys
sys.setrecursionlimit(3000) # 这里设置为3000
```

![](C:\Users\我\Desktop\截图\树形DP.png)

#### 代码
dfs 是先计算底层的最大值，依次向上回溯，需要记录父亲节点，防止再次递归
这里的$f[i]$初值设置为0
```python
##dfs会爆栈
import sys
sys.setrecursionlimit(1000000)

n = int(input())
w = [0] + list(map(int,input().split()))
idx = 0
N = 100010

e = [0 for i in range(2*N)]
ne = [0 for i in range(2*N)]
h = [-1 for i in range(N)]

f = [0 for _ in range(N)]

def add_edge(x, y):
    global idx
    e[idx], ne[idx] = y, h[x]
    h[x] = idx
    idx += 1

def dfs(u, father):
    f[u] = w[u]
    i = h[u]
    while i != -1:
        j = e[i]
        if j != father:
            dfs(j, u)
            f[u] += max(0, f[j])
        i = ne[i]

for i in range(n-1):
    a, b = map(int,input().split())
    add_edge(a, b)
    add_edge(b, a)

dfs(1,-1)
res = f[1]
for i in range(1, n+1):
    res = max(res, f[i])
print(res)
```
#### 拓扑排序
先计算上层的最大值，依次向下层计算
这里的$f[i]$初值设置为树上的结点值
```python
import sys
read=lambda:[int(x) for x in sys.stdin.readline().split()]
n, = read()
w = [0]+read()
##入度字典{a:value}
deg={}
##每个节点的邻接表
h={i:[] for i in range(1,n+1)}
f = [i for i in w]
for i in range(n-1):
    a,b = read()
    deg[a]=deg.get(a,0)+1
    deg[b]=deg.get(b,0)+1
    h[b].append(a)
    h[a].append(b)
queue=[]
for i in deg:
    if deg[i]==1:
        queue.append((i))
visit=set()
while queue:
    u=queue.pop()
    visit.add(u)
    for nxt in h[u]:
        if nxt not in visit:
            deg[nxt]-=1
            f[nxt]+= max(0,f[u])
            if deg[nxt]==1:
                queue.append((nxt))
res = f[1]
for i in range(2,n+1):
    res = max(f[i],res)
print(res)
```
***
### 5. 斐波那契数列前n项和
#### 通用优化方式
对于线性DP方式，优化时间复杂度，可以使用矩阵乘法，一次计算一个向量$f[i]$
#### 解析
利用矩阵快速幂来求；核心是构造一个和为矩阵幂的形式

**向量斐波那契数列**
![](C:\Users\我\Desktop\截图\斐波那契数列向量.png)

**向量斐波那契数列和**
![](C:\Users\我\Desktop\截图\斐波那契数列和.png)

#### 代码
快速幂 + 矩阵乘法求斐波那契数列和（mod  m）
```python
import sys
read=lambda:[int(x) for x in sys.stdin.readline().split()]
n,m=read()

st = [1, 1, 1]
A = [
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 1]
]

def mul1(a,b):
    temp=[0]*3
    for i in range(3):
        for j in range(3):
            temp[i]=(temp[i]+a[j]*b[j][i])%m
    return temp

def mul2(m1,m2):
    temp=[[0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                temp[i][j]=(temp[i][j]+m1[i][k]*m2[k][j])%m
    return temp
    
n -= 1
while n:
    if n&1:
        st=mul1(st,A)
    A=mul2(A,A)
    n>>=1
print(st[2])
```
***
### 6. 包子凑数
#### 解析
1. 给定N个数，有无限个数不能被凑出来，说明这些数的最大公因数（gcd）不为1
2. 在N个数中，不能凑出来的数一定在10000以内，所以数组边界可以设在10010
3. $dp(i,j)$表示前选i项物品任意个，重量为 j **（j<=10000），属性为能否达到重量 j (true/false)**
4. 状态转移：
	$$
	dp(i,j)=dp(i−1,j)| dp(i−1,j−w(i))…|dp(i−1,j−k∗w(i)) \quad (k=j/w(i))
	$$

简化后方程为:

$$
dp(i,j)=dp(i−1,j)|dp(i,j−w(i))(w(i)≤j)
$$

5. 初始化：$dp[0][0]=true$

#### 代码
```python
##一维空间优化
import sys
read = lambda:[int(x) for x in sys.stdin.readline().split()]
n,=read()
N=10010

w=[0]+[int(input()) for _ in range(n)]
f = [False]*N 

def gcd(a,b):
    return a if b==0 else gcd(b,a%b)

d=0
for i in range(1,n+1):
   d = gcd(d,w[i])

if d!=1:
    print("INF")
else:
    f[0]=True
    for i in range(1,n+1):
        for j in range(w[i],N):
            f[j] |= f[j-w[i]]
    res=0
    for i in range(N):
        if not f[i]:
            res+=1
    print(res)
            

```








