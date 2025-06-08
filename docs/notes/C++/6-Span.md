# 6 Span

```admonish danger
:skull: 这段代码会导致未定义的行为,因为基于范围的 for 循环中存在一个 bug,在对临时对象的引用上进行迭代时会使用已经销毁的值
```

```c++
// for the last 3 returned elements:
for (auto s : std::span{arrayOfConst()}.last(3)) 
// fatal runtime ERROR  
```

