# 2 Algorithm
## 2.1 std::transform
- std::transform applies the given function to the elements of the given input range(s), and stores the result in an output range starting from d_first.
## parameters
``` admonish info
first1, last1: the pair of iterators defining the source range of elements to transform ;  
first2: the beginning of the second range of elements to transform, (3,4) only;  
d_first: the beginning of the destination range, may be equal to first1 or first2;  
policy: the execution policy to use;  
unary_op: Ret fun(const Type &a);  
binary_op: Ret fun(const Type1 &a, const Type2 &b);  
```

## possible implementation

``` c++
template<class InputIt, class OutputIt, class UnaryOp>
constexpr //< since C++20
OutputIt transform(InputIt first1, InputIt last1,
                   OutputIt d_first, UnaryOp unary_op)
{
    for (; first1 != last1; ++d_first, ++first1)
        *d_first = unary_op(*first1);
 
    return d_first;
}

template<class InputIt1, class InputIt2, 
         class OutputIt, class BinaryOp>
constexpr //< since C++20
OutputIt transform(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                   OutputIt d_first, BinaryOp binary_op)
{
    for (; first1 != last1; ++d_first, ++first1, ++first2)
        *d_first = binary_op(*first1, *first2);
 
    return d_first;
}
```
## example

```c++
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
 
void print_ordinals(const std::vector<unsigned>& ordinals)
{
    std::cout << "ordinals: ";
    for (unsigned ord : ordinals)
        std::cout << std::setw(3) << ord << ' ';
    std::cout << '\n';
}
 
char to_uppercase(unsigned char c)
{
    return std::toupper(c);
}
 
void to_uppercase_inplace(char& c)
{
    c = to_uppercase(c);
}
 
void unary_transform_example(std::string& hello, std::string world)
{
    // Transform string to uppercase in-place
 
    std::transform(hello.cbegin(), hello.cend(), hello.begin(), to_uppercase);
    std::cout << "hello = " << std::quoted(hello) << '\n';
 
    // for_each version (see Notes above)
    std::for_each(world.begin(), world.end(), to_uppercase_inplace);
    std::cout << "world = " << std::quoted(world) << '\n';
}
 
void binary_transform_example(std::vector<unsigned> ordinals)
{
    // Transform numbers to doubled values
 
    print_ordinals(ordinals);
 
    std::transform(ordinals.cbegin(), ordinals.cend(), ordinals.cbegin(),
                   ordinals.begin(), std::plus<>{});
 
    print_ordinals(ordinals);
}
 
int main()
{
    std::string hello("hello");
    unary_transform_example(hello, "world");
 
    std::vector<unsigned> ordinals;
    std::copy(hello.cbegin(), hello.cend(), std::back_inserter(ordinals));
    binary_transform_example(std::move(ordinals));
}
// OUTPUT
// hello = "HELLO"
// world = "WORLD"
// ordinals:  72  69  76  76  79 
// ordinals: 144 138 152 152 158
```
## 2.2 std::accumulate
```admonish info
Computes the sum of the given value init and the elements in the range [first, last).
```
## parameters

- first, last	-	the pair of iterators defining the range of elements to accumulate
- init	-	initial value of the accumulate
- op	-	Ret fun(const Type1 &a, const Type2 &b);
  ``` admonish example
  如标准库中的std::plus<>
  ```


## possible implementation

```c++

accumulate (1)
template<class InputIt, class T>
constexpr // since C++20
T accumulate(InputIt first, InputIt last, T init)
{
    for (; first != last; ++first)
        init = std::move(init) + *first; // std::move since C++20
 
    return init;
}

template<class InputIt, class T, class BinaryOperation>
constexpr // since C++20
T accumulate(InputIt first, InputIt last, T init, BinaryOperation op)
{
    for (; first != last; ++first)
        init = op(std::move(init), *first); // std::move since C++20
 
    return init;
}
```
## 2.3 std::optinal
- std::optional最高效的写法是触发RVO的写法，即：

    ```c++
    optional<A> optional_best(int n) {
        optional<A> temp(someFn(n));
        return temp;
    }
    
    ```
## 2.4 std::move
- 将 `[first, last)` 范围内的元素移动到以 d_first 开始的目标范围
```c++
template <typename InputIt, typename OutputIt>
OutputIt std::move(InputIt first, InputIt last, OutputIt d_first);
```
- 返回一个迭代器，指向目标范围移动后的结束位置（即 d_first + (last - first)）。
  
|参数|	类型|	描述|
|---|---|---|
|first|	InputIt|	源范围的起始迭代器（指向要移动的第一个元素）|
|last|	InputIt|	源范围的结束迭代器（指向最后一个元素的下一个位置）|
|d_first|	OutputIt|	目标范围的起始迭代器（指向移动后第一个元素位置）|

## 2.5 std::move_backward
- 目标位置：元素会被移动到以 d_last 为结束的目标范围，即目标范围是 `[d_last - N, d_last)`，其中 N = last - first。
```c++
template <class BidirIt1, class BidirIt2>
BidirIt2 std::move_backward(BidirIt1 first, BidirIt1 last, BidirIt2 d_last);
```
- 返回一个迭代器，指向目标范围移动后的起始位置（即 d_last - (last - first)）

|参数|	类型|	描述|
|---|---|---|
|first|	BidirIt1|	源范围的起始迭代器（指向要移动的第一个元素）|
|last|	BidirIt1|	源范围的结束迭代器（指向最后一个元素的下一个位置）|
|d_last|	BidirIt2|	目标范围的结束迭代器（指向移动后最后一个元素的下一个位置）|

## 2.6 std::lower_bound 和 std::upper_bound
- 实际上是二分查找的实现
  
``` admonish info
std::lower_bound: 返回第一个不小于给定值的元素位置。
返回值: 指向第一个满足 *it >= value 的元素的迭代器。若没有这样的元素，则返回 end()
```

``` admonish info
std::upper_bound: 返回第一个大于给定值的元素位置。
返回值: 指向第一个满足 *it > value 的元素的迭代器。若没有这样的元素，则返回 end()。
```

## 2.7 std::distance

```admonish info
std::distance 返回从第一个迭代器到第二个迭代器之间的元素数量。对于随机访问迭代器（如 std::vector、std::deque、std::array 的迭代器），它的时间复杂度是 O(1)；对于其他类型的迭代器（如 std::list、std::forward_list 的迭代器），时间复杂度是 O(n)，因为它需要逐个遍历元素。
```

```c++
template<class InputIterator>
typename std::iterator_traits<InputIterator>::difference_type
    std::distance(InputIterator first, InputIterator last);
```

## 2.8 std::all_of, std::any_of, std::none_of

```admonish
std::all_of: 检查所有元素是否都满足条件（Predicate 返回 true）。  
std::any_of: 检查是否至少有一个元素满足条件。  
std::non_of: 检查是否全部不满足条件
```
- 参数说明
|参数|	说明|
|---|---|
|first|	起始迭代器，指向待检查范围的第一个元素。|
|last|	终止迭代器，指向待检查范围的末尾（最后一个元素的下一个位置）。|
|p|	谓词（Predicate），接受一个元素类型的参数，返回 bool 类型的条件结果。|

## 2.9 std::count, std::count_if
- std::count
  ```admonish
  统计值为 value 的元素个数
  ```

  ```c++
  template<class InputIt, class T>
  typename iterator_traits<InputIt>::difference_type
        count(InputIt first, InputIt last, const T& value);
  ```
- std::count_if
  ```admonish 
  统计符合条件的元素个数
  ```

  ```c++
  template<class InputIt, class Predicate>
  typename iterator_traits<InputIt>::difference_type
    count_if(InputIt first, InputIt last, Predicate p);
  ```
## 2.10 std::find, std::find_if
- std::find: 返回第一个等于 value 的迭代器
    ```admonish
    自定义对象使用std::find，需重载 operator==
    ```
    
    ```c++
    template<class InputIt, class T>
    InputIt find(InputIt first, InputIt last, const T& value);
    ```


- std::find_if: 在范围内查找第一个满足谓词条件的元素。
  ```c++
  template<class InputIt, class Predicate>
  InputIt find_if(InputIt first, InputIt last, Predicate p);
  ```
## 2.11 std::copy, std::copy_if
- std::copy: 完全复制所有元素
    ```c++
    template<class InputIt, class OutputIt, class Predicate>
    OutputIt copy(InputIt first, InputIt last, OutputIt d_first);
    ```

- std::copy_if: 选择性复制元素
    ```c++
    template<class InputIt, class OutputIt, class Predicate>
    OutputIt copy_if(InputIt first, InputIt last, OutputIt d_first, Predicate pred);
    ```
## 2.12 std::fill, std::generate
- std::fill: 用固定值填充范围
  ```c++
  template<class ForwardIt, class T>
  void fill(ForwardIt first, ForwardIt last, const T& value);
  ```
- std::generate: 用生成器函数填充范围
  ```c++
  template<class ForwardIt, class Generator>
  void generate(ForwardIt first, ForwardIt last, Generator gen);
  ```

```admonish example
可以使用随机数生成器来进行填充
```

```c++
#include <vector>
#include <algorithm>
#include <random>

std::vector<int> v(5);
int counter = 0;
std::generate(v.begin(), v.end(), [&counter]() {
    return counter++; // 生成 0, 1, 2, 3, 4
});

// 生成随机数
std::random_device rd;
std::mt19937 rng(rd());
std::uniform_int_distribution<int> dist(1, 10);
std::generate(v.begin(), v.end(), [&]() { return dist(rng); });
```
## 2.13 std::search, std::mismatch
- std::search: 在序列中搜索子序列。
  ```c++
    template<class ForwardIt1, class ForwardIt2>
    ForwardIt1 search(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2);

    // 示例：在 v1 中查找子序列 v2
    std::vector<int> v1 = {1, 2, 3, 4, 5, 6};
    std::vector<int> v2 = {3, 4};
    auto it = std::search(v1.begin(), v1.end(), v2.begin(), v2.end()); // 返回指向3的迭代器
  ```

- std::mismatch: 在比较两个序列，返回第一个不匹配的位置。
  ```c++
    template<class InputIt1, class InputIt2>
    std::pair<InputIt1, InputIt2> mismatch(InputIt1 first1, InputIt1 last1, InputIt2 first2);

    // 示例：比较两个字符串
    std::string s1 = "hello", s2 = "hxllo";
    auto [it1, it2] = std::mismatch(s1.begin(), s1.end(), s2.begin()); 
    // it1指向s1的'e', it2指向s2的'x'
  ```
## 2.14 std::replace, std::replace_if
- std::replace: 替换所有等于 old_value 的元素。
```c++
template<class ForwardIt, class T>
void replace(ForwardIt first, ForwardIt last, const T& old_value, const T& new_value);

// 示例：替换所有3为5
std::vector<int> v = {1, 2, 3, 3, 4};
std::replace(v.begin(), v.end(), 3, 5); // v变为{1, 2, 5, 5, 4}
```
- std::replace_if: 替换满足谓词的元素
```c++
template<class ForwardIt, class Predicate, class T>
void replace_if(ForwardIt first, ForwardIt last, Predicate pred, const T& new_value);

// 示例：替换所有偶数为0
std::replace_if(v.begin(), v.end(), [](int x) { return x % 2 == 0; }, 0);
```

## 2.15 std::remove / std::remove_if
```admonish
移动满足条件的元素到末尾，返回新逻辑结尾。
```

```c++
template<class ForwardIt, class T>
ForwardIt remove(ForwardIt first, ForwardIt last, const T& value);

template<class ForwardIt, class Predicate>
ForwardIt remove_if(ForwardIt first, ForwardIt last, Predicate pred);

// 示例：删除所有3（需结合erase）
std::vector<int> v = {1, 2, 3, 4, 3};
auto new_end = std::remove(v.begin(), v.end(), 3);
v.erase(new_end, v.end()); // v变为{1, 2, 4}
```
## 2.16 std::reverse
- 反转整个序列

```c++
template<class BidirIt>
void reverse(BidirIt first, BidirIt last);

// 示例：反转vector
std::reverse(v.begin(), v.end()); // {1, 2, 3} → {3, 2, 1}
```

## 2.17 std::rotate
- 把middle旋转到开头

```c++
template<class ForwardIt>
ForwardIt rotate(ForwardIt first, ForwardIt middle, ForwardIt last);

// 示例：将中间元素旋转到开头
std::vector<int> v = {1, 2, 3, 4, 5};
std::rotate(v.begin(), v.begin() + 2, v.end()); // v变为{3, 4, 5, 1, 2}
```
## 2.18 std::shuffle
- 使用一个随机生成器，打乱序列

```c++
template<class RandomIt, class RandomGen>
void shuffle(RandomIt first, RandomIt last, RandomGen&& g);

// 示例：使用随机引擎
#include <random>
std::vector<int> v = {1, 2, 3, 4};
std::random_device rd;
std::mt19937 rng(rd());
std::shuffle(v.begin(), v.end(), rng); // 随机排列
```

## 2.19 std::unique
```admonish
删除相邻重复元素（需先排序）。
```
```c++
template<class ForwardIt>
ForwardIt unique(ForwardIt first, ForwardIt last);

// 示例：删除相邻重复项
std::vector<int> v = {1, 1, 2, 3, 3};
auto last = std::unique(v.begin(), v.end());
v.erase(last, v.end()); // v变为{1, 2, 3}
```

## 2.20 std::sort, std::stable_sort, std::partial_sort
- std::partial_sort是部分排序
```c++
template<class RandomIt>
void sort(RandomIt first, RandomIt last);

template<class RandomIt>
void stable_sort(RandomIt first, RandomIt last);

// 示例：降序排序
std::sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

template<class RandomIt>
void partial_sort(RandomIt first, RandomIt middle, RandomIt last);

// 示例：找到前3小的元素
std::vector<int> v = {5, 3, 1, 4, 2};
std::partial_sort(v.begin(), v.begin() + 3, v.end()); // 前3个元素为{1, 2, 3}
```
## 2.21 std::nth_element
- 将第n个元素排序到正确的位置上

```c++
template<class RandomIt>
void nth_element(RandomIt first, RandomIt nth, RandomIt last);

// 示例：找到第3小的元素
std::nth_element(v.begin(), v.begin() + 2, v.end()); 
// v = 3，其他元素相对无序
```

## 2.22 std::merge
- 合并两个**有序序列**
```c++
template<class InputIt1, class InputIt2, class OutputIt>
OutputIt merge(InputIt1 first1, InputIt1 last1, InputIt2 first2, InputIt2 last2, OutputIt d_first);

// 示例：合并两个有序vector
std::vector<int> v1 = {1, 3, 5}, v2 = {2, 4, 6}, result;
std::merge(v1.begin(), v1.end(), v2.begin(), v2.end(), std::back_inserter(result)); 
// result = {1, 2, 3, 4, 5, 6}
```

## 2.23 std::partition, std::stable_partition, std::partition_point
- std::partition: 将满足条件的元素移动到前端

```c++
template<class ForwardIt, class Predicate>
ForwardIt partition(ForwardIt first, ForwardIt last, Predicate pred);

// 示例：按奇偶分区
std::vector<int> v = {1, 2, 3, 4, 5};
auto it = std::partition(v.begin(), v.end(), [](int x) { return x % 2 != 0; });
// 奇数在前，偶数在后（可能改变相对顺序）
```
- std::partition_point: 返回分区点
```c++
template<class ForwardIt, class Predicate>
ForwardIt partition_point(ForwardIt first, ForwardIt last, Predicate pred);

// 示例：找到分区点
auto it = std::partition_point(v.begin(), v.end(), [](int x) { return x % 2 != 0; });
```

## 2.24 std::minelement, std::maxelement, std::clamp
```c++
template<class ForwardIt>
ForwardIt min_element(ForwardIt first, ForwardIt last);

// 示例：找到最大值
auto it = std::max_element(v.begin(), v.end());
```
- std::clamp: 将元素限制在范围内
```c++
template<class T>
const T& clamp(const T& value, const T& lo, const T& hi);

// 示例：限制数值在[0, 100]
int x = 150;
x = std::clamp(x, 0, 100); // x变为100
```

## 2.25 std::sample
- 随机采样，需要插入迭代器
```c++
template<class PopulationIt, class SampleIt, class Distance, class UniformRandomBitGenerator>
SampleIt sample(PopulationIt first, PopulationIt last, SampleIt out, Distance n, UniformRandomBitGenerator&& g);

// 示例：随机采样3个元素
std::vector<int> src = {1, 2, 3, 4, 5}, dest;
std::sample(src.begin(), src.end(), std::back_inserter(dest), 3, std::mt19937{});
```