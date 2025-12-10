# 3 Iterator

- 输入迭代器：只能用来读取指向的值；当该迭代器自加时，之前指向的值就不可访问。
  ```admonish example
  std::istream_iterator 就是这样的迭代器。
  ```
- 前向迭代器：类似于输入迭代器，可以在指示范围迭代多次
  ```admonish example
  std::forward_list 就是这样的迭代器。就像一个单向链表一样，只能向前遍历，不能向后遍历，但可以反复迭代。
  ```
- 双向迭代器：这个迭代器可以自增，也可以自减，迭代器可以向前或向后迭代。
  ```admonish example
  std::list, std::set 和 std::map 都支持双向迭代器。
  ```
- 随机访问迭代器：与其他迭代器不同，随机访问迭代器一次可以跳转到任何容器中的元素上，而非之前的迭代器，一次只能移动一格。
  ```admonish example
  std::vector 和 std::deque 的迭代器就是这种类型。
  ```
- 连续迭代器：这种迭代器具有前述几种迭代器的所有特性，不过需要容器内容在内存上是连续的，类似一个数组或 std::vector 。
- 输出迭代器：该迭代器与其他迭代器不同。因为这是一个单纯用于写出的迭代器，其只能增加，并且将对应内容写入文件当中。如果要读取这个迭代中的数据，那么读取到的值就是未定义的
- 可变迭代器：如果一个迭代器既有输出迭代器的特性，又有其他迭代器的特性，那么这个迭代器就是可变迭代器。
  ```admonish example
  该迭代器可读可写。如果我们从一个非常量容器的实例中获取一个迭代器，那么这个迭代器通常都是可变迭代器。
  ```

## 3.1 std::back_insert_iterator

- 内部调用容器的`push_back`方法

  ```c++
  // 使用 back_insert_iterator 在 destination 的末尾插入元素
  std::copy(source.begin(), source.end(), std::back_inserter(destination));
  ```

## 3.2 std::front_insert_iterator

- 内部调用容器的`push_front`方法

  ```c++
  // 使用 front_insert_iterator 在 destination 的开头插入元素
  std::copy(source.begin(), source.end(), std::front_inserter(destination));  
  ```

## 3.3 std::insert_iterator

- std::insert_iterator 是一个通用的插入迭代器，可以在容器的任意位置插入元素。它需要一个容器和一个插入位置。

  ```c++
  // 在 destination 的第二个位置插入 source 的元素
  std::copy(source.begin(), source.end(), std::inserter(destination, destination.begin() + 1));
  ```

## 3.4 std::istream_iterator

- std::istream_iterator 是一个输入流迭代器，用于从输入流（如 std::cin）读取数据
-

```c++
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm> // std::copy

int main() {
    std::vector<int> numbers;

    std::cout << "Enter some integers (end with EOF):" << std::endl;

    // 使用 istream_iterator 从 std::cin 读取整数
    std::copy(std::istream_iterator<int>(std::cin), std::istream_iterator<int>(), std::back_inserter(numbers));

    std::cout << "You entered:" << std::endl;
    for (int num : numbers) {
        std::cout << num << " ";
    }
    // 输出用户输入的整数
    return 0;
}
```

## 3.5 std::ostream_iterator

- std::ostream_iterator 是一个输出流迭代器，用于将数据写入输出流（如 std::cout）

```c++
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm> // std::copy

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // 使用 ostream_iterator 将 numbers 写入 std::cout，每个元素后加空格
    std::copy(numbers.begin(), numbers.end(), std::ostream_iterator<int>(std::cout, " "));

    // 输出: 1 2 3 4 5
    return 0;
}
```
