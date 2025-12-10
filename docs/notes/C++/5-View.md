# 5 View

```admonish info
通常若引用范围的元素修改,则视图的元素也会修改。  
若视图的元素修改,则引用范围的元素也会修改。
```

```admonish example
视图通常用于在特定的基础上,处理基础范围的元素子集和/或经过一些可选转换后的值。例 如,可以使用一个视图来迭代一个范围的前五个元素
```

```c++
for (const auto& elem : std::views::take(coll, 5)) {  ...  }
```

- 管道语法中让视图对范围进行操作。通过使用操作符 |,可以创建视图的管道:

  ```c++
  auto v = coll 
        | std::views::filter([](auto elem){return elem % 3 == 0;}) 
        | std::views::transform([](auto elem){return elem * elem;}) 
        |std::views::take(3);
  ```

- 通过类模板指定范围结束的值

  ```c++
  template<auto End>
  struct EndValue {
    bool operator== (auto pos) const {
        return *pos == End;  // end is where iterator points to End
    }
  };

  int main() {
    std::vector coll = {42, 8, 0, 15, 7, -1};
    std::ranges::subrange range{coll.begin(), EndValue<7>{}};
    std::ranges::sort(range);
    std::ranges::for_each(coll.begin(), EndValue<-1>{}, 
                          [](auto value){std::cout << ' ' << value;})
  }
  ```

- 支持投影功能，避免写显示的比较器，直接通过投影的方式指定使用Person的age进行排序

  ```c++
  struct Person {
    std::string name;
    int age;
  };
  std::vector<Person> people = {{"Alice", 25}, {"Bob", 30}, {"Charlie", 20}};
  std::sort(people.begin(), people.end(), 
            [](const Person& a, const Person& b) { return a.age < b.age; });
  std::ranges::sort(people, std::less<int>{}, &Person::age);
  ```
