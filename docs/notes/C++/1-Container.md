# 1 Container

## 1.1 std::vector

- 迭代器在vector插入超过容量后，自动扩容时失效
  ```admonish info
  引用语义在使用 span 时必须谨慎。将一个新元素插入 vector 中,该 vector 中保存了跨度所引用的元素。由于 span 的引用语义,若 vector 分配新的内存，会使所有迭代器和指向其元素的指针无效,所以重新分配也会使引用 vector 元素的 span 失效。span 指向了不再存在的元素。  出于这个原因,需要在插入前后都要仔细检查容量 (分配内存的最大元素数量)。若容量发生变化,则重新初始化 span
  ```
