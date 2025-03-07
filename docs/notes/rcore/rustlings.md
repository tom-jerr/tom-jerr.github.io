---
title: rustlings
date: 2024/4/9 23：08
update: 
comments: true
description: rustlings中使用的rust技巧
katex: true
tags: 
- rust
categories: Project
---

# Rustlings

## Option2

- 调用·`vec`的`pop` 方法时，它会从数组的末尾移除一个元素，并返回被移除的元素作为 `Option<T>`。因此，在这个例子中，由于数组的类型是 `Vec<Option<i8>>`，所以 `pop` 方法返回的类型是 `Option<Option<i8>>`

## 错误处理

- `Result<String, String>`：第一个类型是`Ok()`中的数据类型，第二个类型是`Err()`中的类型

~~~rust
use std::fmt::Error;

pub fn generate_nametag_text(name: String) -> Result<String, String> {
    if name.is_empty() {
        // Empty names aren't allowed.
        Err("`name` was empty; it must be nonempty.".into())
    } else {
        Ok(format!("Hi! My name is {}", name))
    }
}
~~~

- `?`相当于如下`match`匹配代码；调用`?`的函数返回值必须为`Result`或`Option`类型，有时候需要改写`main`函数返回值类型

~~~rust
let x = match qty {
    Ok(x) => x * cost_per_item + processing_fee,
    Err(e) => return Err(e),
};

Ok(x)

fn main() -> Result<(), ParseIntError> {
    let mut tokens = 100;
    let pretend_user_input = "8";

    let cost = total_cost(pretend_user_input)?;

    if cost > tokens {
        println!("You can't afford that many!");
    } else {
        tokens -= cost;
        println!("You now have {} tokens.", tokens);
    }
    Ok(())
}
~~~

## Test

- ` #[should_panic]`特性，允许测试代码出现Panic而不终止运行。

## 迭代器

- 消费者适配器是消费掉迭代器，然后返回一个值。那么迭代器适配器，顾名思义，会返回一个新的迭代器，这是实现链式方法调用的关键：`v.iter().map().filter()...`。
- 与消费者适配器不同，迭代器适配器是惰性的，意味着你**需要一个消费者适配器来收尾，最终将迭代器转换成一个具体的值**

~~~rust
let v1: Vec<i32> = vec![1, 2, 3];

let v2: Vec<_> = v1.iter().map(|x| x + 1).collect();

assert_eq!(v2, vec![2, 3, 4]);
~~~

- `collect`会自动根据指定类型对数据进行收集

~~~rust
// Complete the function and return a value of the correct type so the test
// passes.
// Desired output: Ok([1, 11, 1426, 3])
fn result_with_list() -> Result<Vec<i32>, DivisionError> {
    let numbers = vec![27, 297, 38502, 81];
    let division_results = numbers.into_iter().map(|n| divide(n, 27)).collect();
    division_results
}

// Complete the function and return a value of the correct type so the test
// passes.
// Desired output: [Ok(1), Ok(11), Ok(1426), Ok(3)]
fn list_of_results() -> Vec<Result<i32, DivisionError>> {
    let numbers = vec![27, 297, 38502, 81];
    let division_results = numbers.into_iter().map(|n| divide(n, 27)).collect();
    division_results
}
~~~

- `product`直接将迭代器元素相乘，返回一个迭代器

~~~rust
pub fn factorial(num: u64) -> u64 {
    // Complete this function to return the factorial of num
    // Do not use:
    // - return
    // Try not to use:
    // - imperative style loops (for, while)
    // - additional variables
    // For an extra challenge, don't use:
    // - recursion
    // Execute `rustlings hint iterators4` for hints.
    (1..num + 1).product()
}// 实现阶乘
~~~

- `flat_map`将对应的元素展平成为新的迭代器

~~~rust
fn count_collection_iterator(collection: &[HashMap<String, Progress>], value: Progress) -> usize {
    // collection is a slice of hashmaps.
    // collection = [{ "variables1": Complete, "from_str": None, ... },
    //     { "variables2": Complete, ... }, ... ]
    collection
        .into_iter()
        .flat_map(|x| x.values()) // 这里是Vec<&Progress>
        .filter(|&progress| *progress == value)
        .count()
}
~~~

## Threads

### Channel

- 发送者会获取tx所有权，多发送者的线程需要拷贝tx

~~~rust
fn send_tx(q: Queue, tx: mpsc::Sender<u32>) -> () {
    let qc = Arc::new(q);
    let qc1 = Arc::clone(&qc);
    let qc2 = Arc::clone(&qc);
    let tx1 = tx.clone();
    thread::spawn(move || {
        for val in &qc1.first_half {
            println!("sending {:?}", val);
            tx.send(*val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });

    thread::spawn(move || {
        for val in &qc2.second_half {
            println!("sending {:?}", val);
            tx1.send(*val).unwrap();
            thread::sleep(Duration::from_secs(1));
        }
    });
}
~~~

## as_ref_mut

- AsRef 的主要作用是允许我们以统一的方式处理不同类型之间的转换。通过实现 AsRef trait，我们可以定义一个类型转换函数，该函数将一个类型转换为另一个类型的引用，而不是拥有新的所有权。
- 具体来说，如果我们有一个类型 T，并且希望将其转换为类型 U 的引用，我们可以实现 AsRef `<U>` trait 来完成这个转换。在实现中，我们需要提供一个名为 as_ref 的方法，该方法返回类型 &U。这样，我们就可以使用 as_ref 方法来将 T 转换为 U 的引用。

~~~rust
// Obtain the number of bytes (not characters) in the given argument.
// TODO: Add the AsRef trait appropriately as a trait bound.
fn byte_counter<T: AsRef<str>>(arg: T) -> usize {
    arg.as_ref().as_bytes().len()
}

// Obtain the number of characters (not bytes) in the given argument.
// TODO: Add the AsRef trait appropriately as a trait bound.
fn char_counter<T: AsRef<str>>(arg: T) -> usize {
    arg.as_ref().chars().count()
}
~~~

- 当我们在代码中需要将一个类型转换为另一个类型的引用时，可以使用 as_ref 方法来进行转换。这对于接受不同类型引用参数的函数或方法非常有用，因为它使得我们可以使用相同的代码来处理不同的类型。
- AsMut `<T>` 是一个标准库中定义的 trait，它将类型 T 转换为一个指向其内部值的可变引用。当我们在泛型函数中使用 T: AsMut `<u32>` 这个 trait bound 时，我们告诉编译器要求 T 类型必须具备将其内部值作为 u32 类型的可变引用的能力

~~~rust
// Squares a number using as_mut().
// TODO: Add the appropriate trait bound.
fn num_sq<T: AsMut<u32>>(arg: &mut T) {
    // TODO: Implement the function body.
    *arg.as_mut()*=*arg.as_mut()
}
~~~

## 编译成库

- `#[no_mangle]`保证rust编译时函数名不改变
- `#[link_name="str"]`在外部块内，通过属性link_name，指定原生库中函数或静态对象的名称，编译器根据它可以为外部块链接原生库并导入该名称定义的函数或静态对象

~~~rust
// 标准库<stdlib.h>内置的abs函数
extern "C" {
    #[link_name = "abs"]
    fn abs_in_rust(input: i32) -> i32;
}

extern "Rust" {
    fn my_demo_function(a: u32) -> u32;
    #[link_name = "my_demo_function"]
    fn my_demo_function_alias(a: u32) -> u32;
}

mod Foo {
    // No `extern` equals `extern "Rust"`.
    #[no_mangle]
    pub fn my_demo_function(a: u32) -> u32 {
        a
    }
}
~~~

## 反转双向链表

- 从头开始遍历结点，对每个结点交换`prev`和`next`指针
- 最后对整个链表交换`start`和`end`

~~~rust
pub fn reverse(&mut self) {
    // TODO
    let mut current_ptr = self.start;
    while let Some(node_ptr) = current_ptr {
        let mut node = unsafe { *node_ptr.as_ptr().as_mut().unwrap() };
        std::mem::swap(&mut node.prev, &mut node.next);
        current_ptr = node.prev;
    }
    std::mem::swap(&mut self.start, &mut self.end);
}
~~~

