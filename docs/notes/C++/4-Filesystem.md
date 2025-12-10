## 4 FileSystem

- 文件系统库提供对文件系统及其组件（例如路径、常规文件和目录）执行操作的工具。

  ```admonish info
  文件系统库最初开发为boost.filesystem ，并作为技术规范 ISO/IEC TS 18822:2015发布，最终于 C++17 合并到 ISO C++ 中。目前，boost 实现在比 C++17 库更多的编译器和平台上可用。
  ```

- 如果对此库中的函数的调用引发文件系统竞争，即当**多个线程、进程或计算机交错访问和修改文件系统中的同一对象**时，则行为未定义。

### 4.1 定义

- file：保存数据的文件系统对象，可以写入、读取或两者兼而有之。文件具有名称、属性，其中之一是文件类型：
  - 目录：充当目录条目容器的文件，用于标识其他文件（其中一些可能是其他嵌套目录）。在讨论特定文件时，该文件作为条目出现的目录是其父目录。父目录可以用相对路径名表示“……”。
  - 常规文件：将名称与现有文件关联的目录条目（即硬链接）。如果支持多个硬链接，则在删除指向该文件的最后一个硬链接后，该文件将被删除。
  - 符号链接：将名称与路径相关联的目录条目，该路径可能存在也可能不存在。
  - 其他特殊文件类型：块、字符、fifo、套接字。
- 文件名：用于命名文件的字符串。允许的字符、区分大小写、最大长度和不允许的名称由实现定义。名称“.”和“..”在库层面具有特殊含义。
- 路径：标识文件的元素序列。它以可选的根名称开头​​（例如“C：”或者“//server”在 Windows 上），然后是可选的根目录（例如“/”在 Unix 上），后跟零个或多个文件名序列（除最后一个文件名外，其他文件名都必须是目录或目录链接）。路径 (pathname )的字符串表示的本机格式（例如，使用哪些字符作为分隔符）和字符编码是实现定义的，此库提供可移植的路径表示。
  - 绝对路径：明确标识文件位置的路径。
  - 规范路径：不包含符号链接的绝对路径，“.”或者“..”元素。
  - 相对路径：用于标识文件相对于文件系统上某个位置的位置的路径。特殊路径名“.”（点，“当前目录”）和“..”（点点，“父目录”）是相对路径。

### 4.2 directory

```c++
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
 
int main()
{
    const std::filesystem::path sandbox{"sandbox"};
    std::filesystem::create_directories(sandbox/"dir1"/"dir2");
    std::ofstream{sandbox/"file1.txt"};
    std::ofstream{sandbox/"file2.txt"};
 
    std::cout << "directory_iterator:\n";
    // directory_iterator can be iterated using a range-for loop
    for (auto const& dir_entry : std::filesystem::directory_iterator{sandbox}) 
        std::cout << dir_entry.path() << '\n';
 
    std::cout << "\ndirectory_iterator as a range:\n";
    // directory_iterator behaves as a range in other ways, too
    std::ranges::for_each(
        std::filesystem::directory_iterator{sandbox},
        [](const auto& dir_entry) { std::cout << dir_entry << '\n'; });
 
    std::cout << "\nrecursive_directory_iterator:\n";
    for (auto const& dir_entry : std::filesystem::recursive_directory_iterator{sandbox}) 
        std::cout << dir_entry << '\n';
 
    // delete the sandbox dir and all contents within it, including subdirs
    std::filesystem::remove_all(sandbox);
}
// Possible output:
// directory_iterator:
// "sandbox/file2.txt"
// "sandbox/file1.txt"
// "sandbox/dir1"
 
// directory_iterator as a range:
// "sandbox/file2.txt"
// "sandbox/file1.txt"
// "sandbox/dir1"
 
// recursive_directory_iterator:
// "sandbox/file2.txt"
// "sandbox/file1.txt"
// "sandbox/dir1"
// "sandbox/dir1/dir2"
```

### 4.3 space_info

- 确定路径名所在的文件系统的信息页位于，如同通过 POSIX 的 statvfs 操作一样。

- 该对象由 POSIX struct statvfs 内容进行填充如下所示：

  - space_info.capacity设置为f_blocks * f_frsize。
  - space_info.free设置为f_bfree * f_frsize。
  - space_info.available设置为f_bavail * f_frsize。
  - 任何无法确定的成员都设置为static_cast<std::uintmax_t>(-1)

```c++
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <locale>
 
std::uintmax_t disk_usage_percent(const std::filesystem::space_info& si,
                                  bool is_privileged = false) noexcept
{
    if (constexpr std::uintmax_t X(-1);
        si.capacity == 0 || si.free == 0 || si.available == 0 ||
        si.capacity == X || si.free == X || si.available == X
    )
        return 100;
 
    std::uintmax_t unused_space = si.free, capacity = si.capacity;
    if (!is_privileged)
    {
        const std::uintmax_t privileged_only_space = si.free - si.available;
        unused_space -= privileged_only_space;
        capacity -= privileged_only_space;
    }
    const std::uintmax_t used_space{capacity - unused_space};
    return 100 * used_space / capacity;
}
 
void print_disk_space_info(auto const& dirs, int width = 14)
{
    (std::cout << std::left).imbue(std::locale("en_US.UTF-8"));
 
    for (const auto s : {"Capacity", "Free", "Available", "Use%", "Dir"})
        std::cout << "│ " << std::setw(width) << s << ' ';
 
    for (std::cout << '\n'; auto const& dir : dirs)
    {
        std::error_code ec;
        const std::filesystem::space_info si = std::filesystem::space(dir, ec);
        for (auto x : {si.capacity, si.free, si.available, disk_usage_percent(si)})
            std::cout << "│ " << std::setw(width) << static_cast<std::intmax_t>(x) << ' ';
        std::cout << "│ " << dir << '\n';
    }
}
 
int main()
{
    const auto dirs = {"/dev/null", "/tmp", "/home", "/proc", "/null"};
    print_disk_space_info(dirs);
}
// Possible output:

// │ Capacity       │ Free           │ Available      │ Use%           │ Dir            
// │ 84,417,331,200 │ 42,732,986,368 │ 40,156,028,928 │ 50             │ /dev/null
// │ 84,417,331,200 │ 42,732,986,368 │ 40,156,028,928 │ 50             │ /tmp
// │ -1             │ -1             │ -1             │ 100            │ /home
// │ 0              │ 0              │ 0              │ 100            │ /proc
// │ -1             │ -1             │ -1             │ 100            │ /null
```

### 4.4 symlink

- 软链接也叫符号链接，会创建一个新的inode块，里面的数据内容是链接的文件名称
- 创建一个软链接

```c++
#include <cassert>
#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;
 
int main()
{
    fs::create_directories("sandbox/subdir");
    fs::create_symlink("target", "sandbox/sym1");
    fs::create_directory_symlink("subdir", "sandbox/sym2");
 
    for (auto it = fs::directory_iterator("sandbox"); it != fs::directory_iterator(); ++it)
        if (is_symlink(it->symlink_status()))
            std::cout << *it << "->" << read_symlink(*it) << '\n';
 
    assert(std::filesystem::equivalent("sandbox/sym2", "sandbox/subdir"));
    fs::remove_all("sandbox");
}

// Possible output:

// "sandbox/sym1"->"target"
// "sandbox/sym2"->"subdir"
```

```admonish info
读取软链接文件，会获取到被链接的文件本身
```

```c++
#include <filesystem>
#include <iostream>
 
namespace fs = std::filesystem;
 
int main()
{
    for (fs::path p : {"/usr/bin/gcc", "/bin/cat", "/bin/mouse"})
    {
        std::cout << p;
        fs::exists(p) ?
            fs::is_symlink(p) ?
                std::cout << " -> " << fs::read_symlink(p) << '\n' :
                std::cout << " exists but it is not a symlink\n" :
            std::cout << " does not exist\n";
    }
}

// Possible output:

// "/usr/bin/gcc" -> "gcc-5"
// "/bin/cat" exists but it is not a symlink
// "/bin/mouse" does not exist
```

### 4.5 status

```admonish info
就像POSIX中 stat 获取文件方式类似
```

```c++
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
 
namespace fs = std::filesystem;
 
void demo_status(const fs::path& p, fs::file_status s)
{
    std::cout << p;
    // alternative: switch(s.type()) { case fs::file_type::regular: ...}
    if (fs::is_regular_file(s))
        std::cout << " is a regular file\n";
    if (fs::is_directory(s))
        std::cout << " is a directory\n";
    if (fs::is_block_file(s))
        std::cout << " is a block device\n";
    if (fs::is_character_file(s))
        std::cout << " is a character device\n";
    if (fs::is_fifo(s))
        std::cout << " is a named IPC pipe\n";
    if (fs::is_socket(s))
        std::cout << " is a named IPC socket\n";
    if (fs::is_symlink(s))
        std::cout << " is a symlink\n";
    if (!fs::exists(s))
        std::cout << " does not exist\n";
}
 
int main()
{
    // create files of different kinds
    fs::create_directory("sandbox");
    fs::create_directory("sandbox/dir");
    std::ofstream{"sandbox/file"}; // create regular file
    fs::create_symlink("file", "sandbox/symlink");
 
    mkfifo("sandbox/pipe", 0644);
    sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    std::strcpy(addr.sun_path, "sandbox/sock");
    int fd = socket(PF_UNIX, SOCK_STREAM, 0);
    bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
 
    // demo different status accessors
    for (auto it{fs::directory_iterator("sandbox")}; it != fs::directory_iterator(); ++it)
        demo_status(*it, it->symlink_status()); // use cached status from directory entry
    demo_status("/dev/null", fs::status("/dev/null")); // direct calls to status
    demo_status("/dev/sda", fs::status("/dev/sda"));
    demo_status("sandbox/no", fs::status("/sandbox/no"));
 
    // cleanup (prefer std::unique_ptr-based custom deleters)
    close(fd);
    fs::remove_all("sandbox");
}
// Possible output:

// "sandbox/file" is a regular file
// "sandbox/dir" is a directory
// "sandbox/pipe" is a named IPC pipe
// "sandbox/sock" is a named IPC socket
// "sandbox/symlink" is a symlink
// "/dev/null" is a character device
// "/dev/sda" is a block device
// "sandbox/no" does not exist
```

### 4.6 hard link

- 在POSIX系统中，每个目录至少有两个硬链接，自己以及"."

- ".."有三个硬链接，目录本身，"."以及".."

- 多个文件名同时指向同一个索引节点（Inode），只增加i_nlink硬链接计数。

  ```admonish info
  只要文件的索引节点还存在一个以上的链接，删除其中一个链接并不影响索引节点本身和其他的链接（也就是说该文件的实体并未删除），而只有当最后一个链接被删除后，且此时有新数据要存储到磁盘上，那么被删除的文件的数据块及目录的链接才会被释放，存储空间才会被新数据所覆盖。因此，该机制可以有效的防止误删操作。
  ```

- :skull: 硬链接只能在同一类型的文件系统中进行链接，不能跨文件系统。同时它只能对文件进行链接，不能链接目录。

```c++
#include <filesystem>
#include <iostream>
namespace fs = std::filesystem;
 
int main()
{
    // On a POSIX-style filesystem, each directory has at least 2 hard links:
    // itself and the special member pathname "."
    fs::path p = fs::current_path();
    std::cout << "Number of hard links for current path is "
              << fs::hard_link_count(p) << '\n';
 
    // Each ".." is a hard link to the parent directory, so the total number
    // of hard links for any directory is 2 plus number of direct subdirectories
    p = fs::current_path() / ".."; // Each dot-dot is a hard link to parent
    std::cout << "Number of hard links for .. is "
              << fs::hard_link_count(p) << '\n';
}
// Possible output:

// Number of hard links for current path is 2
// Number of hard links for .. is 3
```
