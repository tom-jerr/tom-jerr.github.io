# Pip & uv 使用
## Pip 基础使用
```shell
pip install xxx
```
- 从 PyPI 下载合适的版本(一般是最新，如果没有 requirement)
- 优先使用 wheel 构建，如果没有本地编译
- pip 会创建一个临时隔离环境
- 只安装 pyproject.toml 里声明的 build-system.requires
- 当前环境的 torch / numpy / cuda 都不可见

---
### Pip 相关选项
1. --no-build-isolation：构建时直接用当前环境里的包
2. --no-use-pep517：不用 PEP 517 新构建系统，走 setup.py install
3. --no-deps：只安装这个包，但是相关依赖不安装
4. --no-cache-dir：不用本地缓存
5. --no-binary：强制源码编译；--only-binary :all:： 只允许有源码
6. -i / --index-url：指定 PyPI 镜像

---
### Editable 与正常安装区别
```shell
pip install .
```
发生的事情：
- 构建一个 wheel
- 把二进制的 wheel 拷贝到.venv/lib/python3.x/site-packages/tinyllm/
- 之后改本地源码，不会生效
  
```shell
pip install -e .
```
发生的事情：
- 不拷贝源码
- 在 site-packages 里创建一个 指向源码目录的链接/映射
- 改本地源码，立刻生效
  
**旧时代（setuptools legacy）**
- 生成 .egg-link
- easy-install.pth 把源码路径加进 sys.path

**现代（PEP 660）**
- build backend 支持 editable 安装
- 返回一个可编辑 wheel
- 本质仍然是 路径引用，而不是复制

可以理解为：
```shell
site-packages
└── miniinfer→ /path/to/your/source/miniinfer
```
---
## uv 基础介绍
uv 是一个极快的 Python 包管理与虚拟环境工具，它 替代 pip + venv + pip-tools 的大部分工作
| pip 使用           | uv 里的对应      |
| ------------------ | ---------------- |
| `pip install`      | `uv pip install` |
| `python -m venv`   | `uv venv`        |
| `pip freeze`       | `uv pip freeze`  |
| `requirements.txt` | `uv sync`        |
| `pip-tools`        | `uv` 内建        |

```shell
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
## uv 优势
### uv 缓存基于内容匹配
**pip 的缓存问题**
- 按 URL / 包名缓存
- 项目隔离
- wheel 经常重复构建

**uv 的缓存设计**

```shell
~/.cache/uv/
├── wheels/
│   └── sha256/
├── sdists/
├── builds/
```
- 只要内容 hash 相同就可以全局复用

### 零拷贝安装
**pip 安装 wheel 的传统方式**
- 解压 whl
- 复制文件到 site-packages
- 写 metadata

存在大量：
- IO
- copy
- inode 分配

**uv 的方式**
```shell
wheel 已在 cache
↓
尝试硬链接 / reflink
↓
失败才 fallback 到 copy
```
大致顺序是：
1. reflink（Copy-on-Write）
  - 相比 hard link，reflink 的拷贝语义更安全
  - 文件系统支持（btrfs / xfs / APFS）
  - 零拷贝
  - 几乎瞬时
2. hard link
  - 同一文件系统
  - inode 共享
  - 几乎零成本
3. fast copy
  - mmap / sendfile
  - 并行复制

### Reflink vs hard link

| 维度             | hard link    | reflink               |
| ---------------- | ------------ | --------------------- |
| inode            | 同一个       | 不同                  |
| 数据块           | 共享         | 共享（写时分裂）      |
| 修改文件         | 所有链接都变 | 只影响当前文件        |
| 是否是“拷贝语义” | ❌            | ✅                     |
| 是否需要同一 FS  | ✅            | ❌（部分 FS 可跨子卷） |
| 是否支持目录     | ❌            | ❌                     |
| 支持文件系统     | 所有 Unix FS | 仅 CoW FS             |
| 语义安全性       | ⚠️            | ✅                     | ✅ |