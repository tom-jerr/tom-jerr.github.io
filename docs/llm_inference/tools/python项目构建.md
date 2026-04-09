# Python 项目构建
现代 Python 项目基本围绕 PEP 517/518/621：
- PEP 518：构建工具写在哪里？构建前要先装什么？
- PEP 517：pip / uv 应该如何“调用构建后端”？
- PEP 621：项目元数据应该如何标准化地写？

## 常见 python project 构建配置文件
- pyproject.toml：项目元数据 + 构建配置的统一入口
- build backend（构建后端）：真正构建项目二进制包的实现（setuptools / hatchling / flit / poetry-core…）
- build frontend（构建前端）：调用后端去构建 wheel/sdist 的工具（python -m build、pip、uv 等）
- installer（安装器）：把 wheel 装进环境里（pip、uv）
- publisher（发布工具）：把包上传到 PyPI（twine 等）
- env manager（环境管理）：创建/管理虚拟环境（venv、uv、conda）

[build-system]
- 构建后端使用什么工具
- 构建时的依赖需要什么

[project]
- 项目元数据：name, version, description...
- 运行时依赖，python 版本要求
- 这些依赖会在安装项目二进制的的时候被解析并安装（比如 pip install . 或 uv pip install -e .）

[tools.uv.extra-build-dependencies]
当 uv 需要构建 flash-attn（比如没有合适 wheel、只能源码编译）时，uv 会在它的 build isolation 环境里额外装上 torch，避免 flash-attn 在构建阶段 import torch 失败