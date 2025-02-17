---
title: git commit 规范
tags:
  - git
  - team
math:
---
最近需要和学长在项目上一起开发新功能，git commit 和分支管理就显得比较重要。于是查阅了一下 git commit 的一些规范。基本格式如下：

```text
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

其中，type 表示提交类型，分别为：

- feat: 新增 feature
- fix: 修复 bug
- docs: 仅仅修改了文档，比如 README, CHANGELOG, CONTRIBUTE等等
- style: 仅仅修改了空格、格式缩进、逗号等等，不改变代码逻辑
- refactor: 代码重构，没有加新功能或者修复 bug
- perf: 优化相关，比如提升性能、体验
- test: 测试用例，包括单元测试、集成测试等
- chore: 改变构建流程、或者增加依赖库、工具等
- revert: 回滚到上一个版本

scope 为可选项，表示修改范围。subject 则是对修改的简单描述，一般不超过 50 个字符。

body 为更详细的说明，表示：

1. 为什么这个变更是必须的? 它可能是用来修复一个bug，增加一个feature，提升性能、可靠性、稳定性等等
2. 他如何解决这个问题? 具体描述解决问题的步骤
3. 是否存在副作用、风险?

footer 用于正在处理或者需要关闭某个 issue，可以添加到 issue 的链接。

## 参考资料

1. [Git Commit 规范 \| Feflow](https://feflowjs.com/zh/guide/rule-git-commit.html)