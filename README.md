# Want to be a MlSys wizard

tom-jerr 的技术博客与学习笔记，使用 [Quartz 5](https://quartz.jzhao.xyz/) 构建。

## 本地预览

需要 Node.js 22+ 与 npm 10.9.2+：

当前 Windows 环境如果仍是 Node.js 20，可以使用已安装的 Scoop 配置 `fnm`：

```powershell
scoop install fnm
fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression
fnm install 22.16.0
fnm use 22.16.0
node --version
npm --version
```

如需在后续 PowerShell 会话中自动启用 `fnm`，将下面一行加入 `$PROFILE`：

```powershell
fnm env --use-on-cd --shell powershell | Out-String | Invoke-Expression
```

切换版本后安装依赖并启动：

```shell
npm ci
npx quartz plugin install --from-config
npm run dev
```

站点默认运行在 <http://localhost:8080>。生产构建与资源检查使用：

```shell
npm run verify
```

`verify` 会在构建后检查生成页面中的站内链接、图片、脚本和样式资源是否都能解析。

Markdown、图片和 PDF 等正文资源统一放在 `content/`；Quartz 主题配置位于 `quartz.config.yaml`，定制样式位于 `quartz/styles/custom.scss`。

历史 MkDocs/Zensical 配置保存在 `legacy/mkdocs/`，不再参与构建。
