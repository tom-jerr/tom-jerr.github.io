# Constrained Decoding

SGLang 中关于 "Compressed FSM" (压缩有限状态机) 和 "Jump Forward" (跳跃解码) 的实现主要集中在 sglang/srt/constrained 目录下。

这一机制的核心思想是：当 FSM（有限状态机）处于确定性路径上时（即后续的一系列字符/Token 是唯一的），可以直接“跳过”模型生成，直接将这些确定的 Token 拼接到输出中，从而加速解码。

以下是代码层面的具体实现解析：

### 1. Compressed FSM 的构建 (核心逻辑)

**文件位置**: [outlines_jump_forward.py](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

这是实现“压缩”逻辑的地方。代码通过分析 FSM 的图结构，将线性的、确定性的路径合并为“超边”（Jump Edge）。

- **[init_state_to_jump_forward(regex_string)](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**:
    - 这是入口函数。它首先将正则表达式编译为字节级 FSM ([outlines.fsm.regex.make_deterministic_fsm](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))。
    - **寻找确定性路径**: 它遍历 FSM 的每个状态，检查是否存在“单出度”状态（Singular Transition），即该状态只有一个合法的下一个字符。
    - **路径压缩**: 如果发现连续的单出度状态，它会将这些字符拼接成一个字符串，形成一个从起始状态直接跳转到目标状态的映射。
    - **结果**: 生成 [state_to_jump_forward](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 映射表，Key 是当前状态 ID，Value 是可以直接跳过的字符串及最终状态。

### 2. Jump Forward 的执行 (解码阶段)

**文件位置**: [outlines_backend.py](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

在解码过程中，后端会利用上述构建的映射表来尝试跳过生成步骤。

- **[OutlinesGrammar.try_jump_forward(self, tokenizer)](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)**:
    - **检查**: 每次生成前，检查当前 FSM 状态 ([self.state](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)) 是否在 [jump_forward_map](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 中。
    - **获取**: 如果命中，直接获取预计算好的字节序列 ([jump_forward_bytes](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))。
    - **处理**: 将这些字节转换为 Token ID ([suffix_ids](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)) 并返回。这告诉调度器：“不用让模型跑了，直接把这些 Token 加到输出里，并把 FSM 状态更新到目标状态”。

### 3. 处理 Tokenization Artifacts (重分词问题)

**文件位置**: [outlines_backend.py](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 和 [xgrammar_backend.py](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)

这是你提到的“必须使用原生分词器进行重分词”的关键点。因为直接拼接字符可能会导致分词边界变化（例如 [["a", "b"]](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 拼接后可能变成 `["ab"]`），系统必须确保跳跃后的 Token 序列与模型词表一致。

- **在 [outlines_backend.py](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 中**:
    
    - [try_jump_forward](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 方法中包含处理 **Continuation Bytes** (UTF-8 续字节) 的逻辑。它会小心地处理字节边界，确保生成的 Token 是合法的。
    - 它将跳过的字节序列转换回 Token IDs ([tokenizer.convert_tokens_to_ids](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))，确保符合 Tokenizer 的规范。
- **在 [xgrammar_backend.py](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 中 (另一种后端实现)**:
    
    - 实现了 **[jump_and_retokenize](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)** 方法。
    - 当发生跳跃时，它会对比“旧的输出 Token 序列”和“基于完整字符串重新分词后的新序列”。
    - **Rollback (回滚)**: 如果发现不一致（即 Tokenization Artifacts），它会回滚 FSM 的状态 ([self.matcher.rollback](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html))，以确保状态机与最终的 Token 序列严格同步。

### 总结

SGLang 通过预先分析正则表达式的 FSM 图，找出确定性的“捷径”（[outlines_jump_forward.py](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html)），并在解码时通过 [try_jump_forward](vscode-file://vscode-app/c:/Users/lzy/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) 接口（`out`

lines_backend.py`）直接注入这些 Token，同时利用重分词逻辑确保证 Token 的正确性。`