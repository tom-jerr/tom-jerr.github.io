---
title: DeepGEMM SM100 W4A8 MegaMoE 中文注释源码
created: 2026-10-07
tags:
  - CUDA
  - LLMInference
description: 保留公开 MegaMoE 原代码，按原始行号补充 dispatch、TMA、MMA、epilogue、调度和跨 GPU 同步的中文注释。
---

# DeepGEMM SM100 W4A8 MegaMoE 中文注释源码

配套讲解见 [DeepSeek-V4 的 MegaMoE 推理章节](../../deepseekv4.md#megamoe-源码线程角色与数据布局)。本目录提供 **10 个完整文件、181 处中文注释、4194 行原代码**；仅增加以 `// [中文注释]` 开头的注释，原有代码和英文注释保留。

- DeepGEMM：`057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7`。
- SGLang 前置量化 kernel：`f47d89564dc2479be8e2e866a2b17f149318758e`。
- [下载全部注释源码](DeepGEMM-SM100-W4A8-annotated.zip)。
- [原始 blob hash、SHA-256 与行号映射](manifest.json)。

## 阅读顺序

| 顺序 | 完整源码 | 重点 |
| --- | --- | --- |
| 1 | [host tile / thread 配置](DeepGEMM/csrc/jit_kernels/heuristics/mega_moe.hpp) | 512 threads/CTA、两 CTA cluster、BLOCK_N/BLOCK_K、SMEM stage 数 |
| 2 | [buffer 布局](DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh) | input、两级 ring、full-pool metadata、top-k combine slots |
| 3 | [SGLang pre-dispatch](SGLang/python/sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh) | BF16→FP8、UE8M0、路由复制、padding、PDL |
| 4 | [SM100 W4A8 主 kernel](DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh) | 六阶段数据流、各 warp 的主循环、SwiGLU 和写回 |
| 5 | [scheduler 与 L2KBlockDependency](DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh) | warmup、任务双缓冲、K-block readiness、generation XOR |
| 6 | [grid / NVLink barrier](DeepGEMM/deep_gemm/include/deep_gemm/comm/barrier.cuh) | CTA 汇合、跨 rank release/acquire、phase/sign |
| 7 | [PTX load/store/atomic](DeepGEMM/deep_gemm/include/deep_gemm/ptx/ld_st.cuh) | `gpu` / `sys` scope，普通 atomic 与 release atomic 的区别 |
| 8 | [peer pointer 映射](DeepGEMM/deep_gemm/include/deep_gemm/layout/sym_buffer.cuh) | symmetric offsets 与 `map()` |
| 9 | [host 容量与接口](DeepGEMM/csrc/apis/mega_moe.hpp) | live-block 上界、ring / scale 字节数、dtype / shape 校验 |
| 10 | [UE8M0 数值辅助](DeepGEMM/deep_gemm/include/deep_gemm/common/math.cuh) | power-of-two scale、量化边界、reciprocal exponent |

源码中的 `原始 L123` 指上述 commit 原文件的行号。本地插入注释后行号会变化，`manifest.json` 的 `line_map` 给出注释行与代码行的对应关系。

## 先区分四种“完成”

1. **调度 metadata 读完**：释放 `task_info` 双缓冲 slot。
2. **MMA 不再读输入**：允许复用 SMEM operand stage 或 global activation ring。
3. **epilogue 已读取累加器**：允许复用 TMEM accumulator stage。
4. **所有本轮 L2 远端写入完成**：发布 `combine_ready_grid_idx`，源 rank 才能读取 top-k combine slots。

其中 `l2_empty_count` 在第 2 项成立时增加，可能早于第 4 项。top-k 权重在 Linear1 的 SwiGLU 后、FP8 量化前乘入；最终 combine 对 BF16 partial results 作 FP32 求和，再输出 BF16。

## 版本与使用范围

这是源码阅读包，保留原目录路径便于对照；编译需要完整原仓、CUTLASS / deep_jit 等依赖和相应 CUDA/SM100 环境。本机没有进行 B200 编译、跨 GPU 运行或性能测试。

两个仓库的 pin 分别用于解释框架包装和公开内核，不能据此认为二者可直接作为同一套 build。尤其此 DeepGEMM pin 的 `get_num_sms_for_mega_moe()` 要求 runtime SM 数等于物理 SM 数；SGLang 包装另有 reserved-SM 逻辑，需要核对实际安装的 `sgl-deep-gemm` 版本。

DeepGEMM 原代码的授权保留在 [LICENSE-DeepGEMM.txt](LICENSE-DeepGEMM.txt)，SGLang 原代码的授权保留在 [LICENSE-SGLang.txt](LICENSE-SGLang.txt)。
