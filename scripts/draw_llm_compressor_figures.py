#!/usr/bin/env python3
"""Deterministic, editable SVG figures for the llm-compressor article."""

from __future__ import annotations

import argparse
from pathlib import Path

from draw_hicache_figures import (
    CONTROL, CONTROL_BG, CPU, CPU_BG, GPU, GPU_BG, INK, LINE,
    MUTED, OK, OK_BG, STORE, STORE_BG, WARN, WARN_BG, WHITE, SVG,
)


def card(s: SVG, x: int, y: int, w: int, h: int, title: str,
         detail: str, fill: str, accent: str) -> None:
    s.rect(x, y, w, h, fill, accent, radius=12, stroke_width=2)
    s.text(x + 20, y + 35, title, 21, INK, 700)
    s.text(x + 20, y + 69, detail, 17, MUTED, 500)


def architecture() -> SVG:
    s = SVG(1500, 730, "llm-compressor architecture", "Two entrypoints converge on compressed-tensors checkpoints; only oneshot uses model definitions and modifier calibration.")
    s.header("ARCHITECTURE", "算法、执行路径与部署格式分开看", "蓝色：需要模型定义的 oneshot；绿色：直接处理 safetensors 的 model_free_ptq")

    # Connectors are behind the persistent component boxes.
    s.path("M 322 256 H 394", CPU)
    s.path("M 322 512 H 1102 V 440 H 1170", STORE)
    s.path("M 712 256 H 776", CPU)
    s.path("M 1098 256 H 1170", CPU)
    s.path("M 1318 326 V 434", INK)

    card(s, 64, 184, 258, 142, "oneshot", "HF model + recipe + dataset", CPU_BG, CPU)
    card(s, 394, 184, 318, 142, "CompressionSession", "recipe / lifecycle / events", CONTROL_BG, CONTROL)
    card(s, 776, 184, 322, 142, "CalibrationPipeline", "independent / sequential / datafree", WARN_BG, WARN)
    card(s, 1170, 184, 266, 142, "Modifier", "hooks / observer / 改写权重", GPU_BG, GPU)
    card(s, 64, 440, 258, 142, "model_free_ptq", "safetensors / data-free", STORE_BG, STORE)
    card(s, 1170, 440, 266, 142, "compressed-tensors", "权重 + qparams + config", OK_BG, OK)
    s.line(1303, 582, 1303, 634, OK)
    s.text(1303, 677, "vLLM 加载 → 运行时量化实现", 20, INK, 700, "middle")
    s.text(64, 625, "model_free_ptq 不创建完整 HF Module；不执行 GPTQ / AWQ 等需要校准数据的算法。", 18, MUTED)
    return s


def sequential() -> SVG:
    s = SVG(1500, 710, "Sequential calibration pipeline", "For each traced subgraph, calibrate with hooks, update parameters at the sequential event, then optionally propagate compressed outputs to the next subgraph.")
    s.header("EXECUTION ORDER", "SequentialPipeline：校准前向与可选传播", "图中第二次前向由 propagate_error 或误差记录触发；下一层输入是否更新由 propagate_error 决定")

    steps = [
        (70, "onload 当前 subgraph", "权重进入计算设备", CPU_BG, CPU),
        (352, "校准前向", "hooks 收集 activation / H", WARN_BG, WARN),
        (634, "SEQ_END event", "消费统计，改写权重", CONTROL_BG, CONTROL),
        (916, "传播前向（可选）", "关闭 hooks，运行压缩层", GPU_BG, GPU),
        (1198, "下一 subgraph", "读取更新后的输入", OK_BG, OK),
    ]
    for x, _, _, _, _ in steps[:-1]:
        s.line(x + 234, 260, x + 280, 260, INK)
    for x, title, desc, fill, accent in steps:
        card(s, x, 205, 234, 120, title, desc, fill, accent)

    s.rect(70, 412, 1362, 164, WHITE, LINE, radius=14)
    s.text(94, 454, "IntermediatesCache · CPU / 指定 offload device", 22, INK, 700)
    s.text(94, 494, "保存 batch 和跨 subgraph 激活；当前层按需 fetch，消费后更新 / 删除", 18, MUTED)
    s.path("M 187 412 V 330", CPU, dashed=True)
    s.path("M 1033 330 V 412", GPU, dashed=True)
    s.text(94, 544, "propagate_error=True：写入压缩后输出    False：写入未压缩输出", 18, MUTED)
    s.text(70, 637, "外层：CALIBRATION_EPOCH_START → 逐 subgraph 处理 → CALIBRATION_EPOCH_END", 19, CONTROL, 650)
    return s


def awq() -> SVG:
    s = SVG(1320, 670, "AWQ reparameterization and quantization", "A channel scale is searched using calibration activations, folded into the preceding smooth layer and following weight, then the latter is quantized.")
    s.header("MECHANISM", "AWQ：先保持函数等价，再改变权重量化误差", "按输入通道缩放；搜索目标是校准样本上的 parent 输出重构误差")

    s.text(75, 193, "原始浮点路径", 19, MUTED, 650)
    card(s, 75, 216, 315, 110, "Smooth layer", "输出 X", WHITE, LINE)
    card(s, 505, 216, 315, 110, "Balance layer", "权重 W", WHITE, LINE)
    card(s, 935, 216, 305, 110, "输出", "Y = X Wᵀ", WHITE, LINE)
    s.line(390, 271, 500, 271, INK)
    s.line(820, 271, 930, 271, INK)

    s.text(75, 411, "AWQ 搜索后", 19, CONTROL, 700)
    card(s, 75, 434, 315, 110, "Smooth layer ÷ s", "输出 X / s", CONTROL_BG, CONTROL)
    card(s, 505, 434, 315, 110, "Balance layer × s", "权重 W · s", CPU_BG, CPU)
    card(s, 935, 434, 305, 110, "再量化权重", "Ŷ = (X / s) Q(W · s)ᵀ", OK_BG, OK)
    s.line(390, 489, 500, 489, CONTROL)
    s.line(820, 489, 930, 489, CPU)
    s.text(75, 608, "浮点时 (X / s)(W · s)ᵀ = XWᵀ；s 融入相邻参数，运行时无需单独执行 AWQ 搜索。", 19, INK, 550)
    return s


FIGURES = {
    "llm-compressor-architecture.svg": architecture,
    "llm-compressor-sequential.svg": sequential,
    "llm-compressor-awq.svg": awq,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "content" / "llm_inference" / "img")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in FIGURES.items():
        (args.output_dir / name).write_text(factory().finish(), encoding="utf-8")


if __name__ == "__main__":
    main()
