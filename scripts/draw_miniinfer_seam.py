#!/usr/bin/env python3
"""Generate the MiniInfer Rust-seam article's deterministic SVG diagrams.

Only the standard library is used.  The published SVGs are the artifacts; this
script is their source of truth, so every label here must stay verifiable
against the repository snapshot named in the article.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from draw_hicache_figures import (
    CONTROL,
    CONTROL_BG,
    CPU,
    CPU_BG,
    GPU,
    GPU_BG,
    INK,
    LINE,
    MONO,
    MUTED,
    OK,
    OK_BG,
    SVG,
    WARN,
    WARN_BG,
    WHITE,
)


def box(
    s: SVG,
    x: int,
    y: int,
    w: int,
    h: int,
    lines: list[str],
    fill: str = WHITE,
    stroke: str = LINE,
    size: int = 18,
    weight: int = 600,
    gap: int = 24,
    shadow: bool = False,
) -> None:
    s.rect(x, y, w, h, fill, stroke, radius=12, stroke_width=2, shadow=shadow)
    start = y + h // 2 - ((len(lines) - 1) * gap) // 2 + 6
    s.multiline(x + 18, start, lines, size, INK, weight, "start", gap)


def title_line(s: SVG, x: int, y: int, w: int, text: str, color: str) -> None:
    s.line(x, y, x + w, y, color, 2, arrow=False)


def figure_architecture() -> SVG:
    s = SVG(
        1560,
        970,
        "MiniInfer Rust seam architecture",
        "Rust decides the plan and the page ledger; the Python data plane owns CUDA, the future input map and the release acknowledgements.",
    )
    s.header(
        "SYSTEM ARCHITECTURE",
        "一条接缝，两侧各自的权威",
        "紫色：Rust 控制面；黄色：仅主机内存的 pinned 接缝；蓝/绿：持有 CUDA 的数据面",
    )

    # --- Rust control plane band -------------------------------------------------
    s.rect(64, 150, 1432, 236, CONTROL_BG, CONTROL, radius=18, stroke_width=2, shadow=True)
    s.text(88, 186, "Rust 控制面 · 不创建 CUDA context", 21, CONTROL, 750)
    s.text(88, 214, "miniinfer-engine/src · scheduler.rs / fsm.rs / kv.rs / py.rs", 15, MUTED, 500, family=MONO)

    rust_boxes = [
        (96, 240, 300, [
            "Scheduler::step",
            "退休只由步数决定（不读 token 值）",
            "scheduler.rs:196",
        ], CONTROL),
        (420, 240, 320, [
            "KvPool 页账本",
            "free / staged / fenced / frontier",
            "kv.rs:158  note_device_fence",
        ], CONTROL),
        (764, 240, 300, [
            "PacketArena 写包",
            "Rust 直写 pinned 字节",
            "arena.rs · py.rs",
        ], CONTROL),
        (1088, 240, 380, [
            "StepResult",
            "batch_id · free_pages · retired · preempted",
            "py.rs:32",
        ], CONTROL),
    ]
    for x, y, w, lines, accent in rust_boxes:
        box(s, x, y, w, 118, lines, WHITE, accent, size=17, gap=26)

    # --- Pinned seam band --------------------------------------------------------
    s.rect(64, 420, 1432, 196, WARN_BG, WARN, radius=18, stroke_width=2, shadow=True)
    s.text(88, 456, "共享 pinned 接缝 · 只有主机字节、值、页号", 21, WARN, 750)
    s.text(88, 484, "每侧各持有 depth + 1 份槽位；重写前必须等对应拷贝完成", 15, MUTED, 500)

    seam_boxes = [
        (96, 508, 430, [
            "Arena 环（depth + 1 份）",
            "plan 写入 → H2D 完成前禁止重写",
            "_acquire_arena() · rust_control_plane.py:287",
        ], WARN),
        (560, 508, 440, [
            "pinned staging（每槽一套）",
            "input_ids / positions / out_cache_loc",
            "seq_lens / block_table",
        ], WARN),
        (1034, 508, 434, [
            "ResultSlot 环（depth + 1 份）",
            "pinned int64 + CUDA event",
            "commit 才 synchronize / query",
        ], WARN),
    ]
    for x, y, w, lines, accent in seam_boxes:
        box(s, x, y, w, 92, lines, WHITE, accent, size=17, gap=24)

    # --- Python data plane band --------------------------------------------------
    s.rect(64, 650, 1432, 226, CPU_BG, CPU, radius=18, stroke_width=2, shadow=True)
    s.text(88, 686, "Python / PyTorch 数据面 · 独占 CUDA context 与全部显存", 21, CPU, 750)
    s.text(88, 714, "miniinfer/engine/rust_control_plane.py", 15, MUTED, 500, family=MONO)

    py_boxes = [
        (96, 738, 330, [
            "RustControlPlane",
            "plan → launch → commit",
            "不做任何调度决策",
        ], CPU),
        (462, 738, 330, [
            "ModelRunner.forward",
            "读 native_block_table 与 KV buffer",
            "eager：内部仍同步",
        ], GPU),
        (828, 738, 330, [
            "future_input_map",
            "device int64[slots]",
            "scatter N → gather N+1",
        ], GPU),
        (1194, 738, 274, [
            "Sampler → 结果槽",
            "采样 + 异步 D2H",
            "record(copy_event)",
        ], GPU),
    ]
    for x, y, w, lines, accent in py_boxes:
        box(s, x, y, w, 118, lines, WHITE, accent, size=17, gap=26)

    # --- corridors: Rust -> Python decisions, Python -> Rust acknowledgements ----
    s.path("M 1462 390 L 1462 650", WARN, 3, True, False)
    s.path("M 1512 650 L 1512 390", OK, 3, True, True)

    s.line(96, 916, 156, 916, WARN, 3, arrow=True)
    s.text(172, 922, "CacheAction::FreeDeviceKV{indices}：Rust 只发决策，页由数据面在设备侧释放", 16, WARN, 600)
    s.line(96, 950, 156, 950, OK, 3, arrow=True, dashed=True)
    s.text(172, 956, "device_fence(batch_id)：数据面回执“该批设备工作已完成”，ledger 才把 staged 页放回 free list", 16, OK, 600)

    return s


def figure_pipeline() -> SVG:
    s = SVG(
        1620,
        1040,
        "MiniInfer depth-1 overlap schedule",
        "Step N+1 is planned and launched before step N's tokens come back; the only blocking point is commit, and the release fence advances with it.",
    )
    s.header(
        "RUNTIME SCHEDULE · in_flight_depth = 1",
        "先 launch 下一步，再读回上一步的 token",
        "横向是时间；同一 CUDA stream 内 h2d → forward → sample → scatter 顺序执行。示意图，不按比例",
    )

    x0, x1 = 260, 1560
    lanes = [
        (250, "CPU · 控制面", CONTROL, CONTROL_BG),
        (430, "CUDA stream", GPU, GPU_BG),
        (620, "event / 页栅栏", OK, OK_BG),
    ]
    for y, label, accent, fill in lanes:
        s.rect(64, y, x1 - 64 + 40, 120, fill, accent, radius=14, stroke_width=1)
        s.text(84, y + 34, label, 18, accent, 750)
        s.line(x0, y + 96, x1, y + 96, LINE, 2, arrow=False)

    def bar(x: int, y: int, w: int, label: str, accent: str, fill: str,
            sub: str | None = None, size: int = 15) -> None:
        s.rect(x, y, w, 56, fill, accent, radius=10, stroke_width=2)
        s.text(x + w // 2, y + (26 if sub else 34), label, size, accent, 700, "middle")
        if sub:
            s.text(x + w // 2, y + 46, sub, 12, MUTED, 500, "middle", MONO)

    # Event 的 x 一律由它所标记的条形边界推导，避免“事件和动作对不上”。
    # (key, 显示文本, x, w)：key 唯一，短标签可以重复。
    cpu = [
        ("plan N", "plan N", 280, 108),
        ("launch N", "launch N", 392, 118),
        ("plan N+1", "plan N+1", 534, 118),
        ("launch N+1", "launch N+1", 656, 128),
        ("commit N", "commit N", 816, 148),
        ("plan N+2", "plan N+2", 972, 118),
        ("launch N+2", "launch N+2", 1094, 128),
        ("commit N+1", "commit N+1", 1254, 148),
        ("plan N+3", "plan N+3", 1410, 120),
    ]
    gpu = [
        ("h2d N", "h2d", 392, 60),
        ("forward N", "forward N", 456, 240),
        ("sample N", "sample", 700, 60),
        ("scatter N", "scatter", 764, 60),
        ("h2d N+1", "h2d", 828, 60),
        ("forward N+1", "forward N+1", 892, 240),
        ("sample N+1", "sample", 1136, 60),
        ("scatter N+1", "scatter", 1200, 60),
        ("h2d N+2", "h2d", 1264, 60),
        ("forward N+2", "forward N+2", 1328, 232),
    ]
    bar_end = {key: x + w for key, _, x, w in cpu + gpu}
    bar_start = {key: x for key, _, x, w in cpu + gpu}

    for key, label, x, w in cpu:
        if key.startswith("commit"):
            bar(x, 272, w, label, WARN, WARN_BG, "等 result")
        else:
            bar(x, 272, w, label, CONTROL, WHITE)
    for key, label, x, w in gpu:
        small = not key.startswith("forward")
        bar(x, 452, w, label, OK if small else GPU, OK_BG if small else WHITE,
            size=13 if small else 15)

    # 三个事件：h2d 完成（gate 槽复用）、result 就绪（gate commit 返回）、device_fence（页回 free list）。
    def tick(x: int, label: str, accent: str, above: bool = True) -> None:
        s.line(x, 644, x, 672, accent, 3, arrow=False)
        s.text(x, 628 if above else 696, label, 13, accent, 650, "middle", MONO)

    tick(bar_end["h2d N"], "h2d N done", WARN)
    tick(bar_end["scatter N"], "result N ready", OK)
    tick(bar_end["commit N"], "device_fence(N)", WARN, above=False)
    tick(bar_end["scatter N+1"], "result N+1 ready", OK)

    # commit 的等待段：CPU 到达 → result 就绪；并用虚线导引到 event 行。
    for ready in ("scatter N", "scatter N+1"):
        s.line(bar_end[ready], 272, bar_end[ready], 328, OK, 2, arrow=False, dashed=True)
        s.path(f"M {bar_end[ready]} 676 L {bar_end[ready]} 338", OK, 2, True, True)

    # 槽复用的依赖：h2d N done → plan N+2；连线停在 stream 带上方，避免穿过条形。
    s.path(
        f"M {bar_end['h2d N']} 676 L {bar_end['h2d N']} 700 "
        f"L {bar_start['plan N+2'] + 40} 700 L {bar_start['plan N+2'] + 40} 432",
        MUTED, 2, True, True,
    )

    s.text(84, 582, "同一条 stream 的顺序：h2d N → forward N → sample N → scatter N（写 future_map）"
                    "→ h2d N+1 → forward N+1（gather 读回）→ …", 14, MUTED, 500)
    s.text(84, 714, "arena 环：plan N+2 复用槽 N 前需 h2d N 事件置位；图中早已置位，没有等待。",
           13, MUTED, 500)
    s.text(816, 714, "commit N：816 到达 → 等 result N → 964 返回并回执 device_fence(N)",
           13, MUTED, 500)

    # Release fence movement.
    s.rect(280, 760, 620, 96, OK_BG, OK, radius=12, stroke_width=2)
    s.text(300, 792, "页释放栅栏", 17, OK, 750)
    s.text(300, 820, "plan N+1 期间驱逐的页挂在 frontier N；commit N 调 device_fence(N) 后它们才回到 free list", 13, MUTED, 500)
    s.text(300, 842, "kv.rs:158 note_device_fence → release_now → generation++", 13, MUTED, 500, family=MONO)

    s.rect(940, 760, 590, 96, WARN_BG, WARN, radius=12, stroke_width=2)
    s.text(960, 792, "页不足时的反应路径", 17, WARN, 750)
    s.text(960, 820, "Engine.step 抛 KvOutOfPages → 驱动先 commit 最老的一批（推进栅栏）→ 重试 plan", 13, MUTED, 500)
    s.text(960, 842, "_drain_pipelined · rust_control_plane.py:543", 13, MUTED, 500, family=MONO)

    # Legend.
    s.text(84, 900, "已发出（issue）", 15, MUTED, 500)
    s.rect(214, 886, 40, 18, WHITE, CONTROL, radius=6, stroke_width=2)
    s.text(268, 900, "CPU 控制面动作", 15, MUTED, 500)
    s.rect(430, 886, 40, 18, WHITE, GPU, radius=6, stroke_width=2)
    s.text(484, 900, "GPU 计算", 15, MUTED, 500)
    s.rect(600, 886, 40, 18, OK_BG, OK, radius=6, stroke_width=2)
    s.text(654, 900, "future_map / 栅栏事件", 15, MUTED, 500)
    s.rect(856, 886, 40, 18, WARN_BG, WARN, radius=6, stroke_width=2)
    s.text(910, 900, "阻塞等待", 15, MUTED, 500)

    s.text(84, 966, "读法：CPU 在 forward N 还在执行时就完成了 plan N+1 并发出 launch N+1；", 16, INK, 600)
    s.text(84, 996, "commit N 只等 result N，一旦拿到就回执页栅栏。event 行只画 gate 了可见动作的事件。", 16, INK, 600)
    return s


FIGURES = {
    "miniinfer-seam-architecture.svg": figure_architecture,
    "miniinfer-seam-pipeline.svg": figure_pipeline,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "content/llm_inference/img",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in FIGURES.items():
        svg = factory()
        path = args.output_dir / name
        path.write_text(svg.finish(), encoding="utf-8", newline="\n")
        print(f"{path}: {svg.width} x {svg.height}")


if __name__ == "__main__":
    main()
