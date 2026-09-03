#!/usr/bin/env python3
"""Generate the HiCache article's deterministic SVG diagrams.

The script intentionally uses only the Python standard library.  The generated
SVG files are the published artifacts; this script remains their source of truth.
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path


FONT = "Inter, Noto Sans SC, Microsoft YaHei, PingFang SC, sans-serif"
MONO = "JetBrains Mono, Cascadia Code, Consolas, monospace"

INK = "#172033"
MUTED = "#637083"
LINE = "#CAD3E1"
BG = "#F8FAFD"
WHITE = "#FFFFFF"
GPU = "#E45756"
GPU_BG = "#FDE9E7"
CPU = "#2F6FE4"
CPU_BG = "#E8F0FF"
STORE = "#138A7E"
STORE_BG = "#E2F5F1"
CONTROL = "#7656D8"
CONTROL_BG = "#EEE9FF"
WARN = "#C47B10"
WARN_BG = "#FFF2D6"
OK = "#2E8B57"
OK_BG = "#E4F4EA"
GREY_BG = "#EEF2F7"


@dataclass
class SVG:
    width: int
    height: int
    title: str
    desc: str

    def __post_init__(self) -> None:
        self.parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{html.escape(self.title)}</title>",
            f"<desc id=\"desc\">{html.escape(self.desc)}</desc>",
            "<defs>",
            f'<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{INK}"/></marker>',
            f'<marker id="arrow-muted" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{MUTED}"/></marker>',
            f'<marker id="arrow-gpu" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{GPU}"/></marker>',
            f'<marker id="arrow-cpu" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{CPU}"/></marker>',
            f'<marker id="arrow-store" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{STORE}"/></marker>',
            f'<marker id="arrow-control" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{CONTROL}"/></marker>',
            f'<marker id="arrow-ok" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 Z" fill="{OK}"/></marker>',
            f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="4" stdDeviation="7" flood-color="#172033" flood-opacity="0.09"/></filter>',
            "</defs>",
            f'<rect width="{self.width}" height="{self.height}" fill="{BG}"/>',
        ]

    def add(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x: int, y: int, w: int, h: int, fill: str = WHITE, stroke: str = LINE,
             radius: int = 18, stroke_width: int = 2, dashed: bool = False,
             shadow: bool = False, opacity: float = 1.0) -> None:
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        filt = ' filter="url(#shadow)"' if shadow else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"{dash}{filt}/>'
        )

    def line(self, x1: int, y1: int, x2: int, y2: int, color: str = INK,
             width: int = 3, arrow: bool = True, dashed: bool = False) -> None:
        marker_id = {
            INK: "arrow", MUTED: "arrow-muted", GPU: "arrow-gpu", CPU: "arrow-cpu",
            STORE: "arrow-store", CONTROL: "arrow-control", OK: "arrow-ok",
        }.get(color)
        marker = f' marker-end="url(#{marker_id})"' if arrow and marker_id else ""
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round"{dash}{marker}/>'
        )

    def path(self, d: str, color: str = INK, width: int = 3,
             arrow: bool = True, dashed: bool = False, fill: str = "none") -> None:
        marker_id = {
            INK: "arrow", MUTED: "arrow-muted", GPU: "arrow-gpu", CPU: "arrow-cpu",
            STORE: "arrow-store", CONTROL: "arrow-control", OK: "arrow-ok",
        }.get(color)
        marker = f' marker-end="url(#{marker_id})"' if arrow and marker_id else ""
        dash = ' stroke-dasharray="8 7"' if dashed else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{dash}{marker}/>'
        )

    def text(self, x: int, y: int, value: str, size: int = 22, color: str = INK,
             weight: int = 500, anchor: str = "start", family: str = FONT) -> None:
        self.add(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(value)}</text>'
        )

    def multiline(self, x: int, y: int, lines: list[str], size: int = 20,
                  color: str = INK, weight: int = 500, anchor: str = "start",
                  gap: int = 30) -> None:
        safe = [html.escape(line) for line in lines]
        spans = "".join(
            f'<tspan x="{x}" dy="{0 if i == 0 else gap}">{line}</tspan>'
            for i, line in enumerate(safe)
        )
        self.add(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{spans}</text>'
        )

    def circle(self, cx: int, cy: int, r: int, fill: str, stroke: str = WHITE,
               stroke_width: int = 3) -> None:
        self.add(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def header(self, kicker: str, title: str, subtitle: str | None = None) -> None:
        self.text(64, 46, kicker.upper(), 16, CONTROL, 700)
        self.text(64, 83, title, 30, INK, 750)
        if subtitle:
            self.text(64, 114, subtitle, 18, MUTED, 450)

    def finish(self) -> str:
        return "\n".join([*self.parts, "</svg>", ""])


def pill(svg: SVG, x: int, y: int, w: int, label: str, fill: str, color: str) -> None:
    svg.rect(x, y, w, 36, fill, fill, radius=18, stroke_width=0)
    svg.text(x + w // 2, y + 25, label, 16, color, 700, "middle")


def step(svg: SVG, x: int, y: int, w: int, h: int, number: str, title: str,
         subtitle: str, fill: str, accent: str) -> None:
    svg.rect(x, y, w, h, fill, accent, radius=18, shadow=True)
    svg.circle(x + 32, y + 31, 18, accent)
    svg.text(x + 32, y + 38, number, 15, WHITE, 750, "middle")
    svg.text(x + 60, y + 36, title, 21, INK, 700)
    svg.text(x + 60, y + 68, subtitle, 17, MUTED, 450)


def figure_layers() -> SVG:
    s = SVG(1200, 660, "HiCache three-tier architecture", "HiRadixTree tracks prefix metadata while KV data moves among GPU, host memory and storage.")
    s.header("ARCHITECTURE", "一棵前缀树，三级 KV 数据", "树回答“在哪里”；数据层回答“怎么搬”")
    s.rect(64, 155, 300, 430, CONTROL_BG, CONTROL, shadow=True)
    pill(s, 92, 184, 122, "控制面", CONTROL_BG, CONTROL)
    s.text(92, 252, "HiRadixTree", 28, INK, 750)
    s.multiline(92, 305, ["Prefix", "State", "Index / Hash"], 22, INK, 600, gap=54)
    s.line(250, 300, 250, 430, CONTROL, 3, False)
    for cy, color in [(300, GPU), (365, CPU), (430, STORE)]:
        s.circle(250, cy, 9, color, color, 0)
    s.text(92, 530, "不复制 KV", 18, MUTED, 500)
    s.text(92, 557, "只维护位置与状态", 18, MUTED, 500)

    tiers = [
        (452, 156, GPU_BG, GPU, "L1 · GPU HBM", "命中即复用", "最低延迟"),
        (452, 310, CPU_BG, CPU, "L2 · Host DRAM", "扩展容量", "Pinned"),
        (452, 464, STORE_BG, STORE, "L3 · Storage", "跨实例共享", "RDMA / FS"),
    ]
    for x, y, fill, accent, title, main, tail in tiers:
        s.rect(x, y, 660, 112, fill, accent, shadow=True)
        s.circle(x + 42, y + 56, 19, accent)
        s.text(x + 78, y + 44, title, 23, INK, 700)
        s.text(x + 78, y + 78, main, 19, MUTED, 500)
        pill(s, x + 500, y + 38, 126, tail, WHITE, accent)
    s.path("M 364 275 C 405 275, 405 212, 452 212", CONTROL, 3, True, True)
    s.path("M 364 365 C 410 365, 410 366, 452 366", CONTROL, 3, True, True)
    s.path("M 364 455 C 405 455, 405 520, 452 520", CONTROL, 3, True, True)
    s.line(780, 268, 780, 304, MUTED, 3)
    s.text(804, 291, "H2D / D2H", 16, MUTED, 600)
    s.line(780, 422, 780, 458, MUTED, 3)
    s.text(804, 445, "get / set", 16, MUTED, 600)
    return s


def figure_decoupling() -> SVG:
    s = SVG(1200, 760, "Decoupling HiCache CPU control work", "Before, CPU preparation and submission elongate the scheduler critical path. After, workers prepare and submit transfers while scheduling continues.")
    s.header("MOTIVATION", "异步 CUDA，不等于异步 Scheduler", "瓶颈常在 DMA 之前的 CPU 控制面")
    s.text(64, 165, "BEFORE", 17, GPU, 750)
    s.rect(64, 184, 1072, 198, GPU_BG, GPU, radius=22)
    s.text(88, 222, "Scheduler", 18, INK, 700)
    before = [
        (210, 206, 150, "match", CONTROL_BG, CONTROL),
        (390, 206, 195, "索引 / 分配", WARN_BG, WARN),
        (615, 206, 175, "提交 DMA", WARN_BG, WARN),
        (820, 206, 180, "event / commit", WARN_BG, WARN),
    ]
    for x, y, w, label, fill, accent in before:
        s.rect(x, y, w, 58, fill, accent, radius=12)
        s.text(x + w // 2, y + 37, label, 18, INK, 650, "middle")
    for x1, x2 in [(360, 390), (585, 615), (790, 820)]:
        s.line(x1, 235, x2, 235, GPU, 3)
    s.rect(210, 304, 790, 42, WHITE, LINE, radius=10)
    s.rect(210, 304, 610, 42, "#F7C9C5", "#F7C9C5", radius=10, stroke_width=0)
    s.text(88, 333, "GPU", 18, INK, 700)
    s.text(512, 332, "bubble", 17, GPU, 700, "middle")
    s.text(910, 332, "forward", 17, OK, 700, "middle")
    pill(s, 944, 144, 166, "队首阻塞", GPU_BG, GPU)

    s.text(64, 443, "AFTER", 17, OK, 750)
    s.rect(64, 462, 1072, 238, OK_BG, OK, radius=22)
    lanes = [(512, "Scheduler"), (584, "CPU Worker"), (656, "DMA / GPU")]
    for y, label in lanes:
        s.text(88, y, label, 18, INK, 700)
        s.line(205, y - 8, 1094, y - 8, LINE, 2, False)
    blocks = [
        (222, 482, 132, 42, "match", CONTROL_BG, CONTROL),
        (380, 482, 130, 42, "enqueue", CPU_BG, CPU),
        (538, 482, 170, 42, "next batch", WHITE, OK),
        (538, 554, 190, 42, "索引 / 合并", CPU_BG, CPU),
        (754, 554, 140, 42, "submit", CPU_BG, CPU),
        (754, 626, 244, 42, "copy  ∥  forward", WHITE, OK),
    ]
    for x, y, w, h, label, fill, accent in blocks:
        s.rect(x, y, w, h, fill, accent, radius=10)
        s.text(x + w // 2, y + 28, label, 17, INK, 650, "middle")
    s.line(354, 503, 380, 503, OK, 3, True)
    s.line(510, 503, 538, 503, OK, 3, True)
    s.path("M 445 524 L 445 575 L 538 575", MUTED, 2, True, True)
    s.line(728, 575, 754, 575, MUTED, 3)
    s.line(824, 596, 824, 626, MUTED, 3)
    s.path("M 998 647 C 1045 647, 1045 503, 930 503", MUTED, 2, True, True)
    pill(s, 924, 422, 186, "Scheduler 早返回", OK_BG, OK)
    return s


def figure_controller() -> SVG:
    s = SVG(1200, 700, "HiCache ownership boundaries", "The scheduler owns tree state, bounded queues carry operations, and workers own preparation and transfer submission.")
    s.header("OWNERSHIP", "后台线程搬数据，Scheduler 提交状态", "把并发边界放在 Operation / Ack 上")
    s.rect(64, 160, 310, 460, CONTROL_BG, CONTROL, shadow=True)
    pill(s, 90, 184, 136, "Scheduler", CONTROL_BG, CONTROL)
    s.multiline(90, 260, ["匹配 Prefix", "预留 Slot", "锁定 Node", "轮询 Ack", "Commit / Free"], 21, INK, 600, gap=58)
    s.text(90, 575, "唯一树状态写者", 18, CONTROL, 700)

    s.rect(446, 160, 286, 460, GREY_BG, LINE, shadow=True)
    pill(s, 472, 184, 124, "边界", GREY_BG, MUTED)
    queue_items = [
        (472, 250, "H2D Queue", CPU),
        (472, 322, "D2H Queue", GPU),
        (472, 394, "L3 Queue", STORE),
        (472, 488, "Ack Queue", CONTROL),
    ]
    for x, y, label, accent in queue_items:
        s.rect(x, y, 234, 48, WHITE, accent, radius=12)
        s.text(x + 117, y + 31, label, 18, INK, 650, "middle")
    s.text(589, 575, "有界 · 可批量 · 可优先级", 17, MUTED, 600, "middle")

    s.rect(804, 160, 332, 460, CPU_BG, CPU, shadow=True)
    pill(s, 830, 184, 146, "CPU Workers", CPU_BG, CPU)
    worker_items = [
        (830, 250, "索引整理", "cat / sort / clone"),
        (830, 335, "批量提交", "CUDA stream / I/O"),
        (830, 420, "完成通知", "event → ack"),
    ]
    for x, y, title, sub in worker_items:
        s.rect(x, y, 280, 64, WHITE, CPU, radius=12)
        s.text(x + 18, y + 27, title, 18, INK, 700)
        s.text(x + 18, y + 50, sub, 15, MUTED, 500, family=MONO)
    s.text(970, 575, "不直接改 Radix Tree", 18, CPU, 700, "middle")

    s.line(374, 298, 446, 274, INK, 3)
    s.line(732, 274, 804, 282, INK, 3)
    s.line(804, 512, 732, 512, MUTED, 3)
    s.line(446, 512, 374, 512, MUTED, 3)
    return s


def figure_read() -> SVG:
    s = SVG(1200, 760, "HiCache read pipeline", "Storage prefetch, host-to-device transfer and layer-wise forward execution overlap across scheduler, workers and GPU lanes.")
    s.header("READ PATH", "L3 → L2 → L1：两段流水线", "预取先落 Host；进入 Batch 后再回 GPU")
    lane_y = [190, 310, 430, 550]
    labels = ["Scheduler", "L3 Worker", "H2D Worker", "GPU"]
    accents = [CONTROL, STORE, CPU, GPU]
    for y, label, accent in zip(lane_y, labels, accents):
        pill(s, 64, y - 20, 142, label, WHITE, accent)
        s.line(232, y, 1128, y, LINE, 2, False)
    s.text(1115, 145, "time →", 16, MUTED, 600, "end")

    blocks = [
        (250, 165, 150, "match", CONTROL_BG, CONTROL),
        (420, 165, 138, "prefetch", CONTROL_BG, CONTROL),
        (650, 165, 110, "poll", CONTROL_BG, CONTROL),
        (785, 165, 132, "load", CONTROL_BG, CONTROL),
        (1025, 165, 92, "ack", CONTROL_BG, CONTROL),
        (430, 285, 165, "exists", STORE_BG, STORE),
        (610, 285, 210, "get → L2", STORE_BG, STORE),
        (795, 405, 160, "index", CPU_BG, CPU),
        (970, 405, 148, "H2D", CPU_BG, CPU),
        (1000, 525, 116, "forward", GPU_BG, GPU),
    ]
    for x, y, w, label, fill, accent in blocks:
        s.rect(x, y, w, 50, fill, accent, radius=11)
        s.text(x + w // 2, y + 33, label, 18, INK, 650, "middle")
    s.path("M 489 215 L 489 285", MUTED, 2, True, True)
    s.path("M 760 310 L 700 190", MUTED, 2, True, True)
    s.path("M 851 215 L 851 405", MUTED, 2, True, True)
    s.path("M 1044 455 L 1044 525", MUTED, 2, True, True)
    s.path("M 1100 550 C 1140 550, 1140 190, 1117 190", MUTED, 2, True, True)
    s.rect(610, 605, 506, 72, WHITE, LINE, radius=14, dashed=True)
    s.text(863, 635, "Layer 0 ready → Forward 先启动", 19, OK, 700, "middle")
    s.text(863, 661, "后续 Layer 继续搬", 17, MUTED, 500, "middle")
    s.path("M 1038 525 C 1010 505, 1010 475, 1016 455", GPU, 3, False, True)
    return s


def figure_write() -> SVG:
    s = SVG(1200, 680, "HiCache write pipeline", "The scheduler publishes a write operation, D2H and storage workers perform transfers, and acknowledgements return for commit and release.")
    s.header("WRITE PATH", "L1 → L2 → L3：非关键写入向后移", "write-through 可流水化；write-back 释放 GPU Slot 前必须完成 D2H")
    stages = [
        (64, 180, 220, 120, "1", "Scheduler", "reserve + lock", CONTROL_BG, CONTROL),
        (332, 180, 220, 120, "2", "D2H Worker", "merge + submit", GPU_BG, GPU),
        (600, 180, 220, 120, "3", "L2 Host", "Pinned pool", CPU_BG, CPU),
        (868, 180, 268, 120, "4", "Backup Worker", "batch_set → L3", STORE_BG, STORE),
    ]
    for args in stages:
        step(s, *args)
    for x1, x2 in [(284, 332), (552, 600), (820, 868)]:
        s.line(x1, 240, x2, 240, INK, 3)
    s.rect(64, 380, 1072, 190, WHITE, LINE, radius=22)
    s.text(92, 420, "完成路径", 20, INK, 700)
    s.line(202, 450, 1010, 450, MUTED, 3)
    for cx, title, sub, color in [
        (260, "D2H done", "Ack", GPU),
        (520, "L2 可复用", "node.backuped", CPU),
        (780, "L3 done", "Ack", STORE),
        (1010, "Commit", "unlock / free", CONTROL),
    ]:
        s.circle(cx, 450, 13, color, color, 0)
        s.text(cx, 493, title, 18, INK, 700, "middle")
        s.text(cx, 520, sub, 15, MUTED, 500, "middle", MONO)
    pill(s, 88, 596, 176, "write-through", OK_BG, OK)
    s.text(286, 620, "Scheduler 不等 L3", 18, MUTED, 600)
    pill(s, 690, 596, 156, "write-back", WARN_BG, WARN)
    s.text(870, 620, "D2H 后才能回收源 Slot", 18, WARN, 650)
    return s


def figure_state() -> SVG:
    s = SVG(1200, 680, "HiCache node state transitions", "A node transitions among miss, GPU-only, GPU plus host backup, host-only tombstone and removed states.")
    s.header("STATE", "一个 TreeNode 的分层状态", "树结构保留 Prefix；value / host_value 决定驻留层级")
    nodes = [
        (64, 230, 180, 100, GREY_BG, MUTED, "MISS", "无数据"),
        (310, 230, 220, 100, GPU_BG, GPU, "GPU only", "value"),
        (596, 230, 236, 100, "#F1ECFF", CONTROL, "GPU + L2", "value + host"),
        (898, 230, 238, 100, CPU_BG, CPU, "L2 only", "tombstone"),
    ]
    for x, y, w, h, fill, accent, title, sub in nodes:
        s.rect(x, y, w, h, fill, accent, shadow=True)
        s.text(x + w // 2, y + 43, title, 22, INK, 700, "middle")
        s.text(x + w // 2, y + 74, sub, 16, MUTED, 500, "middle", MONO)
    for x1, x2, label, color in [
        (244, 310, "insert", MUTED),
        (530, 596, "backup", MUTED),
        (832, 898, "evict L1", MUTED),
    ]:
        s.line(x1, 280, x2, 280, color, 3)
        s.text((x1 + x2) // 2, 257, label, 15, MUTED, 600, "middle")
    s.path("M 1017 330 C 1017 410, 714 410, 714 330", CPU, 3, True)
    s.text(866, 440, "load back", 16, CPU, 650, "middle")
    s.rect(465, 492, 270, 92, WHITE, LINE, radius=16, dashed=True)
    s.text(600, 530, "REMOVED", 21, INK, 700, "middle")
    s.text(600, 559, "L2 evict", 16, MUTED, 500, "middle")
    s.path("M 1017 330 C 1017 520, 790 538, 735 538", MUTED, 3, True)
    s.path("M 420 330 C 420 460, 465 500, 505 513", MUTED, 2, True, True)
    pill(s, 64, 594, 230, "父先备份，子才能备份", WARN_BG, WARN)
    pill(s, 906, 594, 230, "从最深叶子开始驱逐", GREY_BG, MUTED)
    return s


def figure_host_memory() -> SVG:
    s = SVG(1200, 720, "NUMA-aware pinned hugepage initialization", "Bind the worker and memory to the GPU-local NUMA node, allocate huge pages, prefault them, and register the pool once.")
    s.header("HOST MEMORY", "Pinned + HugeTLB + Prefault + NUMA", "四件事各自解决不同问题，顺序也重要")
    xs = [64, 286, 508, 730, 952]
    items = [
        ("1", "定位 GPU", "NUMA node", CONTROL_BG, CONTROL),
        ("2", "绑定", "CPU + memory", CONTROL_BG, CONTROL),
        ("3", "分配", "HugeTLB", WARN_BG, WARN),
        ("4", "预触页", "first-touch", CPU_BG, CPU),
        ("5", "注册一次", "Pinned / MR", STORE_BG, STORE),
    ]
    for x, (n, title, sub, fill, accent) in zip(xs, items):
        step(s, x, 178, 184, 112, n, title, sub, fill, accent)
    for x1, x2 in zip([248, 470, 692, 914], [286, 508, 730, 952]):
        s.line(x1, 234, x2, 234, INK, 3)

    cards = [
        (64, 370, 248, CPU_BG, CPU, "Pinned", "真正异步 DMA", "避免临时 staging"),
        (338, 370, 248, WARN_BG, WARN, "HugeTLB", "更少 TLB / 映射", "2 MiB ≠ 自动 Pinned"),
        (612, 370, 248, CONTROL_BG, CONTROL, "Prefault", "启动期支付缺页", "压低首轮抖动"),
        (886, 370, 250, STORE_BG, STORE, "NUMA", "本地内存路径", "少一次跨 Socket"),
    ]
    for x, y, w, fill, accent, title, main, sub in cards:
        s.rect(x, y, w, 150, fill, accent, shadow=True)
        s.text(x + 22, y + 38, title, 23, accent, 750)
        s.text(x + 22, y + 82, main, 19, INK, 650)
        s.text(x + 22, y + 116, sub, 16, MUTED, 500)
    s.rect(64, 580, 1072, 70, WHITE, LINE, radius=16, dashed=True)
    s.text(600, 610, "HugeTLB 是稀缺、不可换出的资源", 19, WARN, 700, "middle")
    s.text(600, 636, "先预留容量，再按 NUMA 节点验证实际落点", 17, MUTED, 500, "middle")
    return s


def figure_storage() -> SVG:
    s = SVG(1200, 690, "HiCache storage backend interface", "Longest-prefix metadata queries are separate from zero-copy and generic data paths between the host pool and storage backends.")
    s.header("STORAGE", "统一接口，两条数据路径", "控制面查最长前缀；数据面负责 get / set")
    s.rect(64, 170, 1072, 92, CONTROL_BG, CONTROL, shadow=True)
    s.text(92, 207, "控制面", 18, CONTROL, 750)
    s.text(232, 207, "batch_exists(keys)", 21, INK, 700, family=MONO)
    s.text(232, 238, "遇到首个 miss 即停止", 17, MUTED, 500)
    pill(s, 940, 198, 164, "Longest Prefix", WHITE, CONTROL)

    s.rect(64, 324, 300, 250, CPU_BG, CPU, shadow=True)
    s.text(92, 367, "L2 Host Pool", 24, INK, 750)
    s.text(92, 405, "预分配 Pinned Buffer", 18, MUTED, 500)
    s.multiline(92, 467, ["host_indices", "buffer pointer"], 18, CPU, 650, gap=34)

    s.rect(452, 324, 286, 108, STORE_BG, STORE, shadow=True)
    s.text(595, 364, "Zero-copy", 22, INK, 750, "middle")
    s.text(595, 399, "batch_get/set_v1", 17, STORE, 650, "middle", MONO)
    s.rect(452, 466, 286, 108, GREY_BG, MUTED, shadow=True)
    s.text(595, 506, "Generic", 22, INK, 750, "middle")
    s.text(595, 541, "tensor / block copy", 17, MUTED, 650, "middle", MONO)

    s.rect(826, 324, 310, 250, STORE_BG, STORE, shadow=True)
    s.text(854, 367, "L3 Backends", 24, INK, 750)
    s.multiline(854, 418, ["Mooncake · NIXL", "HF3FS · SiMM", "AIBrix · File"], 18, INK, 600, gap=40)

    s.line(364, 378, 452, 378, STORE, 3)
    s.line(738, 378, 826, 378, STORE, 3)
    s.path("M 364 520 L 452 520", MUTED, 3, True)
    s.path("M 738 520 L 826 520", MUTED, 3, True)
    s.text(600, 637, "v1：Backend 直接读写 L2 地址", 18, STORE, 700, "middle")
    return s


def figure_mempool_layout() -> SVG:
    s = SVG(
        1200,
        650,
        "HiCache host memory layout",
        "GPU KV remains layer-first while host and storage use page-first layout for contiguous page transfer.",
    )
    s.header("MEMORY LAYOUT", "GPU 保持 Layer-first，Host 改为 Page-first", "计算布局不动；搬运布局按 Page 连续")

    s.rect(64, 158, 420, 290, GPU_BG, GPU, shadow=True)
    pill(s, 90, 184, 184, "L1 · GPU", GPU_BG, GPU)
    s.text(90, 253, "Layer-first", 27, INK, 750)
    for row, layer in enumerate(["L0", "L1", "L2"]):
        y = 286 + row * 54
        s.text(92, y + 31, layer, 18, GPU, 750)
        for col, page in enumerate(["A", "B", "C"]):
            x = 146 + col * 92
            s.rect(x, y, 78, 38, WHITE, GPU, radius=8, stroke_width=1)
            s.text(x + 39, y + 25, page, 17, INK, 650, "middle")

    s.rect(716, 158, 420, 290, CPU_BG, CPU, shadow=True)
    pill(s, 742, 184, 204, "L2 · Host", CPU_BG, CPU)
    s.text(742, 253, "Page-first", 27, INK, 750)
    for row, page in enumerate(["A", "B", "C"]):
        y = 286 + row * 54
        s.text(744, y + 31, page, 18, CPU, 750)
        for col, layer in enumerate(["L0", "L1", "L2"]):
            x = 798 + col * 92
            s.rect(x, y, 78, 38, WHITE, CPU, radius=8, stroke_width=1)
            s.text(x + 39, y + 25, layer, 17, INK, 650, "middle")

    s.line(514, 303, 686, 303, CONTROL, 4)
    pill(s, 523, 326, 154, "IO kernel", CONTROL_BG, CONTROL)

    s.rect(64, 506, 250, 80, GPU_BG, GPU, radius=16)
    s.text(189, 555, "L1 · GPU", 22, INK, 750, "middle")
    s.rect(475, 506, 250, 80, CPU_BG, CPU, radius=16)
    s.text(600, 555, "L2 · Host", 22, INK, 750, "middle")
    s.rect(886, 506, 250, 80, STORE_BG, STORE, radius=16)
    s.text(1011, 555, "L3 · Storage", 22, INK, 750, "middle")
    s.line(314, 546, 475, 546, CONTROL, 4)
    s.text(394, 529, "layout transform", 15, CONTROL, 650, "middle")
    s.line(725, 546, 886, 546, STORE, 4)
    s.text(805, 529, "zero-copy page", 15, STORE, 650, "middle")
    return s


FIGURES = {
    "hicache-layers.svg": figure_layers,
    "hicache-thread-decoupling.svg": figure_decoupling,
    "hicache-controller-ownership.svg": figure_controller,
    "hicache-read-pipeline.svg": figure_read,
    "hicache-write-pipeline.svg": figure_write,
    "hicache-node-state.svg": figure_state,
    "hicache-host-memory.svg": figure_host_memory,
    "hicache-storage-backend.svg": figure_storage,
    "hicache-mempool-layout.svg": figure_mempool_layout,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="content/sglang/img",
        help="Directory for generated SVG files (default: content/sglang/img)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in FIGURES.items():
        (output_dir / name).write_text(factory().finish(), encoding="utf-8", newline="\n")
        print(output_dir / name)


if __name__ == "__main__":
    main()
