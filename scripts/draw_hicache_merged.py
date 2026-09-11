#!/usr/bin/env python3
"""Generate editable SVG diagrams for the merged HiCache article.

The figures are reconstructed from the article's textual call flow. This script
never reads or embeds the article's previous image assets.
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
    GREY_BG,
    INK,
    LINE,
    MONO,
    MUTED,
    OK,
    OK_BG,
    STORE,
    STORE_BG,
    SVG,
    WARN,
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
    shadow: bool = False,
) -> None:
    s.rect(x, y, w, h, fill, stroke, radius=12, stroke_width=2, shadow=shadow)
    gap = 25
    start = y + h // 2 - ((len(lines) - 1) * gap) // 2 + 6
    s.multiline(x + w // 2, start, lines, size, INK, weight, "middle", gap)


def tag(s: SVG, x: int, y: int, w: int, text: str, fill: str, color: str) -> None:
    s.rect(x, y, w, 32, fill, color, radius=16, stroke_width=1)
    s.text(x + w // 2, y + 22, text, 14, color, 700, "middle", MONO)


def lane(s: SVG, y: int, label: str, x1: int, x2: int) -> None:
    s.text(42, y + 34, label, 18, INK, 650)
    s.line(x1, y + 42, x2, y + 42, LINE, 1, arrow=False)


def figure_architecture() -> SVG:
    s = SVG(
        2100,
        1710,
        "HiCache component architecture",
        "Scheduler owns queue state, allocates L2 host pages after an L3 hit, publishes completed prefetch data as an L2 prefix, and re-matches waiting requests.",
    )
    s.header(
        "SYSTEM ARCHITECTURE",
        "HiCache：一棵逻辑前缀树，三层物理驻留",
        "紫色虚线是控制与状态提交；红、蓝、绿色实线是 KV 数据路径",
    )

    # Connectors are drawn before component faces.
    s.path("M 500 292 L 620 292", CONTROL, 3, True, True)
    s.path("M 1040 292 L 1160 292", CONTROL, 3, True, True)
    s.path("M 1370 415 L 1370 515", CPU, 4)
    s.path("M 1540 610 L 1690 610", STORE, 4)
    s.path("M 1160 330 L 1090 330 L 1090 610 L 860 610", GPU, 4)
    s.path("M 1160 365 L 1130 365 L 1130 745 L 860 745", CPU, 4)
    s.path("M 1160 292 L 1070 292 L 1070 835 L 1690 835", STORE, 3, True, True)
    s.path("M 620 340 L 570 340 L 570 845 L 1160 845", CONTROL, 2, True, True)

    box(
        s,
        70,
        180,
        430,
        225,
        [
            "Scheduler",
            "waiting_queue / ongoing_prefetch",
            "L3 hit → alloc L2 host pages",
            "safe-point publish / re-match",
        ],
        CONTROL_BG,
        CONTROL,
        size=17,
        shadow=True,
    )
    box(
        s,
        620,
        190,
        420,
        205,
        ["HiRadixCache", "match / insert / evict", "唯一逻辑 Prefix Tree"],
        WHITE,
        CONTROL,
        shadow=True,
    )
    box(
        s,
        1160,
        170,
        420,
        245,
        ["HiCacheController", "Operation / Queue / ACK", "跨 Rank 连续长度对齐"],
        WHITE,
        CONTROL,
        shadow=True,
    )

    s.rect(70, 480, 940, 380, WHITE, LINE, radius=18, stroke_width=2)
    s.text(96, 519, "LOCAL RESIDENCY", 17, MUTED, 750, family=MONO)
    box(s, 110, 560, 750, 100, ["L1 · GPU HBM", "node.value = device_indices"], GPU_BG, GPU)
    box(
        s,
        110,
        695,
        750,
        100,
        ["L2 · Pinned Host DRAM", "node.host_value = host_indices"],
        CPU_BG,
        CPU,
    )

    s.rect(1160, 480, 420, 380, WHITE, LINE, radius=18, stroke_width=2)
    s.text(1186, 519, "TRANSFER & WORKERS", 17, MUTED, 750, family=MONO)
    box(s, 1200, 535, 340, 80, ["H2D / D2H", "L2TransferEngine"], CPU_BG, CPU)
    box(s, 1200, 650, 340, 80, ["prefetch / I/O / sync", "storage threads"], STORE_BG, STORE)
    box(s, 1200, 770, 340, 70, ["backup thread"], STORE_BG, STORE)

    s.rect(1690, 480, 340, 380, STORE_BG, STORE, radius=18, stroke_width=2)
    s.text(1715, 519, "L3 · EXTERNAL STORAGE", 16, STORE, 750, family=MONO)
    box(s, 1725, 545, 270, 82, ["batch_exists", "连续 page prefix"], WHITE, STORE, size=17)
    box(s, 1725, 660, 270, 82, ["batch_get(_v1)", "L3 → L2"], WHITE, STORE, size=17)
    box(s, 1725, 775, 270, 65, ["batch_set(_v1)"], WHITE, STORE, size=17)

    tag(s, 508, 258, 104, "CALL", CONTROL_BG, CONTROL)
    tag(s, 1048, 258, 104, "OP", CONTROL_BG, CONTROL)
    tag(s, 890, 568, 100, "D2H", GPU_BG, GPU)
    tag(s, 890, 703, 100, "H2D", CPU_BG, CPU)
    tag(s, 1570, 572, 110, "GET/SET", STORE_BG, STORE)
    s.text(
        1160,
        905,
        "不变量：后台线程搬数据；Scheduler 在安全点修改 Radix 节点与引用计数",
        18,
        MUTED,
        600,
    )

    # Scheduler-owned L3 prefetch state. The two columns make the asynchronous
    # completion path and the subsequent waiting-queue scan explicit.
    s.rect(70, 950, 1960, 700, WHITE, CONTROL, radius=18, stroke_width=2)
    s.text(100, 992, "SCHEDULER · L3 PREFETCH STATE", 18, CONTROL, 750, family=MONO)
    s.text(
        100,
        1022,
        "L3 完成先在安全点发布成 L2 prefix；waiting_queue 中的请求随后重新匹配",
        17,
        MUTED,
        550,
    )

    s.rect(100, 1045, 900, 590, CONTROL_BG, CONTROL, radius=16, stroke_width=1)
    s.text(130, 1082, "① ENQUEUE → HIT → PUBLISH", 16, CONTROL, 750, family=MONO)
    left_flow = [
        (1094, ["_prefetch_kvcache(req)", "match L1 / L2"], CONTROL_BG, CONTROL),
        (1155, ["waiting_queue.append(req)"], GREY_BG, MUTED),
        (1216, ["global L3 hit length", "prefetch_hit_queue"], STORE_BG, STORE),
        (1277, ["alloc L2 host pages", "仅按 agreed hit 分配"], CPU_BG, CPU),
        (1338, ["ongoing_prefetch[req]"], CONTROL_BG, CONTROL),
        (1399, ["completed_req"], OK_BG, OK),
        (1460, ["_handle_prefetch_result()"], CONTROL_BG, CONTROL),
        (1521, ["publish: L3 bytes → L2 prefix"], OK_BG, OK),
        (1582, ["ongoing_prefetch.delete(req)"], CONTROL_BG, CONTROL),
    ]
    for y, lines, fill, stroke in left_flow:
        box(s, 270, y, 560, 50, lines, fill, stroke, size=14, weight=650)
    for (y1, _, _, color), (y2, _, _, _) in zip(left_flow, left_flow[1:]):
        s.line(550, y1 + 50, 550, y2, color, 2)

    s.rect(1050, 1045, 930, 590, CONTROL_BG, CONTROL, radius=16, stroke_width=1)
    s.text(1080, 1082, "② WAITING QUEUE → RE-MATCH", 16, CONTROL, 750, family=MONO)
    right_flow = [
        (1100, ["get_new_batch_prefill()"], CONTROL_BG, CONTROL),
        (1166, ["for req in waiting_queue"], GREY_BG, MUTED),
        (1232, ["check_prefetch_progress(req)"], CONTROL_BG, CONTROL),
        (1320, ["prefetch done / partial accepted"], OK_BG, OK),
        (1386, ["pop_prefetch_loaded_span()"], CPU_BG, CPU),
        (1452, ["req.init_next_round_input()"], CONTROL_BG, CONTROL),
        (1518, ["match_prefix() again"], CONTROL_BG, CONTROL),
        (1584, ["L1 hit  /  new L2 hit", "包括刚刚 L3 → L2 的 prefix"], CPU_BG, CPU),
    ]
    for y, lines, fill, stroke in right_flow:
        box(s, 1235, y, 560, 52, lines, fill, stroke, size=14, weight=650)
    for (y1, _, _, color), (y2, _, _, _) in zip(right_flow, right_flow[1:]):
        if y1 == 1232:
            continue
        s.line(1515, y1 + 52, 1515, y2, color, 2)
    s.path("M 1515 1284 L 1515 1302 L 1515 1320", OK, 2)
    box(s, 1075, 1260, 140, 78, ["还必须等待 L3", "continue"], GREY_BG, MUTED, size=13, weight=650)
    s.path("M 1235 1258 L 1218 1258 L 1218 1299 L 1215 1299", MUTED, 2)
    s.path("M 830 1546 L 1020 1546 L 1020 1610 L 1235 1610", CPU, 2, True, True)
    s.text(1020, 1533, "刚 publish 的 L2 prefix 在此可见", 13, CPU, 650, "middle", MONO)
    return s


def figure_memory_model() -> SVG:
    s = SVG(
        1700,
        690,
        "HiCache residency and memory layout",
        "Radix metadata maps a shared prefix to GPU, host and hashed storage residency.",
    )
    s.header(
        "RESIDENCY MODEL",
        "逻辑命中与物理可读是两回事",
        "L1/L2 共享 Radix 节点；L3 通过链式 page hash 查询",
    )
    specs = [
        (70, "L1 · GPU", GPU_BG, GPU, ["Layer-first", "node.value", "可直接供 Attention 读取"]),
        (585, "L2 · Host", CPU_BG, CPU, ["Page-first", "node.host_value", "必须先完成 H2D"]),
        (1100, "L3 · Storage", STORE_BG, STORE, ["page hash → object", "backend owns placement", "先 L3→L2→L1"]),
    ]
    for x, title, fill, stroke, lines in specs:
        s.rect(x, 165, 430, 250, fill, stroke, radius=18, stroke_width=2, shadow=True)
        s.text(x + 26, 207, title, 25, INK, 750)
        s.multiline(x + 26, 262, lines, 19, INK, 550, "start", 40)
    s.line(500, 270, 585, 270, GPU, 3)
    s.line(585, 310, 500, 310, CPU, 3)
    s.line(1015, 270, 1100, 270, STORE, 3)
    s.line(1100, 310, 1015, 310, STORE, 3)
    tag(s, 494, 230, 98, "D2H", GPU_BG, GPU)
    tag(s, 494, 330, 98, "H2D", CPU_BG, CPU)
    tag(s, 1006, 230, 98, "SET", STORE_BG, STORE)
    tag(s, 1006, 330, 98, "GET", STORE_BG, STORE)

    s.rect(70, 480, 1460, 145, WHITE, LINE, radius=16, stroke_width=2)
    s.text(98, 520, "三个完成点", 19, INK, 750)
    milestones = [
        (315, "L3 exists hit", "只有 metadata", STORE),
        (780, "L3→L2 finish + publish", "Host bytes 有效", CPU),
        (1260, "layer i H2D event", "该层才可读取", GPU),
    ]
    s.line(280, 565, 1320, 565, LINE, 3, arrow=False)
    for x, title, note, color in milestones:
        s.circle(x, 565, 10, color)
        s.text(x, 550, title, 17, INK, 700, "middle")
        s.text(x, 597, note, 16, MUTED, 550, "middle")
    return s


def figure_lifecycle() -> SVG:
    s = SVG(
        2100,
        1310,
        "HiCache request lifecycle",
        "End-to-end request path showing L3 query, hit, prefetch, L2 load, forward, L2 write and L3 write.",
    )
    s.header(
        "END-TO-END PIPELINE",
        "HiCache 嵌入 Scheduler 的请求主干",
        "时间向右；每个彩色阶段标注发生线程与完成语义，箱宽不代表耗时",
    )

    s.text(42, 145, "READ · request admission", 18, CONTROL, 750, family=MONO)
    for y, label in [
        (175, "Scheduler"),
        (345, "L3 query"),
        (515, "L3 I/O"),
        (685, "H2D stream"),
        (855, "GPU compute"),
    ]:
        lane(s, y, label, 215, 2040)

    stages = [
        (235, 175, 205, ["HTTP / tokenizer", "handle_generate_request", "_add_request_to_queue"], CONTROL_BG, CONTROL),
        (475, 175, 220, ["_prefetch_kvcache", "match L1 + L2"], CONTROL_BG, CONTROL),
        (735, 175, 220, ["waiting_queue.append", "其他请求可继续运行"], GREY_BG, MUTED),
        (735, 345, 220, ["L3 QUERY", "batch_exists"], STORE_BG, STORE),
        (995, 345, 220, ["L3 HIT", "MIN #1 → hit_queue"], STORE_BG, STORE),
        (1255, 515, 220, ["L3 PREFETCH", "GET → L2 slots"], STORE_BG, STORE),
        (1515, 515, 220, ["PREFETCH FINISH", "MIN #2 → publish L2"], OK_BG, OK),
        (1755, 175, 220, ["second match", "admission / alloc L1"], CONTROL_BG, CONTROL),
        (1755, 685, 220, ["L2 LOAD", "layer-wise H2D"], CPU_BG, CPU),
        (1755, 855, 220, ["Forward", "只计算 true miss"], GPU_BG, GPU),
    ]
    for x, y, w, lines, fill, stroke in stages:
        box(s, x, y, w, 85, lines, fill, stroke, size=16)

    s.line(440, 217, 475, 217, CONTROL, 3)
    s.line(695, 217, 735, 217, CONTROL, 3)
    s.path("M 695 240 L 715 240 L 715 387 L 735 387", STORE, 3)
    s.line(955, 387, 995, 387, STORE, 3)
    s.path("M 1215 387 L 1235 387 L 1235 557 L 1255 557", STORE, 3)
    s.line(1475, 557, 1515, 557, STORE, 3)
    s.path("M 1735 557 L 1745 557 L 1745 217 L 1755 217", CONTROL, 3)
    s.line(1865, 260, 1865, 685, CPU, 3)
    s.line(1865, 770, 1865, 855, GPU, 3)

    s.path("M 955 217 L 1705 217", MUTED, 2, False, True)
    s.text(1080, 204, "request remains in waiting queue", 14, MUTED, 550, family=MONO)
    s.text(1260, 655, "drain ACK 后，L3 bytes 才发布为 L2 prefix", 16, MUTED, 550)
    s.text(1755, 650, "PrefillAdder 通过预算后才启动", 15, CPU, 650)

    s.text(42, 1015, "WRITE · after Forward / cache insertion", 18, CONTROL, 750, family=MONO)
    lane(s, 1045, "Scheduler + D2H", 215, 2040)
    lane(s, 1175, "backup thread", 215, 2040)
    write_stages = [
        (300, 1045, ["cache_unfinished / finished", "insert L1 Radix"]),
        (680, 1045, ["L2 WRITE", "alloc Host + D2H"]),
        (1060, 1045, ["D2H ACK", "Host bytes 有效"]),
        (1440, 1175, ["L3 WRITE", "batch_set(_v1)"]),
        (1810, 1175, ["Storage ACK", "解除 Host 保护"]),
    ]
    colors = [
        (CONTROL_BG, CONTROL),
        (CPU_BG, CPU),
        (OK_BG, OK),
        (STORE_BG, STORE),
        (OK_BG, OK),
    ]
    for (x, y, lines), (fill, stroke) in zip(write_stages, colors):
        box(s, x, y, 250, 80, lines, fill, stroke, size=16)
    s.line(550, 1085, 680, 1085, CONTROL, 3)
    s.line(930, 1085, 1060, 1085, CPU, 3, dashed=True)
    s.path("M 1310 1085 L 1370 1085 L 1370 1215 L 1440 1215", STORE, 3)
    s.line(1690, 1215, 1810, 1215, STORE, 3, dashed=True)
    s.text(680, 1152, "write-through：insert 后；write-back：L1 eviction 时", 16, MUTED, 550)
    return s


def sequence_event(
    s: SVG,
    xs: list[int],
    row: int,
    a: int,
    b: int,
    lines: list[str],
    color: str,
    dashed: bool = False,
) -> None:
    y = 205 + row * 78
    s.text(32, y + 24, f"{row + 1:02d}", 14, MUTED, 650, family=MONO)
    if a == b:
        box(s, xs[a] - 132, y - 8, 264, 62, lines, WHITE, color, size=14)
    else:
        s.line(xs[a], y + 38, xs[b], y + 38, color, 2, True, dashed)
        s.multiline(
            (xs[a] + xs[b]) // 2,
            y + 8,
            lines,
            14,
            INK,
            550,
            "middle",
            20,
        )


def figure_prefetch_sequence() -> SVG:
    s = SVG(
        1900,
        1370,
        "HiCache L3 query and prefetch sequence",
        "Detailed sequence from local match through two rank synchronizations and publication as L2 prefix.",
    )
    s.header(
        "L3 QUERY & PREFETCH",
        "L3 命中不是一个时刻，而是两次跨 Rank 收敛",
        "时间向下；实线为调用/提交，虚线为完成或 ACK",
    )
    xs = [200, 610, 1030, 1460, 1780]
    names = ["Scheduler", "HiRadixCache", "Query thread", "I/O + sync", "L3 backend"]
    for x, name in zip(xs, names):
        box(s, x - 150, 115, 300, 55, [name], GREY_BG, CONTROL, size=15)
        s.line(x, 170, x, 1305, LINE, 1, arrow=False, dashed=True)
    events = [
        (0, 1, ["init_next_round_input", "match L1 / L2"], CONTROL, False),
        (1, 0, ["device_indices + host_hit_length", "last_host_node"], MUTED, True),
        (0, 1, ["prefetch_from_storage", "suffix page-align + enqueue"], CONTROL, False),
        (1, 2, ["prefetch_queue", "protect Host anchor"], CONTROL, False),
        (2, 4, ["L3 QUERY · batch_exists", "hash chain"], STORE, False),
        (4, 2, ["local consecutive hit pages"], MUTED, True),
        (2, 3, ["MIN all-reduce #1", "global L3 hit length"], STORE, False),
        (3, 0, ["prefetch_hit_queue"], MUTED, True),
        (0, 0, ["仅按 agreed hit 分配 L2", "不足则 evict / shorten / revoke"], CPU, False),
        (0, 3, ["prefetch_buffer", "Host slots + hashes"], STORE, False),
        (3, 4, ["L3 PREFETCH · batch_get", "write target Host slots"], STORE, False),
        (4, 3, ["completed_tokens"], MUTED, True),
        (3, 3, ["MIN all-reduce #2", "agreed completed prefix"], STORE, False),
        (3, 0, ["ack_prefetch_queue", "completed_req / partial"], MUTED, True),
    ]
    for row, (a, b, lines, color, dashed) in enumerate(events):
        sequence_event(s, xs, row, a, b, lines, color, dashed)
    s.text(
        82,
        1335,
        "Scheduler 随后 _handle_prefetch_result()：把完成 bytes 发布为 L2 host-only nodes；请求再 match 一次。",
        17,
        INK,
        650,
    )
    return s


def figure_layer_overlap() -> SVG:
    s = SVG(
        1700,
        590,
        "Layer-wise H2D and forward overlap",
        "Each layer's forward waits only for that layer's restored prefix KV.",
    )
    s.header(
        "L2 LOAD",
        "H2D 与 Forward 是逐层生产—消费",
        "逐层 event 是模型读取门槛；整次 ACK 只负责生命周期清理",
    )
    for y, label in [(165, "H2D stream"), (315, "Compute stream"), (465, "Scheduler")]:
        lane(s, y, label, 220, 1650)
    tx = [250, 555, 860, 1165]
    cx = [555, 860, 1165, 1470]
    for i, (x, c) in enumerate(zip(tx, cx)):
        box(s, x, 165, 225, 72, [f"Layer {i} · H2D"], CPU_BG, CPU, size=16)
        box(s, c, 315, 225, 72, [f"Layer {i} · Forward"], GPU_BG, GPU, size=16)
        s.path(f"M {x + 225} 201 L {c - 18} 201 L {c - 18} 351 L {c} 351", CPU, 2)
        s.text(c - 8, 292, f"event {i}", 14, CPU, 650, "end", MONO)
    box(
        s,
        920,
        455,
        520,
        75,
        ["loading_check：整次 H2D ACK 后清理 ongoing_load_back"],
        WHITE,
        CONTROL,
        size=16,
    )
    s.path("M 1390 201 L 1580 201 L 1580 493 L 1440 493", MUTED, 2, True, True)
    return s


def figure_write_policies() -> SVG:
    s = SVG(
        1880,
        770,
        "HiCache write policies",
        "Write-through and write-back differ at the trigger, but share D2H completion before L3 write.",
    )
    s.header(
        "WRITE PIPELINE",
        "三种写策略只改变触发点，不改变数据依赖",
        "L2 write = GPU→Host D2H；L3 write = Host→Storage batch_set",
    )
    rows = [
        (165, "write_through", ["insert / first hit", "threshold = 1"]),
        (345, "selective", ["insert / reuse", "threshold = 2"]),
        (525, "write_back", ["L1 eviction", "no Host copy"]),
    ]
    xs = [245, 565, 885, 1205, 1525]
    for y, label, trigger in rows:
        s.text(35, y + 37, label, 18, INK, 700, family=MONO)
        stages = [
            (trigger, CONTROL_BG, CONTROL),
            (["L2 WRITE", "alloc Host + D2H"], CPU_BG, CPU),
            (["D2H ACK", "Host bytes 有效"], OK_BG, OK),
            (["L3 WRITE", "batch_set(_v1)"], STORE_BG, STORE),
            (["Storage ACK", "release Host ref"], OK_BG, OK),
        ]
        for j in range(4):
            s.line(
                xs[j] + 250,
                y + 42,
                xs[j + 1],
                y + 42,
                CPU if j < 2 else STORE,
                3,
                True,
                dashed=(j in (1, 3)),
            )
        for x, (lines, fill, stroke) in zip(xs, stages):
            box(s, x, y, 250, 84, lines, fill, stroke, size=15)
        if label == "write_back":
            s.text(885, y + 119, "D2H 完成后才能 free 源 GPU slots；无需等待 L3", 15, WARN, 650)
        else:
            s.text(245, y + 119, "pending 期间节点仍受 L1 lock_ref 保护", 15, MUTED, 550)
    return s


def figure_storage() -> SVG:
    s = SVG(
        1740,
        690,
        "HiCache storage abstraction",
        "The controller dispatches zero-copy and generic-copy paths to pluggable L3 backends.",
    )
    s.header(
        "STORAGE ABSTRACTION",
        "L3 placement 被统一接口隐藏，但数据路径并不相同",
        "绿色是 backend 直接读写 L2 buffer；灰色是显式 tensor / block copy",
    )
    box(
        s,
        70,
        220,
        330,
        170,
        ["HiCacheController", "batch_exists", "batch_get / batch_set"],
        CONTROL_BG,
        CONTROL,
        shadow=True,
    )
    box(
        s,
        535,
        160,
        330,
        150,
        ["Zero-copy adapter", "batch_get/set_v1", "Host pointer / view"],
        STORE_BG,
        STORE,
        shadow=True,
    )
    box(
        s,
        535,
        395,
        330,
        150,
        ["Generic adapter", "tensor / block", "explicit copy"],
        GREY_BG,
        MUTED,
        shadow=True,
    )
    box(
        s,
        1010,
        145,
        650,
        170,
        ["Direct-buffer backends", "Mooncake · NIXL · SiMM · HF3FS", "backend 直接读写预分配 L2 地址"],
        STORE_BG,
        STORE,
        shadow=True,
    )
    box(
        s,
        1010,
        390,
        650,
        170,
        ["Generic / manager backends", "File · AIBrix", "本地文件或 Global KV Manager"],
        GREY_BG,
        MUTED,
        shadow=True,
    )
    s.line(400, 275, 535, 235, STORE, 3)
    s.line(400, 335, 535, 470, MUTED, 3)
    s.line(865, 235, 1010, 235, STORE, 3)
    s.line(865, 470, 1010, 470, MUTED, 3)
    s.rect(70, 590, 1590, 62, WHITE, LINE, radius=12, stroke_width=1)
    s.text(
        865,
        628,
        "共同 contract：batch_exists 从 keys[0] 开始返回连续存在的 page 数；中间 miss 后立即截断",
        17,
        INK,
        650,
        "middle",
    )
    return s


FIGURES = {
    "hicache-merged-architecture.svg": figure_architecture,
    "hicache-merged-memory-model.svg": figure_memory_model,
    "hicache-merged-lifecycle.svg": figure_lifecycle,
    "hicache-merged-prefetch-sequence.svg": figure_prefetch_sequence,
    "hicache-merged-layer-overlap.svg": figure_layer_overlap,
    "hicache-merged-write-policies.svg": figure_write_policies,
    "hicache-merged-storage.svg": figure_storage,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "content/sglang/img",
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
