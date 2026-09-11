#!/usr/bin/env python3
"""Deterministic, editable SVG figures for the KV cache comparison article."""
from __future__ import annotations

import argparse
from pathlib import Path
from draw_hicache_figures import (
    SVG, INK, MUTED, LINE, WHITE, GPU, GPU_BG, CPU, CPU_BG,
    STORE, STORE_BG, CONTROL, CONTROL_BG, GREY_BG, WARN_BG,
)

PALETTE = {
    'control': (CONTROL_BG, CONTROL), 'copy': (CPU_BG, CPU),
    'io': (STORE_BG, STORE), 'compute': (GPU_BG, GPU),
    'neutral': (GREY_BG, LINE), 'wait': (WARN_BG, MUTED),
}


def box(s, x, y, w, lines, kind='control', h=64, size=19):
    fill, edge = PALETTE[kind]
    s.rect(x, y, w, h, fill, edge, radius=9, stroke_width=1.5)
    if isinstance(lines, str):
        lines = [lines]
    first = y + h / 2 + 7 - (len(lines) - 1) * 12
    s.multiline(x + w / 2, first, lines, size, INK, 550, 'middle', gap=24)


def heading(s, title, note):
    s.text(40, 43, title, 28, INK, 700)
    s.text(40, 75, note, 17, MUTED)


def flow_row(s, y, label, steps):
    s.text(40, y + 27, label, 20, INK, 700)
    start, w, gap = 234, 200, 26
    for i in range(len(steps) - 1):
        x = start + i * (w + gap)
        s.line(x + w, y + 32, x + w + gap - 5, y + 32, INK, 1.5)
    for i, (lines, kind) in enumerate(steps):
        box(s, start + i * (w + gap), y, w, lines, kind, size=17)


def lookup():
    s = SVG(1380, 540, 'Prefix lookup comparison', 'Four integration paths distinguish metadata lookup, prefetch, GPU allocation and consumption.')
    heading(s, 'Prefix lookup：查到位置之后，还要决定何时可用', '同一行从左向右；LMCache V1 指传统进程内集成，MP 指标准 multiprocess 路径。')
    rows = [
        ('vLLM + FlexKV', [(['GPU APC', '屏蔽本地已命中'], 'control'), (['各 tier prefix', 'match + plan'], 'control'), (['分配 GPU blocks', '绑定 graph 目标'], 'control'), (['外部 KV → GPU', '轮询接收完成'], 'copy'), (['剩余 suffix', '进入 prefill'], 'compute')]),
        ('SGLang HiCache', [(['HiRadixTree', 'GPU + Host match'], 'control'), (['L3 batch_exists', '确认命中再分配 Host'], 'control'), (['L3 → Host', '按预取策略收口'], 'io'), (['Host → GPU', '逐层就绪事件'], 'copy'), (['每层 KV 就绪', '该层继续 forward'], 'compute')]),
        ('vLLM + LMCache V1', [(['GPU APC', 'lookup client'], 'control'), (['chunk contains', '位置映射 + pin'], 'control'), (['GPU 分配 / metadata', '可选异步预取'], 'control'), (['retrieve / layerwise', '填充 GPU KV'], 'copy'), (['整批或逐层等待', '取决于配置'], 'compute')]),
        ('vLLM + LMCache MP', [(['GPU APC', 'LOOKUP RPC'], 'control'), (['本地 L1 读锁', '缺失部分查 L2'], 'control'), (['L2 → L1', '查询预取完成'], 'io'), (['分配 GPU / RETRIEVE', '异步搬运与回报'], 'copy'), (['请求解除等待', '进入 prefill'], 'compute')]),
    ]
    for i, (label, steps) in enumerate(rows):
        flow_row(s, 112 + i * 99, label, steps)
    s.text(40, 520, '层级命名不同：HiCache L1 = GPU；LMCache MP L1 = 本地存储层（本文按 CPU DRAM 配置）。', 17, MUTED)
    return s


def transfer():
    s = SVG(1380, 666, 'FlexKV GET and PUT dependencies', 'Ordinary non-GDS global path. GET has one joined H2D; PUT releases GPU ownership before background downstream writes finish.')
    heading(s, 'FlexKV：GET 汇合，PUT 分叉', '普通 global 路径的 op 依赖；箭头表示完成依赖，不代表实测耗时。')
    s.text(40, 128, 'GET', 24, CONTROL, 700)
    # Draw connectors before boxes. CPU f1 participates in the joined H2D.
    s.path('M 392 161 H 696 V 231 H 736', INK, 1.6)
    s.path('M 630 231 H 736', INK, 1.6)
    s.path('M 630 301 H 696 V 231 H 736', INK, 1.6)
    s.line(967, 231, 1003, 231, INK, 1.6)
    box(s, 156, 131, 236, ['CPU 命中 f1', '已有 host 数据'], 'copy', h=60)
    box(s, 156, 201, 474, ['SSD 命中 f2 → DISK2H', '补入 host staging'], 'io', h=60)
    box(s, 156, 271, 474, ['Remote 命中 f3 → REMOTE2H', '补入 host staging'], 'io', h=60)
    box(s, 736, 199, 231, ['一个 H2D op', '搬 f1 + f2 + f3'], 'copy')
    box(s, 1008, 199, 324, ['GPU load 完成 → 通知 vLLM', '可调度未命中 suffix'], 'compute', size=18)
    s.text(735, 297, '可选：REMOTE2H → H2DISK promotion', 17, MUTED)
    s.line(40, 359, 1340, 359, LINE, 1, False)
    s.text(40, 408, 'PUT', 24, CONTROL, 700)
    s.path('M 382 446 H 438', INK, 1.6)
    s.path('M 675 446 H 727 V 414 H 775', INK, 1.6)
    s.path('M 675 446 H 727 V 499 H 775', INK, 1.6)
    s.path('M 675 446 H 727 V 584 H 775', INK, 1.6)
    box(s, 156, 414, 226, ['put_match', '预留空间 / 去重'], 'control')
    box(s, 443, 414, 232, ['D2H', 'GPU → CPU'], 'copy')
    box(s, 780, 384, 552, ['task_end：GPU 源已安全搬走', 'finished_sending → vLLM 可释放 GPU blocks'], 'control', size=18)
    box(s, 780, 469, 552, ['H2DISK：CPU → SSD', '完成后该 tier 发布可用状态'], 'io', size=18)
    box(s, 780, 554, 552, ['H2REMOTE：CPU → Remote', '完成后该 tier 发布可用状态'], 'io', size=18)
    s.text(40, 649, '普通索引：insert(unready) → op callback set_ready；Mooncake Store 分支使用 deferred insert，完成点需另看。', 17, MUTED)
    return s


def read_timeline():
    s = SVG(1380, 780, 'Read pipeline timeline', 'Conceptual timing, not a benchmark. Distinguishes inter-request overlap from layerwise same-request overlap.')
    heading(s, '读 pipeline：异步加载，与同一请求逐层重叠，是两件事', '示意时间轴，不按比例；每层 forward 计算的是未命中 suffix，并读取已恢复的 prefix KV。')
    s.line(270, 109, 1333, 109, MUTED, 1.5)
    s.text(1280, 99, '时间 →', 16, MUTED)
    def lane(y, label):
        s.text(43, y + 27, label, 18, INK, 600)
        s.line(270, y + 25, 1334, y + 25, LINE, 1, False)
    # Data records define stage order; paired lanes show real overlap.
    groups = [
        (140, 'FlexKV / global', [
            (0, 280, 278, 'SSD ∥ Remote → Host', 'io'), (0, 576, 210, '统一 H2D', 'copy'),
            (1, 280, 506, '该请求等待；其他请求可执行', 'wait'), (1, 813, 505, '完成回报 → suffix prefill', 'compute')]),
        (300, 'HiCache', [
            (0, 280, 242, 'L3 → Host / 收口', 'io'), (0, 540, 151, 'H2D 层 0', 'copy'), (0, 709, 151, 'H2D 层 1', 'copy'), (0, 878, 151, 'H2D 层 2', 'copy'),
            (1, 709, 151, 'forward 0', 'compute'), (1, 878, 151, 'forward 1', 'compute'), (1, 1047, 151, 'forward 2', 'compute')]),
        (460, 'LMCache V1 / layerwise', [
            (0, 280, 242, '选定 backend 数据', 'io'), (0, 540, 151, 'load 层 0', 'copy'), (0, 709, 151, 'load 层 1', 'copy'), (0, 878, 151, 'load 层 2', 'copy'),
            (1, 709, 151, 'forward 0', 'compute'), (1, 878, 151, 'forward 1', 'compute'), (1, 1047, 151, 'forward 2', 'compute')]),
        (620, 'LMCache MP / standard', [
            (0, 280, 278, 'LOOKUP + L2 → L1', 'io'), (0, 576, 210, 'RETRIEVE / H2D', 'copy'),
            (1, 280, 506, '该请求等待；其他请求可执行', 'wait'), (1, 813, 505, '完成回报 → suffix prefill', 'compute')]),
    ]
    for y, name, stages in groups:
        s.text(40, y - 8, name, 20, INK, 700)
        lane(y, '                  I/O')
        lane(y + 57, '                  Compute')
        for lane_id, x, w, text, kind in stages:
            box(s, x, y + lane_id * 57, w, text, kind, h=45, size=17)
        if name in ('HiCache', 'LMCache V1 / layerwise'):
            for x in (691, 860, 1029):
                s.path(f'M {x} {y+22} H {x+9} V {y+79} H {x+15}', INK, 1.2)
    s.text(40, 765, '逐层箭头是该层 KV 就绪依赖；能否隐藏延迟还取决于传输速度、模型计算时间和 stream 资源竞争。', 17, MUTED)
    return s


def write_lifecycle():
    s = SVG(1380, 692, 'Write trigger comparison', 'Request completion, hit admission, memory pressure and worker forward hooks trigger different write behavior.')
    heading(s, '写入与驱逐：先找触发事件，再看数据路径', '这张图是事件顺序，不是耗时比较；异步提交不等于下层存储已经可见。')
    rows = [
        ('FlexKV / vLLM', [(['请求正常结束', '或显式 abort offload'], 'control'), (['put_match', '保留 GPU 源'], 'control'), (['D2H 完成', '可释放 GPU'], 'copy'), (['CPU → SSD / Remote', '后台继续'], 'io'), (['外部 tier ready', '可供后续查询'], 'control')]),
        ('HiCache / through', [(['插入 / 访问树节点', '达到准入阈值'], 'control'), (['D2H backup', '保护节点'], 'copy'), (['DMA ack', '再提交 L3 backup'], 'control'), (['以后 GPU 驱逐', '已有备份则降级'], 'neutral'), (['Host 驱逐', '释放未保护叶子'], 'neutral')]),
        ('HiCache / back', [(['GPU 容量不足', '选择可驱逐节点'], 'control'), (['无 Host 备份', '尝试 D2H'], 'copy'), (['等待 D2H 完成', '才能释放 GPU'], 'wait'), (['Host → L3', '已启用时后台写'], 'io'), (['Host 空间不足', '可能直接丢弃'], 'neutral')]),
        ('LMCache V1 / eager', [(['本次计算产生 KV', '完整可保存 chunk'], 'compute'), (['逐层 save', '或 forward 后 store'], 'copy'), (['MemoryObj', '交给后端任务'], 'control'), (['按配置提交 put', 'CPU / Disk / Remote'], 'io'), (['后端独立驱逐', '不自动逐级下沉'], 'neutral')]),
        ('LMCache MP / eager', [(['step 中新 KV', 'wait_for_save 提交'], 'compute'), (['STORE / D2H', 'reserve_write'], 'copy'), (['finish_write', '本地对象 ready'], 'control'), (['StoreController', '按策略写 L2'], 'io'), (['EvictionController', '按水位回收 L1'], 'neutral')]),
    ]
    for i, (name, stages) in enumerate(rows):
        flow_row(s, 108 + i * 101, name, stages)
    s.text(40, 640, 'MP lazy offload（可选）：请求结束进入 FIFO → 达到阈值 → 校验 block hash → 保护并异步保存。', 18, MUTED)
    s.text(40, 671, '它不是 GPU LRU 驱逐回调；入选前 blocks 未 pin，已被复用的数据会跳过保存。', 18, MUTED)
    return s


def prefix():
    s = SVG(1200, 476, 'Prefix coverage and source selection', 'Ten illustrative blocks, GPU hits three, CPU four, SSD six, remote eight. External reuse adds five blocks, not the sum of tier hits.')
    heading(s, '同一条 prefix：总命中长度，不能跨层相加', '示例统一采用 16 tokens / block；不是三个系统的默认配置。')
    x0, cell = 256, 85
    for i in range(10):
        s.text(x0 + i * cell + 38, 119, f'B{i}', 19, MUTED, 500, 'middle')
    for row, (label, n, kind) in enumerate([('GPU', 3, 'compute'), ('CPU', 4, 'copy'), ('SSD', 6, 'io'), ('Remote', 8, 'io')]):
        y = 134 + row * 58
        s.text(43, y + 31, f'{label} prefix = {n}', 21, INK, 600)
        for i in range(10):
            box(s, x0 + i * cell, y, 76, 'hit' if i < n else '—', kind if i < n else 'neutral', h=44, size=17)
    y = 375
    for start, n, label, kind in [(0, 3, 'GPU 复用', 'compute'), (3, 1, 'CPU', 'copy'), (4, 2, 'SSD', 'io'), (6, 2, 'Remote', 'io'), (8, 2, '重算', 'neutral')]:
        box(s, x0 + start * cell, y, n * cell - 9, label, kind, h=44, size=18)
    s.text(40, 456, '外部新增 = 8 − 3 = 5 blocks = 80 tokens；剩余 prefill = (10 − 8) × 16 = 32 tokens。', 20, INK, 600)
    return s


FIGURES = {'kv-lookup-comparison.svg': lookup, 'flexkv-transfer-dependencies.svg': transfer,
           'kv-read-timeline.svg': read_timeline, 'kv-write-lifecycle.svg': write_lifecycle,
           'kv-prefix-coverage.svg': prefix}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output', nargs='?', type=Path, default=Path(__file__).resolve().parents[1] / 'content/llm_inference/img/kv-cache-comparison')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, make in FIGURES.items():
        s = make()
        (args.output / name).write_text(s.finish(), encoding='utf-8')
        print(f'{name}: {s.width} x {s.height}')


if __name__ == '__main__':
    main()
