#!/usr/bin/env python3
"""Editable 800 × 400 project cards. Evidence: project-architecture-sources.md.

Usage: python scripts/draw_project_architectures.py [output_directory]
Only standard-library SVG primitives; no remote assets, raster, or gradients.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from draw_hicache_figures import FONT, INK, MUTED, LINE, BG, WHITE, CPU_BG, STORE_BG

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "content/img/projects"


class Figure:
    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
        self.regions: list[str] = []
        self.connectors: list[str] = []
        self.nodes: list[str] = []
        self.labels: list[str] = []

    def text(self, x, y, value, size=26, *, anchor="middle", muted=False, weight=550):
        self.labels.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" '
            f'font-size="{size}" font-weight="{weight}" fill="{MUTED if muted else INK}">'
            f'{html.escape(value)}</text>'
        )

    def rect(self, x, y, w, h, fill, layer):
        assert x >= 12 and y >= 12 and x + w <= 788 and y + h <= 388
        layer.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
                     f'fill="{fill}" stroke="{LINE}" stroke-width="2"/>')

    def panel(self, x, y, w, h, title):
        self.rect(x, y, w, h, WHITE, self.regions)
        self.text(x + 18, y + 34, title, 25, anchor="start", weight=650)

    def box(self, x, y, w, h, title, detail=None, fill=CPU_BG, size=28):
        self.rect(x, y, w, h, fill, self.nodes)
        self.text(x + w / 2, y + h / 2 + (9 if detail is None else -4), title, size)
        if detail:
            self.text(x + w / 2, y + h / 2 + 25, detail, 21, muted=True, weight=450)

    def edge(self, *points, dashed=False, arrow=True):
        assert all(a[0] == b[0] or a[1] == b[1] for a, b in zip(points, points[1:]))
        d = "M" + " L".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        self.connectors.append(f'<path d="{d}" fill="none" stroke="{INK}" '
                               f'stroke-width="2.4" stroke-linejoin="round"{dash}{marker}/>')

    def svg(self):
        return "\n".join([
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400" '
            'viewBox="0 0 800 400" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(self.title)}</title>',
            f'<desc id="desc">{html.escape(self.description)}</desc>',
            '<defs><marker id="arrow" viewBox="0 0 8 8" refX="8" refY="4" '
            'markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">'
            f'<path d="M0 0 L8 4 L0 8 Z" fill="{INK}"/></marker></defs>',
            f'<rect width="800" height="400" fill="{BG}"/>',
            # Parent surfaces precede wires; nodes and labels sit above wires.
            *self.regions, *self.connectors, *self.nodes, *self.labels, '</svg>', ''
        ])


def sglang():
    f = Figure("SGLang：混合批次与资源归属", "据 feature/eagle-mixed-fa3-cuda-graph。"
               "Scheduler 生成 MIXED parent 与 Prefill、Verify 非拥有式子视图。"
               "EAGLEWorkerV2 拥有 MixedWorker 和已有 runner、KV、graph 资源；"
               "MixedWorker 复用它们，不创建第二套模型。图示所有权，不表示 target 与 draft 并行。")
    f.panel(24, 24, 254, 352, "Scheduler")
    f.box(46, 104, 210, 240, "", fill=BG)
    f.text(151, 139, "MIXED batch", 25)
    f.box(62, 163, 178, 62, "Prefill")
    f.box(62, 260, 178, 62, "Verify")
    f.panel(328, 24, 448, 352, "EAGLEWorkerV2")
    f.box(350, 104, 404, 76, "MixedWorker", "EagleMixedWorkerV2")
    f.edge((278, 222), (302, 222), (302, 142), (350, 142))
    f.edge((552, 180), (552, 265))
    f.text(573, 230, "复用", 22, anchor="start", muted=True)
    f.box(350, 265, 404, 88, "Target / Draft runner", "KV pool · Graph runner", fill=STORE_BG)
    return f


def cuda():
    f = Figure("cuda-learn：统一算子绑定", "Python 测试和 benchmark 调用 PyTorch binding；"
               "binding 通过 DLPack 与 TVM FFI 调用注册的 CUDA kernels，共用 torch stream。")
    f.panel(24, 24, 330, 352, "Python / PyTorch")
    f.panel(422, 24, 354, 352, "CUDA 插件")
    f.box(48, 100, 282, 78, "测试 · Benchmark", size=26)
    f.box(48, 260, 282, 78, "PyTorch binding", "ops.py", size=26)
    f.box(446, 100, 306, 78, "TVM FFI", "全局函数注册")
    f.box(446, 260, 306, 78, "CUDA kernels", "GEMM · FlashAttention")
    f.edge((189, 178), (189, 260))
    f.edge((330, 299), (388, 299), (388, 139), (446, 139))
    f.edge((599, 178), (599, 260))
    return f


def miniinfer():
    f = Figure("MiniInfer：推理引擎核心", "LLMEngine 的 Scheduler 调用 ModelRunner；"
               "KVCacheManager 提供分页与 Radix 前缀缓存。省略可选多进程部署。")
    f.panel(24, 24, 752, 352, "LLMEngine")
    f.box(48, 106, 266, 102, "Scheduler", "Continuous batching")
    f.box(426, 106, 326, 102, "ModelRunner", "CUDA Graph / FlashInfer")
    f.box(193, 281, 414, 73, "KVCacheManager", "Paged KV · Radix prefix", fill=STORE_BG)
    f.edge((314, 157), (426, 157))
    f.edge((181, 208), (181, 244), (300, 244), (300, 281))
    f.edge((589, 208), (589, 244), (500, 244), (500, 281))
    return f


def nebula():
    f = Figure("Nebula：向量检索组件", "graphd 经 RPC 访问 storaged，metad 提供 schema 和索引元数据。"
               "storaged 的内存 ANN 索引封装 HNSWlib 和 Faiss IVF；RocksDB 保存向量属性和 ID 映射，"
               "不是 ANN 图的存储容器。元数据接口以虚线表示。")
    f.box(24, 96, 252, 106, "graphd", "nGQL · 查询计划")
    f.box(24, 267, 252, 87, "metad", "Schema · 索引元数据", fill=BG)
    f.panel(352, 24, 424, 352, "storaged")
    f.box(374, 104, 380, 90, "内存 ANN 索引", "HNSWlib / Faiss IVF")
    f.box(374, 267, 380, 87, "RocksDB", "向量属性 · ID 映射", fill=STORE_BG)
    f.edge((276, 149), (374, 149))
    f.text(314, 134, "RPC", 20, muted=True)
    f.edge((150, 202), (150, 267), dashed=True)
    f.edge((276, 310), (352, 310), dashed=True)
    f.edge((564, 194), (564, 267))
    return f


def blog():
    f = Figure("个人博客：Quartz 内容架构", "本地 Markdown 由 Quartz 转换并生成页面与内容索引；"
               "本地图片、PDF 由资源发射器复制到静态站点。对应当前工作区的 Quartz 迁移版。")
    f.panel(24, 24, 222, 352, "本地内容")
    f.panel(292, 24, 248, 352, "Quartz")
    f.panel(586, 24, 190, 352, "静态站点")
    f.box(42, 102, 186, 82, "Markdown", size=27)
    f.box(42, 268, 186, 82, "图片 · PDF", fill=STORE_BG, size=26)
    f.box(314, 102, 204, 82, "Transformer", size=25)
    f.box(314, 268, 204, 82, "Emitter")
    f.box(604, 102, 154, 82, "页面", size=28)
    f.box(604, 268, 154, 82, "搜索 · 标签", fill=STORE_BG, size=23)
    f.edge((228, 143), (314, 143))
    f.edge((416, 184), (416, 268))
    f.edge((228, 309), (314, 309))
    f.edge((518, 309), (564, 309), (564, 143), (604, 143))
    f.edge((564, 309), (604, 309))
    return f


def bustub():
    f = Figure("BusTub：数据库内核", "查询执行使用 B+ 树与表存储，经缓冲池和磁盘层访问页面。"
               "事务管理以 MVCC/OCC 和 Undo Log 协调执行与数据可见性，虚线表示协调接口。"
               "依据个人主页项目经历及 BusTub 通关指北，而非宣称独立开发整个上游数据库。")
    f.panel(24, 24, 752, 352, "BusTub")
    f.box(48, 82, 414, 68, "查询执行", "Volcano · Hash Join")
    f.box(48, 184, 414, 68, "B+ 树 · 表存储", fill=STORE_BG)
    f.box(48, 286, 704, 68, "缓冲池 · 磁盘调度", "LRU-K · Disk Scheduler", fill=STORE_BG)
    f.box(528, 82, 224, 170, "", fill=BG)
    f.text(640, 123, "事务管理", 28)
    f.text(640, 174, "MVCC / OCC", 24)
    f.text(640, 221, "Undo Log", 24)
    f.edge((255, 150), (255, 184))
    f.edge((255, 252), (255, 286))
    f.edge((528, 116), (462, 116), dashed=True)
    f.edge((528, 218), (462, 218), dashed=True)
    return f


def tinykv():
    f = Figure("TinyKV：调度与 Raft 副本", "TinyScheduler 通过心跳和调度任务管理存储节点；"
               "图中仅展示同一 Region 的两个 Raft 副本，Leader 向 Follower 复制日志，"
               "节点使用 BadgerDB 持久化。两节点仅为局部示意，不代表完整集群或容错部署建议。"
               "虚线是调度控制接口，实线是复制与存储接口。")
    f.box(226, 24, 348, 65, "TinyScheduler", fill=BG)
    f.panel(24, 160, 332, 216, "存储节点 A")
    f.panel(444, 160, 332, 216, "存储节点 B")
    f.box(46, 210, 288, 58, "Raft · Leader")
    f.box(466, 210, 288, 58, "Raft · Follower")
    f.box(46, 308, 288, 48, "BadgerDB", fill=STORE_BG)
    f.box(466, 308, 288, 48, "BadgerDB", fill=STORE_BG)
    f.edge((400, 89), (400, 120), (190, 120), (190, 160), dashed=True)
    f.edge((400, 120), (610, 120), (610, 160), dashed=True)
    f.edge((334, 239), (466, 239))
    f.text(400, 222, "日志复制", 19, muted=True)
    f.edge((190, 268), (190, 308))
    f.edge((610, 268), (610, 308))
    return f


def minilsm():
    f = Figure("MiniLSM：已实现的学习模块", "当前 master 的 scan 合并活跃与不可变 Memtable 的迭代器；"
               "另有 Block、SST 编解码和表迭代器模块。SST 尚未接入 scan，flush 仍有 unimplemented。"
               "两栏分别展示内存读取和表文件结构，不画成已完成的持久化 LSM 系统。")
    f.panel(24, 24, 352, 352, "内存读取")
    f.panel(424, 24, 352, 352, "SST 模块")
    f.box(46, 102, 308, 90, "Memtables", "活跃表 · 不可变表", fill=STORE_BG)
    f.box(46, 268, 308, 86, "MergeIterator", "统一有序视图", size=27)
    f.box(446, 102, 308, 90, "SST", "Blocks · Metadata", fill=STORE_BG)
    f.box(446, 268, 308, 86, "SsTableIterator", "表内遍历", size=26)
    f.edge((200, 192), (200, 268))
    f.edge((600, 192), (600, 268))
    return f


FIGURES = {
    "sglang": sglang, "cuda-learn": cuda, "miniinfer": miniinfer,
    "nebula": nebula, "blog": blog, "bustub": bustub,
    "tinykv": tinykv, "minilsm": minilsm,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, factory in FIGURES.items():
        destination = args.output / f"{name}-architecture.svg"
        destination.write_text(factory().svg(), encoding="utf-8", newline="\n")
        print(destination)


if __name__ == "__main__":
    main()
