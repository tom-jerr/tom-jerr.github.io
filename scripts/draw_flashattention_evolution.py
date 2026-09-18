"""Reproducible, editable FlashAttention scheduling diagrams (logical, not measured).

Sources: FA2 section 3, FA3 figures 1/2 + algorithm 2, FA4 figures 1/2;
NVIDIA PTX: mma, wgmma, tcgen05, cp.async and cp.async.bulk.tensor.
Run: python scripts/draw_flashattention_evolution.py [output_directory]
"""
from pathlib import Path
from html import escape
import argparse

FONT = "'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif"
COLORS = dict(copy="#e1f3ef", mma="#e9eefb", soft="#fff0cc", corr="#f6e4ed", neutral="#f4f5f7")
INK = "#263238"


class SVG:
    def __init__(self, width, height, title, desc):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
                      '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 Z" fill="#637381"/></marker></defs>',
                      f'<rect width="{width}" height="{height}" fill="white"/>']

    def text(self, x, y, label, size=18, anchor="start", bold=False, color=INK):
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" font-weight="{700 if bold else 400}" fill="{color}">{escape(label)}</text>')

    def rect(self, x, y, width, height, kind="neutral"):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{COLORS[kind]}" stroke="#a7b3bd" stroke-width="1.2"/>')

    def line(self, x1, y1, x2, y2, arrow=False, dashed=False):
        self.parts.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="#637381" stroke-width="1.3"' + (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="4 5"' if dashed else '') + '/>')

    def heading(self, title, subtitle):
        self.text(32, 39, title, 26, bold=True)
        self.text(32, 69, subtitle, 16, color="#586975")

    def lane(self, y, label, end=1190):
        self.text(185, y + 30, label, 18, "end", True)
        self.line(210, y + 48, end, y + 48, dashed=True)

    def op(self, y, start, end, label, kind, scale=94, size=18):
        x, width = 218 + start * scale, (end - start) * scale - 5
        self.rect(x, y, width, 44, kind)
        self.text(x + width / 2, y + 28, label, size, "middle")

    def axis(self, y, end=1190):
        self.line(218, y, end, y, arrow=True)
        self.text(end, y + 24, "逻辑时间 →（非实测比例）", 15, "end", color="#586975")

    def save(self, path):
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


def copy_pipeline(out):
    s = SVG(1240, 440, "FA2：copy–compute overlap", "双缓冲复用必须在旧消费者读完之后；tile 内部的 QK、softmax、PV 仍有依赖。")
    s.heading("FA2 · 下一块搬运藏在当前块计算后面", "Ampere-style cp.async + mma.sync；画出两块 SMEM buffer 的保守复用顺序。")
    for y, label in [(118, "异步 copy"), (208, "计算线程")]:
        s.lane(y, label)
    for a, b, name in [(0, 1, "KV₀ → A"), (1, 3, "KV₁ → B"), (4, 6, "KV₂ → A"), (7, 9, "KV₃ → B")]:
        s.op(118, a, b, name, "copy", size=16)
    for j in range(3):
        for k, (label, kind) in enumerate([("QK", "mma"), ("softmax", "soft"), ("PV", "mma")]):
            s.op(208, 1 + 3*j+k, 2 + 3*j+k, f"{label}{'₀₁₂'[j]}", kind, size=16)
    s.line(309, 163, 309, 205, True)
    s.text(322, 188, "ready", 15)
    s.line(594, 207, 594, 164, True)
    s.text(606, 188, "A 可复用", 15)
    s.axis(290)
    s.text(32, 362, "预取负责提前准备数据；softmax 不会因为 copy 异步而自动与本 tile 的 MMA 并行。", 19)
    s.text(32, 397, "A/B 是物理 buffer；下标 0/1/2/3 是逻辑 KV tile。实际实现可更早释放已读完的 K 或 V。", 17, color="#586975")
    s.save(out / "fa2-copy-overlap.svg")


def partition(out):
    s = SVG(1240, 480, "FA2 sliced-Q ownership", "左侧不同 warp 产生同一输出行的部分和；右侧不同 warp 拥有不同输出行。")
    s.heading("FA2 · 把输出行的所有权交给一个 warp", "这里只画两个 warp；sliced-K 指 KV token 方向的分工，不是 QK 点积的 head-dimension split-K。")
    for x, title in [(32, "sliced-K：同一批 Q，拆开 KV"), (650, "sliced-Q：拆开 Q，共用 KV")]:
        s.text(x, 116, title, 22, bold=True)
    for j in range(2):
        y = 147 + 98*j
        s.rect(32, y, 240, 62, "mma")
        s.text(152, y+26, f"warp {j}：Q × K{j}ᵀ", 19, "middle")
        s.text(152, y+50, f"P{j} × V{j} → partial O", 17, "middle")
        s.line(274, y+31, 365, 245, True)
        s.rect(650, y, 253, 62, "mma")
        s.text(776, y+26, f"warp {j}：Q{j} × Kᵀ", 19, "middle")
        s.text(776, y+50, f"P{j} × V → O{j}", 17, "middle")
        s.line(906, y+31, 957, y+31, True)
        s.rect(962, y, 240, 62, "copy")
        s.text(1082, y+37, f"独立输出 Q{j} 对应的行", 19, "middle")
    s.rect(370, 196, 226, 99, "corr")
    s.text(483, 228, "SMEM 交换", 21, "middle")
    s.text(483, 258, "同步 / 合并 partial O", 17, "middle")
    s.text(483, 282, "（同一输出行）", 16, "middle")
    s.text(32, 368, "左：多个 warp 参与同一输出行，需要在片上合并贡献。", 20)
    s.text(32, 405, "右：不同输出行互不归约；warp 内仍可能为 rowmax / rowsum 做 shuffle。", 20)
    s.text(32, 447, "图示 ownership，而非完整指令布局；收益来自减少跨 warp 通信，不是消除所有同步。", 17, color="#586975")
    s.save(out / "fa2-warp-partition.svg")


def hopper(out):
    s = SVG(1240, 690, "FA3 两种 GEMM–softmax overlap", "上部为跨 warpgroup 的 ping-pong，下部为同一 warpgroup 跨迭代的 PV 与下一块 softmax 重叠。")
    s.heading("FA3 · WGMMA 异步后，找到独立的 softmax 工作", "稳态示意，省略 TMA lane 和 prologue / tail；所有 PV 都必须等待自己的 P。")
    s.text(32, 111, "A. 两个 consumer warpgroup：ping-pong", 21, bold=True)
    for y, label in [(139, "consumer A"), (204, "consumer B")]:
        s.lane(y, label)
    for start in (0, 4, 8):
        s.op(139, start, start+2, "PV → QK", "mma")
        s.op(204, start, start+2, "softmax", "soft")
    for start in (2, 6):
        s.op(139, start, start+2, "softmax", "soft")
        s.op(204, start, start+2, "PV → QK", "mma")
    s.text(218, 288, "一个组使用 Tensor Core 时，让另一个组使用 MUFU；两组交替进入重计算区。", 17)
    s.text(32, 346, "B. 一个 consumer warpgroup：跨 KV 迭代的软件流水", 21, bold=True)
    for y, label in [(378, "Tensor Core"), (460, "标量 / MUFU")]:
        s.lane(y, label)
    for a,b,label in [(0,2,"QKⱼ"),(2,4,"Pⱼ₋₁Vⱼ₋₁"),(5,7,"QKⱼ₊₁"),(7,9,"PⱼVⱼ")]:
        s.op(378,a,b,label,"mma")
    for a,b,label,kind in [(2,4,"softmaxⱼ","soft"),(4,5,"校正 O","corr"),(7,9,"softmaxⱼ₊₁","soft"),(9,10,"校正 O","corr")]:
        s.op(460,a,b,label,kind,size=17)
    s.line(403,422,403,457,True)
    s.line(591,422,591,457,True)
    s.line(873,422,873,457,True)
    s.axis(545)
    s.text(32, 615, "QKⱼ 完成后可读 Sⱼ；PVⱼ₋₁ 完成后才可改写 O 或复用 Pⱼ₋₁ 的寄存器。", 20)
    s.text(32, 652, "同时在途的 accumulator / P buffer 增加寄存器压力；SMEM stage 数不等于这层流水深度。", 17, color="#586975")
    s.save(out / "fa3-gemm-softmax-overlap.svg")


def blackwell_forward(out):
    s = SVG(1340, 710, "FA4 forward overlap", "L/H 为两个独立 query tile。MMA、softmax、correction 通过 TMEM 和 barrier 交接；O 校正必须在对应 PV 之前完成。")
    s.heading("FA4 · TMEM 把 MMA、softmax 与输出校正分开", "L / H 各是一个 128-row Q tile；j 是 KV tile。仅表示合法依赖，省略 P 分段提交及边界阶段。")
    scale=103
    for y,label in [(108,"TMA"),(195,"Tensor Core"),(293,"softmax L"),(381,"softmax H"),(482,"correction")]:
        s.lane(y,label,end=1280)
    for a,b,label in [(0,2,"预取 K/Vⱼ₊₁"),(4,6,"预取 K/Vⱼ₊₂")]:
        s.op(108,a,b,label,"copy",scale)
    tc=[(0,1,"QKᴸⱼ"),(1,2,"QKᴴⱼ"),(3,4,"PVᴸⱼ"),(4,5,"QKᴸⱼ₊₁"),(5,6,"PVᴴⱼ"),(6,7,"QKᴴⱼ₊₁"),(7,8,"PVᴸⱼ₊₁"),(8,9,"QKᴸⱼ₊₂"),(9,10,"PVᴴⱼ₊₁")]
    for a,b,label in tc:s.op(195,a,b,label,"mma",scale,16)
    for a,b,label in [(1,3,"Sᴸⱼ → Pᴸⱼ"),(5,7,"Sᴸⱼ₊₁ → Pᴸⱼ₊₁")]:s.op(293,a,b,label,"soft",scale)
    for a,b,label in [(3,5,"Sᴴⱼ → Pᴴⱼ"),(7,9,"Sᴴⱼ₊₁ → Pᴴⱼ₊₁")]:s.op(381,a,b,label,"soft",scale)
    for a,b,label in [(2,3,"αᴸⱼ Oᴸ"),(4,5,"αᴴⱼ Oᴴ"),(6,7,"αᴸⱼ₊₁ Oᴸ"),(8,9,"αᴴⱼ₊₁ Oᴴ")]:s.op(482,a,b,label,"corr",scale,15)
    s.line(318,239,318,290,True)
    s.text(330,271,"S ready",15)
    # A dependency gate at L's first PV, with no arrow passing through another operation.
    s.line(527,482,527,241,True,True)
    s.text(540,456,"P ready 且 O rescaled",17)
    s.axis(572,end=1280)
    s.text(32,635,"两个 softmax 组错开指数计算；correction 可与另一 Q tile 的 MMA / softmax 重叠。",20)
    s.text(32,674,"同一 O 的 rescale 与 PV 写回仍互斥。TMEM 提供交接位置，不会自动解除数据依赖。",18,color="#586975")
    s.save(out / "fa4-forward-overlap.svg")


def blackwell_backward(out):
    s=SVG(1340,540,"FA4 backward overlap","1-CTA 主循环：S_j、dK_(j-1)、dQ_(j-1)、dP_j、dV_j；softmax 重算与上一迭代的梯度 MMA 重叠。")
    s.heading("FA4 backward · 用上一轮的梯度 MMA 覆盖本轮 softmax", "1-CTA 稳态的逻辑调度；j 表示固定 KV tile 下扫描的 Q tile。2-CTA 的 dP / dQ 顺序另有调整。")
    scale=103
    for y,label in [(133,"Tensor Core"),(246,"标量 / MUFU")]:s.lane(y,label,end=1280)
    labels=["Sⱼ","dKⱼ₋₁","dQⱼ₋₁","dPⱼ","dVⱼ","Sⱼ₊₁","dKⱼ","dQⱼ","dPⱼ₊₁","dVⱼ₊₁"]
    for i,label in enumerate(labels):s.op(133,i,i+1,label,"mma",scale,18)
    for a,b,label in [(1,3,"exp(Sⱼ − LSE) → Pⱼ"),(4,5,"dSⱼ"),(6,8,"exp(Sⱼ₊₁ − LSE) → Pⱼ₊₁"),(9,10,"dSⱼ₊₁")]:s.op(246,a,b,label,"soft",scale,16)
    for t in [1,4,6,9]:s.line(218+t*scale-3,177,218+t*scale-3,243,True)
    s.axis(329,end=1280)
    s.text(32,397,"TMEM 长期保留：dK、dV；轮转区：S / P，以及 dP / dS / dQ。",20)
    s.text(32,434,"覆盖旧名字之前，必须完成读取 / staging / reduction；图中不展开这些 buffer 交接。",19)
    s.text(32,483,"prologue 先产生第一轮 P、dP、dS；tail 排空最后的 dK / dQ。不能直接删掉首尾。",18,color="#586975")
    s.save(out / "fa4-backward-overlap.svg")


def instruction_flows(out):
    """Hardware-labeled data paths; dashed arrows carry instruction issue only.

    PTX contracts: cp.async; cp.async.bulk.tensor; mma.sync; wgmma;
    tcgen05 issue granularity and memory consistency. No physical floorplan claim.
    """
    records = [
        dict(name="ampere-mma-flow", title="Ampere · mma.sync：warp 持有输入和输出 fragment",
             sub="示例为 FP16/BF16 → FP32 的 warp MMA；数据已先搬到 SMEM。实线是数据，虚线是发起。",
             issuer="32-thread warp → SM instruction issue", issue=2,
             boxes=[("Shared Memory", "A / B tile", "copy"), ("Register File", "A / B / C fragments", "neutral"),
                    ("第 3 代 Tensor Core", "mma.sync", "mma"), ("Register File", "D：FP32 accumulator", "neutral"), ("CUDA Core / MUFU", "softmax / 行统计量", "soft")],
             edges=["ldmatrix", "operand read", "writeback", "依赖就绪后读取"],
             protocol="执行顺序：准备 fragment → 全 warp 发起 mma.sync → 结果依赖由 scoreboard 追踪 → 消费 D。",
             notes=[".sync 约束参与线程会合，不是整个 CTA / GPU 同步；其他 warp 或独立指令仍可推进。",
                    "本路径没有 WGMMA 式 commit / wait_group；C 与 D 可以使用同一组寄存器做累加。"]),
        dict(name="hopper-mma-flow", title="Hopper · WGMMA：异步读 SMEM，结果仍由 warpgroup 的寄存器持有",
             sub="图示 SS 输入路径；RS 变体的 A 来自寄存器，B 仍来自 SMEM。WGMMA 是 128-thread collective。",
             issuer="4 warps / 128 threads → collective issue", issue=1,
             boxes=[("Shared Memory", "A / B：SMEM descriptor", "copy"), ("第 4 代 Tensor Core", "wgmma.mma_async", "mma"),
                    ("Register File", "D：warpgroup accumulator", "neutral"), ("等待对应 async group", "wgmma.wait_group", "corr"), ("CUDA Core / MUFU", "softmax 使用 D", "soft")],
             edges=["直接读 operand", "异步写回", "结果就绪门槛", "安全访问 fragment"],
             protocol="执行顺序：operand ready + fence → issue MMA → commit_group → 独立工作 → wait_group → 读 D。",
             notes=["wait 之前不能读写在途 accumulator；SMEM operand 也不能在异步读取完成前被覆盖。",
                    "wgmma.fence 管寄存器访问次序；普通 SMEM store → async reader 还要检查 proxy ordering。"]),
        dict(name="blackwell-mma-flow", title="Blackwell SM100 · tcgen05.mma：单线程发起，accumulator 写入 TMEM",
             sub="图示 SS 输入路径；TS 变体让 A 直接来自 TMEM。寄存器不再长期承载整个 MMA accumulator。",
             issuer="一个 elected thread → tcgen05.mma issue", issue=1,
             boxes=[("Shared Memory", "A / B：SMEM descriptor", "copy"), ("第 5 代 Tensor Core", "tcgen05.mma", "mma"),
                    ("Tensor Memory", "D / 累加值：TMEM", "copy"), ("Register File", "tcgen05.ld 后的 fragment", "neutral"), ("CUDA Core / MUFU", "softmax / correction", "soft")],
             edges=["直接读 operand", "异步写 TMEM", "完成后 tcgen05.ld", "wait::ld 后使用"],
             protocol="完成协议：MMA → tcgen05.commit / mbarrier → consumer wait / ordering → tcgen05.ld → wait::ld。",
             notes=["MMA 单线程发起 ≠ TMEM load/store 单线程完成；tcgen05.ld / st 是 warp collective。",
                    "2-CTA MMA 可由 CTA pair 中一个线程发起，peer CTA 必须存活；相关资源按配对协议管理。"]),
        dict(name="ampere-load-flow", title="Ampere · cp.async：还没有 TMA，由线程生成每段地址",
             sub="单条 cp.async 搬 4 / 8 / 16 bytes（取决于变体）；多个线程的请求组成 tile。数据绕过通用 RF 中转。",
             issuer="各 copy thread：地址计算 / predicate → cp.async", issue=2,
             boxes=[("HBM", "global 源数据", "neutral"), ("L2 / cache path", ".ca 与 .cg 策略不同", "neutral"),
                    ("异步 global→shared 路径", "硬件推进 copy", "copy"), ("Shared Memory", "本 CTA 的 tile buffer", "copy"), ("MMA consumer", "ldmatrix → RF → mma.sync", "mma")],
             edges=["L2 miss 时取数", "cache 数据供给", "直接写 SMEM", "copy ready 后读取"],
             protocol="完成协议：cp.async → commit_group → 独立计算 → wait_group → 必要的 CTA 同步 → consume。",
             notes=["每线程保留地址 / predicate 寄存器，但没有 payload 的 global-load → RF → shared-store 中转。",
                    "cp.async.wait_group 追踪发起线程的 copy groups；它不是整个 CTA 的通用 barrier。"]),
        dict(name="hopper-load-flow", title="Hopper · TMA load：一个线程提交 tensor map 和 tile 坐标",
             sub="TMA 根据 descriptor 处理多维地址、边界填充与支持的 SMEM swizzle；并非任意 layout 都能自动转换。",
             issuer="一个 producer thread：tensor map + coords → TMA", issue=1,
             boxes=[("HBM / L2", "global tensor", "neutral"), ("TMA 搬运单元", "地址生成 / tile transfer", "copy"),
                    ("Shared Memory", "Q / K / V tile", "copy"), ("第 4 代 Tensor Core", "WGMMA 消费 SMEM", "mma"), ("Register File", "MMA accumulator", "neutral")],
             edges=["TMA 取数", "写 SMEM / swizzle", "barrier ready 后读取", "MMA 结果"],
             protocol="完成协议：arm mbarrier / expect bytes → TMA load → complete_tx → consumer wait → consume。",
             notes=["load 完成通知按 transaction bytes 记账；thread arrival、copy completion、buffer release 是不同事件。",
                    "Hopper 已支持 cluster multicast：同一 tile 可送往选中的多个 CTA；TMA 不直接写 accumulator。"]),
        dict(name="blackwell-load-flow", title="Blackwell SM100 · TMA load：保留 G2S 路径，扩展 CTA-pair 完成通知",
             sub="普通 1-CTA load 仍类似 Hopper；图下重点说明 .cta_group::2，不把它画成自动搬两份数据。",
             issuer="一个 producer thread → cp.async.bulk.tensor", issue=1,
             boxes=[("HBM / L2", "global tensor", "neutral"), ("TMA 搬运单元", "tensor map + coordinates", "copy"),
                    ("Shared Memory", "选定 destination CTA", "copy"), ("第 5 代 Tensor Core", "tcgen05.mma / CTA pair", "mma"), ("Tensor Memory", "MMA 的输出", "copy")],
             edges=["TMA 取数", "仍然先写 SMEM", "等待需要的 operand", "MMA 写入 TMEM"],
             protocol=".cta_group::2：load 的 complete_tx 可通知 destination CTA 或其 peer CTA 中的 mbarrier。",
             notes=["例如数据写 CTA1 的 SMEM，完成信号交给 CTA0 的 barrier；consumer 须等齐全部所需事务。",
                    "TMA load ≠ global→TMEM；cta_group::2 ≠ multicast；tcgen05.cp 是另一条 SMEM→TMEM 指令。"]),
    ]
    for rec in records:
        s=SVG(1440,440,rec["title"],rec["sub"]+" "+rec["protocol"])
        s.heading(rec["title"],rec["sub"])
        center=40+rec["issue"]*276+128
        s.rect(center-285,96,570,43,"neutral")
        s.text(center,123,rec["issuer"],18,"middle")
        s.line(center,139,center,176,True,True)
        for i,(title,detail,kind) in enumerate(rec["boxes"]):
            x=40+i*276
            s.rect(x,180,256,84,kind)
            s.text(x+128,211,title,19,"middle",True)
            s.text(x+128,244,detail,16,"middle")
            if i<4:
                s.line(x+257,221,x+273,221,True)
                s.text(x+265,286,rec["edges"][i],15,"middle")
        s.text(40,335,rec["protocol"],18)
        s.text(40,381,rec["notes"][0],18)
        s.text(40,416,rec["notes"][1],17,color="#586975")
        s.save(out/(rec["name"]+".svg"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",nargs="?",type=Path,default=Path(__file__).resolve().parents[1]/"content/cuda/img/flashattention-evolution")
    out=parser.parse_args().output
    out.mkdir(parents=True,exist_ok=True)
    for draw in [partition,copy_pipeline,hopper,blackwell_forward,blackwell_backward,instruction_flows]:draw(out)
    print(f"Wrote 11 editable SVG diagrams to {out}")


if __name__=="__main__":main()
