"""Generate editable DeepSeek V4 diagrams. Usage: python script.py [output_dir]."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import draw_dsv4_flash

NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)
C = {"ink": "#17212e", "line": "#54677c", "muted": "#526171",
     "neutral": "#f5f7fa", "blue": "#e8f1fa", "green": "#e9f5ef",
     "amber": "#fff3db", "purple": "#f1ebfa"}


def el(p, tag, **a):
    return ET.SubElement(p, f"{{{NS}}}{tag}", {k.replace("_", "-"): str(v) for k, v in a.items()})


class Figure:
    def __init__(self, w, h, title):
        self.root = ET.Element(f"{{{NS}}}svg", width=str(w), height=str(h),
                               viewBox=f"0 0 {w} {h}", role="img", **{"aria-label": title})
        el(self.root, "title").text = title
        defs = el(self.root, "defs")
        m = el(defs, "marker", id="arrow", viewBox="0 0 10 10", refX=9, refY=5,
               markerWidth=7, markerHeight=7, orient="auto-start-reverse")
        el(m, "path", d="M0 0 L10 5 L0 10 Z", fill=C["line"])
        el(defs, "style").text = ('text{font-family:"Microsoft YaHei","PingFang SC",'
                                   '"Noto Sans CJK SC",Arial,sans-serif}')
        el(self.root, "rect", x=0, y=0, width=w, height=h, fill="white")
        self.t(35, 34, title, 25, bold=True, anchor="start")

    def t(self, x, y, s, size=19, bold=False, anchor="middle", color="ink"):
        el(self.root, "text", x=x, y=y, fill=C.get(color, color), font_size=size,
           font_weight=650 if bold else 400, text_anchor=anchor,
           dominant_baseline="middle").text = s

    def box(self, x, y, w, h, lines, color="neutral", size=19):
        el(self.root, "rect", x=x, y=y, width=w, height=h, rx=9, fill=C[color],
           stroke="#91a0b1", stroke_width=1.4)
        if isinstance(lines, str): lines = [lines]
        for i, line in enumerate(lines):
            self.t(x+w/2, y+h/2+(i-(len(lines)-1)/2)*27, line,
                   size if i == 0 else size-2, bold=i == 0)

    def panel(self, x, y, w, h, label):
        el(self.root, "rect", x=x, y=y, width=w, height=h, rx=11, fill="none",
           stroke="#aeb8c5", stroke_width=1.4)
        self.t(x+18, y+25, label, 20, bold=True, anchor="start", color="muted")

    def arrow(self, *points, dash=False):
        el(self.root, "polyline", points=" ".join(f"{x},{y}" for x,y in points),
           fill="none", stroke=C["line"], stroke_width=2,
           stroke_dasharray="6 5" if dash else "none", marker_end="url(#arrow)")

    def save(self, path):
        ET.indent(self.root)
        ET.ElementTree(self.root).write(path, encoding="utf-8", xml_declaration=True)


def overview(path):
    f=Figure(1450, 770, "DeepSeek-V4：模型主干、mHC 与 Attention 的包裹关系")
    f.panel(30,70,310,655,"模型入口 / 输出")
    for y,lines,color in [(125,["Embedding 表 [V,D]","查表 → [T,D]"],"neutral"),
                          (245,["hc_broadcast","复制 → [T,4,D]"],"purple"),
                          (420,["DecoderLayer × L","始终维护四路残差"],"neutral"),
                          (570,["hc_head → Norm","[T,4,D] → [T,D]"],"purple"),
                          (665,["lm_head → logits"],"neutral")]:
        f.box(55,y,260,65,lines,color)
    for a,b in [(190,245),(310,420),(485,570),(635,665)]:f.arrow((185,a),(185,b))
    f.panel(375,70,1035,655,"一个 DecoderLayer：Attention 与 MoE 各自有 mHC pre / post")
    stages=[(420,130,205,"mHC-pre (Attn)","purple"),(665,130,230,"Norm + Attention","blue"),
            (935,130,245,"mHC-post (Attn)","purple"),(420,270,205,"mHC-pre (FFN)","purple"),
            (665,270,230,"Norm + MoE","amber"),(935,270,245,"mHC-post (FFN)","purple")]
    for x,y,w,s,c in stages:f.box(x,y,w,65,s,c)
    f.arrow((625,162),(665,162));f.arrow((895,162),(935,162))
    f.arrow((1180,162),(1230,162),(1230,240),(395,240),(395,302),(420,302))
    f.arrow((625,302),(665,302));f.arrow((895,302),(935,302))
    f.t(440,365,"pre 合成 D 维子层输入；post 混合原残差并写回四路",18,anchor="start",color="muted")
    f.box(420,405,390,105,["Attention：共享 K/V MQA","SWA + C4/CSA 或 C128/HCA","Q heads 不复制 KV latent"],"blue")
    f.box(865,405,490,105,["MoE：路由专家 + 共享专家","Flash：256 routed / top-6","Pro：384 routed / top-6"],"amber")
    f.t(420,556,"Flash 43 层：0, 0, (4,128) ×20, 4",20,bold=True,anchor="start")
    f.t(420,597,"Pro 61 层：128, 128, (4,128) ×29, 4",20,bold=True,anchor="start")
    f.t(420,639,"0 = 纯 SWA；4 / 128 = 对应压缩率，且都保留 SWA",18,anchor="start",color="muted")
    f.t(420,680,"原始 Flash / Pro 的额外 MTP 层不计入主干 L",18,anchor="start",color="muted")
    f.save(path)


def attention(path, c4):
    f=Figure(1400,565,"C4 / CSA：独立的 Core 与 Indexer 压缩" if c4 else "C128 / HCA：全量压缩历史 + 局部 SWA")
    f.box(35,230,175,90,["Attention 输入","x [T,D]"],"purple")
    f.box(290,80,245,75,["SWA KV 投影","每 token 一条 512D"],"blue")
    f.box(290,230,245,90,["Core compressor", "C4：4 新 token → 1 条" if c4 else "C128：128 token → 1 条"],"green")
    f.box(630,80,275,75,["本地 SWA KV cache","最近 128 token"],"blue")
    f.box(630,230,275,90,["Core 压缩 KV cache","512D；共享 K / V"],"green")
    f.box(1030,165,330,110,["共享 K/V MQA", "SWA + top-k C4 条目" if c4 else "SWA + 全部 C128 条目","然后 inverse RoPE + O 投影"],"blue")
    f.arrow((210,275),(250,275),(250,117),(290,117));f.arrow((535,117),(630,117))
    f.arrow((210,275),(290,275));f.arrow((535,275),(630,275))
    f.arrow((905,117),(985,117),(985,190),(1030,190))
    f.arrow((905,275),(985,275),(985,248),(1030,248))
    if c4:
        f.box(290,390,245,85,["Indexer compressor","C4；输出 128D index K"],"amber")
        f.box(630,390,275,85,["Indexer Q × 历史 index K","打分 → top-k 条目索引"],"amber")
        f.arrow((250,275),(250,432),(290,432));f.arrow((535,432),(630,432))
        f.arrow((770,390),(770,320),dash=True)
        f.t(1030,415,"Flash top-k = 512",19,anchor="start")
        f.t(1030,452,"Pro top-k = 1024",19,anchor="start")
        f.t(35,527,"实线：张量数据；虚线：选取索引。Indexer 不向 Core compressor 传递输入。",18,anchor="start",color="muted")
    else:
        f.box(290,390,615,85,["每个完整 128-token block 产生一个长期条目","不重叠、不建 Indexer；未完成尾块由 SWA 保持可见"],"green")
        f.t(35,527,"C4 与 C128 在主干中交错；HCA 的 dense 指压缩条目全部参与。",18,anchor="start",color="muted")
    f.box(1030,65,330,65,["当前 token 的 Q"],"neutral")
    f.arrow((1195,130),(1195,165))
    f.save(path)


def quant(path):
    f=Figure(1440,670,"DeepSeek-V4：逻辑 KV 条目、物理页与压缩状态")
    f.panel(30,75,825,240,"共享 KV latent：512 维，不乘 K/V ×2 或 Query head 数")
    widths=[(55,160,445,"448D NoPE / FP8","448 B","blue"),
            (500,160,200,"64D RoPE / BF16","128 B","green"),
            (700,160,130,"Scale + pad","7 + 1 B","amber")]
    for x,y,w,a,b,c in widths:f.box(x,y,w,90,[a,b],c,size=18)
    f.t(55,280,"逻辑条目：448 + 128 + 7 + 1 = 584 B",20,bold=True,anchor="start")
    f.panel(880,75,525,240,"物理页按 576 B 对齐")
    for y,s in [(135,"SWA：256 条 / 页 → 585 B / 条"),(182,"C4：64 条 / 页 → 585 B / 条"),
                (229,"C128：2 条 / 页 → 864 B / 条")]:f.t(900,y,s,20,anchor="start")
    f.t(900,278,"ceil(page_size ×584 /576) ×576",17,anchor="start",color="muted")
    f.panel(30,345,680,270,"序列压缩改变条目数量")
    f.box(55,400,245,80,["C4：当前4 + 前块4","逐维 softmax(score + APE)"],"green",size=18)
    f.box(400,400,275,80,["每4新 token 写1条","Core 512D + index K 128D"],"green",size=18)
    f.arrow((300,440),(400,440))
    f.box(55,510,245,75,["C128：当前128","逐维 softmax(score + APE)"],"blue",size=18)
    f.box(400,510,275,75,["每128 token 写1条","Core 512D；无 Indexer"],"blue",size=18)
    f.arrow((300,547),(400,547))
    f.panel(740,345,665,270,"其他独立收费的状态")
    f.t(765,406,"Indexer K：FP8 132 B / 条；FP4 实际 68 B / 条",18,anchor="start")
    f.t(765,449,"C4 ring：Core / Indexer 各自保存 KV + score",18,anchor="start")
    f.t(765,492,"C128 ring：按请求预留未完成 block 的状态",18,anchor="start")
    f.t(765,535,"压缩状态默认 FP32；不属于 584 B 的最终 KV",18,anchor="start")
    f.t(765,578,"该图限定 V4 paged FlashMLA 布局",18,anchor="start",color="muted")
    f.t(35,642,"压缩率降低历史条目数；量化降低每条宽度；页对齐、SWA 和状态池决定真实容量。",18,anchor="start",color="muted")
    f.save(path)


def cache(path):
    f=Figure(1440,670,"Shadow prefix 思路：Full 历史与 SWA 窗口独立管理")
    f.panel(30,75,655,535,"当前源码：UnifiedRadixCache + Full / SWA component")
    f.box(65,135,580,70,["同一个 radix node","Full component value + SWA component value"],"neutral")
    f.box(65,265,250,85,["Full 路径锁 / 生命周期","保留长期压缩历史"],"green",size=18)
    f.box(395,265,250,85,["SWA 窗口锁 / LRU","近期窗口独立回收"],"blue",size=18)
    f.arrow((195,205),(195,265));f.arrow((520,205),(520,265))
    f.box(65,410,580,95,["SWATokenToKVPoolAllocator","full slot → SWA slot 映射","Full 页同时确定 C4 / C128 压缩页地址"],"amber")
    f.arrow((195,350),(195,410));f.arrow((520,350),(520,410))
    f.t(65,550,"压缩页随 Full 页重用；压缩历史不靠 ring 自然淘汰",18,anchor="start",color="muted")
    f.panel(725,75,680,535,"窗口 = 4 的概念例子；不代表 DSV4 的实际页粒度")
    f.box(775,135,220,75,["A：token 0,1","Full 有 / SWA 无"],"amber")
    f.box(1080,135,270,75,["B：token 2,3","Full 有 / SWA 有"],"blue")
    f.arrow((995,172),(1080,172))
    f.box(1080,260,270,75,["C：token 4,5","Full 有 / SWA 有"],"blue")
    f.arrow((1215,210),(1215,260))
    f.box(775,390,575,85,["匹配到 C：SWA 连续覆盖4 token，可复用","只匹配到 B：覆盖2 token，回退到安全边界"],"green",size=18)
    f.t(775,528,"内部 SWA 驱逐 → tombstone，保留 Full / radix key",18,anchor="start")
    f.t(775,566,"叶子被驱逐 → 回收整节点；必要时清理 tombstone 祖先",18,anchor="start")
    f.t(35,642,"HiCache / request-ring 等路径另有恢复规则；图示为常规 paged SWA 的设备缓存。",18,anchor="start",color="muted")
    f.save(path)


def parallel(path):
    f=Figure(1450,735,"DSV4 的并行组：PP 跨 stage，CP / EP 在 stage 内复用 ranks")
    f.t(35,77,"结构示例 PP=2、--tp=4：8 GPU；开启 prefill CP 后 Attention TP=1、CP=4",19,anchor="start",color="muted")
    for sx,label,start in [(35,"PP stage 0",0),(755,"PP stage 1",4)]:
        f.panel(sx,115,655,240,label)
        for i in range(4):
            x=sx+22+i*158
            f.box(x,170,140,105,[f"global rank {start+i}",f"CP slot {i}",f"t{i}, t{i+4}, …"],"blue",size=17)
        f.t(sx+25,316,"每个 CP rank 都有本 stage 的完整 KV；EP 后端覆盖本组4 ranks",17,anchor="start")
    f.arrow((690,230),(755,230));f.t(722,190,"PP",18,bold=True)
    f.t(35,392,"PP group：(0,4)、(1,5)、(2,6)、(3,7)；传递对应 slot 的 [Tlocal,4D] 残差",19,anchor="start")
    f.panel(35,430,655,235,"CP + TP-MoE（moe-a2a-backend=none）")
    f.box(55,490,180,75,["本地 token 输入","约 T/4 行"],"blue",size=18)
    f.box(270,490,185,75,["All-gather","按 rank 拼接"],"neutral",size=18)
    f.box(490,490,175,75,["TP-sharded experts","所有专家各1/4宽"],"amber",size=17)
    f.arrow((235,527),(270,527));f.arrow((455,527),(490,527))
    f.t(65,613,"SUM reduce-scatter → 还原本 rank 的 token 输出",18,anchor="start")
    f.panel(755,430,655,235,"CP + EP-MoE（DeepEP / MegaMoE）")
    f.box(775,490,175,75,["本地 token 输入","约 T/4 行"],"blue",size=18)
    f.box(985,490,195,75,["按 top-k Dispatch","跨 ranks 路由"],"neutral",size=18)
    f.box(1215,490,175,75,["完整本地 experts","专家集合按 rank 分"],"amber",size=17)
    f.arrow((950,527),(985,527));f.arrow((1180,527),(1215,527))
    f.t(785,613,"Combine → 回到 token 源 rank；EP=4，不再额外 ×4 GPU",18,anchor="start")
    f.t(35,704,"PP+CP 是源码结构示例；官方 CP 测试为 PP=1。部署组合仍需完整正确性与性能验证。",18,anchor="start",color="muted")
    f.save(path)


def megamoe(path):
    f=Figure(1480,700,"MegaMoE：在一个 kernel 内重叠 NVLink 通信与专家计算")
    f.box(35,90,285,75,["Router / HashTopK","topk_ids + topk_weights"],"neutral")
    f.box(390,90,320,75,["pre_dispatch","量化激活、填 symmetric buffer"],"amber")
    f.box(795,90,640,75,["预先转换的 expert 权重","交错 gate/up；Scale 按 UTCCP 访问布局重排"],"neutral")
    f.arrow((320,127),(390,127))
    f.panel(35,215,1400,380,"一个 MegaMoE launch：图中方块表示设备内角色与依赖，非独立 kernel")
    f.box(65,275,245,120,["Dispatch / pull","读源 rank token + scale","写本地 expert ring"],"blue",size=18)
    f.box(365,275,240,120,["Linear 1 / Tensor Core","gate + up → SwiGLU","乘路由权重","再量化为 FP8"],"green",size=18)
    f.box(665,275,240,120,["Linear 2 / Tensor Core","逐 K block 等 readiness","down projection"],"green",size=18)
    f.box(965,275,245,120,["Write-back / Combine","写回源 rank 的 top-k 槽","等 peer 本轮写完","FP32 求和 → BF16"],"blue",size=18)
    f.arrow((550,165),(550,215));f.arrow((188,395),(188,448),(485,448),(485,395))
    f.arrow((605,335),(665,335));f.arrow((905,335),(965,335))
    f.box(1260,305,145,60,"BF16 y", "purple",size=20);f.arrow((1210,335),(1260,335))
    f.arrow((1435,127),(1450,127),(1450,260),(485,260),(485,275))
    f.arrow((785,260),(785,275))
    f.t(68,488,"Dispatch 与 TMA / UMMA / epilogue 使用不同线程角色，可同时推进不同 tiles。",19,anchor="start")
    f.t(68,530,"计数器、readiness mask 和 barrier 确认生产者完成；消费者完成后才能复用 ring slot。",19,anchor="start")
    f.t(68,571,"Router 与 pre_dispatch 位于 mega kernel 之前；共享专家是否融合取决于后端和权重格式。",18,anchor="start",color="muted")
    f.t(35,630,"适用域：有 peer 访问的 NVLink EP 组；跨节点需另选通信路径。",19,anchor="start")
    f.t(35,673,"容量：每 rank 的 prefill / decode / verify 行数都必须 ≤ 配置 cap；增大 cap 会增加 HBM workspace。",19,anchor="start",color="muted")
    f.save(path)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    overview(out/"dsv4_overview.svg")
    attention(out/"dsv4_csa.svg",True)
    attention(out/"dsv4_hca.svg",False)
    draw_dsv4_flash.build(out/"dsv4_inference.svg")
    quant(out/"dsv4_quant.svg")
    cache(out/"dsv4_swaradixcache.svg")
    parallel(out/"dsv4_parallel.svg")
    megamoe(out/"dsv4_megamoe.svg")


if __name__ == "__main__":
    default=Path(__file__).resolve().parents[1]/"content/paperreadings/llm/img"
    main(Path(sys.argv[1]) if len(sys.argv)>1 else default)
