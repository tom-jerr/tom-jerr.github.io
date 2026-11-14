"""Draw the DeepSeek-V4-Flash backbone and SGLang attention execution paths."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET


W, H = 1660, 650
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def el(parent, tag, **attrs):
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {k.replace("_", "-"): str(v) for k, v in attrs.items()})


def text(parent, x, y, value, size=22, weight=400, fill="#17212e", anchor="middle"):
    node = el(parent, "text", x=x, y=y, fill=fill, font_size=size,
              font_weight=weight, text_anchor=anchor, dominant_baseline="middle")
    node.text = value
    return node


def box(parent, x, y, w, h, title, subtitle="", fill="#f6f8fb", stroke="#778698"):
    el(parent, "rect", x=x, y=y, width=w, height=h, rx=12, fill=fill,
       stroke=stroke, stroke_width=1.6)
    if subtitle:
        text(parent, x + w / 2, y + h / 2 - 13, title, 20, 650)
        text(parent, x + w / 2, y + h / 2 + 17, subtitle, 15, 400, "#526171")
    else:
        text(parent, x + w / 2, y + h / 2, title, 20, 650)


def arrow(parent, x1, y1, x2, y2, color="#54677c", dash=None):
    attrs = dict(x1=x1, y1=y1, x2=x2, y2=y2, stroke=color,
                 stroke_width=2.4, marker_end="url(#arrow)")
    if dash:
        attrs["stroke_dasharray"] = dash
    el(parent, "line", **attrs)


def build(out):
    root = ET.Element(f"{{{SVG}}}svg", {"width": str(W), "height": str(H),
                   "viewBox": f"0 0 {W} {H}", "role": "img",
                   "aria-label": "DeepSeek V4 Flash backbone and C4 C128 attention execution order"})
    defs = el(root, "defs")
    marker = el(defs, "marker", id="arrow", viewBox="0 0 10 10", refX=9,
                refY=5, markerWidth=7, markerHeight=7, orient="auto-start-reverse")
    el(marker, "path", d="M 0 0 L 10 5 L 0 10 z", fill="#54677c")
    style = el(root, "style")
    style.text = ('text{font-family:"Microsoft YaHei","PingFang SC",'
                  '"Noto Sans CJK SC",Arial,sans-serif}')
    el(root, "rect", x=0, y=0, width=W, height=H, fill="#ffffff")

    text(root, 55, 37, "DeepSeek-V4-Flash：主干层序与单层执行顺序", 27, 700, anchor="start")
    text(root, 55, 80, "43 个主干层", 18, 700, "#526171", "start")
    el(root, "rect", x=215, y=61, width=180, height=39, rx=7, fill="#f3f5f8", stroke="#9aa7b6")
    text(root, 305, 81, "0–1  纯 SWA", 17, 650)
    el(root, "rect", x=409, y=61, width=880, height=39, rx=7, fill="#e8f1fa", stroke="#80a9cb")
    text(root, 849, 81, "2–42 交错：C4 × 21 / C128 × 20", 17, 650)
    el(root, "rect", x=1303, y=61, width=300, height=39, rx=7, fill="#f3f5f8", stroke="#9aa7b6")
    text(root, 1453, 81, "另有 MTP ratio=0", 17, 650)

    text(root, 55, 139, "Embedding ×4 后：每个主干 DecoderLayer", 19, 700, "#526171", "start")
    macro = [
        (55, 160, 180, "4 路残差流"),
        (259, 160, 180, "mHC-pre (Attn)"),
        (463, 160, 220, "Norm + Attention"),
        (707, 160, 180, "mHC-post (Attn)"),
        (911, 160, 180, "mHC-pre (FFN)"),
        (1115, 160, 180, "Norm + MoE"),
        (1319, 160, 235, "mHC-post (FFN)"),
    ]
    for i, (x, y, w, name) in enumerate(macro):
        box(root, x, y, w, 61, name, fill="#eaf2fb" if "Attention" in name else "#f5f7f9")
        if i:
            prev_x, _, prev_w, _ = macro[i-1]
            arrow(root, prev_x + prev_w + 3, 190, x - 5, 190)

    el(root, "line", x1=573, y1=222, x2=573, y2=264,
       stroke="#80a9cb", stroke_width=2, stroke_dasharray="5 5")
    text(root, 1554, 246, "末层 → hc_head → norm → lm_head", 16, 400,
         "#526171", "end")
    text(root, 55, 272, "Attention 内部：默认单流路径", 19, 700, "#526171", "start")

    # The two attention variants share SWA; indexer exists only for C4.
    text(root, 65, 354, "C4 / CSA", 21, 700, "#2871aa", "start")
    c4 = [
        (205, 315, 206, "Q + SWA KV", "写 128-token 窗口"),
        (440, 315, 280, "Indexer compressor", "index K → score → top-k"),
        (749, 315, 275, "Core compressor", "4:1 overlap → C4 KV"),
        (1053, 315, 270, "稀疏 MQA + SWA", "Flash top-k 512"),
        (1352, 315, 245, "Grouped O", "投影回 hidden"),
    ]
    for i, (x, y, w, a, b) in enumerate(c4):
        box(root, x, y, w, 75, a, b, fill="#eaf2fb", stroke="#80a9cb")
        if i:
            px, _, pw, _, _ = c4[i-1]
            arrow(root, px + pw + 4, 352, x - 6, 352,
                  dash="5 4" if i == 2 else None)

    text(root, 65, 505, "C128 / HCA", 21, 700, "#397759", "start")
    c128 = [
        (205, 466, 206, "Q + SWA KV", "写 128-token 窗口"),
        (749, 466, 275, "Core compressor", "128:1 → C128 KV"),
        (1053, 466, 270, "密集 MQA + SWA", "读取已完成的块"),
        (1352, 466, 245, "Grouped O", "投影回 hidden"),
    ]
    text(root, 578, 447, "不经过 Indexer", 17, 400, "#397759")
    for i, (x, y, w, a, b) in enumerate(c128):
        box(root, x, y, w, 75, a, b, fill="#eaf5ee", stroke="#7dad8e")
        if i:
            px, _, pw, _, _ = c128[i-1]
            arrow(root, px + pw + 4, 503, x - 6, 503)

    el(root, "line", x1=55, y1=575, x2=1603, y2=575, stroke="#d8dee5")
    text(root, 55, 603, "虚线仅表示默认单流提交次序；Indexer top-k 与 Core C4 KV 均在 MQA 前就绪，多流实现可重叠。",
         17, 400, "#526171", "start")
    text(root, 55, 629, "mHC 包裹 Attention 与 MoE 子层；C4/C128 是不同层，不在一个层内串行执行。",
         17, 400, "#526171", "start")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "content/paperreadings/llm/img/dsv4_inference.svg"
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
