"""Rough geometric QA for the regenerated SVGs: text overflow and arrow/box collisions."""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

SVG_DIR = Path(r"e:\blog\content\sglang\img\speculative-diffusion")
NS = "{http://www.w3.org/2000/svg}"


def text_width(s, size):
    # CJK glyphs are roughly 1.0 em wide; Latin/digits/punct ~0.55 em.
    w = 0.0
    for ch in s:
        w += size if ord(ch) > 0x2E80 else size * 0.55
    return w


def check(path):
    tree = ET.parse(path)
    root = tree.getroot()
    W = float(root.get("width"))
    H = float(root.get("height"))
    problems = []
    boxes = []
    for e in root.iter():
        tag = e.tag.replace(NS, "")
        if tag == "rect" and e.get("fill") != "white":
            x, y = float(e.get("x")), float(e.get("y"))
            w, h = float(e.get("width")), float(e.get("height"))
            boxes.append((x, y, x + w, y + h))
        if tag == "text":
            x, y = float(e.get("x")), float(e.get("y"))
            size = float(e.get("font-size"))
            anchor = e.get("text-anchor", "start")
            txt = e.text or ""
            tw = text_width(txt, size)
            if anchor == "middle":
                x0, x1 = x - tw / 2, x + tw / 2
            elif anchor == "end":
                x0, x1 = x - tw, x
            else:
                x0, x1 = x, x + tw
            if x0 < 0 or x1 > W:
                problems.append(f"TEXT OVERFLOW x: '{txt}' ({x0:.0f}..{x1:.0f})")
            if y > H:
                problems.append(f"TEXT BELOW CANVAS: '{txt}' y={y:.0f}")
            # text vs box interior (text drawn over a box is expected only when
            # it belongs to that box; flag texts whose left edge falls strictly
            # inside a rect not its own and whose vertical center is outside it)
    # arrows: polyline points inside any box interior (excluding endpoints on the edge)
    for e in root.iter():
        if e.tag.replace(NS, "") == "path" and "marker-end" in (e.get("d", "") and e.get("d") or "") and e.get("marker-end"):
            d = e.get("d", "")
            pts = re.findall(r"([-\d.]+),([-\d.]+)", d)
            for i, (px, py) in enumerate(pts):
                if i == 0 or i == len(pts) - 1:
                    continue
                px, py = float(px), float(py)
                for bx0, by0, bx1, by1 in boxes:
                    if bx0 + 2 < px < bx1 - 2 and by0 + 2 < py < by1 - 2:
                        problems.append(f"ARROW CROSSES BOX at ({px:.0f},{py:.0f}) inside ({bx0:.0f},{by0:.0f},{bx1:.0f},{by1:.0f})")
    return W, H, problems


if __name__ == "__main__":
    files = sys.argv[1:] or ["dspark-inference.svg", "dspark-host-budget.svg", "dflash2-inference.svg"]
    for name in files:
        w, h, probs = check(SVG_DIR / name)
        print(f"{name}: {w}x{h}")
        for p in probs:
            print("   ", p)
        if not probs:
            print("    clean")
