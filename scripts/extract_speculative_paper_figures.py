"""Extract attributed source figures for the EAGLE/DFlash article.

Requires PyMuPDF. PDF coordinates use points from the top-left; source pages
were inspected with Poppler before choosing these crop rectangles.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "tmp/pdfs/speculative-figures"
OUT = ROOT / "content/sglang/img/speculative-diffusion"


def fetch(url: str, name: str) -> Path:
    path = CACHE / name
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            path.write_bytes(response.read())
    return path


def main():
    import pymupdf

    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    sources = []
    for name, url, cached, page, clip, figure in [
        ("dflash-paper-fig2.png", "https://arxiv.org/pdf/2602.06036v2",
         "dflash-v2.pdf", 4, [82, 64, 516, 267], "Figure 2"),
        ("dspark-paper-fig1.png", "https://arxiv.org/pdf/2607.05147v1",
         "dspark-v1.pdf", 5, [92, 83, 503, 350], "Figure 1"),
    ]:
        source = fetch(url, cached)
        with pymupdf.open(source) as pdf:
            pix = pdf[page - 1].get_pixmap(matrix=pymupdf.Matrix(3, 3),
                                          clip=pymupdf.Rect(clip), alpha=False)
            pix.save(OUT / name)
        sources.append(dict(file=name, url=url, figure=figure, page=page,
                            crop_points=clip, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                            changes="Crop from source PDF at 216 dpi; diagram content unchanged."))

    url = "https://inco.ai/blog/dflash2/"
    source = fetch(url, "dflash2-official.html")
    html = source.read_text(encoding="utf-8")
    start = html.index('<svg', html.index('id="figure-4"'))
    end = html.index('</svg>', start) + len('</svg>')
    svg = html[start:end]
    # Resolve the source website's light-theme CSS variables so the extracted
    # vector is self-contained. First occurrence is the light-theme value.
    styles = html
    for i, href in enumerate(re.findall(r'href="([^"]+\.css[^"]*)"', html)):
        styles += fetch("https://inco.ai" + href, f"dflash2-css-{i}.css").read_text(encoding="utf-8")
    variables = {}
    for key, val in re.findall(r'(--[\w-]+)\s*:\s*([^;}]+)', styles):
        variables.setdefault(key, val.strip())
    for _ in range(8):
        svg = re.sub(r'var\((--[\w-]+)\)', lambda m: variables.get(m[1], m[0]), svg)
    if "var(" in svg:
        raise ValueError("Unresolved source CSS variable")
    root = ET.fromstring(svg)
    root.set("xmlns", "http://www.w3.org/2000/svg")
    root.set("width", "760")
    root.set("height", "380")
    root.set("style", "font-family:Arial,Helvetica,sans-serif;background:#faf9f7")
    root.attrib.pop("class", None)
    name = "dflash2-official-fig4.svg"
    (OUT / name).write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    sources.append(dict(file=name, url=url + "#figure-4", figure="Figure 4",
                        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        changes="Extracted inline SVG; resolved light-theme colors and standalone font. Original labels and geometry retained.",
                        note="Official technical blog, not an independently published DFlash 2 paper PDF."))
    (OUT / "sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Extracted 3 source figures and recorded provenance.")


if __name__ == "__main__":
    main()
