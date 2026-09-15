"""Structural QA for a single Quartz markdown article.

Checks: code fences, LaTeX delimiter pairing, heading levels, link-reference
definitions/usages, image file existence, and duplicated H1 titles.
"""
import re
import sys
from pathlib import Path

DOC = Path(r"e:\blog\content\sglang\Eagle2 in SGLang.md")
problems = []

text = DOC.read_text(encoding="utf-8")
lines = text.splitlines()

# 1. Code fences balanced
fences = [i for i, ln in enumerate(lines, 1) if ln.strip().startswith("```")]
if len(fences) % 2:
    problems.append(f"odd code-fence count: {len(fences)}")

# 2. LaTeX delimiters (outside code fences)
in_fence = False
body_dollars = 0
for i, ln in enumerate(lines, 1):
    if ln.strip().startswith("```"):
        in_fence = not in_fence
        continue
    if in_fence:
        continue
    stripped = ln.replace("$$", "")
    dollars = stripped.count("$")
    if dollars % 2:
        problems.append(f"line {i}: odd inline $ count ({dollars})")
    body_dollars += dollars
if text.count("$$") % 2:
    problems.append(f"odd $$ count ({text.count('$$')})")

# 3. Headings: no skipped levels, single H1 (outside code fences)
in_fence = False
prev = 0
h1s = []
for i, ln in enumerate(lines, 1):
    if ln.strip().startswith("```"):
        in_fence = not in_fence
        continue
    if in_fence:
        continue
    m = re.match(r"^(#{1,6})\s", ln)
    if not m:
        continue
    level = len(m.group(1))
    if level == 1:
        h1s.append((i, ln))
    if prev and level > prev + 1:
        problems.append(f"line {i}: heading jumps {prev}->{level}")
    prev = max(prev, level) if level > prev else prev
if len(h1s) != 1:
    problems.append(f"expected exactly 1 H1, found {len(h1s)}: {h1s}")

# 4. Link reference definitions and usages
defs = set(re.findall(r"^\[([\w-]+)\]:\s+\S+", text, re.M))
if len(defs) != len(set(re.findall(r"^\[([\w-]+)\]:\s+\S+", text, re.M))):
    problems.append("duplicate link reference definitions")
usages = set(re.findall(r"\]\[([\w-]+)\]", text))
missing = usages - defs
if missing:
    problems.append(f"undefined link refs: {sorted(missing)}")
# Footnote-style definitions must start at line begin with a non-space after colon is URL
bare_defs = [ln for ln in lines if re.match(r"^\[[\w-]+\]:\s*$", ln.strip())]
if bare_defs:
    problems.append(f"empty link definitions: {bare_defs[:5]}")

# 5. Images referenced must exist
base = DOC.parent
for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)\)", text):
    p = (base / m.group(1)).resolve()
    if not p.exists():
        problems.append(f"missing image: {m.group(1)}")

# 6. Front matter fields
fm = text.split("---", 2)
if len(fm) < 3:
    problems.append("missing front matter")
else:
    head = fm[1]
    for key in ("title", "created", "tags", "description"):
        if not re.search(rf"^{key}:", head, re.M):
            problems.append(f"front matter missing '{key}'")
    title = re.search(r"^title:\s*(.+)$", head, re.M)
    body_title = re.search(r"^#\s+(.+)$", text.split("---", 2)[2], re.M)
    if title and body_title and title.group(1).strip() != body_title.group(1).strip():
        problems.append(f"title/H1 mismatch: {title.group(1)!r} vs {body_title.group(1)!r}")

if problems:
    print("PROBLEMS:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("article clean")
