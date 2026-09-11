"""Generate the editable SVG/PNG figures used by the CuTe FA2 case study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


COLORS = {
    "cute_s2": "#3f51b5",
    "cute_s3": "#00897b",
    "official_fa2": "#d97706",
    "q": "#dbeafe",
    "kv0": "#dcfce7",
    "kv1": "#fef3c7",
    "kv2": "#fce7f3",
    "compute": "#e5e7eb",
    "sync": "#fee2e2",
    "ink": "#172033",
}


def set_font():
    candidates = ["Microsoft YaHei", "Noto Sans CJK SC", "DejaVu Sans"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((x for x in candidates if x in installed), "DejaVu Sans")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.hashsalt"] = "tom-jerr-cute-fa2-case"


def save(fig, output: Path):
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight", metadata={"Date": None})


def load_rows(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def performance(rows, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    names = [("cute_s2", "CuTe 2-stage"), ("cute_s3", "CuTe 3-stage"),
             ("official_fa2", "FlashAttention 2.8.3")]
    for ax, causal, title in zip(axes, (False, True), ("Non-causal", "Causal")):
        subset = [r for r in rows if r["batch"] == 1 and r["heads"] == 1
                  and r["causal"] is causal]
        for name, label in names:
            values = sorted((r for r in subset if r["implementation"] == name), key=lambda x: x["n"])
            ax.plot([x["n"] for x in values], [x["median_us"] for x in values],
                    marker="o", markersize=4.5, linewidth=2, label=label,
                    color=COLORS[name])
            ax.fill_between([x["n"] for x in values], [x["min_us"] for x in values],
                            [x["max_us"] for x in values], color=COLORS[name], alpha=.08)
        ax.set(xscale="log", yscale="log", xlabel="Sequence length N", ylabel="Latency (μs)", title=title)
        ax.set_xticks([64, 128, 256, 512, 1024, 2048, 4096],
                      ["64", "128", "256", "512", "1K", "2K", "4K"])
        ax.grid(True, which="both", alpha=.25, linewidth=.7)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("D=64, B=1, H=1 — CUDA Graph median; band = observed min–max", fontsize=12)
    save(fig, output)
    plt.close(fig)


def stage_ratio(rows, output: Path):
    cases = [(1, 8, 512), (1, 8, 2048), (1, 8, 4096), (4, 8, 512), (4, 8, 2048)]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True, sharey=True)
    for ax, causal, title in zip(axes, (False, True), ("Non-causal", "Causal")):
        ratios = []
        labels = []
        for b, h, n in cases:
            selected = {(r["implementation"]): r["median_us"] for r in rows
                        if (r["batch"], r["heads"], r["n"], r["causal"]) == (b, h, n, causal)}
            ratios.append(selected["cute_s3"] / selected["cute_s2"])
            labels.append(f"B{b} H{h}\nN{n}")
        bars = ax.bar(labels, ratios, color=COLORS["cute_s3"], width=.65)
        ax.axhline(1, color=COLORS["ink"], linewidth=1)
        for bar, value in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, value + .015, f"{value:.2f}×",
                    ha="center", va="bottom", fontsize=9)
        ax.set(title=title, ylabel="3-stage latency / 2-stage latency", ylim=(0, max(1.85, max(ratios)+.18)))
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("An extra shared-memory stage usually hurts this kernel on SM89", fontsize=12)
    save(fig, output)
    plt.close(fig)


def box(ax, x, y, w, h, text, color, fontsize=10):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=.06",
                           facecolor=color, edgecolor=COLORS["ink"], linewidth=.9)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, color=COLORS["ink"])


def pipeline(output: Path):
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")
    lanes = [(4.75, "cp.async producer"), (3.55, "stage buffer"),
             (2.35, "Tensor Core + softmax"), (1.15, "ownership / visibility")]
    for y, label in lanes:
        ax.text(.25, y+.3, label, ha="left", va="center", fontsize=10, color=COLORS["ink"])
        ax.plot([1.9, 12.6], [y, y], color="#d1d5db", linewidth=.8)
    xs = [2.45, 4.9, 7.35, 9.8]
    for i, x in enumerate(xs):
        ax.text(x+.9, 5.35, f"iteration {i}", ha="center", fontsize=10, color="#4b5563")
        if i < 3:
            box(ax, x, 4.45, 1.85, .62, f"load K/V tile {i+1}", COLORS[f"kv{(i+1)%3}"])
        else:
            box(ax, x, 4.45, 1.85, .62, "pipeline drain", COLORS["sync"])
        box(ax, x, 3.25, 1.85, .62, f"read stage {i%2}\nK/V tile {i}", COLORS[f"kv{i%3}"])
        box(ax, x, 2.05, 1.85, .62, f"QK tile {i} → softmax\n→ PV", COLORS["compute"], 8.9)
        box(ax, x, .85, 1.85, .62, "wait_group +\n__syncthreads", COLORS["sync"], 9.2)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x+1.87, 4.75), (xs[i+1]-.05, 3.58),
                                         arrowstyle="-|>", mutation_scale=11, linewidth=1,
                                         connectionstyle="arc3,rad=-.12", color=COLORS["ink"]))
        ax.add_patch(FancyArrowPatch((x+.9, 3.22), (x+.9, 2.7), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1, color=COLORS["ink"]))
        ax.add_patch(FancyArrowPatch((x+.9, 2.02), (x+.9, 1.5), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1, color=COLORS["ink"]))
    ax.text(2.45, .23, "2-stage ring: while tile t is consumed, tile t+1 is produced into the other stage.\n"
            "wait_group controls async-copy completion; __syncthreads transfers ownership between producer and consumer threads.",
            fontsize=10, color=COLORS["ink"], va="bottom")
    save(fig, output)
    plt.close(fig)


def cta_warp_thread(output: Path):
    """Show how a 64x64 score tile is owned by CTA, warp, lane group and lane."""
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.5), constrained_layout=True)
    ink = COLORS["ink"]
    warp_colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#fce7f3"]

    ax = axes[0]
    ax.set(xlim=(0, 64), ylim=(64, 0), aspect="equal", title="CTA: one 64 × 64 score tile")
    for w, color in enumerate(warp_colors):
        ax.add_patch(Rectangle((0, 16*w), 64, 16, facecolor=color, edgecolor=ink, linewidth=1.1))
        ax.text(32, 16*w+8, f"warp {w}: rows {16*w}…{16*w+15}", ha="center", va="center", fontsize=9)
    ax.set_xlabel("N / key columns"); ax.set_ylabel("M / query rows")
    ax.set_xticks([0, 16, 32, 48, 64]); ax.set_yticks([0, 16, 32, 48, 64])

    ax = axes[1]
    ax.set(xlim=(0, 64), ylim=(16, 0), aspect="auto", title="One warp: eight 4-lane groups")
    for g in range(8):
        color = "#bfdbfe" if g % 2 == 0 else "#dbeafe"
        for row in (g, g+8):
            ax.add_patch(Rectangle((0, row), 64, 1, facecolor=color, edgecolor="white", linewidth=.6))
        ax.text(32, g+.5, f"group {g}", ha="center", va="center", fontsize=7.5, color=ink)
        ax.text(32, g+8+.5, f"group {g}", ha="center", va="center", fontsize=7.5, color=ink)
    ax.set_xlabel("all 64 columns are completed by q = lane mod 4")
    ax.set_ylabel("warp-local row")
    ax.set_xticks([0, 16, 32, 48, 64]); ax.set_yticks([0, 8, 16])

    ax = axes[2]
    ax.set(xlim=(-1, 65), ylim=(4.2, -.7), title="One 4-lane group, one row")
    lane_colors = ["#2563eb", "#059669", "#d97706", "#db2777"]
    for q, color in enumerate(lane_colors):
        for j in range(8):
            for c in range(2):
                n = 8*j + 2*q + c
                ax.add_patch(Rectangle((n, q), 1, .78, facecolor=color, edgecolor="white", linewidth=.25))
        ax.text(-.7, q+.39, f"q={q}", ha="right", va="center", fontsize=8.5, color=ink)
    ax.set_xlabel("n = 8j + 2q + c   (j=0…7, c=0,1)")
    ax.set_yticks([]); ax.set_xticks([0, 8, 16, 24, 32, 40, 48, 56, 64])
    ax.text(32, 4.05, "4 lanes × 16 values = one complete 64-value row",
            ha="center", va="center", fontsize=9.5, color=ink)
    fig.suptitle("TiledMMA ownership: CTA → warp → 4-lane reduction group", fontsize=13)
    save(fig, output)
    plt.close(fig)


def fragment_layout(output: Path):
    """Derive the C fragment and its register-only C-to-A re-interpretation."""
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 8.3), constrained_layout=True)
    ink = COLORS["ink"]

    ax = axes[0, 0]
    ax.set(xlim=(0, 8), ylim=(16, 0), aspect="equal", title="PTX m16n8k16 C/D atom")
    q, g = 1, 2
    for y in range(16):
        for x in range(8):
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=.45))
    labels = [(2*q, g, "c0"), (2*q+1, g, "c1"), (2*q, g+8, "c2"), (2*q+1, g+8, "c3")]
    for x, y, label in labels:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#60a5fa", edgecolor=ink, linewidth=.9))
        ax.text(x+.5, y+.5, label, ha="center", va="center", fontsize=8, color=ink)
    ax.set_xlabel("atom n"); ax.set_ylabel("atom m")
    ax.set_xticks(range(9)); ax.set_yticks([0, 2, 8, 10, 16])
    ax.text(4, 15.3, "example lane: g=2, q=1", ha="center", fontsize=9, color=ink)

    ax = axes[0, 1]; ax.axis("off"); ax.set_title("CuTe C fragment")
    box(ax, .04, .67, .92, .20, "trS: ((_2,_2),_1,_8)\nstride: ((_1,_2),_0,_4)", "#dbeafe", 12)
    box(ax, .04, .37, .92, .18, "coordinate ((c,r), 0, j)\nslot = c + 2r + 4j", "#dcfce7", 11)
    box(ax, .04, .07, .92, .18, "m = 16w + g + 8r\nn = 8j + 2q + c", "#fef3c7", 11)
    ax.add_patch(FancyArrowPatch((.5,.66),(.5,.56), arrowstyle="-|>", mutation_scale=12, color=ink))
    ax.add_patch(FancyArrowPatch((.5,.36),(.5,.26), arrowstyle="-|>", mutation_scale=12, color=ink))

    ax = axes[1, 0]; ax.axis("off"); ax.set_title("The same 32 half registers, viewed as MMA A")
    box(ax, .03, .64, .94, .22, "A view: (((_2,_2),_2),_1,_4)\nslot = c + 2r + 4h + 8κ", "#fce7f3", 11)
    box(ax, .03, .33, .94, .20, "m = 16w + g + 8r\nk = 16κ + 2q + c + 8h", "#ede9fe", 11)
    box(ax, .03, .04, .94, .18, "j = 2κ+h\nC slot = A slot", "#dcfce7", 11)
    ax.add_patch(FancyArrowPatch((.5,.63),(.5,.54), arrowstyle="-|>", mutation_scale=12, color=ink))
    ax.add_patch(FancyArrowPatch((.5,.32),(.5,.23), arrowstyle="-|>", mutation_scale=12, color=ink))

    ax = axes[1, 1]; ax.axis("off"); ax.set_title("What moves data and what only changes a view")
    box(ax, .04, .68, .92, .17, "trS FP32 → trP_as_c FP16", "#fee2e2", 11)
    ax.text(.5, .61, "element conversion writes registers", ha="center", fontsize=9, color=ink)
    box(ax, .04, .38, .92, .17, "left_inverse(C) ∘ A", "#dbeafe", 11)
    ax.text(.5, .31, "compile-time coordinate map", ha="center", fontsize=9, color=ink)
    box(ax, .04, .08, .92, .17, "trP_as_c.compose(a_to_c) → trP_as_a", "#dcfce7", 10.5)
    ax.text(.5, .01, "same pointer; no shuffle, copy or shared memory", ha="center", fontsize=9, color=ink)
    for y0, y1 in ((.68,.56),(.38,.26)):
        ax.add_patch(FancyArrowPatch((.5,y0),(.5,y1), arrowstyle="-|>", mutation_scale=12, color=ink))
    fig.suptitle("From PTX lane ABI to trS, softmax ownership and P-as-A", fontsize=13)
    save(fig, output)
    plt.close(fig)


def main():
    set_font()
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results)
    performance(rows, args.output / "fa2_latency")
    stage_ratio(rows, args.output / "fa2_stage_ratio")
    pipeline(args.output / "fa2_two_stage_pipeline")
    cta_warp_thread(args.output / "fa2_cta_warp_thread")
    fragment_layout(args.output / "fa2_fragment_layout")


if __name__ == "__main__":
    main()
