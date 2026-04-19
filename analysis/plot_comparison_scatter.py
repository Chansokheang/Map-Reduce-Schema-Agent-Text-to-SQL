"""
Method Comparison Scatter Plots (Square, Single-Column)
========================================================
Generates two separate square scatter plots:
  Fig_vs_foundation_sq — MRS-Agent vs Foundation Models
  Fig_vs_methods_sq    — MRS-Agent vs Text-to-SQL Methods

Features:
  - Boxed labels inside axes (no clipping)
  - Axes fraction positioning for labels
  - Horizontal grid lines
  - Pastel palette, Arial font, closed box spines

Usage (run from analysis/ directory):
  python plot_comparison_scatter.py
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path

warnings.filterwarnings("ignore")
FIGURES_DIR = Path("./figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
COL = 84 * MM
DPI = 600

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica Neue", "Helvetica",
                           "Liberation Sans", "DejaVu Sans"],
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "axes.linewidth":     0.6,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

colors_map = {"foundation": "#8BB8D9", "pipeline": "#8DC9A8", "ours": "#E8937A"}


def save_fig(fig, stem):
    for ext in ("svg", "tiff", "pdf"):
        p = FIGURES_DIR / f"{stem}.{ext}"
        kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
        if ext == "tiff":
            kwargs["dpi"] = DPI
        fig.savefig(p, **kwargs)
    plt.close(fig)
    print(f"   -> {stem}.svg / .tiff / .pdf")


def make_fig(data, label_pos, filename, xlabel, legend_labels, marker_type):
    fig, ax = plt.subplots(figsize=(COL * 1.15, COL * 0.85))
    data_sorted = sorted(data, key=lambda d: d[1])
    x = np.arange(len(data_sorted))

    # Horizontal grid
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, color="#E0E0E0", zorder=0)

    # Plot dots
    for i, (name, score) in enumerate(data_sorted):
        is_ours = name == "MRS-Agent (Ours)"
        color = "#E8937A" if is_ours else colors_map.get(marker_type, "#8DC9A8")
        marker = "o" if is_ours else ("s" if marker_type == "foundation" else "o")
        ax.scatter(x[i], score, s=100 if is_ours else 40, c=color,
                   marker=marker, edgecolors="white", linewidths=0.5,
                   zorder=5 if is_ours else 3)

    # Boxed labels at axes-fraction positions
    for i, (name, score) in enumerate(data_sorted):
        if name not in label_pos:
            continue
        is_ours = name == "MRS-Agent (Ours)"
        xf, yf = label_pos[name]
        fs = 7 if is_ours else 6
        fw = "bold" if is_ours else "normal"
        clr = "#C0392B" if is_ours else "#333333"
        bbox_clr = "#FFF0ED" if is_ours else "#F5F5F5"
        border_clr = "#C0392B" if is_ours else "#DDDDDD"
        arrow_clr = "#C0392B" if is_ours else "#CCCCCC"
        ax.annotate(name, (x[i], score),
                    xytext=(xf, yf), textcoords="axes fraction",
                    fontsize=fs, fontweight=fw, color=clr,
                    ha="left", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=bbox_clr,
                              edgecolor=border_clr, linewidth=0.4),
                    arrowprops=dict(arrowstyle="-", color=arrow_clr, linewidth=0.4),
                    annotation_clip=True)

    ax.set_xticks([])
    ax.set_xlabel(xlabel, fontsize=9, labelpad=6)
    ax.set_ylabel("Execution Accuracy (%)", fontsize=9)
    ax.set_ylim(48, 82)
    ax.yaxis.set_major_locator(MultipleLocator(5))

    legend_handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor=c,
                   markersize=sz, label=l)
        for m, c, sz, l in legend_labels
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=7)
    plt.tight_layout(pad=0.3)
    save_fig(fig, filename)


# ══════════════════════════════════════════════════════════════════════════════
# Fig_vs_foundation_sq
# ══════════════════════════════════════════════════════════════════════════════
print("[1/2] Fig_vs_foundation_sq ...")

foundation = [
    ("Qwen2.5-coder-32b",   53.52),
    ("GPT-4.1-mini",         53.97),
    ("DeepSeek-V3",          59.71),
    ("Kimi-K2",              60.63),
    ("DeepSeek-R1",          61.67),
    ("Gemini-2.0-flash",     62.97),
    ("GLM-4.7",              63.82),
    ("Gemini-2.0-flash-exp", 64.69),
    ("GPT-4o",               64.72),
    ("Qwen3-Coder-480B",    66.17),
    ("Claude 4.5 Sonnet",    67.34),
    ("Claude Opus 4.6",      68.77),
    ("MRS-Agent (Ours)",     77.04),
]

foundation_labels = {
    "Qwen2.5-coder-32b":  (0.08, 0.08),
    "GPT-4.1-mini":        (0.22, 0.14),
    "DeepSeek-V3":         (0.08, 0.30),
    "Kimi-K2":             (0.50, 0.22),
    "DeepSeek-R1":         (0.12, 0.40),
    "Gemini-2.0-flash":    (0.55, 0.32),
    "GLM-4.7":             (0.15, 0.50),
    "Gemini-2.0-flash-exp":(0.55, 0.42),
    "GPT-4o":              (0.60, 0.52),
    "Qwen3-Coder-480B":   (0.55, 0.62),
    "Claude 4.5 Sonnet":   (0.60, 0.72),
    "Claude Opus 4.6":     (0.62, 0.82),
    "MRS-Agent (Ours)":    (0.52, 0.93),
}

make_fig(
    foundation, foundation_labels,
    "Fig_vs_foundation_sq", "Foundation Models",
    [("s", "#8BB8D9", 6, "Foundation Model"),
     ("o", "#E8937A", 8, "MRS-Agent (Ours)")],
    "foundation",
)


# ══════════════════════════════════════════════════════════════════════════════
# Fig_vs_methods_sq
# ══════════════════════════════════════════════════════════════════════════════
print("[2/2] Fig_vs_methods_sq ...")

pipeline = [
    ("DIN-SQL",              50.72),
    ("DAIL-SQL",             54.76),
    ("MAC-SQL",              61.08),
    ("SQL-PaLM",             61.93),
    ("CHESS",                65.00),
    ("RSL-SQL",              67.21),
    ("OpenSearch-SQL",       69.30),
    ("Alpha-SQL",            69.70),
    ("CHASE-SQL",            73.00),
    ("G2SQL",                73.16),
    ("XiYan-SQL",            73.34),
    ("Agentar-Scale-SQL",    74.90),
    ("AskData",              77.64),
    ("MRS-Agent (Ours)",     77.04),
]

pipeline_labels = {
    "DIN-SQL":           (0.08, 0.06),
    "DAIL-SQL":          (0.18, 0.16),
    "MAC-SQL":           (0.08, 0.35),
    "SQL-PaLM":          (0.25, 0.30),
    "CHESS":             (0.18, 0.46),
    "RSL-SQL":           (0.42, 0.46),
    "OpenSearch-SQL":    (0.15, 0.58),
    "Alpha-SQL":         (0.50, 0.56),
    "CHASE-SQL":         (0.15, 0.70),
    "G2SQL":             (0.55, 0.68),
    "XiYan-SQL":         (0.50, 0.78),
    "Agentar-Scale-SQL": (0.68, 0.75),
    "AskData":           (0.55, 0.88),
    "MRS-Agent (Ours)":  (0.52, 0.95),
}

make_fig(
    pipeline, pipeline_labels,
    "Fig_vs_methods_sq", "Text-to-SQL Methods",
    [("o", "#8DC9A8", 6, "Text-to-SQL Method"),
     ("o", "#E8937A", 8, "MRS-Agent (Ours)")],
    "pipeline",
)

print("\nDone. All files in:", FIGURES_DIR.resolve())
