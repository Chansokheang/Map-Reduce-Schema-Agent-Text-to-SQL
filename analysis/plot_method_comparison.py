"""
Method Comparison Scatter Plot (1x2 split)
==========================================
(a) MRS-Agent vs Foundation Models
(b) MRS-Agent vs Text-to-SQL Methods

Uses adjustText to auto-resolve label overlaps.
Output: Fig_method_comparison.svg / .tiff / .pdf
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from adjustText import adjust_text
from pathlib import Path

warnings.filterwarnings("ignore")
FIGURES_DIR = Path("./figures")
MM = 1 / 25.4
FULL = 174 * MM
DPI = 600

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica Neue", "Helvetica",
                           "Liberation Sans", "DejaVu Sans"],
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    7,
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

# ── Foundation models (sorted by EX) ──
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

# ── Text-to-SQL methods (sorted by EX) ──
pipeline = [
    ("DIN-SQL",              50.72),
    ("DAIL-SQL",             54.76),
    ("DTS-SQL",              55.80),
    ("TA-SQL",               56.19),
    ("CodeS",                58.47),
    ("SuperSQL",             58.50),
    ("MAC-SQL",              61.08),
    ("SQL-PaLM",             61.93),
    ("MCS-SQL",              63.36),
    ("CHESS",                65.00),
    ("E-SQL",                65.58),
    ("MSc-SQL",              65.60),
    ("RSL-SQL",              67.21),
    ("OpenSearch-SQL",       69.30),
    ("OmniSQL",              69.23),
    ("Alpha-SQL",            69.70),
    ("CHASE-SQL",            73.00),
    ("G2SQL",                73.16),
    ("XiYan-SQL",            73.34),
    ("Agentar-Scale-SQL",    74.90),
    ("AskData",              77.64),
    ("MRS-Agent (Ours)",     77.04),
]


def draw_panel(ax, data, base_color, title):
    names  = [d[0] for d in data]
    scores = [d[1] for d in data]
    x = np.arange(len(names))

    # Draw dots
    for i, (name, score) in enumerate(data):
        is_ours = "MRS" in name
        ax.scatter(x[i], score,
                   s=130 if is_ours else 45,
                   c="#E8937A" if is_ours else base_color,
                   edgecolors="white", linewidths=0.6,
                   zorder=5 if is_ours else 3)

    # Top-cluster labels need manual placement to avoid overlap
    manual_offsets = {
        "MRS-Agent (Ours)": (-3.0, 4.0, 7, "bold", "#C0392B"),
        "AskData":          (2.0, -4.5, 5.5, "normal", "#333333"),
        "Agentar-Scale-SQL":(-6.0, -4.0, 5.5, "normal", "#333333"),
        "G2SQL":            (-2.5,  3.5, 5.5, "normal", "#333333"),
        "XiYan-SQL":        (2.5, -6.0, 5.5, "normal", "#333333"),
        "CHASE-SQL":        (-5.0, 3.0, 5.5, "normal", "#333333"),
        "Alpha-SQL":        (3.0, -3.0, 5.5, "normal", "#333333"),
        "OpenSearch-SQL":   (-5.5, 1.0, 5.5, "normal", "#333333"),
        "OmniSQL":          (2.5,  2.5, 5.5, "normal", "#333333"),
    }

    texts = []
    for i, (name, score) in enumerate(data):
        if name in manual_offsets:
            continue
        t = ax.text(x[i], score, name,
                    fontsize=5.5, fontweight="normal", color="#333333",
                    ha="center", va="center")
        texts.append(t)

    # Auto-adjust non-manual labels
    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#CCCCCC", linewidth=0.4),
                expand=(3.0, 3.0),
                force_text=(2.0, 2.5),
                force_points=(1.0, 1.2),
                ensure_inside_axes=True,
                iterations=200)

    # Manually annotate top-cluster labels
    for i, (name, score) in enumerate(data):
        if name in manual_offsets:
            xoff, yoff, fs, fw, clr = manual_offsets[name]
            arrow_clr = "#C0392B" if "MRS" in name else "#CCCCCC"
            ax.annotate(name, (x[i], score),
                        xytext=(x[i] + xoff, score + yoff),
                        fontsize=fs, fontweight=fw, color=clr,
                        ha="center", va="center",
                        arrowprops=dict(arrowstyle="-", color=arrow_clr,
                                        linewidth=0.5))

    # Reference lines
    #ax.axhline(y=77.04, color="#E8937A", linewidth=0.5, linestyle=":", zorder=0)
    #ax.axhline(y=67.34, color=base_color, linewidth=0.5, linestyle=":", zorder=0)

    # No arrow -- reference lines are sufficient

    ax.set_xticks([])
    ax.set_ylabel("EX (%)", fontsize=9)
    ax.set_ylim(48, 84)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel(title, fontsize=8, labelpad=6)


# ── (a) vs Foundation Models ──
fig1, ax1 = plt.subplots(figsize=(FULL, 2.2))
draw_panel(ax1, foundation, "#8BB8D9", "Foundation Models")
plt.tight_layout(pad=0.3)
for ext in ("svg", "tiff", "pdf"):
    p = FIGURES_DIR / f"Fig_vs_foundation.{ext}"
    kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
    if ext == "tiff":
        kwargs["dpi"] = DPI
    fig1.savefig(p, **kwargs)
plt.close(fig1)
print("Done: Fig_vs_foundation.svg / .tiff / .pdf")

# ── (b) vs Text-to-SQL Methods ──
fig2, ax2 = plt.subplots(figsize=(FULL, 2.8))
draw_panel(ax2, pipeline, "#8DC9A8", "Text-to-SQL Methods")
plt.tight_layout(pad=0.3)
for ext in ("svg", "tiff", "pdf"):
    p = FIGURES_DIR / f"Fig_vs_methods.{ext}"
    kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
    if ext == "tiff":
        kwargs["dpi"] = DPI
    fig2.savefig(p, **kwargs)
plt.close(fig2)
print("Done: Fig_vs_methods.svg / .tiff / .pdf")
