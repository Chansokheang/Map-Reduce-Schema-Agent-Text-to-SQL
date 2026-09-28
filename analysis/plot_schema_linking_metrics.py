"""
Schema Linking Metrics (1x4 Panel)
====================================
Shows Recall, Precision, F1, and Perfect Recall across difficulty levels.
Each panel = one metric, x-axis = difficulty (Simple, Moderate, Challenging, Overall).
Bars colored by difficulty level, consistent across panels.

Output: Fig_schema_linking.svg / .tiff / .pdf

Usage (run from analysis/ directory):
  python plot_schema_linking_metrics.py
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
FIGURES_DIR = Path("./figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
FULL = 174 * MM
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

# Colors per difficulty level (consistent across all panels)
DIFF_COLORS  = ["#A195C1", "#8DC9A8", "#F2D88C", "#8BB8D9"]
DIFF_HATCHES = ["", "///", "\\\\\\", "..."]
DIFF_LABELS  = ["Simple", "Moderate", "Challenging", "Overall"]
DIFF_SHORT   = ["Sim", "Mod", "Chal", "All"]

# Metrics data: [Simple, Moderate, Challenging, Overall]
metrics = {
    "Recall":         [84.28, 86.15, 82.07, 84.64],
    "Precision":      [77.37, 78.65, 75.30, 77.56],
    "F1":             [79.00, 80.46, 76.62, 79.22],
    "Perfect Recall": [78.30, 78.70, 73.80, 78.00],
}

metric_names = ["Recall", "Precision", "F1", "Perfect Recall"]

print("[Fig] Schema Linking Metrics (1x4) ...")

fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.2))

for idx, metric in enumerate(metric_names):
    ax = axes[idx]
    vals = metrics[metric]
    x = np.arange(len(DIFF_LABELS))

    # Each bar colored by difficulty level
    for i in range(len(vals)):
        ax.bar(x[i], vals[i], width=0.65,
               color=DIFF_COLORS[i], hatch=DIFF_HATCHES[i],
               edgecolor="white", linewidth=0.5)
        ax.text(x[i], vals[i] + 0.3, f"{vals[i]:.1f}",
                ha="center", va="bottom", fontsize=6, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(DIFF_SHORT, fontsize=6.5)
    ax.set_ylabel("Score (%)", fontsize=8)
    ax.set_ylim(65, 92)
    ax.yaxis.set_major_locator(MultipleLocator(5))

    label_char = chr(ord("a") + idx)
    ax.set_xlabel(f"({label_char}) {metric}", fontsize=8, labelpad=4)

# Shared legend for difficulty levels
legend_handles = [
    mpatches.Patch(facecolor=c, hatch=h, edgecolor="white",
                   linewidth=0.4, label=l)
    for c, h, l in zip(DIFF_COLORS, DIFF_HATCHES, DIFF_LABELS)
]
fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=False,
           fontsize=6.5, bbox_to_anchor=(0.5, 1.0))

plt.tight_layout(pad=0.4, w_pad=1.8, rect=[0, 0, 1, 0.88])

for ext in ("svg", "tiff", "pdf"):
    p = FIGURES_DIR / f"Fig_schema_linking.{ext}"
    kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
    if ext == "tiff":
        kwargs["dpi"] = DPI
    fig.savefig(p, **kwargs)
plt.close(fig)
print("   -> Fig_schema_linking.svg / .tiff / .pdf")
