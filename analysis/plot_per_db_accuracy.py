"""
Per-Database Execution Accuracy
================================
Bar chart showing EX accuracy per database with average line.
Thrombosis (lowest) highlighted in coral.

Output: Fig_per_db_accuracy.svg / .tiff / .pdf

Usage (run from analysis/ directory):
  python plot_per_db_accuracy.py
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

C_NORMAL  = "#8DC9A8"
C_LOW     = "#E8937A"

# Sorted by total table count (descending) to match schema filtering figure
data = [
    ("Formula 1",    75.29),
    ("Superhero",    93.80),
    ("Financial",    81.13),
    ("Codebase",     78.49),
    ("EU Football",  78.29),
    ("Student Club", 82.91),
    ("Card Games",   73.30),
    ("Debit Card",   73.44),
    ("Toxicology",   76.55),
    ("CA Schools",   78.65),
    ("Thrombosis",   63.19),
]

names = [d[0] for d in data]
acc   = [d[1] for d in data]

print("[Fig] Per-Database EX Accuracy ...")

fig, ax = plt.subplots(figsize=(FULL, 2.2))

x = np.arange(len(names))
bars = ax.bar(x, acc, width=0.6,
              color=C_NORMAL, edgecolor="white", linewidth=0.5)

# Highlight lowest in coral
for i, bar in enumerate(bars):
    if acc[i] < 65:
        bar.set_color(C_LOW)

# Value labels
for bar, val in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

# Average line
avg = np.mean(acc)
ax.axhline(avg, color="#888888", linewidth=0.6, linestyle="--")
ax.text(len(names) - 0.3, avg + 0.5, f"Avg: {avg:.1f}%",
        ha="right", va="bottom", fontsize=7, color="#888888")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=7, rotation=35, ha="right")
ax.set_ylabel("EX Accuracy (%)")
ax.set_ylim(55, 100)
ax.yaxis.set_major_locator(MultipleLocator(10))

plt.tight_layout(pad=0.3)

for ext in ("svg", "tiff", "pdf"):
    p = FIGURES_DIR / f"Fig_per_db_accuracy.{ext}"
    kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
    if ext == "tiff":
        kwargs["dpi"] = DPI
    fig.savefig(p, **kwargs)
plt.close(fig)
print("   -> Fig_per_db_accuracy.svg / .tiff / .pdf")
