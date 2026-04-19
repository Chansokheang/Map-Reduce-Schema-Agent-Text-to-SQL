"""
Schema Filtering Horizontal Stacked Bar
=========================================
Shows Total vs Relevant tables per database as horizontal stacked bars
with reduction percentages.

Output: Fig_schema_filtering.svg / .tiff / .pdf

Usage (run from analysis/ directory):
  python plot_schema_filtering_bar.py
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

C_RELEVANT = "#8DC9A8"
C_FILTERED = "#D5D5D5"
C_PCT      = "#E8937A"

# Data sorted by total tables descending
# (name, total_tables, avg_relevant_tables)
data = [
    ("Formula 1",    14, 2.5),
    ("Superhero",    10, 2.4),
    ("Financial",     8, 3.3),
    ("Codebase",      8, 1.8),
    ("EU Football",   8, 1.9),
    ("Student Club",  8, 2.0),
    ("Card Games",    7, 1.8),
    ("Debit Card",    6, 2.1),
    ("Toxicology",    4, 2.6),
    ("CA Schools",    3, 2.0),
    ("Thrombosis",    3, 2.1),
]

names    = [d[0] for d in data]
total    = [d[1] for d in data]
relevant = [round(d[2]) for d in data]
filtered = [t - r for t, r in zip(total, relevant)]
reduction_pct = [(1 - r / t) * 100 for r, t in zip(relevant, total)]

# Reverse for horizontal (top = largest schema)
names    = names[::-1]
total    = total[::-1]
relevant = relevant[::-1]
filtered = filtered[::-1]
reduction_pct = reduction_pct[::-1]

print("[Fig] Schema Filtering Horizontal Bar ...")

fig, ax = plt.subplots(figsize=(FULL, 3.0))

y = np.arange(len(names))
h = 0.6

# Stacked: Relevant (green) + Filtered (gray)
b1 = ax.barh(y, relevant, h, label="Relevant Tables (kept)",
             color=C_RELEVANT, edgecolor="white", linewidth=0.5)
b2 = ax.barh(y, filtered, h, left=relevant, label="Filtered Tables (removed)",
             color=C_FILTERED, edgecolor="white", linewidth=0.5)

# Relevant count inside green bar
for i, (bar, r) in enumerate(zip(b1, relevant)):
    ax.text(r / 2, y[i], f"{r}", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#1A1A1A")

# Filtered count inside gray bar
for i, (bar, r, f) in enumerate(zip(b2, relevant, filtered)):
    if f > 0:
        ax.text(r + f / 2, y[i], f"{f}", ha="center", va="center",
                fontsize=7.5, color="#666666")

# Reduction % at end
for i, (t, pct) in enumerate(zip(total, reduction_pct)):
    ax.text(t + 0.3, y[i], f"-{pct:.0f}%",
            ha="left", va="center", fontsize=7.5,
            fontweight="bold", color=C_PCT)

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Number of Tables", fontsize=9)
ax.set_xlim(0, max(total) + 3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
ax.legend(loc="lower right", frameon=False, fontsize=7.5)

plt.tight_layout(pad=0.3)

for ext in ("svg", "tiff", "pdf"):
    p = FIGURES_DIR / f"Fig_schema_filtering.{ext}"
    kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
    if ext == "tiff":
        kwargs["dpi"] = DPI
    fig.savefig(p, **kwargs)
plt.close(fig)
print("   -> Fig_schema_filtering.svg / .tiff / .pdf")
