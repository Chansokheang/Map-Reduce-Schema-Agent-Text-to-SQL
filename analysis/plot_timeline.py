"""
Timeline Scatter Plot: Text-to-SQL Methods vs EX Accuracy
==========================================================
Shows the evolution of methods over time with MRS-Agent positioned.
  x-axis: publication date
  y-axis: EX accuracy (%)
  color:  foundation model (blue) vs pipeline method (green) vs ours (coral)
  size:   category emphasis

Output: Fig_timeline.svg / .tiff / .pdf
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# ── Data from Tables 2 & 3 ──
# (name, year, month, EX%, category)
methods = [
    # Foundation models (Table 2)
    ("GPT-4.1-mini",         2024,  4, 53.97, "foundation"),
    ("Qwen2.5-coder-32b",   2024,  9, 53.52, "foundation"),
    ("DeepSeek-V3",          2024, 12, 59.71, "foundation"),
    ("Kimi-K2",              2025,  6, 60.63, "foundation"),
    ("DeepSeek-R1",          2025,  1, 61.67, "foundation"),
    ("Gemini-2.0-flash",     2024, 12, 62.97, "foundation"),
    ("GLM-4.7",              2025,  3, 63.82, "foundation"),
    ("Gemini-2.0-flash-exp", 2025,  1, 64.69, "foundation"),
    ("GPT-4o",               2024,  5, 64.72, "foundation"),
    ("Qwen3-Coder-480B",    2025,  6, 66.17, "foundation"),
    ("Claude 4.5 Sonnet",    2025,  3, 67.34, "foundation"),
    ("Claude Opus 4.6",      2025,  5, 68.77, "foundation"),

    # Pipeline / SOTA methods (Table 3)
    ("DIN-SQL",              2023,  6, 50.72, "pipeline"),
    ("DAIL-SQL",             2024,  1, 54.76, "pipeline"),
    ("DTS-SQL",              2024,  2, 55.80, "pipeline"),
    ("TA-SQL",               2024,  6, 56.19, "pipeline"),
    ("CodeS",                2024,  3, 58.47, "pipeline"),
    ("SuperSQL",             2024,  6, 58.50, "pipeline"),
    ("MAC-SQL",              2025,  1, 61.08, "pipeline"),
    ("SQL-PaLM",             2024,  1, 61.93, "pipeline"),
    ("MCS-SQL",              2025,  1, 63.36, "pipeline"),
    ("CHESS",                2024,  6, 65.00, "pipeline"),
    ("E-SQL",                2024,  9, 65.58, "pipeline"),
    ("MSc-SQL",              2025,  3, 65.60, "pipeline"),
    ("RSL-SQL",              2024, 10, 67.21, "pipeline"),
    ("OpenSearch-SQL",       2025,  3, 69.30, "pipeline"),
    ("OmniSQL",              2025,  6, 69.23, "pipeline"),
    ("Alpha-SQL",            2025,  6, 69.70, "pipeline"),
    ("CHASE-SQL",            2025,  1, 73.00, "pipeline"),
    ("G2SQL",                2026,  1, 73.16, "pipeline"),
    ("XiYan-SQL",            2026,  1, 73.34, "pipeline"),
    ("Agentar-Scale-SQL",    2025,  6, 74.90, "pipeline"),
    ("AskData",              2025,  6, 77.64, "pipeline"),

    # Ours
    ("MRS-Agent",            2026,  4, 77.04, "ours"),
]


def to_x(year, month):
    return year + (month - 1) / 12.0


# ── Plot ──
fig, ax = plt.subplots(figsize=(FULL, FULL * 0.45))

colors  = {"foundation": "#8BB8D9", "pipeline": "#8DC9A8", "ours": "#E8937A"}
zorders = {"foundation": 2, "pipeline": 3, "ours": 5}
sizes   = {"foundation": 50, "pipeline": 65, "ours": 140}

for cat, label in [("foundation", "Foundation Model"),
                   ("pipeline", "Text-to-SQL Method"),
                   ("ours", "MRS-Agent (Ours)")]:
    subset = [m for m in methods if m[4] == cat]
    xs = [to_x(m[1], m[2]) for m in subset]
    ys = [m[3] for m in subset]
    ax.scatter(xs, ys, s=sizes[cat], c=colors[cat],
               edgecolors="white", linewidths=0.6,
               label=label, zorder=zorders[cat])

# ── Labels for key methods ──
label_offsets = {
    "DIN-SQL":            (0.08, -1.2),
    "DAIL-SQL":           (0.08, 0.8),
    "CodeS":              (0.08, 0.8),
    "SQL-PaLM":           (0.08, -1.5),
    "MAC-SQL":             (0.08, 0.8),
    "CHASE-SQL":          (0.08, -1.8),
    "G2SQL":              (-0.15, -1.8),
    "XiYan-SQL":          (0.08, 0.8),
    "Agentar-Scale-SQL":  (-0.35, -1.8),
    "AskData":            (-0.15, -1.8),
    "OpenSearch-SQL":     (0.08, 0.8),
    "Alpha-SQL":          (0.08, -1.5),
    "GPT-4o":             (0.08, -1.2),
    "Claude 4.5 Sonnet":  (0.08, -1.5),
    "Claude Opus 4.6":    (0.08, 0.8),
    "DeepSeek-R1":        (0.08, 0.8),
    "Gemini-2.0-flash":   (0.08, -1.5),
    "MRS-Agent":          (-0.15, 1.5),
}

for m in methods:
    name, year, month, ex, cat = m
    if name in label_offsets:
        xp = to_x(year, month)
        xoff, yoff = label_offsets[name]

        fontsize = 7.5 if cat == "ours" else 6
        fontweight = "bold" if cat == "ours" else "normal"
        color = "#C0392B" if cat == "ours" else "#333333"

        use_arrow = abs(xoff) > 0.12 or abs(yoff) > 1.5
        arrow_props = dict(arrowstyle="-", color="#CCCCCC", linewidth=0.4) if use_arrow else None

        ax.annotate(name, (xp, ex),
                    xytext=(xp + xoff, ex + yoff),
                    fontsize=fontsize, fontweight=fontweight, color=color,
                    arrowprops=arrow_props)

# ── Reference lines ──
ax.axhline(y=77.04, color=colors["ours"], linewidth=0.5, linestyle=":", zorder=0)

ax.set_xlabel("Publication Timeline", fontsize=9)
ax.set_ylabel("Execution Accuracy (%)", fontsize=9)
ax.set_ylim(48, 82)
ax.set_xlim(to_x(2023, 3), to_x(2026, 7))

# X-axis date ticks
tick_positions = sorted([to_x(y, 1) for y in range(2023, 2027)]
                        + [to_x(y, 7) for y in range(2023, 2026)])
tick_labels = []
for tp in tick_positions:
    y = int(tp)
    m = int(round((tp - y) * 12)) + 1
    tick_labels.append(f"Jan {y}" if m == 1 else f"Jul {y}")
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=6.5, rotation=30, ha="right")

ax.legend(loc="upper left", frameon=False, fontsize=7.5)

plt.tight_layout(pad=0.3)

for ext in ("svg", "tiff", "pdf"):
    p = FIGURES_DIR / f"Fig_timeline.{ext}"
    kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
    if ext == "tiff":
        kwargs["dpi"] = DPI
    fig.savefig(p, **kwargs)
plt.close(fig)
print("Done: Fig_timeline.svg / .tiff / .pdf")
