"""
QA-SQL: Evaluation Figures (Springer VLDB Journal format)
==========================================================
Generates two must-have evaluation figures:

  Fig8  — Per-Strategy EX Comparison (Ablation)
            Grouped bar: simple / moderate / challenging / overall
            for Q1–Q5 strategies + full MRS-Agent pipeline

  Fig9  — Main EX vs. Baselines
            Overall Execution Accuracy compared to published BIRD baselines
            *** Update BASELINES dict below with your actual cited numbers ***

Springer format applied (matches plot_figures_springer.py):
  Font  : Arial/Helvetica, 8–9 pt labels, 7–8 pt ticks/legend
  Width : 84 mm (Fig8 single-col) / 174 mm (Fig9 full-width)
  DPI   : 600 (combination art)
  Color : RGB, Wong (2011) colorblind-safe palette + hatching

Usage (run from analysis/ directory):
  python plot_eval_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Springer Format Constants (same as plot_figures_springer.py) ──────────────
MM   = 1 / 25.4
COL  = 84  * MM        # 84 mm  ≈ 3.31" — single column
FULL = 174 * MM        # 174 mm ≈ 6.85" — full text width
DPI  = 600

plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica Neue", "Helvetica",
                              "Liberation Sans", "DejaVu Sans"],
    "font.size":             9,
    "axes.labelsize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       7.5,
    "legend.title_fontsize": 8,
    "lines.linewidth":       1.0,
    "axes.linewidth":        0.6,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "patch.linewidth":       0.5,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "savefig.dpi":           DPI,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
})

# Pastel Palette (VLDB Journal / Springer style)
WONG    = ["#A195C1", "#8DC9A8", "#F2D88C", "#E8937A",
           "#8BB8D9", "#C9A9D4", "#B5D99C", "#666666"]
HATCHES = ["", "///", "\\\\\\", "xxx", "...", "---", "|||", "+++"]


def save_fig(fig, stem):
    for ext in ("eps", "tiff", "pdf"):
        p = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(p, format=ext, **({"dpi": DPI} if ext == "tiff" else {}))
    plt.close(fig)
    print(f"   → {stem}.eps / .tiff / .pdf")


# ══════════════════════════════════════════════════════════════════════════════
# ── DATA ──────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# Per-strategy EX scores (from your evaluation)
# Columns: simple, moderate, challenging, overall
STRATEGIES = {
    "Q1: Full\nSchema":     {"simple": 73.84, "moderate": 57.67, "challenging": 60.69, "overall": 67.71},
    "Q2: SME\nMetadata":    {"simple": 74.16, "moderate": 58.32, "challenging": 58.62, "overall": 67.91},
    "Q3: Minimal\nProfile": {"simple": 74.27, "moderate": 61.12, "challenging": 57.93, "overall": 68.75},
    "Q4: Focused\nSchema":  {"simple": 79.24, "moderate": 64.79, "challenging": 64.83, "overall": 73.52},
    "Q5: Full\nProfile":    {"simple": 73.95, "moderate": 59.18, "challenging": 59.31, "overall": 68.10},
    "MRS-Agent\n(Full)":    {"simple": 83.24, "moderate": 68.25, "challenging": 65.52, "overall": 77.04},
}

# ─────────────────────────────────────────────────────────────────────────────
# BASELINES — update these with your actual cited numbers before submission
# Format: "Label": overall_EX_score (%)
# Sources to cite: BIRD leaderboard, original papers
# ─────────────────────────────────────────────────────────────────────────────
BASELINES = {
    # ── Replace / extend with your actual baseline citations ──────────────────
    "Zero-shot\nGPT-4":     46.35,   # [ref]
    "DIN-SQL\n+GPT-4":      50.72,   # NEURIPS 2023
    "DAIL-SQL\n+GPT-4":     54.76,   # VLDB 2024
    "CHESS\n+GPT-4":        65.00,   # [ref]  ← update
    "ACT-SQL\n+GPT-4":      74.40,   # [ref]  ← update
    # ─────────────────────────────────────────────────────────────────────────
    "MRS-Agent\n(Ours)":    77.04,   # this work
}


# ══════════════════════════════════════════════════════════════════════════════
# Fig8 — Per-Strategy EX Comparison (Ablation)  [full width]
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig8] Per-Strategy EX Comparison …")

DIFF_LEVELS = ["simple", "moderate", "challenging", "overall"]
DIFF_LABELS = ["Simple", "Moderate", "Challenging", "Overall"]

strat_names = list(STRATEGIES.keys())
n_strat     = len(strat_names)
n_diff      = len(DIFF_LEVELS)
x           = np.arange(n_strat)
w           = 0.19
offsets     = np.linspace(-(n_diff - 1) / 2 * w, (n_diff - 1) / 2 * w, n_diff)

fig, ax = plt.subplots(figsize=(FULL, 2.8))

for i, (level, label) in enumerate(zip(DIFF_LEVELS, DIFF_LABELS)):
    vals = [STRATEGIES[s][level] for s in strat_names]
    bars = ax.bar(
        x + offsets[i], vals, w,
        label=label,
        color=WONG[i], hatch=HATCHES[i],
        edgecolor="white", linewidth=0.4
    )

# Annotate overall bar (last group) with bold value
for i, s in enumerate(strat_names):
    ov = STRATEGIES[s]["overall"]
    is_full = "MRS-Agent" in s
    ax.text(
        x[i] + offsets[-1],                    # above the "overall" bar
        ov + 0.5,
        f"{ov:.1f}",
        ha="center", va="bottom",
        fontsize=6.5 if not is_full else 7,
        fontweight="bold" if is_full else "normal",
        color=WONG[3] if is_full else "black"
    )

# Highlight MRS-Agent column with background shading
ax.axvspan(x[-1] - 0.5, x[-1] + 0.5,
           color=WONG[0], alpha=0.07, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(strat_names, fontsize=7.5)
ax.set_ylabel("Execution Accuracy (%)")
ax.set_ylim(50, 92)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
ax.tick_params(axis="y", which="minor", length=2)
ax.legend(title="Difficulty", loc="upper left",
          frameon=False, ncol=4, handlelength=1.0,
          handletextpad=0.4, columnspacing=0.8)

# Dashed baseline: best single-strategy
best_single = max(STRATEGIES[s]["overall"] for s in strat_names
                  if "MRS-Agent" not in s)
ax.axhline(best_single, color="gray", linewidth=0.8,
           linestyle="--", alpha=0.6, zorder=0)
ax.text(n_strat - 0.1, best_single + 0.3,
        f"Best single strategy ({best_single:.1f}%)",
        ha="right", va="bottom", fontsize=6.5, color="gray")

plt.tight_layout(pad=0.3)
save_fig(fig, "Fig8")

# Print summary
print(f"   Best single strategy  : {best_single:.2f}%")
mrs_overall = STRATEGIES["MRS-Agent\n(Full)"]["overall"]
print(f"   MRS-Agent (full)       : {mrs_overall:.2f}%")
gain = mrs_overall - best_single
print(f"   Gain from full pipeline: +{gain:.2f} pp")


# ══════════════════════════════════════════════════════════════════════════════
# Fig9 — Main EX vs. Baselines  [full width]
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig9] Main EX vs. Baselines …")

bl_names  = list(BASELINES.keys())
bl_values = list(BASELINES.values())
is_ours   = ["MRS" in n or "Ours" in n for n in bl_names]

colors  = [WONG[3] if o else WONG[0]  for o in is_ours]
hats    = ["xxx"   if o else ""        for o in is_ours]
alphas  = [1.0     if o else 0.85      for o in is_ours]

fig, ax = plt.subplots(figsize=(FULL, 2.5))
x    = np.arange(len(bl_names))
bars = ax.bar(x, bl_values, width=0.6,
              color=colors, hatch=hats,
              edgecolor="white", linewidth=0.5)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)

# Value labels on each bar
for bar, val, o in zip(bars, bl_values, is_ours):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{val:.2f}",
            ha="center", va="bottom",
            fontsize=7 if not o else 7.5,
            fontweight="bold" if o else "normal",
            color=WONG[3] if o else "black")

# Legend patches
legend_handles = [
    mpatches.Patch(facecolor=WONG[0], label="Baseline methods"),
    mpatches.Patch(facecolor=WONG[3], hatch="xxx", label="MRS-Agent (ours)"),
]
ax.legend(handles=legend_handles, frameon=False, loc="upper left",
          handlelength=1.2)

ax.set_xticks(x)
ax.set_xticklabels(bl_names, fontsize=7.5)
ax.set_ylabel("Execution Accuracy (%)")
ax.set_ylim(0, max(bl_values) * 1.2)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))

plt.tight_layout(pad=0.3)
save_fig(fig, "Fig9")
print(f"   MRS-Agent overall EX: {BASELINES['MRS-Agent\n(Ours)']:.2f}%")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nDone. Files in {FIGURES_DIR}:")
for stem in ["Fig8", "Fig9"]:
    for ext in ["eps", "tiff", "pdf"]:
        p = FIGURES_DIR / f"{stem}.{ext}"
        if p.exists():
            print(f"  {p.name:<20s} ({p.stat().st_size // 1024}KB)")
