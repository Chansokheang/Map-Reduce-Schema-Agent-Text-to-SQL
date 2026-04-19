"""
QA-SQL: VLDB-Style Multi-Panel Figures
=======================================
Generates multi-panel comparison figures matching VLDB Journal design:
  - 2x2 subplot grids with (a), (b), (c), (d) labels
  - 1x2 side-by-side panels like Fig 18/19/21 in reference papers
  - Value annotations on top of bars
  - Consistent color palette, hatching for B&W accessibility

Figures:
  Fig10 — Pipeline Latency by Database (2×2 grid, top-4 DBs)
  Fig11 — Schema Complexity vs. Performance (1×2 panel)
  Fig12 — Per-Difficulty Strategy Ablation (2×2 grid)
  Fig13 — Strategy Contribution Analysis (1×2 panel)

Springer format:
  Font  : Arial/Helvetica, 8–9 pt labels, 7–8 pt ticks/legend
  Width : 84 mm (single-col) / 174 mm (full-width)
  DPI   : 600 (combination art)
  Color : RGB, Wong (2011) colorblind-safe palette + hatching
  Output: .eps (vector) + .tiff (600 dpi) + .pdf

Usage (run from analysis/ directory):
  python plot_vldb_multipanel.py
"""

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, MultipleLocator
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("../output/claude_headless_v2")
FIGURES_DIR = Path("./figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Springer Figure Dimensions ────────────────────────────────────────────────
MM   = 1 / 25.4
COL  = 84  * MM       # 84 mm  ≈ 3.31" — single column
FULL = 174 * MM       # 174 mm ≈ 6.85" — full text width
HMAX = 234 * MM       # 234 mm ≈ 9.21" — max height
DPI  = 600

# ── Global Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Arial", "Helvetica Neue", "Helvetica",
                              "Liberation Sans", "DejaVu Sans"],
    "font.size":             9,
    "axes.labelsize":        9,
    "axes.titlesize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       7.5,
    "legend.title_fontsize": 8,
    "lines.linewidth":       1.0,
    "axes.linewidth":        0.6,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "patch.linewidth":       0.5,
    "axes.spines.top":       True,
    "axes.spines.right":     True,
    "savefig.dpi":           DPI,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "pdf.fonttype":          42,
    "ps.fonttype":           42,
})

# ── Pastel Palette (VLDB Journal / Springer style) ───────────────────────────
# Matches the soft lavender–mint–cream aesthetic from published VLDB figures
WONG = [
    "#A195C1",   # lavender / muted purple
    "#8DC9A8",   # mint green
    "#F2D88C",   # cream / warm yellow
    "#E8937A",   # soft coral / salmon
    "#8BB8D9",   # soft sky-blue
    "#C9A9D4",   # light mauve
    "#B5D99C",   # light lime
    "#666666",   # neutral gray
]
HATCHES = ["", "///", "\\\\\\", "xxx", "...", "---", "|||", "+++"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]

def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def normalize_sql(sql):
    return " ".join(sql.strip().lower().split()) if sql else ""

def save_fig(fig, stem):
    """Save SVG (vector) + TIFF (600 dpi) + PDF for LaTeX."""
    for ext in ("svg", "tiff", "pdf"):
        p = FIGURES_DIR / f"{stem}.{ext}"
        kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
        if ext == "tiff":
            kwargs["dpi"] = DPI
        fig.savefig(p, **kwargs)
    plt.close(fig)
    print(f"   -> {stem}.svg / .tiff / .pdf")

def db_short(name):
    """Short display name for databases."""
    mapping = {
        "california_schools":     "CA Schools",
        "card_games":             "Card Games",
        "codebase_community":     "Codebase",
        "debit_card_specializing":"Debit Card",
        "european_football_2":    "EU Football",
        "financial":              "Financial",
        "formula_1":              "Formula 1",
        "student_club":           "Student Club",
        "superhero":              "Superhero",
        "thrombosis_prediction":  "Thrombosis",
        "toxicology":             "Toxicology",
    }
    return mapping.get(name, name.replace("_", " ").title())

def annotate_bars(ax, bars, fmt="{:.1f}", fontsize=7, offset=0.3, bold=False):
    """Add value labels on top of bars."""
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize,
                fontweight="bold" if bold else "normal")


# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data ...")
selected   = load_json(OUTPUT_DIR / "selected.json")
timings    = load_jsonl(OUTPUT_DIR / "timings.jsonl")
schema_out = load_jsonl(OUTPUT_DIR / "schema_agent_output.jsonl")
decomp     = load_jsonl(OUTPUT_DIR / "agentic_decomposition.jsonl")

STRAT_NAMES = ["Full Schema", "SME Metadata", "Minimal Profile",
               "Focused Schema", "Full Profile"]
STRAT_FILES = ["candidate_full_schema.json", "candidate_sme_metadata.json",
               "candidate_minimal_profile.json", "candidate_focused_schema.json",
               "candidate_full_profile.json"]
candidates = {n: load_json(OUTPUT_DIR / f)
              for n, f in zip(STRAT_NAMES, STRAT_FILES)}

N = len(selected)
print(f"  {N} questions loaded\n")


# ══════════════════════════════════════════════════════════════════════════════
# Fig10 — Pipeline Latency by Database  (2×2 grid, VLDB Fig 21 style)
# ══════════════════════════════════════════════════════════════════════════════
print("[Fig10] Pipeline Latency by Database (2x2 grid) ...")

# Group timings by database
db_timings = defaultdict(lambda: defaultdict(list))
for t in timings:
    db = t["database"]
    for key in ["schema_agent_ms", "candidate_generation_ms", "judge_ms"]:
        db_timings[db][key].append(t[key] / 1000.0)

# Select 4 databases with diverse schema sizes
panel_dbs = ["california_schools", "financial", "formula_1", "codebase_community"]
stage_labels = ["Schema\nAgent", "Candidate\nGen.", "Judge"]
stage_keys   = ["schema_agent_ms", "candidate_generation_ms", "judge_ms"]

fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 0.62))
axes = axes.flatten()

for idx, db in enumerate(panel_dbs):
    ax = axes[idx]
    means = [np.mean(db_timings[db][k]) for k in stage_keys]
    x = np.arange(len(stage_labels))

    bars = ax.bar(x, means, width=0.55,
                  color=[WONG[0], WONG[1], WONG[2]],
                  hatch=[HATCHES[0], HATCHES[1], HATCHES[2]],
                  edgecolor="white", linewidth=0.4)

    annotate_bars(ax, bars, fmt="{:.1f}", fontsize=7, offset=0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=7)
    ax.set_ylabel("Latency (s)", fontsize=8)
    ax.set_ylim(0, max(means) * 1.25)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Subfigure label: (a), (b), (c), (d)
    label_char = chr(ord('a') + idx)
    ax.set_xlabel(f"({label_char}) {db_short(db)}", fontsize=8, labelpad=4)

plt.tight_layout(pad=0.4, h_pad=1.2, w_pad=0.8)
save_fig(fig, "Fig10")
print("   Databases shown:", [db_short(d) for d in panel_dbs])


# ══════════════════════════════════════════════════════════════════════════════
# Fig11a — Schema Reduction vs. EX Accuracy  (scatter, single column)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig11a] Schema Reduction vs. EX Accuracy ...")

db_total    = defaultdict(list)
db_relevant = defaultdict(list)
for rec in schema_out:
    db = rec["database"]
    db_total[db].append(rec["total_tables_evaluated"])
    db_relevant[db].append(rec["relevant_tables_count"])

all_dbs      = sorted(db_total.keys())
avg_total    = {d: np.mean(db_total[d]) for d in all_dbs}
avg_relevant = {d: np.mean(db_relevant[d]) for d in all_dbs}
reduction    = {d: (1 - avg_relevant[d] / avg_total[d]) * 100 for d in all_dbs}

# Per-database EX accuracy (computed from BIRD evaluation)
DB_ACCURACY = {
    "california_schools":      78.65,
    "card_games":              73.30,
    "codebase_community":      78.49,
    "debit_card_specializing": 73.44,
    "european_football_2":     78.29,
    "financial":               81.13,
    "formula_1":               75.29,
    "student_club":            82.91,
    "superhero":               93.80,
    "thrombosis_prediction":   63.19,
    "toxicology":              76.55,
}

fig, ax = plt.subplots(figsize=(COL * 1.6, COL * 1.35))

# Scatter: x = reduction%, y = EX accuracy, size = num questions
# Count questions per DB
db_qcount = Counter(rec["database"] for rec in schema_out)

for db in all_dbs:
    if db not in DB_ACCURACY:
        continue
    rx = reduction[db]
    ey = DB_ACCURACY[db]
    sz = db_qcount[db]
    ax.scatter(rx, ey, s=sz * 0.6, color=WONG[0], edgecolor="white",
               linewidth=0.4, alpha=0.85, zorder=3)
    # Label each point
    ax.annotate(db_short(db),
                xy=(rx, ey), xytext=(4, 4),
                textcoords="offset points",
                fontsize=5.5, color="black")

# Trend line
x_vals = [reduction[d] for d in all_dbs if d in DB_ACCURACY]
y_vals = [DB_ACCURACY[d] for d in all_dbs if d in DB_ACCURACY]
if len(x_vals) > 2:
    coeffs = np.polyfit(x_vals, y_vals, 1)
    xl = np.linspace(min(x_vals) - 2, max(x_vals) + 2, 200)
    ax.plot(xl, np.polyval(coeffs, xl),
            color="gray", linewidth=0.8, linestyle="--", alpha=0.7,
            label=f"Trend (slope={coeffs[0]:.2f})")

    # Pearson correlation
    corr = np.corrcoef(x_vals, y_vals)[0, 1]
    ax.text(0.05, 0.05, f"r = {corr:.3f}",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="gray", linewidth=0.4))

ax.set_xlabel("Schema Reduction (%)")
ax.set_ylabel("Execution Accuracy (%)")
ax.set_xlim(min(x_vals) - 5, max(x_vals) + 5)
ax.set_ylim(min(y_vals) - 5, max(y_vals) + 5)
ax.legend(loc="upper right", frameon=False, fontsize=7)
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig11a")
print(f"   Pearson r = {corr:.3f}")
print(f"   Avg reduction: {np.mean(list(reduction.values())):.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Fig11b — Relevance Score Distribution per Database  (standalone, full width)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig11b] Relevance Score Distribution per Database ...")

# Collect all relevance scores per database (relevant vs filtered)
db_above = defaultdict(int)   # tables scoring >= threshold (kept)
db_below = defaultdict(int)   # tables scoring < threshold (filtered)
db_avg_score = defaultdict(list)
for rec in schema_out:
    db = rec["database"]
    threshold = rec.get("relevance_threshold", 0.5)
    for tbl in rec.get("table_relevances", []):
        score = tbl.get("relevance_score", 0)
        db_avg_score[db].append(score)
        if score >= threshold:
            db_above[db] += 1
        else:
            db_below[db] += 1

# Sort by avg relevance score
sorted_dbs = sorted(db_avg_score.keys(),
                    key=lambda d: np.mean(db_avg_score[d]), reverse=True)

above_vals = [db_above[d] for d in sorted_dbs]
below_vals = [db_below[d] for d in sorted_dbs]
avg_scores = [np.mean(db_avg_score[d]) for d in sorted_dbs]

fig, ax = plt.subplots(figsize=(FULL, 2.6))

x = np.arange(len(sorted_dbs))
w = 0.35
b1 = ax.bar(x - w/2, above_vals, w, label=r"Relevant ($\geq\tau$)",
            color=WONG[1], hatch="///", edgecolor="white", linewidth=0.4)
b2 = ax.bar(x + w/2, below_vals, w, label=r"Filtered ($<\tau$)",
            color=WONG[3], hatch="xxx", edgecolor="white", linewidth=0.4)

for bar in (*b1, *b2):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(above_vals + below_vals) * 0.01,
                f"{h}", ha="center", va="bottom", fontsize=6)

# Avg score annotation on top
for i, (xi, s) in enumerate(zip(x, avg_scores)):
    total = above_vals[i] + below_vals[i]
    ax.text(xi, max(above_vals[i], below_vals[i]) + max(above_vals + below_vals) * 0.06,
            f"$\\bar{{s}}$={s:.2f}", ha="center", va="bottom",
            fontsize=6, color=WONG[0])

ax.set_xticks(x)
ax.set_xticklabels([db_short(d) for d in sorted_dbs], fontsize=7,
                    rotation=45, ha="right")
ax.set_ylabel("Number of Table Evaluations")
ax.legend(loc="upper right", frameon=False, fontsize=7.5)
ax.set_ylim(0, max(above_vals + below_vals) * 1.2)
ax.axhline(0, color="black", linewidth=0.4)
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig11b")
print(f"   Overall avg relevance score: {np.mean([s for v in db_avg_score.values() for s in v]):.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig12 — Per-Difficulty Strategy Ablation (2×2 grid, VLDB Fig 23 style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig12] Per-Difficulty Strategy Ablation (2x2) ...")

STRATEGIES = {
    "Q1: Full\nSchema":     {"simple": 73.84, "moderate": 57.67, "challenging": 60.69, "overall": 67.71},
    "Q2: SME\nMetadata":    {"simple": 74.16, "moderate": 58.32, "challenging": 58.62, "overall": 67.91},
    "Q3: Minimal\nProfile": {"simple": 74.27, "moderate": 61.12, "challenging": 57.93, "overall": 68.75},
    "Q4: Focused\nSchema":  {"simple": 79.24, "moderate": 64.79, "challenging": 64.83, "overall": 73.52},
    "Q5: Full\nProfile":    {"simple": 73.95, "moderate": 59.18, "challenging": 59.31, "overall": 68.10},
    "MRS-Agent":            {"simple": 83.24, "moderate": 68.25, "challenging": 65.52, "overall": 77.04},
}

strat_names = list(STRATEGIES.keys())
DIFF_LEVELS = ["simple", "moderate", "challenging", "overall"]
DIFF_TITLES = ["Simple", "Moderate", "Challenging", "Overall"]

# 1×4 horizontal layout — hatched bars, legend, baseline line, with gaps
fig, axes = plt.subplots(1, 4, figsize=(FULL, 2.2))

STRAT_COLORS  = [WONG[0], WONG[1], WONG[2], WONG[4], WONG[5], WONG[3]]
STRAT_HATCHES = [HATCHES[0], HATCHES[1], HATCHES[2], HATCHES[4], HATCHES[5], HATCHES[3]]
STRAT_LABELS  = ["Q1: Full Schema", "Q2: SME Metadata", "Q3: Minimal Profile",
                 "Q4: Focused Schema", "Q5: Full Profile", "MRS-Agent (Full)"]
SHORT_XLABELS = ["Q1: Full\nSchema", "Q2: SME\nMetadata", "Q3: Minimal\nProfile",
                 "Q4: Focused\nSchema", "Q5: Full\nProfile", "MRS-Agent"]

for idx, (level, title) in enumerate(zip(DIFF_LEVELS, DIFF_TITLES)):
    ax = axes[idx]
    vals = [STRATEGIES[s][level] for s in strat_names]
    x = np.arange(len(strat_names))

    bars = ax.bar(x, vals, width=0.65,
                  color=STRAT_COLORS, hatch=STRAT_HATCHES,
                  edgecolor="white", linewidth=0.4)

    # Value annotations on top
    for bar, val, sn in zip(bars, vals, strat_names):
        is_mrs = "MRS" in sn
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}",
                ha="center", va="bottom",
                fontsize=5.5 if not is_mrs else 6,
                fontweight="bold" if is_mrs else "normal",
                color=WONG[3] if is_mrs else "black")

    ax.set_xticks(x)
    ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4", "Q5", "MRS"], fontsize=6.5)

    ax.set_ylabel("EX (%)", fontsize=8)
    ax.set_ylim(50, 90)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    # Best single strategy baseline
    best_single = max(vals[:-1])
    ax.axhline(best_single, color="#BBBBBB", linewidth=0.6,
               linestyle="--", zorder=0)

    label_char = chr(ord('a') + idx)
    ax.set_xlabel(f"({label_char}) {title}", fontsize=8, labelpad=4)

# Shared legend at top
legend_handles = [mpatches.Patch(facecolor=c, hatch=h, edgecolor="white",
                                 linewidth=0.4, label=l)
                  for c, h, l in zip(STRAT_COLORS, STRAT_HATCHES, STRAT_LABELS)]
legend_handles.append(
    plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=0.6,
               label="Best Single Strategy")
)
fig.legend(handles=legend_handles, loc="upper center",
           ncol=4, frameon=False, fontsize=6,
           bbox_to_anchor=(0.5, 1.0))

plt.tight_layout(pad=0.4, w_pad=1.8, rect=[0, 0, 1, 0.82])
save_fig(fig, "Fig12")

print("   Strategy scores:")
for s in strat_names:
    print(f"     {s:20s} overall={STRATEGIES[s]['overall']:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Fig13 — Strategy Contribution Analysis  (1×2 panel, VLDB Fig 19 style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig13] Strategy Contribution Analysis (1x2) ...")

# (a) Win Rate — which strategy's SQL was selected by the judge
wins     = {k: 0 for k in STRAT_NAMES}
modified = 0
agreed   = 0
for qid, sel_sql in selected.items():
    sel_norm = normalize_sql(sel_sql)
    matched = [n for n in STRAT_NAMES
               if normalize_sql(candidates[n].get(qid, "")) == sel_norm]
    if not matched:
        modified += 1
    elif len(matched) == 1:
        wins[matched[0]] += 1
    else:
        agreed += 1  # multiple strategies produced the same SQL

# (b) SQL Diversity — avg pairwise Jaccard distance per question
pairwise_distances = defaultdict(list)
for qid in selected.keys():
    sqls = [normalize_sql(candidates[n].get(qid, "")) for n in STRAT_NAMES]
    for i in range(len(STRAT_NAMES)):
        for j in range(i+1, len(STRAT_NAMES)):
            sa = set(sqls[i].split()) if sqls[i] else set()
            sb = set(sqls[j].split()) if sqls[j] else set()
            if sa or sb:
                jacc = len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0
                dist = 1.0 - jacc
                pair_key = f"{STRAT_NAMES[i][:4]}-{STRAT_NAMES[j][:4]}"
                pairwise_distances["all"].append(dist)

# Per-database diversity
db_diversity = defaultdict(list)
for qid, sel_sql in selected.items():
    parts = sel_sql.split("\t----- bird -----\t")
    db = parts[1].strip() if len(parts) == 2 else "unknown"
    sqls = [normalize_sql(candidates[n].get(qid, "")) for n in STRAT_NAMES]
    unique_sqls = len(set(sqls))
    db_diversity[db].append(unique_sqls)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, 2.6))

# Panel (a): Strategy Win Rate — focused on unique wins (exclude Agreed)
# Show only the cases where a SINGLE strategy uniquely won or judge modified
# Add a text annotation for the "Agreed" count
win_labels = ["Full\nSchema", "SME\nMeta", "Minimal\nProfile",
              "Focused\nSchema", "Full\nProfile", "Judge\nModified"]
win_values = list(wins.values()) + [modified]
win_colors = WONG[:5] + [WONG[7]]
win_hats   = HATCHES[:6]

x = np.arange(len(win_labels))
bars = ax1.bar(x, win_values, width=0.6,
               color=win_colors, hatch=win_hats,
               edgecolor="white", linewidth=0.4)

for bar, val in zip(bars, win_values):
    pct = val / N * 100
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(win_values) * 0.02,
             f"{val}\n({pct:.0f}%)",
             ha="center", va="bottom", fontsize=6)

ax1.set_xticks(x)
ax1.set_xticklabels(win_labels, fontsize=6)
ax1.set_ylabel("Number of Questions")
ax1.set_ylim(0, max(win_values) * 1.28)

# Add annotation for the "Agreed" portion
agreed_pct = agreed / N * 100
ax1.text(0.98, 0.95,
         f"Strategies agreed: {agreed} ({agreed_pct:.0f}%)",
         transform=ax1.transAxes, ha="right", va="top",
         fontsize=7, color="gray",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                   edgecolor="gray", linewidth=0.4))
ax1.set_xlabel("(a) Unique Strategy Win Rate", fontsize=8, labelpad=4)

# Panel (b): Avg Unique SQL Candidates per Database
sorted_dbs = sorted(db_diversity.keys(),
                    key=lambda d: np.mean(db_diversity[d]), reverse=True)
db_labels  = [db_short(d) for d in sorted_dbs]
db_means   = [np.mean(db_diversity[d]) for d in sorted_dbs]

x2 = np.arange(len(sorted_dbs))
bars2 = ax2.bar(x2, db_means, width=0.6,
                color=WONG[0], hatch="", edgecolor="white", linewidth=0.4)

annotate_bars(ax2, bars2, fmt="{:.2f}", fontsize=6, offset=0.02)

ax2.set_xticks(x2)
ax2.set_xticklabels(db_labels, fontsize=5.5, rotation=45, ha="right")
ax2.set_ylabel("Avg. Unique SQL Candidates (of 5)")
ax2.set_ylim(0, 5.5)
ax2.axhline(np.mean(db_means), color="gray", linewidth=0.6,
            linestyle="--", alpha=0.6)
ax2.text(len(sorted_dbs) - 0.5, np.mean(db_means) + 0.08,
         f"Mean: {np.mean(db_means):.2f}", ha="right", va="bottom",
         fontsize=6.5, color="gray")
ax2.set_xlabel("(b) SQL Candidate Diversity per Database", fontsize=8, labelpad=4)

plt.tight_layout(pad=0.4, w_pad=1.5)
save_fig(fig, "Fig13")

print(f"   Win distribution: {wins}")
print(f"   Agreed (multi):   {agreed}")
print(f"   Judge-modified:   {modified}")
print(f"   Total:            {sum(wins.values()) + agreed + modified}")
print(f"   Avg diversity:    {np.mean(db_means):.2f} unique candidates/question")


# ══════════════════════════════════════════════════════════════════════════════
# Fig14 — Latency Distribution by Database  (2×2 box/bar, VLDB Fig 22 style)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig14] Latency Distribution by Pipeline Stage (2x2) ...")

stage_info = [
    ("Schema Agent",   "schema_agent_ms"),
    ("Candidate Gen.", "candidate_generation_ms"),
    ("Judge",          "judge_ms"),
    ("Execution",      "execution_ms"),
]

# Group all timings per database for each stage
fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 0.65))
axes = axes.flatten()

sorted_all_dbs = sorted(set(t["database"] for t in timings),
                        key=lambda d: db_short(d))
db_short_labels = [db_short(d) for d in sorted_all_dbs]

for idx, (stage_name, stage_key) in enumerate(stage_info):
    ax = axes[idx]

    data_per_db = []
    for db in sorted_all_dbs:
        vals = [t[stage_key] / 1000.0 for t in timings if t["database"] == db]
        data_per_db.append(vals)

    # Use median bar height + IQR (25th-75th) whiskers — robust to outliers
    medians = [np.median(v) for v in data_per_db]
    q25     = [np.percentile(v, 25) for v in data_per_db]
    q75     = [np.percentile(v, 75) for v in data_per_db]
    err_lo  = [med - q for med, q in zip(medians, q25)]
    err_hi  = [q - med for med, q in zip(medians, q75)]

    x = np.arange(len(sorted_all_dbs))
    bars = ax.bar(x, medians, width=0.6,
                  color=WONG[idx], hatch=HATCHES[idx],
                  edgecolor="white", linewidth=0.4,
                  yerr=[err_lo, err_hi], capsize=2,
                  error_kw={"linewidth": 0.5, "capthick": 0.5})

    # Annotate with median values above whisker
    for bar, val, eh in zip(bars, medians, err_hi):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + eh + max(medians) * 0.04,
                f"{val:.1f}", ha="center", va="bottom", fontsize=5.5)

    ax.set_xticks(x)
    ax.set_xticklabels(db_short_labels, fontsize=4.5, rotation=45, ha="right")
    ax.set_ylabel("Time (s)", fontsize=8)
    ax.set_ylim(bottom=0, top=max(q75) * 1.35)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    label_char = chr(ord('a') + idx)
    ax.set_xlabel(f"({label_char}) {stage_name}", fontsize=8, labelpad=4)

plt.tight_layout(pad=0.4, h_pad=1.5, w_pad=0.8)
save_fig(fig, "Fig14")


# ══════════════════════════════════════════════════════════════════════════════
# Fig15 — Component Type Distribution (2×2 grid showing top-4 DBs)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig15] Query Component Distribution by Database (2x2) ...")

COMP_TYPES = ["entity", "filter", "aggregation", "projection"]
COMP_LABELS = ["Entity", "Filter", "Aggregation", "Projection"]

# Compute per-database
db_comp = defaultdict(Counter)
db_comp_avg = defaultdict(lambda: defaultdict(float))
db_qcount = Counter()
for rec in decomp:
    db = rec["database"]
    db_qcount[db] += 1
    for comp in rec.get("components", []):
        ct = comp.get("type", "")
        if ct in COMP_TYPES:
            db_comp[db][ct] += 1

# Average components per question per database
for db in db_comp:
    for ct in COMP_TYPES:
        db_comp_avg[db][ct] = db_comp[db][ct] / db_qcount[db]

# Pick 4 diverse databases
comp_panel_dbs = ["california_schools", "formula_1", "financial", "card_games"]

fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 0.62))
axes = axes.flatten()

for idx, db in enumerate(comp_panel_dbs):
    ax = axes[idx]
    vals = [db_comp_avg[db][ct] for ct in COMP_TYPES]
    x = np.arange(len(COMP_TYPES))

    bars = ax.bar(x, vals, width=0.6,
                  color=[WONG[i] for i in range(4)],
                  hatch=[HATCHES[i] for i in range(4)],
                  edgecolor="white", linewidth=0.4)

    annotate_bars(ax, bars, fmt="{:.2f}", fontsize=6.5, offset=0.02)

    ax.set_xticks(x)
    ax.set_xticklabels(COMP_LABELS, fontsize=7)
    ax.set_ylabel("Avg. per Query", fontsize=8)
    ax.set_ylim(0, max(vals) * 1.3)

    label_char = chr(ord('a') + idx)
    ax.set_xlabel(f"({label_char}) {db_short(db)}", fontsize=8, labelpad=4)

plt.tight_layout(pad=0.4, h_pad=1.2, w_pad=0.8)
save_fig(fig, "Fig15")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"All multi-panel figures saved to: {FIGURES_DIR.resolve()}")
print(f"{'='*60}")
print("Figures generated:")
for stem in ["Fig10", "Fig11", "Fig12", "Fig13", "Fig14", "Fig15"]:
    for ext in ["eps", "tiff", "pdf"]:
        p = FIGURES_DIR / f"{stem}.{ext}"
        if p.exists():
            size = p.stat().st_size // 1024
            print(f"  {p.name:<20s} ({size}KB)")
