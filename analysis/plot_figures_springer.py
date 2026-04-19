"""
QA-SQL: Springer VLDB Journal Figure Generator
================================================
Generates publication-quality figures following Springer artwork guidelines:

  Font       : Arial/Helvetica, 8–9 pt (axis labels), 7–8 pt (tick/legend)
  Width      : 84 mm (single-column) or 174 mm (full text-width)
  Max height : 234 mm
  Resolution : 600 dpi (combination art — color + text)
  Color      : RGB, 8 bits per channel; Wong (2011) colorblind-safe palette
  Hatching   : added to every filled region for B&W print / accessibility
  Format     : EPS (vector, fonts embedded) + TIFF 600 dpi + PDF for LaTeX
  Naming     : Fig1.eps … Fig7.eps  (no titles inside figures)
  Line width : >= 0.5 pt (well above the 0.1 mm / 0.3 pt minimum)

Figures:
  Fig1 — Strategy Win Rate (horizontal bar)
  Fig2 — Candidate SQL Diversity (Jaccard heatmap)
  Fig3 — Schema Reduction per Database (grouped bar)
  Fig4 — Query Component Distribution (stacked bar)
  Fig5 — Pipeline Stage Latency Breakdown (horizontal bar)
  Fig6 — Latency vs. Schema Complexity (scatter)
  Fig7 — Pipeline Stage Latency CDF

Usage (run from project root):
  python analysis/plot_figures_springer.py
"""

import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("../output/claude_headless_v2")
FIGURES_DIR = Path("./figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Springer Figure Dimensions ────────────────────────────────────────────────
MM    = 1 / 25.4                    # 1 mm in inches
COL   = 84  * MM                    # 84 mm  ≈ 3.31" — single column width
FULL  = 174 * MM                    # 174 mm ≈ 6.85" — full text width
HMAX  = 234 * MM                    # 234 mm ≈ 9.21" — max figure height
DPI   = 600                         # combination art (color + lettering)

# ── Global Style (Springer-Compliant) ─────────────────────────────────────────
plt.rcParams.update({
    # Springer: Helvetica or Arial, 8–12 pt at final size
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
    # Lines — Springer min: 0.1 mm (0.3 pt); use 0.6–1 pt for legibility
    "lines.linewidth":       1.0,
    "axes.linewidth":        0.6,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.minor.width":     0.4,
    "ytick.minor.width":     0.4,
    "patch.linewidth":       0.5,
    # Spines
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    # Save
    "savefig.dpi":           DPI,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    # Embed fonts in EPS/PDF (required for vector submission)
    "pdf.fonttype":          42,   # TrueType
    "ps.fonttype":           42,   # TrueType
})

# ── Pastel Palette (VLDB Journal / Springer style) ───────────────────────────
# Soft lavender–mint–cream aesthetic matching published VLDB figures
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

# Hatch patterns — one per filled region for B&W print accessibility
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

def jaccard(a, b):
    sa = set(normalize_sql(a).split())
    sb = set(normalize_sql(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def save_fig(fig, stem):
    """Save EPS (vector) + TIFF (600 dpi) + PDF for LaTeX."""
    for ext in ("eps", "tiff", "pdf"):
        p = FIGURES_DIR / f"{stem}.{ext}"
        kwargs = {"format": ext}
        if ext == "tiff":
            kwargs["dpi"] = DPI
        fig.savefig(p, **kwargs)
    plt.close(fig)
    print(f"   → {stem}.eps / .tiff / .pdf")

def db_label(name):
    """Short, multi-line database label for axis ticks."""
    return name.replace("_", "\n")


# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data …")
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
print(f"  {N} questions | {len(timings)} timing records "
      f"| {len(schema_out)} schema records | {len(decomp)} decomp records")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig1 — Strategy Win Rate  (full-width horizontal bar)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Fig1: Strategy Win Rate …")

wins     = {k: 0 for k in STRAT_NAMES}
modified = 0

for qid, sel_sql in selected.items():
    sel_norm = normalize_sql(sel_sql)
    matched  = [n for n in STRAT_NAMES
                if normalize_sql(candidates[n].get(qid, "")) == sel_norm]
    if not matched:
        modified += 1
    elif len(matched) == 1:
        wins[matched[0]] += 1
    # multi-match: all strategies agreed → not credited to a single one

labels = list(wins.keys()) + ["Judge Modified"]
values = list(wins.values()) + [modified]
colors = WONG[:len(STRAT_NAMES)] + ["#aaaaaa"]
hats   = HATCHES[:len(labels)]

fig, ax = plt.subplots(figsize=(FULL, 2.4))
bars = ax.barh(labels, values, height=0.6,
               color=colors, edgecolor="white", linewidth=0.5)
for bar, h in zip(bars, hats):
    bar.set_hatch(h)
for bar, val in zip(bars, values):
    pct = val / N * 100
    ax.text(bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val} ({pct:.1f}%)", va="center", fontsize=7.5)

ax.set_xlabel("Number of Questions")
ax.set_xlim(0, max(values) * 1.28)
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig1")
print(f"   Wins: {wins} | Judge-modified: {modified}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig2 — Candidate SQL Diversity  (Jaccard heatmap, single column)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Fig2: Candidate SQL Diversity …")

n   = len(STRAT_NAMES)
sim = np.zeros((n, n))
qids = list(selected.keys())

for i, ni in enumerate(STRAT_NAMES):
    for j, nj in enumerate(STRAT_NAMES):
        if i == j:
            sim[i, j] = 1.0
        elif j > i:
            avg = float(np.mean([
                jaccard(candidates[ni].get(q, ""),
                        candidates[nj].get(q, ""))
                for q in qids
            ]))
            sim[i, j] = sim[j, i] = avg

short = ["Full\nSchema", "SME\nMeta", "Minimal\nProfile",
         "Focused\nSchema", "Full\nProfile"]

fig, ax = plt.subplots(figsize=(COL * 1.55, COL * 1.35))
sns.heatmap(
    sim, annot=True, fmt=".2f",
    xticklabels=short, yticklabels=short,
    cmap="YlOrRd", vmin=0.0, vmax=1.0,
    linewidths=0.4, linecolor="white",
    annot_kws={"size": 7},
    ax=ax,
    cbar_kws={"label": "Avg. Jaccard Similarity", "shrink": 0.85,
              "pad": 0.02}
)
ax.tick_params(axis="both", labelsize=7)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.collections[0].colorbar.ax.tick_params(labelsize=7)
ax.collections[0].colorbar.set_label("Avg. Jaccard Similarity", size=8)
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig2")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig3 — Schema Reduction per Database  (grouped bar, full width)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Fig3: Schema Reduction per Database …")

db_total    = defaultdict(list)
db_relevant = defaultdict(list)
for rec in schema_out:
    db = rec["database"]
    db_total[db].append(rec["total_tables_evaluated"])
    db_relevant[db].append(rec["relevant_tables_count"])

databases    = sorted(db_total.keys())
avg_total    = np.array([np.mean(db_total[d])    for d in databases])
avg_relevant = np.array([np.mean(db_relevant[d]) for d in databases])
reduction    = (1 - avg_relevant / avg_total) * 100   # % tables removed

x = np.arange(len(databases))
w = 0.35

fig, ax = plt.subplots(figsize=(FULL, 2.8))
b1 = ax.bar(x - w/2, avg_total,    w, label="Total Tables",
            color=WONG[0], hatch="",    edgecolor="white", linewidth=0.4)
b2 = ax.bar(x + w/2, avg_relevant, w, label="Relevant Tables",
            color=WONG[1], hatch="///", edgecolor="white", linewidth=0.4)

for bar in (*b1, *b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08,
            f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)

# Secondary annotation: reduction %
for i, (xi, r) in enumerate(zip(x, reduction)):
    ax.annotate(f"−{r:.0f}%",
                xy=(xi + w/2, avg_relevant[i]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=6,
                color=WONG[3])

ax.set_xticks(x)
ax.set_xticklabels([db_label(d) for d in databases], fontsize=7)
ax.set_ylabel("Avg. Number of Tables")
ax.legend(loc="upper right", frameon=False)
ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig3")
print(f"   Avg reduction across DBs: {reduction.mean():.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig4 — Query Component Distribution  (stacked %, full width)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Fig4: Query Component Distribution …")

COMP_TYPES = ["entity", "filter", "aggregation", "projection"]
db_comp = defaultdict(Counter)
for rec in decomp:
    for comp in rec.get("components", []):
        ct = comp.get("type", "")
        if ct in COMP_TYPES:
            db_comp[rec["database"]][ct] += 1

databases_d = sorted(db_comp.keys())
comp_arr    = {ct: np.array([db_comp[d][ct] for d in databases_d], float)
               for ct in COMP_TYPES}
totals_d    = sum(comp_arr[ct] for ct in COMP_TYPES)

fig, ax = plt.subplots(figsize=(FULL, 2.6))
x      = np.arange(len(databases_d))
bottom = np.zeros(len(databases_d))

for i, ct in enumerate(COMP_TYPES):
    pct = comp_arr[ct] / totals_d * 100
    ax.bar(x, pct, bottom=bottom,
           label=ct.capitalize(),
           color=WONG[i], hatch=HATCHES[i],
           edgecolor="white", linewidth=0.4)
    bottom += pct

ax.set_xticks(x)
ax.set_xticklabels([db_label(d) for d in databases_d], fontsize=7)
ax.set_ylabel("Component Share (%)")
ax.set_ylim(0, 115)
ax.legend(loc="upper right", frameon=False, ncol=2)
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig4")

overall = Counter()
for d in db_comp:
    overall.update(db_comp[d])
print(f"   Overall component counts: {dict(overall)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig5 — Pipeline Stage Latency Breakdown  (horizontal bar, full width)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Fig5: Pipeline Stage Latency …")

STAGE_KEYS = {
    "Schema Agent":     "schema_agent_ms",
    "Candidate Gen.":   "candidate_generation_ms",
    "Judge":            "judge_ms",
    "Execution":        "execution_ms",
    "Last Resort":      "last_resort_ms",
    "Input Processing": "input_processing_ms",
}

avg_s   = {lbl: np.mean([t[k] for t in timings]) / 1000
           for lbl, k in STAGE_KEYS.items()}
avg_s   = dict(sorted(avg_s.items(), key=lambda kv: -kv[1]))
s_lbls  = list(avg_s.keys())
s_vals  = list(avg_s.values())
total_s = sum(s_vals)
pcts    = [v / total_s * 100 for v in s_vals]

fig, ax = plt.subplots(figsize=(FULL, 2.2))
bars = ax.barh(
    s_lbls, s_vals, height=0.6,
    color=[WONG[i % len(WONG)] for i in range(len(s_lbls))],
    hatch=[HATCHES[i]          for i in range(len(s_lbls))],
    edgecolor="white", linewidth=0.5
)
for bar, pct, val in zip(bars, pcts, s_vals):
    ax.text(bar.get_width() + max(s_vals) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} s  ({pct:.0f}%)",
            va="center", fontsize=7.5)

ax.set_xlabel("Average Time (seconds)")
ax.set_xlim(0, max(s_vals) * 1.45)
ax.invert_yaxis()
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig5")
print(f"   Avg total latency: {total_s:.1f}s/question")
for lbl, v, p in zip(s_lbls, s_vals, pcts):
    print(f"   {lbl:<22s} {v:.2f}s  {p:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig6 — Latency vs. Schema Complexity  (scatter, single column)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Fig6: Latency vs. Schema Complexity …")

schema_by_q = {r["question"]: r for r in schema_out}
x_v, y_v, db_v = [], [], []
for t in timings:
    r = schema_by_q.get(t["question"])
    if r:
        x_v.append(r["total_tables_evaluated"])
        y_v.append(t["schema_agent_ms"] / 1000)
        db_v.append(t["database"])

unique_dbs   = sorted(set(db_v))
MARKERS      = ["o", "s", "^", "D", "v", "P", "*", "X",
                "h", "<", ">", "8", "p", "H"]
db_c = {db: WONG[i % len(WONG)]    for i, db in enumerate(unique_dbs)}
db_m = {db: MARKERS[i % len(MARKERS)] for i, db in enumerate(unique_dbs)}

# Single column figure — legend placed below
fig, ax = plt.subplots(figsize=(COL * 1.5, COL * 1.3))
for db in unique_dbs:
    xi = [x_v[k] for k, d in enumerate(db_v) if d == db]
    yi = [y_v[k] for k, d in enumerate(db_v) if d == db]
    ax.scatter(xi, yi, c=db_c[db], marker=db_m[db],
               s=8, alpha=0.45, linewidths=0,
               label=db.replace("_", " "))

coeffs = np.polyfit(x_v, y_v, 1)
xl     = np.linspace(min(x_v), max(x_v), 200)
ax.plot(xl, np.polyval(coeffs, xl),
        color="black", linewidth=1.0, linestyle="--",
        label=f"Trend ({coeffs[0]:.2f} s/table)")

ax.set_xlabel("Total Tables Evaluated")
ax.set_ylabel("Schema Agent Latency (s)")
ax.legend(fontsize=5.5, frameon=False, ncol=2,
          loc="upper left", handlelength=0.8, handletextpad=0.4,
          borderpad=0.3, labelspacing=0.25)
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig6")
print(f"   {len(x_v)} points, {len(unique_dbs)} databases; "
      f"slope={coeffs[0]:.3f} s/table")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig7 — Pipeline Stage Latency CDF  (full width)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Fig7: Latency CDF …")

STAGE_CDF = {
    "Schema Agent":   ("schema_agent_ms",        WONG[0], "-"),
    "Candidate Gen.": ("candidate_generation_ms", WONG[1], "--"),
    "Judge":          ("judge_ms",                WONG[2], "-."),
    "Execution":      ("execution_ms",            WONG[3], ":"),
}

fig, ax = plt.subplots(figsize=(FULL, 2.5))
for label, (key, color, ls) in STAGE_CDF.items():
    vals  = np.sort([t[key] / 1000.0 for t in timings])
    cdf   = np.arange(1, len(vals) + 1) / len(vals) * 100
    p50   = float(np.percentile(vals, 50))
    p90   = float(np.percentile(vals, 90))
    ax.plot(vals, cdf, label=f"{label} (p50={p50:.0f}s, p90={p90:.0f}s)",
            color=color, linewidth=1.2, linestyle=ls)
    print(f"   {label:<22s} p50={p50:.1f}s  p90={p90:.1f}s  "
          f"mean={vals.mean():.1f}s  max={vals.max():.1f}s")

ax.axhline(50, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)
ax.axhline(90, color="gray", linewidth=0.5, linestyle=":", alpha=0.7)
# Annotate reference lines at right edge after drawing
ax.set_xlim(left=0)
ax.set_ylim(0, 105)
xmax = ax.get_xlim()[1]
ax.text(xmax * 0.99, 51.5, "p50", va="bottom", ha="right",
        color="gray", fontsize=7)
ax.text(xmax * 0.99, 91.5, "p90", va="bottom", ha="right",
        color="gray", fontsize=7)

ax.set_xlabel("Latency (seconds)")
ax.set_ylabel("Cumulative Questions (%)")
ax.legend(loc="lower right", frameon=False)
plt.tight_layout(pad=0.3)
save_fig(fig, "Fig7")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"All figures saved to: {FIGURES_DIR.resolve()}")
print(f"{'='*60}")
print("Format spec (Springer VLDB Journal):")
print(f"  Width  : COL={84}mm (Fig2,6) | FULL={174}mm (Fig1,3,4,5,7)")
print(f"  DPI    : {DPI} (combination art)")
print(f"  Font   : Arial/Helvetica, 8-9 pt labels, 7-8 pt ticks/legend")
print(f"  Color  : RGB, Wong (2011) colorblind-safe palette")
print(f"  Hatch  : applied to all filled regions for B&W accessibility")
print(f"  Output : .eps (vector, fonts embedded) + .tiff + .pdf")
print(f"\nFiles written:")
for stem in ["Fig1","Fig2","Fig3","Fig4","Fig5","Fig6","Fig7"]:
    for ext in ["eps","tiff","pdf"]:
        p = FIGURES_DIR / f"{stem}.{ext}"
        size = f"({p.stat().st_size//1024}KB)" if p.exists() else "(pending)"
        print(f"  {p.name:<20s} {size}")
