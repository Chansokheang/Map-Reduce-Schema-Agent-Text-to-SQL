"""
QA-SQL Analysis Figures
========================
Generates 6 figures from output/claude_headless_v2/ for the scientific paper.

Figures:
  1. Strategy Win Rate
  2. Candidate SQL Diversity (heatmap)
  3. Schema Reduction per Database
  4. Query Component Type Distribution
  5. Pipeline Stage Latency Breakdown
  6. Latency vs. Schema Complexity
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output/claude_headless_v2")
FIGURES_DIR = Path("analysis/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})
PALETTE = sns.color_palette("tab10")

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def normalize_sql(sql):
    """Lowercase + collapse whitespace for comparison."""
    return " ".join(sql.strip().lower().split()) if sql else ""

def jaccard(sql_a, sql_b):
    """Token-level Jaccard similarity between two SQL strings."""
    a = set(normalize_sql(sql_a).split())
    b = set(normalize_sql(sql_b).split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
selected    = load_json(OUTPUT_DIR / "selected.json")
timings     = load_jsonl(OUTPUT_DIR / "timings.jsonl")
schema_out  = load_jsonl(OUTPUT_DIR / "schema_agent_output.jsonl")
decomp      = load_jsonl(OUTPUT_DIR / "agentic_decomposition.jsonl")

STRATEGY_NAMES = [
    "Full Schema",
    "SME Metadata",
    "Minimal Profile",
    "Focused Schema",
    "Full Profile",
]
STRATEGY_FILES = [
    "candidate_full_schema.json",
    "candidate_sme_metadata.json",
    "candidate_minimal_profile.json",
    "candidate_focused_schema.json",
    "candidate_full_profile.json",
]

candidates = {}
for name, fname in zip(STRATEGY_NAMES, STRATEGY_FILES):
    candidates[name] = load_json(OUTPUT_DIR / fname)

print(f"  Loaded {len(selected)} questions, {len(timings)} timing records, "
      f"{len(schema_out)} schema records, {len(decomp)} decomposition records.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Strategy Win Rate
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Strategy Win Rate...")

exclusive_wins = {k: 0 for k in STRATEGY_NAMES}
no_match = 0

for qid, sel_sql in selected.items():
    sel_norm = normalize_sql(sel_sql)
    matched = [
        name for name in STRATEGY_NAMES
        if normalize_sql(candidates[name].get(qid, "")) == sel_norm
    ]
    if not matched:
        no_match += 1
    elif len(matched) == 1:
        exclusive_wins[matched[0]] += 1
    # multi-match: strategies agreed, not credited to any single one

total_exclusive = sum(exclusive_wins.values())
labels = list(exclusive_wins.keys()) + ["Judge Modified"]
values = list(exclusive_wins.values()) + [no_match]
colors = PALETTE[:len(STRATEGY_NAMES)] + ["#aaaaaa"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"{val}", va="center", ha="left", fontsize=10)

ax.set_xlabel("Number of Questions")
ax.set_title("Figure 1: Strategy Win Rate\n"
             "(exclusive matches between judge selection and each strategy)")
ax.set_xlim(0, max(values) * 1.18)
ax.invert_yaxis()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_strategy_win_rate.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig1_strategy_win_rate.png", bbox_inches="tight")
plt.close()
print(f"   Exclusive wins: {exclusive_wins}, No match: {no_match}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Candidate SQL Diversity (Jaccard Similarity Heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Candidate SQL Diversity...")

n = len(STRATEGY_NAMES)
sim_matrix = np.zeros((n, n))

all_qids = list(selected.keys())
for i, name_i in enumerate(STRATEGY_NAMES):
    for j, name_j in enumerate(STRATEGY_NAMES):
        if i == j:
            sim_matrix[i][j] = 1.0
        elif j > i:
            sims = [
                jaccard(candidates[name_i].get(qid, ""),
                        candidates[name_j].get(qid, ""))
                for qid in all_qids
            ]
            avg = float(np.mean(sims))
            sim_matrix[i][j] = avg
            sim_matrix[j][i] = avg

short_names = ["Full\nSchema", "SME\nMeta", "Minimal\nProfile",
               "Focused\nSchema", "Full\nProfile"]

fig, ax = plt.subplots(figsize=(6, 5))
mask = np.zeros_like(sim_matrix, dtype=bool)  # show full matrix

sns.heatmap(
    sim_matrix,
    annot=True, fmt=".2f",
    xticklabels=short_names, yticklabels=short_names,
    cmap="YlOrRd", vmin=0.0, vmax=1.0,
    linewidths=0.5, linecolor="white",
    ax=ax, cbar_kws={"label": "Avg. Jaccard Similarity"}
)
ax.set_title("Figure 2: Candidate SQL Diversity\n"
             "(avg. token-level Jaccard similarity across 1,534 questions)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_candidate_diversity.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig2_candidate_diversity.png", bbox_inches="tight")
plt.close()
print(f"   Similarity matrix computed over {len(all_qids)} questions.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Schema Reduction per Database
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Schema Reduction per Database...")

db_total    = defaultdict(list)
db_relevant = defaultdict(list)

for rec in schema_out:
    db = rec["database"]
    db_total[db].append(rec["total_tables_evaluated"])
    db_relevant[db].append(rec["relevant_tables_count"])

databases = sorted(db_total.keys())
avg_total    = [np.mean(db_total[db]) for db in databases]
avg_relevant = [np.mean(db_relevant[db]) for db in databases]

x = np.arange(len(databases))
width = 0.38

fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar(x - width/2, avg_total,    width, label="Total Tables",    color=PALETTE[0], alpha=0.85)
b2 = ax.bar(x + width/2, avg_relevant, width, label="Relevant Tables", color=PALETTE[1], alpha=0.85)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([db.replace("_", "\n") for db in databases], fontsize=8.5)
ax.set_ylabel("Average Number of Tables")
ax.set_title("Figure 3: Schema Reduction per Database\n"
             "(avg. total vs. relevant tables selected by Map-Reduce Schema Agent)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_schema_reduction.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig3_schema_reduction.png", bbox_inches="tight")
plt.close()
print(f"   Processed {len(databases)} databases.")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Query Component Type Distribution per Database
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Query Component Type Distribution...")

COMP_TYPES = ["entity", "filter", "aggregation", "projection"]
db_comp_counts = defaultdict(lambda: Counter())

for rec in decomp:
    db = rec["database"]
    for comp in rec.get("components", []):
        ctype = comp.get("type", "unknown")
        if ctype in COMP_TYPES:
            db_comp_counts[db][ctype] += 1

databases_d = sorted(db_comp_counts.keys())
comp_data = {
    ctype: [db_comp_counts[db][ctype] for db in databases_d]
    for ctype in COMP_TYPES
}

x = np.arange(len(databases_d))
width = 0.18
offsets = np.linspace(-1.5 * width, 1.5 * width, len(COMP_TYPES))
comp_colors = PALETTE[:len(COMP_TYPES)]

fig, ax = plt.subplots(figsize=(10, 4.5))
for i, (ctype, color, offset) in enumerate(zip(COMP_TYPES, comp_colors, offsets)):
    bars = ax.bar(x + offset, comp_data[ctype], width,
                  label=ctype.capitalize(), color=color, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([db.replace("_", "\n") for db in databases_d], fontsize=8.5)
ax.set_ylabel("Component Count")
ax.set_title("Figure 4: Query Component Type Distribution per Database\n"
             "(entity, filter, aggregation, projection from Map-Reduce decomposition)")
ax.legend(title="Component Type")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_component_distribution.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig4_component_distribution.png", bbox_inches="tight")
plt.close()

overall = Counter()
for db in db_comp_counts:
    overall.update(db_comp_counts[db])
print(f"   Overall: {dict(overall)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Pipeline Stage Latency Breakdown
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Pipeline Stage Latency Breakdown...")

STAGES = {
    "Schema Agent":       "schema_agent_ms",
    "Candidate Gen.":     "candidate_generation_ms",
    "Judge":              "judge_ms",
    "Execution":          "execution_ms",
    "Last Resort":        "last_resort_ms",
    "Input Processing":   "input_processing_ms",
}

avg_stage = {
    label: np.mean([t[key] for t in timings])
    for label, key in STAGES.items()
}

# Sort descending
avg_stage = dict(sorted(avg_stage.items(), key=lambda x: -x[1]))
labels  = list(avg_stage.keys())
values  = [v / 1000 for v in avg_stage.values()]  # convert to seconds
total_s = sum(values)
pcts    = [v / total_s * 100 for v in values]
colors  = PALETTE[:len(labels)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: horizontal bar
bars = ax1.barh(labels, values, color=colors, edgecolor="white")
for bar, pct in zip(bars, pcts):
    ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
             f"{pct:.1f}%", va="center", fontsize=9)
ax1.set_xlabel("Average Time (seconds)")
ax1.set_title("Stage Latency (avg. per question)")
ax1.invert_yaxis()
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_xlim(0, max(values) * 1.22)

# Right: pie
wedges, texts, autotexts = ax2.pie(
    values, labels=None, autopct="%1.1f%%",
    colors=colors, startangle=140,
    pctdistance=0.75, wedgeprops={"edgecolor": "white"}
)
for at in autotexts:
    at.set_fontsize(8)
ax2.legend(wedges, labels, loc="lower center",
           bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
ax2.set_title("Stage Proportion (%)")

fig.suptitle("Figure 5: Pipeline Stage Latency Breakdown\n"
             f"(avg. total = {total_s:.1f}s per question, n={len(timings)})",
             y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig5_latency_breakdown.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig5_latency_breakdown.png", bbox_inches="tight")
plt.close()
print(f"   Avg total latency: {total_s:.1f}s")
for label, val, pct in zip(labels, values, pcts):
    print(f"   {label}: {val:.2f}s ({pct:.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Latency vs. Schema Complexity (Scatter)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Latency vs. Schema Complexity...")

# Join timings and schema_agent by question text (both have 'question' field)
schema_by_q = {r["question"]: r for r in schema_out}

x_vals, y_vals, db_labels = [], [], []
for t in timings:
    q = t["question"]
    if q in schema_by_q:
        x_vals.append(schema_by_q[q]["total_tables_evaluated"])
        y_vals.append(t["schema_agent_ms"] / 1000)  # seconds
        db_labels.append(t["database"])

# Color by database
unique_dbs = sorted(set(db_labels))
db_color_map = {db: PALETTE[i % len(PALETTE)] for i, db in enumerate(unique_dbs)}
point_colors = [db_color_map[db] for db in db_labels]

fig, ax = plt.subplots(figsize=(7, 5))
scatter = ax.scatter(x_vals, y_vals, c=point_colors, alpha=0.35, s=18, linewidths=0)

# Regression line
x_arr = np.array(x_vals)
y_arr = np.array(y_vals)
coeffs = np.polyfit(x_arr, y_arr, 1)
x_line = np.linspace(x_arr.min(), x_arr.max(), 100)
ax.plot(x_line, np.polyval(coeffs, x_line), color="black",
        linewidth=1.5, linestyle="--", label=f"Trend (slope={coeffs[0]:.2f}s/table)")

# Legend for databases
handles = [mpatches.Patch(color=db_color_map[db],
                           label=db.replace("_", " ")) for db in unique_dbs]
ax.legend(handles=handles, title="Database", fontsize=7.5,
          title_fontsize=8, loc="upper left", ncol=2)

ax.set_xlabel("Total Tables Evaluated by Schema Agent")
ax.set_ylabel("Schema Agent Latency (seconds)")
ax.set_title("Figure 6: Latency vs. Schema Complexity\n"
             "(each point = one question; dashed = linear trend)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig6_latency_vs_complexity.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig6_latency_vs_complexity.png", bbox_inches="tight")
plt.close()
print(f"   Plotted {len(x_vals)} points across {len(unique_dbs)} databases.")
print(f"   Regression slope: {coeffs[0]:.2f}s per additional table")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")
print("Files:")
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  {f.name}")
