"""
Pipeline Stage Latency CDF
===========================
CDF plot of schema agent, candidate generation, judge, and execution
latency across all 1,534 questions.

Reads from: output/claude_headless_v2/timings.jsonl
Saves to:   analysis/figures/fig7_latency_cdf.pdf / .png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("output/claude_headless_v2")
FIGURES_DIR = Path("analysis/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":  150,
})

# ── Load timings ──────────────────────────────────────────────────────────────
with open(OUTPUT_DIR / "timings.jsonl", encoding="utf-8") as f:
    timings = [json.loads(l) for l in f if l.strip()]

print(f"Loaded {len(timings)} timing records.")

# ── Stages to plot ────────────────────────────────────────────────────────────
STAGES = {
    "Schema Agent":    ("schema_agent_ms",       "#1f77b4"),  # blue
    "Candidate Gen.":  ("candidate_generation_ms","#ff7f0e"),  # orange
    "Judge":           ("judge_ms",              "#2ca02c"),  # green
    "Execution":       ("execution_ms",           "#d62728"),  # red
}

# ── Compute CDF per stage ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

for label, (key, color) in STAGES.items():
    values_s = np.array([t[key] / 1000.0 for t in timings])  # ms → seconds
    sorted_v = np.sort(values_s)
    cdf      = np.arange(1, len(sorted_v) + 1) / len(sorted_v)

    ax.plot(sorted_v, cdf * 100, label=label, color=color, linewidth=2)

    # Annotate p50 and p90
    p50 = np.percentile(sorted_v, 50)
    p90 = np.percentile(sorted_v, 90)
    ax.axvline(p50, color=color, linewidth=0.6, linestyle=":", alpha=0.5)

    print(f"{label:20s}  p50={p50:.1f}s  p90={p90:.1f}s  "
          f"max={sorted_v.max():.1f}s  mean={sorted_v.mean():.1f}s")

# ── Reference lines ───────────────────────────────────────────────────────────
ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax.axhline(90, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 200,
        51, "p50", va="bottom", ha="right", color="gray", fontsize=9)
ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 200,
        91, "p90", va="bottom", ha="right", color="gray", fontsize=9)

ax.set_xlabel("Latency (seconds)")
ax.set_ylabel("Cumulative % of Questions")
ax.set_title(f"Figure 7: Pipeline Stage Latency CDF\n"
             f"(n = {len(timings)} questions)")
ax.set_ylim(0, 102)
ax.set_xlim(left=0)
ax.legend(loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig7_latency_cdf.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig7_latency_cdf.png", bbox_inches="tight")
plt.close()

print(f"\nSaved: {FIGURES_DIR / 'fig7_latency_cdf.pdf'}")
print(f"Saved: {FIGURES_DIR / 'fig7_latency_cdf.png'}")
