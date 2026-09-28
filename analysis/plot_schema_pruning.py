"""
Schema Pruning Before & After Network Diagram
===============================================
Generates a side-by-side network graph showing:
  (a) Full database schema (all tables and FK relationships)
  (b) Pruned schema after Map-Reduce Schema Agent filtering

Accessibility fixes:
  - Black/dark text on all light backgrounds (WCAG compliant)
  - Large readable font sizes (scaled for 84mm column width)
  - Consistent Arial font matching other figures
  - No transparency (EPS/print safe)

Uses Formula 1 database (14 tables -> 2 relevant) as the example.

Output: Fig_pruning.svg / .tiff / .pdf
"""

import sqlite3
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths & Constants ─────────────────────────────────────────────────────────
DB_PATH     = Path("../data/bird_data/dev_databases/formula_1/formula_1.sqlite")
FIGURES_DIR = Path("./figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MM   = 1 / 25.4
FULL = 174 * MM
DPI  = 600

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica Neue", "Helvetica",
                           "Liberation Sans", "DejaVu Sans"],
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.linewidth":     0.6,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "savefig.dpi":        DPI,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── Colors (high contrast, no transparency) ──────────────────────────────────
C_NODE_FULL    = "#A195C1"   # lavender — all nodes in (a)
C_RELEVANT     = "#8DC9A8"   # mint green — kept tables
C_FILTERED     = "#E0E0E0"   # light gray — filtered tables
C_EDGE_ACTIVE  = "#444444"   # dark gray — active FK
C_EDGE_FADED   = "#CCCCCC"   # light gray — pruned FK
C_BORDER_REL   = "#2E7D52"   # dark green — relevant node border
C_BORDER_FILT  = "#BBBBBB"   # gray — filtered node border
C_TEXT_DARK    = "#1A1A1A"   # near-black — all node labels
C_TEXT_FADED   = "#888888"   # medium gray — filtered labels in (b)


# ── Build Graph from SQLite ──────────────────────────────────────────────────
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
all_tables = [r[0] for r in cursor.fetchall() if r[0] != "sqlite_sequence"]

fk_edges = []
for t in all_tables:
    cursor.execute(f"PRAGMA foreign_key_list({t})")
    for row in cursor.fetchall():
        target = row[2]
        if target in all_tables:
            fk_edges.append((t, target))
conn.close()

fk_edges = list(set(fk_edges))

# Relevant tables from actual pipeline output (schema_agent_output.jsonl)
relevant_tables = {"drivers", "qualifying"}

# Build networkx graph
G = nx.Graph()
G.add_nodes_from(all_tables)
G.add_edges_from(fk_edges)

# ── Layout (fixed seed for reproducibility) ──────────────────────────────────
pos = nx.spring_layout(G, seed=42, k=3.8, iterations=200)


def short_name(name):
    """Compact labels that fit inside node circles."""
    mapping = {
        "constructorResults":   "constr.\nResults",
        "constructorStandings": "constr.\nStndgs",
        "driverStandings":      "driver\nStndgs",
        "lapTimes":             "lapTimes",
        "pitStops":             "pitStops",
    }
    return mapping.get(name, name)


def label_metrics(name):
    """Return (max_chars_in_line, n_lines) for the label."""
    label = short_name(name)
    lines = label.split("\n")
    return max(len(l) for l in lines), len(lines)


def node_size_for(name, base=500, char_scale=75, line_bonus=180,
                  emphasize=False):
    """Vary node area to fit its label. Emphasized (relevant) nodes get
    an extra boost so they read as visually dominant."""
    max_chars, n_lines = label_metrics(name)
    size = base + char_scale * max_chars + line_bonus * (n_lines - 1)
    if emphasize:
        size = int(size * 1.30)
    return size


# matplotlib node_size is area in points². The empirical conversion from
# node area to a "radius" in axis units (panel spans roughly [-1,1] inside
# an ~87mm wide axes) is ~0.0072 * sqrt(area). Used by resolve_overlap.
NODE_AXIS_SCALE = 0.0072


def resolve_overlap_varied(pos, sizes, scale=NODE_AXIS_SCALE,
                           padding=0.08, max_iter=600):
    """Push nodes apart until every pair clears (r_i + r_j + padding).
    No rescaling — that would shrink resolved gaps back under the diameter."""
    keys = list(pos.keys())
    arr = np.array([pos[k] for k in keys], dtype=float)
    radii = np.array([scale * np.sqrt(sizes[k]) for k in keys])
    for _ in range(max_iter):
        moved = False
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                d = arr[j] - arr[i]
                dist = np.linalg.norm(d)
                min_dist = radii[i] + radii[j] + padding
                if dist < min_dist:
                    if dist < 1e-9:
                        d = np.array([1e-3, 0.0])
                        dist = 1e-3
                    push = (min_dist - dist) / 2.0 + 5e-3
                    unit = d / dist
                    arr[i] -= unit * push
                    arr[j] += unit * push
                    moved = True
        if not moved:
            break
    # Re-center only (no scaling) so layout stays around origin.
    arr -= arr.mean(axis=0)
    return {k: tuple(arr[i]) for i, k in enumerate(keys)}


# Per-node sizes. Panel (b) gets a separate, more dramatic sizing where the
# relevant tables are inflated and filtered tables shrink. We resolve overlap
# using the larger of the two so the same layout works for both panels.
sizes_full   = {t: node_size_for(t) for t in all_tables}
sizes_pruned = {
    t: node_size_for(t, base=400, char_scale=65, line_bonus=160,
                     emphasize=(t in relevant_tables))
    for t in all_tables
}
sizes_layout = {t: max(sizes_full[t], sizes_pruned[t]) for t in all_tables}

pos = resolve_overlap_varied(pos, sizes_layout)


# ── Draw Figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, FULL * 0.52))

# ─────────────────────────────────────────────
# (a) BEFORE — Full Schema (all tables equal)
# ─────────────────────────────────────────────

# Edges
nx.draw_networkx_edges(G, pos, ax=ax1,
                       edge_color=C_EDGE_ACTIVE, width=0.9)

# Nodes — lavender, dark border; size varies per label so text fits
node_list_full = list(all_tables)
nx.draw_networkx_nodes(G, pos, nodelist=node_list_full, ax=ax1,
                       node_color=C_NODE_FULL,
                       node_size=[sizes_full[t] for t in node_list_full],
                       edgecolors="#7A6FA0", linewidths=1.2)

# Labels — BLACK text on light background (accessibility fix)
def _label_fontsize(size, n_lines, base=8.0):
    """Font scales gently with circle area; multi-line labels shrink slightly."""
    fs = base * (size / 2100.0) ** 0.40
    if n_lines > 1:
        fs *= 0.92
    return float(np.clip(fs, 6.0, 10.0))

for t in all_tables:
    x, y = pos[t]
    _, n_lines = label_metrics(t)
    ax1.text(x, y, short_name(t), ha="center", va="center",
             fontsize=_label_fontsize(sizes_full[t], n_lines),
             fontweight="medium", color=C_TEXT_DARK,
             fontfamily="Arial", linespacing=0.95)

ax1.set_xlabel(f"(a) Full Schema ({len(all_tables)} tables, {len(fk_edges)} edges)",
               fontsize=9, labelpad=8)
ax1.margins(0.18)
ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# ─────────────────────────────────────────────
# (b) AFTER — Pruned Schema
# ─────────────────────────────────────────────

# Classify edges
active_edges = [(u, v) for u, v in fk_edges
                if u in relevant_tables and v in relevant_tables]
faded_edges  = [(u, v) for u, v in fk_edges
                if not (u in relevant_tables and v in relevant_tables)]

# Draw faded edges first (dotted, light)
nx.draw_networkx_edges(G, pos, edgelist=faded_edges, ax=ax2,
                       edge_color=C_EDGE_FADED, width=0.5, style="dotted")
# Draw active edges (solid, bold)
nx.draw_networkx_edges(G, pos, edgelist=active_edges, ax=ax2,
                       edge_color=C_EDGE_ACTIVE, width=2.0)

# Draw filtered nodes (variably sized, gray) — sized to contain labels
filtered_tables = [t for t in all_tables if t not in relevant_tables]
nx.draw_networkx_nodes(G, pos, nodelist=filtered_tables, ax=ax2,
                       node_color=C_FILTERED,
                       node_size=[sizes_pruned[t] for t in filtered_tables],
                       edgecolors=C_BORDER_FILT, linewidths=0.8)

# Draw relevant nodes (emphasized, green, thick dark border)
relevant_list = [t for t in all_tables if t in relevant_tables]
nx.draw_networkx_nodes(G, pos, nodelist=relevant_list, ax=ax2,
                       node_color=C_RELEVANT,
                       node_size=[sizes_pruned[t] for t in relevant_list],
                       edgecolors=C_BORDER_REL, linewidths=2.0)

# Labels — dark text everywhere (accessibility fix); font scales with circle
for t in all_tables:
    x, y = pos[t]
    _, n_lines = label_metrics(t)
    if t in relevant_tables:
        ax2.text(x, y, short_name(t), ha="center", va="center",
                 fontsize=_label_fontsize(sizes_pruned[t], n_lines, base=9.2),
                 fontweight="bold", color=C_TEXT_DARK,
                 fontfamily="Arial", linespacing=0.95)
    else:
        ax2.text(x, y, short_name(t), ha="center", va="center",
                 fontsize=_label_fontsize(sizes_pruned[t], n_lines, base=7.2),
                 fontweight="regular", color=C_TEXT_FADED,
                 fontfamily="Arial", linespacing=0.95)

ax2.set_xlabel(f"(b) Pruned Schema ({len(relevant_tables)} relevant, "
               f"{len(filtered_tables)} filtered)",
               fontsize=9, labelpad=8)
ax2.margins(0.18)
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# ── Legend (inside figure bounds) ────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=C_NODE_FULL, edgecolor="#7A6FA0",
                   linewidth=1.0, label="Schema Table"),
    mpatches.Patch(facecolor=C_RELEVANT, edgecolor=C_BORDER_REL,
                   linewidth=1.5, label=r"Relevant ($\geq \tau$)"),
    mpatches.Patch(facecolor=C_FILTERED, edgecolor=C_BORDER_FILT,
                   linewidth=0.8, label=r"Filtered ($< \tau$)"),
    plt.Line2D([0], [0], color=C_EDGE_ACTIVE, linewidth=1.5,
               label="FK Relationship"),
    plt.Line2D([0], [0], color=C_EDGE_FADED, linewidth=0.8, linestyle="dotted",
               label="Pruned FK"),
]
fig.legend(handles=legend_handles, loc="upper center",
           ncol=5, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, 1.0))

# ── Example query annotation ────────────────────────────────────────────────
query_text = ('Query: "List the reference names of the drivers '
              'who are eliminated in the first period of qualifying"')
fig.text(0.5, -0.01, query_text, ha="center", va="top",
         fontsize=7.5, style="italic", color="#444444")

plt.tight_layout(pad=0.4, w_pad=2.0, rect=[0, 0.03, 1, 0.92])

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("svg", "tiff", "pdf"):
    p = FIGURES_DIR / f"Fig_pruning.{ext}"
    kwargs = {"format": ext, "bbox_inches": "tight", "pad_inches": 0.05}
    if ext == "tiff":
        kwargs["dpi"] = DPI
    fig.savefig(p, **kwargs)

plt.close(fig)
print(f"Done: Fig_pruning.svg / .tiff / .pdf")
print(f"  Database: Formula 1 ({len(all_tables)} tables -> {len(relevant_tables)} relevant)")
print(f"  FK edges: {len(fk_edges)} total, {len(active_edges)} active after pruning")
print(f"  Accessibility: black text on light backgrounds, no transparency")
