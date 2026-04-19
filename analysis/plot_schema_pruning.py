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
pos = nx.spring_layout(G, seed=42, k=1.8, iterations=80)

def short_name(name):
    """Readable multi-line labels for long table names."""
    mapping = {
        "constructorResults":   "constructor\nResults",
        "constructorStandings": "constructor\nStandings",
        "driverStandings":      "driver\nStandings",
        "lapTimes":             "lapTimes",
        "pitStops":             "pitStops",
    }
    return mapping.get(name, name)


# ── Draw Figure ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL, FULL * 0.42))

# ─────────────────────────────────────────────
# (a) BEFORE — Full Schema (all tables equal)
# ─────────────────────────────────────────────

# Edges
nx.draw_networkx_edges(G, pos, ax=ax1,
                       edge_color=C_EDGE_ACTIVE, width=0.9)

# Nodes — uniform lavender, dark border
nx.draw_networkx_nodes(G, pos, ax=ax1,
                       node_color=C_NODE_FULL,
                       node_size=550,
                       edgecolors="#7A6FA0", linewidths=1.2)

# Labels — BLACK text on light background (accessibility fix)
for t in all_tables:
    x, y = pos[t]
    ax1.text(x, y, short_name(t), ha="center", va="center",
             fontsize=6.5, fontweight="medium", color=C_TEXT_DARK,
             fontfamily="Arial")

ax1.set_xlabel(f"(a) Full Schema ({len(all_tables)} tables, {len(fk_edges)} edges)",
               fontsize=9, labelpad=8)
ax1.margins(0.12)
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

# Draw filtered nodes (smaller, gray)
filtered_tables = [t for t in all_tables if t not in relevant_tables]
filtered_pos = {t: pos[t] for t in filtered_tables}
nx.draw_networkx_nodes(G, pos, nodelist=filtered_tables, ax=ax2,
                       node_color=C_FILTERED,
                       node_size=350,
                       edgecolors=C_BORDER_FILT, linewidths=0.8)

# Draw relevant nodes (larger, green, thick dark border)
relevant_list = [t for t in all_tables if t in relevant_tables]
nx.draw_networkx_nodes(G, pos, nodelist=relevant_list, ax=ax2,
                       node_color=C_RELEVANT,
                       node_size=800,
                       edgecolors=C_BORDER_REL, linewidths=2.0)

# Labels — dark text everywhere (accessibility fix)
for t in all_tables:
    x, y = pos[t]
    if t in relevant_tables:
        ax2.text(x, y, short_name(t), ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color=C_TEXT_DARK,
                 fontfamily="Arial")
    else:
        ax2.text(x, y, short_name(t), ha="center", va="center",
                 fontsize=5.5, fontweight="regular", color=C_TEXT_FADED,
                 fontfamily="Arial")

ax2.set_xlabel(f"(b) Pruned Schema ({len(relevant_tables)} relevant, "
               f"{len(filtered_tables)} filtered)",
               fontsize=9, labelpad=8)
ax2.margins(0.12)
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
