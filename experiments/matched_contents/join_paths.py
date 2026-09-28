"""Join columns for a database, read from its declared foreign keys.

Companion to retriever.py: the retriever says which table and column a value lives in, this
module says how those tables can be joined. Both are lookups — the edges come from the
database itself (`PRAGMA foreign_key_list`, plus the BIRD tables file when it is present),
never from hand-written name rules.

  python -m experiments.matched_contents.join_paths --db california_schools --tables frpm,schools
"""
import argparse
import functools
import json
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_DATABASES = ROOT / "data/bird_data/dev_databases"
DEFAULT_TABLES_JSON = ROOT / "data/bird_data/dev_tables.json"
DEFAULT_JOIN_LIMIT = 12


def databases_dir():
    return Path(os.environ.get("QASQL_DATABASES_DIR", DEFAULT_DATABASES))


def tables_json():
    return Path(os.environ.get("QASQL_TABLES_JSON", DEFAULT_TABLES_JSON))


def database_path(db_id):
    path = databases_dir() / db_id / f"{db_id}.sqlite"
    return path if path.exists() else None


def _edges_from_sqlite(db_id):
    path = database_path(db_id)
    if path is None:
        return []
    edges = []
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            tables = [t for (t,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            for table in tables:
                for row in conn.execute(f'PRAGMA foreign_key_list("{table}")'):
                    target, from_column, to_column = row[2], row[3], row[4]
                    if to_column is None:            # FK to the target's primary key
                        pk = [c[1] for c in conn.execute(f'PRAGMA table_info("{target}")') if c[5]]
                        to_column = pk[0] if pk else None
                    if from_column and to_column:
                        edges.append((table, from_column, target, to_column))
    except sqlite3.Error:
        return []
    return edges


def _edges_from_tables_json(db_id):
    path = tables_json()
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entry = next((e for e in entries if e.get("db_id") == db_id), None)
    if not entry:
        return []
    names = entry.get("column_names_original") or []
    tables = entry.get("table_names_original") or []
    edges = []
    for pair in entry.get("foreign_keys") or []:
        try:
            (ti, col), (tj, other) = names[pair[0]], names[pair[1]]
            edges.append((tables[ti], col, tables[tj], other))
        except (IndexError, TypeError, ValueError):
            continue
    return edges


@functools.lru_cache(maxsize=64)
def foreign_keys(db_id):
    """[(table, column, other_table, other_column)] for the database, deduplicated."""
    seen, edges = set(), []
    for table, column, other_table, other_column in _edges_from_sqlite(db_id) + _edges_from_tables_json(db_id):
        if table.casefold() == other_table.casefold():
            continue
        key = frozenset({(table.casefold(), column.casefold()), (other_table.casefold(), other_column.casefold())})
        if key in seen:
            continue
        seen.add(key)
        edges.append((table, column, other_table, other_column))
    return edges


def join_columns(db_id, tables=None, focus=None, limit=DEFAULT_JOIN_LIMIT):
    """Declared joins among `tables`, the ones touching `focus` first.

    tables: names the caller can still use (None = every table in the database).
    focus:  names the question already points at — tables holding matched values, or the one
            table a schema worker is scoring. Edges between two focus tables come first, then
            edges from a focus table to a reachable one, then the rest.
    """
    allowed = {t.casefold() for t in tables} if tables else None
    wanted = {t.casefold() for t in focus} if focus else set()
    ranked = []
    for table, column, other_table, other_column in foreign_keys(db_id):
        left, right = table.casefold(), other_table.casefold()
        if allowed and (left not in allowed or right not in allowed):
            continue
        touching = (left in wanted) + (right in wanted)
        rank = {2: 0, 1: 1}.get(touching, 2)
        if wanted and rank == 2:
            continue
        ranked.append({"rank": rank, "table": table, "column": column,
                       "other_table": other_table, "other_column": other_column})
    ranked.sort(key=lambda e: (e["rank"], e["table"].casefold(), e["column"].casefold()))
    for edge in ranked:
        edge.pop("rank")
    return ranked[:limit]


def format_block(edges):
    """The '# Join columns' block for a prompt. Empty string when the database declares none."""
    if not edges:
        return ""
    lines = ["# Join columns",
             "Foreign keys declared in this database. Use them as the join keys when the question "
             "needs more than one table; they are lookups, not instructions."]
    for edge in edges:
        lines.append(f"- {edge['table']}.{edge['column']} = {edge['other_table']}.{edge['other_column']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True)
    parser.add_argument("--tables", default=None, help="Comma-separated tables still in play")
    parser.add_argument("--focus", default=None, help="Comma-separated tables the question points at")
    parser.add_argument("--limit", type=int, default=DEFAULT_JOIN_LIMIT)
    args = parser.parse_args()
    edges = join_columns(args.db,
                         tables=args.tables.split(",") if args.tables else None,
                         focus=args.focus.split(",") if args.focus else None,
                         limit=args.limit)
    print(format_block(edges) or "(no declared foreign keys)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
