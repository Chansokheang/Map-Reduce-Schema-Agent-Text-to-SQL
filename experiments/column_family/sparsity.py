"""Which numbered-family columns are near-empty? Measured from the databases themselves.

A family member (X2, X3 ... with an X1 sibling in the same table) counts as sparse when it is
non-empty on fewer than THRESHOLD of rows. That distinguishes the case RULE P is for from the ones
it must not touch:

  california_schools  AdmFName1/2/3     66.2% / 2.4% / 0.2%   <- sparse: a primary/secondary contact
  formula_1           q1/q2/q3          98.4% / 48.2% / 28.6% <- qualifying rounds, all meaningful
  european_football_2 home_player_1..11 ~95% each             <- eleven players, all meaningful
  financial           A12/A13           98.7% / 100%          <- per-year statistics

WORKS ON ANY DATABASE SET, including the BIRD test set: the column list comes from
`PRAGMA table_info` on each .sqlite file, so no pre-extracted schema JSON is needed, and the
directory is a parameter. The cache file is named after the directory, so a dev cache can never be
used by accident for a test run.

  python -m experiments.column_family.sparsity                       # dev (default)
  python -m experiments.column_family.sparsity <path-to-databases>   # e.g. test_databases
  QASQL_DB_DIR=/path/to/test_databases  ...                          # same, via the environment

Each database directory is expected in BIRD's layout: <dir>/<db_id>/<db_id>.sqlite
"""
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_DB_DIR = ROOT / "data/bird_data/dev_databases"
THRESHOLD = 0.10
FAMILY = re.compile(r"(.*?)(\d{1,2})$")


def db_dir(explicit=None):
    return Path(explicit or os.environ.get("QASQL_DB_DIR") or DEFAULT_DB_DIR)


def cache_path(explicit=None):
    """Named after the database directory, so dev and test caches cannot be confused."""
    return HERE / f"sparse_columns__{db_dir(explicit).name}.json"


def columns_of(con, table):
    return [row[1] for row in con.execute(f'PRAGMA table_info("{table}")')]


def tables_of(con):
    return [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]


def families(names):
    """Members X2, X3 ... that have an X1 sibling in the same table."""
    lower = {n.lower() for n in names}
    out = []
    for n in names:
        m = FAMILY.fullmatch(n)
        if m and m.group(2) != "1" and (m.group(1) + "1").lower() in lower:
            out.append(n)
    return sorted(out)


def sparse_for_db(sqlite_path, threshold=THRESHOLD):
    out = {}
    con = sqlite3.connect(sqlite_path)
    try:
        for table in tables_of(con):
            members = families(columns_of(con, table))
            if not members:
                continue
            try:
                total = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            except sqlite3.Error:
                continue
            if not total:
                continue
            for col in members:
                try:
                    n = con.execute(
                        f'SELECT COUNT(*) FROM "{table}" '
                        f'WHERE "{col}" IS NOT NULL AND TRIM(CAST("{col}" AS TEXT)) <> \'\''
                    ).fetchone()[0]
                except sqlite3.Error:
                    continue
                if n / total < threshold:
                    out.setdefault(table, []).append(
                        {"column": col, "filled": round(100 * n / total, 1)})
    finally:
        con.close()
    return out


def sparse_for(db_id, directory=None, threshold=THRESHOLD):
    path = db_dir(directory) / db_id / f"{db_id}.sqlite"
    return sparse_for_db(path, threshold) if path.exists() else {}


def load(directory=None):
    path = cache_path(directory)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def build(directory=None, threshold=THRESHOLD):
    base = db_dir(directory)
    out = {}
    for child in sorted(base.iterdir()) if base.exists() else []:
        sqlite_file = child / f"{child.name}.sqlite"
        if child.is_dir() and sqlite_file.exists():
            got = sparse_for_db(sqlite_file, threshold)
            if got:
                out[child.name] = got
    cache_path(directory).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    base = db_dir(directory)
    if not base.exists():
        sys.exit(f"no such database directory: {base}")
    data = build(directory)
    print(f"scanned {base}")
    print(f"sparse family columns (<{int(THRESHOLD*100)}% filled) -> {cache_path(directory).name}")
    for db, tables in data.items():
        for table, cols in tables.items():
            print(f"  {db}.{table}: " + ", ".join(f"{c['column']} ({c['filled']}%)" for c in cols))
    if not data:
        print("  none found - RULE P will not be added for these databases")
