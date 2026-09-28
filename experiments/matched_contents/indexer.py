"""Build a per-database index of stored text values ("matched contents" source).

Read-only over the BIRD databases. For every text column that is not free text, the distinct
values are copied into output/matched_contents/index/<db>.sqlite with a normalized form, so a
question phrase can be looked up by exact or prefix match without scanning the databases again.

  python -m experiments.matched_contents.indexer --databases-dir data/bird_data/dev_databases
"""
import argparse
import re
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "output/matched_contents/index"
MAX_VALUE_CHARS = 80          # longer cells are free text, not lookup values
MAX_DISTINCT_PER_COLUMN = 200000
PUNCT = re.compile(r"[^0-9a-z]+")


def normalize(value):
    return PUNCT.sub(" ", str(value).casefold()).strip()


def text_columns(conn):
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        for row in conn.execute(f'PRAGMA table_info("{table}")'):
            column, declared = row[1], (row[2] or "").upper()
            if "CHAR" in declared or "TEXT" in declared or "CLOB" in declared or declared == "":
                yield table, column


def build(db_path, out_path, log=print):
    db_path, out_path = Path(db_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    started = time.monotonic()
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as source, \
         closing(sqlite3.connect(out_path)) as index:
        index.execute("CREATE TABLE value (norm TEXT NOT NULL, value TEXT NOT NULL, "
                      "table_name TEXT NOT NULL, column_name TEXT NOT NULL)")
        columns = list(text_columns(source))
        kept = skipped = rows = 0
        for table, column in columns:
            try:
                cursor = source.execute(
                    f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL '
                    f'AND length("{column}") BETWEEN 1 AND {MAX_VALUE_CHARS} LIMIT {MAX_DISTINCT_PER_COLUMN + 1}')
                values = [r[0] for r in cursor.fetchall()]
            except sqlite3.Error:
                skipped += 1
                continue
            if not values or len(values) > MAX_DISTINCT_PER_COLUMN:
                skipped += 1
                continue
            batch = [(normalize(v), str(v), table, column) for v in values if normalize(v)]
            index.executemany("INSERT INTO value VALUES (?,?,?,?)", batch)
            kept += 1
            rows += len(batch)
        index.execute("CREATE INDEX value_norm ON value(norm)")
        index.commit()
    log(f"{db_path.stem}: {kept} columns indexed, {skipped} skipped, {rows} values, "
        f"{time.monotonic() - started:.1f}s")
    return {"database": db_path.stem, "columns_indexed": kept, "columns_skipped": skipped, "values": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--databases-dir", default="data/bird_data/dev_databases")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--only", default=None, help="Index a single database by name")
    args = parser.parse_args()
    out = Path(args.out)
    databases = sorted(Path(args.databases_dir).glob("*/*.sqlite"))
    if args.only:
        databases = [p for p in databases if p.stem == args.only]
    for path in databases:
        build(path, out / f"{path.stem}.sqlite")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
