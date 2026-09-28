"""One searchable document per column, built from the documentation shipped with the database.

Three sources, merged per (table, column):
  * data/bird_data/dev_databases/<db>/database_description/<table>.csv — BIRD's own column
    description, data format and value description.
  * data/column_meaning.json — the prose meaning already used by the pipeline, keyed
    "<db>|<table>|<column>".
  * the column name itself.

No model calls, no gold. Everything here ships with the benchmark, so the same code works on
the test set.

  python -m experiments.column_meaning.corpus --db california_schools --table frpm
"""
import argparse
import csv
import functools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_DATABASES = ROOT / "data/bird_data/dev_databases"
DEFAULT_MEANINGS = ROOT / "data/column_meaning.json"
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


def databases_dir():
    return Path(os.environ.get("QASQL_DATABASES_DIR", DEFAULT_DATABASES))


def meanings_path():
    return Path(os.environ.get("QASQL_COLUMN_MEANING_JSON", DEFAULT_MEANINGS))


def _read_text(path):
    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def _clean(value):
    return " ".join(str(value or "").split()).strip(" #")


@functools.lru_cache(maxsize=1)
def _meanings():
    """{(db, table, column) casefolded: prose meaning}"""
    path = meanings_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for key, text in raw.items():
        parts = str(key).split("|")
        if len(parts) == 3:
            db, table, column = (p.strip() for p in parts)
            out[(db.casefold(), table.casefold(), column.casefold())] = _clean(text)
    return out


def _description_rows(db_id, table):
    """Rows of <table>.csv, as dicts with lowercase keys. [] when the file is missing."""
    path = databases_dir() / db_id / "database_description" / f"{table}.csv"
    if not path.exists():
        matches = [p for p in (databases_dir() / db_id / "database_description").glob("*.csv")
                   if p.stem.casefold() == table.casefold()] \
            if (databases_dir() / db_id / "database_description").exists() else []
        if not matches:
            return []
        path = matches[0]
    text = _read_text(path)
    if not text:
        return []
    rows = []
    for row in csv.DictReader(text.splitlines()):
        rows.append({(k or "").strip().casefold(): _clean(v) for k, v in row.items()})
    return rows


@functools.lru_cache(maxsize=32)
def columns(db_id, tables=None):
    """[{table, column, type, description, value_description, meaning, text}] for the database.

    `text` is the concatenation the retriever searches. `tables` limits the result and must be a
    tuple (the result is cached).
    """
    allowed = {t.casefold() for t in tables} if tables else None
    description_dir = databases_dir() / db_id / "database_description"
    table_files = sorted(description_dir.glob("*.csv")) if description_dir.exists() else []
    meanings = _meanings()
    out = []
    for path in table_files:
        table = path.stem
        if allowed and table.casefold() not in allowed:
            continue
        for row in _description_rows(db_id, table):
            name = row.get("original_column_name") or row.get("column_name") or ""
            if not name:
                continue
            readable = row.get("column_name") or ""
            description = row.get("column_description") or ""
            value_description = row.get("value_description") or ""
            meaning = meanings.get((db_id.casefold(), table.casefold(), name.casefold()), "")
            parts = [name, readable, description, value_description, meaning]
            out.append({
                "table": table,
                "column": name,
                "type": row.get("data_format") or "",
                "description": description,
                "value_description": value_description,
                "meaning": meaning,
                "text": " ".join(p for p in parts if p),
            })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True)
    parser.add_argument("--table", default=None)
    args = parser.parse_args()
    rows = columns(args.db, (args.table,) if args.table else None)
    print(f"{len(rows)} columns")
    for row in rows[:40]:
        print(f"  {row['table']}.{row['column']} ({row['type']}): {row['text'][:140]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
