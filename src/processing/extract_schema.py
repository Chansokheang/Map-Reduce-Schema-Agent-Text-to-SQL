import sqlite3
import json
from pathlib import Path


def load_dev_tables(dev_tables_path):
    """Load dev_tables.json and create lookup dictionaries for readable names."""
    with open(dev_tables_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lookup = {}
    for db in data:
        db_id = db["db_id"]

        # Build table name mapping: original -> readable
        table_names_orig = db.get("table_names_original", [])
        table_names_readable = db.get("table_names", [])
        table_map = {}
        for orig, readable in zip(table_names_orig, table_names_readable):
            table_map[orig] = readable

        # Build column name mapping: (table_idx, col_orig) -> col_readable
        col_names_orig = db.get("column_names_original", [])
        col_names_readable = db.get("column_names", [])
        col_map = {}
        for orig, readable in zip(col_names_orig, col_names_readable):
            table_idx = orig[0]
            col_orig = orig[1]
            col_readable = readable[1]
            if table_idx >= 0:  # Skip -1 which is "*"
                table_name = table_names_orig[table_idx] if table_idx < len(table_names_orig) else None
                if table_name:
                    col_map[(table_name, col_orig)] = col_readable

        lookup[db_id] = {
            "table_map": table_map,
            "col_map": col_map
        }

    return lookup


def get_schema_with_samples(db_path, dev_tables_lookup, sample_limit=3):
    """Extract all tables, columns, data types, keys, and top N sample rows from a database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    db_name = db_path.stem
    db_lookup = dev_tables_lookup.get(db_name, {"table_map": {}, "col_map": {}})

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {}
    for table in tables:
        # Get readable table name
        table_readable = db_lookup["table_map"].get(table, table)

        # Get columns with data types and primary key info
        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        cursor.execute(f'PRAGMA table_info("{table}");')
        table_info = cursor.fetchall()

        columns = []
        primary_keys = []
        for col in table_info:
            col_name = col[1]
            col_type = col[2] if col[2] else "unknown"
            is_pk = col[5] > 0  # pk column is > 0 if it's a primary key

            # Get readable column name
            col_readable = db_lookup["col_map"].get((table, col_name), col_name)

            columns.append({
                "name": col_name,
                "readable_name": col_readable,
                "type": col_type
            })

            if is_pk:
                primary_keys.append(col_name)

        # Get foreign keys
        # PRAGMA foreign_key_list returns: (id, seq, table, from, to, on_update, on_delete, match)
        cursor.execute(f'PRAGMA foreign_key_list("{table}");')
        fk_info = cursor.fetchall()

        foreign_keys = []
        for fk in fk_info:
            foreign_keys.append({
                "column": fk[3],
                "references_table": fk[2],
                "references_column": fk[4]
            })

        # Get top N sample rows
        try:
            cursor.execute(f'SELECT * FROM "{table}" LIMIT {sample_limit};')
            rows = [list(row) for row in cursor.fetchall()]
        except Exception as e:
            rows = [{"error": str(e)}]

        schema[table] = {
            "table_readable_name": table_readable,
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "sample_rows": rows
        }

    conn.close()
    return schema


def main():
    base_dir = Path(__file__).parent.parent.parent
    db_dir = base_dir / "data" / "bird_data" / "dev_databases"
    dev_tables_path = base_dir / "data" / "bird_data" / "dev_tables.json"
    output_dir = base_dir / "data" / "bird_data" / "schemas"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load readable names from dev_tables.json
    print("Loading dev_tables.json for readable names...")
    dev_tables_lookup = load_dev_tables(dev_tables_path)
    print(f"Loaded readable names for {len(dev_tables_lookup)} databases\n")

    db_files = list(db_dir.glob("**/*.sqlite"))
    print(f"Found {len(db_files)} databases\n")

    for db_path in db_files:
        db_name = db_path.stem
        schema = get_schema_with_samples(db_path, dev_tables_lookup, sample_limit=5)

        # Save to JSON
        output_path = output_dir / f"{db_name}_schema.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"database": db_name, "tables": schema}, f, indent=2, default=str)

        # Print summary
        print(f"=== {db_name} ===")
        for table, info in schema.items():
            col_names = [c["name"] for c in info["columns"]]
            pk_str = f" [PK: {', '.join(info['primary_keys'])}]" if info["primary_keys"] else ""
            fk_count = len(info["foreign_keys"])
            fk_str = f" [FK: {fk_count}]" if fk_count > 0 else ""
            print(f"  {table}{pk_str}{fk_str}: {col_names}")
        print()

    print(f"Schemas saved to: {output_dir}")


if __name__ == "__main__":
    main()
