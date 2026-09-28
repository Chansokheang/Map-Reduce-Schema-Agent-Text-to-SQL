"""
Schema Extractor - Extract database schema and save to file.

Supports SQLite databases. Extracts tables, columns, data types,
primary keys, foreign keys, and optionally sample data.

Usage:
    python schema_extractor.py <database_path> [--output <output_path>] [--format json|txt|sql] [--samples <n>]
"""

import sqlite3
import json
import argparse
from pathlib import Path
from datetime import datetime


def extract_schema(db_path: str, sample_limit: int = 0) -> dict:
    """
    Extract complete schema from a SQLite database.

    Args:
        db_path: Path to the SQLite database file
        sample_limit: Number of sample rows to include (0 = no samples)

    Returns:
        Dictionary containing database schema information
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables (excluding sqlite internal tables)
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
    """)
    tables = [row[0] for row in cursor.fetchall()]

    schema = {
        "database": Path(db_path).stem,
        "extracted_at": datetime.now().isoformat(),
        "tables": {}
    }

    for table in tables:
        # Get column info: (cid, name, type, notnull, default_value, pk)
        cursor.execute(f'PRAGMA table_info("{table}");')
        table_info = cursor.fetchall()

        columns = []
        primary_keys = []

        for col in table_info:
            col_name = col[1]
            col_type = col[2] if col[2] else "TEXT"
            not_null = bool(col[3])
            default_val = col[4]
            is_pk = col[5] > 0

            columns.append({
                "name": col_name,
                "type": col_type,
                "not_null": not_null,
                "default": default_val,
                "is_primary_key": is_pk
            })

            if is_pk:
                primary_keys.append(col_name)

        # Get foreign keys: (id, seq, table, from, to, on_update, on_delete, match)
        cursor.execute(f'PRAGMA foreign_key_list("{table}");')
        fk_info = cursor.fetchall()

        foreign_keys = []
        for fk in fk_info:
            foreign_keys.append({
                "column": fk[3],
                "references_table": fk[2],
                "references_column": fk[4],
                "on_update": fk[5],
                "on_delete": fk[6]
            })

        # Get indexes
        cursor.execute(f'PRAGMA index_list("{table}");')
        index_info = cursor.fetchall()

        indexes = []
        for idx in index_info:
            idx_name = idx[1]
            is_unique = bool(idx[2])
            cursor.execute(f'PRAGMA index_info("{idx_name}");')
            idx_columns = [col[2] for col in cursor.fetchall()]
            indexes.append({
                "name": idx_name,
                "unique": is_unique,
                "columns": idx_columns
            })

        # Get row count
        cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
        row_count = cursor.fetchone()[0]

        table_data = {
            "columns": columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "indexes": indexes,
            "row_count": row_count
        }

        # Get sample rows if requested
        if sample_limit > 0:
            try:
                cursor.execute(f'SELECT * FROM "{table}" LIMIT {sample_limit};')
                col_names = [c["name"] for c in columns]
                rows = cursor.fetchall()
                table_data["sample_rows"] = [
                    dict(zip(col_names, row)) for row in rows
                ]
            except Exception as e:
                table_data["sample_rows"] = [{"error": str(e)}]

        schema["tables"][table] = table_data

    conn.close()
    return schema


def format_as_json(schema: dict) -> str:
    """Format schema as JSON string."""
    return json.dumps(schema, indent=2, default=str)


def format_as_text(schema: dict) -> str:
    """Format schema as readable text."""
    lines = [
        f"Database: {schema['database']}",
        f"Extracted: {schema['extracted_at']}",
        "=" * 60,
        ""
    ]

    for table_name, table_info in schema["tables"].items():
        lines.append(f"TABLE: {table_name}")
        lines.append(f"  Row Count: {table_info['row_count']}")
        lines.append("  Columns:")

        for col in table_info["columns"]:
            pk_marker = " [PK]" if col["is_primary_key"] else ""
            null_marker = " NOT NULL" if col["not_null"] else ""
            default_marker = f" DEFAULT {col['default']}" if col["default"] else ""
            lines.append(f"    - {col['name']}: {col['type']}{pk_marker}{null_marker}{default_marker}")

        if table_info["foreign_keys"]:
            lines.append("  Foreign Keys:")
            for fk in table_info["foreign_keys"]:
                lines.append(f"    - {fk['column']} -> {fk['references_table']}.{fk['references_column']}")

        if table_info["indexes"]:
            lines.append("  Indexes:")
            for idx in table_info["indexes"]:
                unique_marker = " (UNIQUE)" if idx["unique"] else ""
                lines.append(f"    - {idx['name']}: ({', '.join(idx['columns'])}){unique_marker}")

        if "sample_rows" in table_info and table_info["sample_rows"]:
            lines.append(f"  Sample Rows ({len(table_info['sample_rows'])}):")
            for row in table_info["sample_rows"]:
                lines.append(f"    {row}")

        lines.append("")

    return "\n".join(lines)


def format_as_sql(schema: dict) -> str:
    """Format schema as SQL DDL statements."""
    lines = [
        f"-- Database: {schema['database']}",
        f"-- Extracted: {schema['extracted_at']}",
        ""
    ]

    for table_name, table_info in schema["tables"].items():
        col_defs = []
        for col in table_info["columns"]:
            col_def = f'    "{col["name"]}" {col["type"]}'
            if col["not_null"]:
                col_def += " NOT NULL"
            if col["default"] is not None:
                col_def += f" DEFAULT {col['default']}"
            col_defs.append(col_def)

        # Add primary key constraint
        if table_info["primary_keys"]:
            pk_cols = ", ".join(f'"{pk}"' for pk in table_info["primary_keys"])
            col_defs.append(f"    PRIMARY KEY ({pk_cols})")

        # Add foreign key constraints
        for fk in table_info["foreign_keys"]:
            fk_def = f'    FOREIGN KEY ("{fk["column"]}") REFERENCES "{fk["references_table"]}"("{fk["references_column"]}")'
            if fk.get("on_update"):
                fk_def += f" ON UPDATE {fk['on_update']}"
            if fk.get("on_delete"):
                fk_def += f" ON DELETE {fk['on_delete']}"
            col_defs.append(fk_def)

        lines.append(f'CREATE TABLE "{table_name}" (')
        lines.append(",\n".join(col_defs))
        lines.append(");")
        lines.append("")

    return "\n".join(lines)


def save_schema(schema: dict, output_path: str, output_format: str = "json"):
    """
    Save schema to file in specified format.

    Args:
        schema: Schema dictionary from extract_schema()
        output_path: Path to output file
        output_format: Output format - 'json', 'txt', or 'sql'
    """
    formatters = {
        "json": format_as_json,
        "txt": format_as_text,
        "sql": format_as_sql
    }

    formatter = formatters.get(output_format, format_as_json)
    content = formatter(schema)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def extract_all_schemas(db_directory: str, output_directory: str,
                        output_format: str = "json", sample_limit: int = 0):
    """
    Extract schemas from all SQLite databases in a directory.

    Args:
        db_directory: Directory containing .sqlite files
        output_directory: Directory to save schema files
        output_format: Output format - 'json', 'txt', or 'sql'
        sample_limit: Number of sample rows to include

    Returns:
        List of output file paths
    """
    db_dir = Path(db_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_files = list(db_dir.glob("**/*.sqlite")) + list(db_dir.glob("**/*.db"))
    output_files = []

    for db_path in db_files:
        print(f"Extracting schema from: {db_path.name}")
        schema = extract_schema(str(db_path), sample_limit)

        ext = {"json": ".json", "txt": ".txt", "sql": ".sql"}[output_format]
        output_path = output_dir / f"{db_path.stem}_schema{ext}"

        save_schema(schema, str(output_path), output_format)
        output_files.append(str(output_path))

        # Print summary
        table_count = len(schema["tables"])
        total_cols = sum(len(t["columns"]) for t in schema["tables"].values())
        print(f"  -> {table_count} tables, {total_cols} columns")

    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Extract database schema and save to file"
    )
    parser.add_argument(
        "database",
        help="Path to SQLite database file or directory containing databases"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: <database>_schema.<format>)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "txt", "sql"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "-s", "--samples",
        type=int,
        default=0,
        help="Number of sample rows to include (default: 0)"
    )

    args = parser.parse_args()
    db_path = Path(args.database)

    if db_path.is_dir():
        # Extract from all databases in directory
        output_dir = args.output or str(db_path / "schemas")
        files = extract_all_schemas(
            str(db_path), output_dir, args.format, args.samples
        )
        print(f"\nSaved {len(files)} schema files to: {output_dir}")
    else:
        # Extract from single database
        schema = extract_schema(str(db_path), args.samples)

        ext = {"json": ".json", "txt": ".txt", "sql": ".sql"}[args.format]
        output_path = args.output or f"{db_path.stem}_schema{ext}"

        save_schema(schema, output_path, args.format)

        # Print summary
        print(f"Database: {schema['database']}")
        print(f"Tables: {len(schema['tables'])}")
        for table_name, table_info in schema["tables"].items():
            cols = len(table_info["columns"])
            rows = table_info["row_count"]
            print(f"  - {table_name}: {cols} columns, {rows} rows")
        print(f"\nSchema saved to: {output_path}")


if __name__ == "__main__":
    main()
