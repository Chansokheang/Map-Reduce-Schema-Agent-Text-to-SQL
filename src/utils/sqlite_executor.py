"""
SQLite Executor - Execute SQL queries against SQLite databases.

Usage:
    python sqlite_executor.py <database_path> <sql_query_or_file> [options]

Examples:
    python sqlite_executor.py mydb.sqlite "SELECT * FROM users LIMIT 10"
    python sqlite_executor.py mydb.sqlite query.sql -o results.json
    python sqlite_executor.py mydb.sqlite "SELECT * FROM orders" -f csv
"""

import sqlite3
import argparse
import json
from pathlib import Path
from typing import Optional
from datetime import datetime, date
from decimal import Decimal


class SQLiteExecutor:
    """Execute SQL queries against SQLite databases."""

    def __init__(self, db_path: str):
        """
        Initialize SQLite connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """Establish database connection."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        return self.connection

    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute(
        self,
        sql: str,
        params: Optional[tuple] = None,
        fetch: bool = True,
        as_dict: bool = True,
    ) -> dict:
        """
        Execute a SQL query.

        Args:
            sql: SQL query string
            params: Query parameters (for parameterized queries)
            fetch: Whether to fetch results (False for INSERT/UPDATE/DELETE)
            as_dict: Return rows as dictionaries (True) or tuples (False)

        Returns:
            Dictionary with execution results
        """
        conn = self.connect()

        result = {
            "success": False,
            "query": sql,
            "rows": [],
            "columns": [],
            "row_count": 0,
            "execution_time_ms": 0,
            "error": None,
        }

        try:
            start_time = datetime.now()

            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            if fetch and cursor.description:
                result["columns"] = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                if as_dict:
                    result["rows"] = [dict(row) for row in rows]
                else:
                    result["rows"] = [tuple(row) for row in rows]
                result["row_count"] = len(result["rows"])
            else:
                result["row_count"] = cursor.rowcount

            conn.commit()

            end_time = datetime.now()
            result["execution_time_ms"] = (end_time - start_time).total_seconds() * 1000
            result["success"] = True

        except sqlite3.Error as e:
            conn.rollback()
            result["error"] = str(e)

        return result

    def execute_file(self, file_path: str, **kwargs) -> dict:
        """
        Execute SQL from a file.

        Args:
            file_path: Path to SQL file
            **kwargs: Additional arguments passed to execute()

        Returns:
            Dictionary with execution results
        """
        with open(file_path, "r", encoding="utf-8") as f:
            sql = f.read()
        return self.execute(sql, **kwargs)

    def execute_many(
        self, queries: list[str], stop_on_error: bool = True
    ) -> list[dict]:
        """
        Execute multiple SQL queries.

        Args:
            queries: List of SQL query strings
            stop_on_error: Stop execution if a query fails

        Returns:
            List of execution results
        """
        results = []
        for sql in queries:
            result = self.execute(sql)
            results.append(result)
            if not result["success"] and stop_on_error:
                break
        return results

    def get_tables(self) -> list[str]:
        """Get list of tables in database."""
        sql = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """
        result = self.execute(sql)
        return [row["name"] for row in result["rows"]] if result["success"] else []

    def get_schema(self, table: str) -> list[dict]:
        """Get column information for a table."""
        sql = f'PRAGMA table_info("{table}");'
        result = self.execute(sql)
        if result["success"]:
            return [
                {
                    "column_name": row["name"],
                    "data_type": row["type"],
                    "is_nullable": "YES" if not row["notnull"] else "NO",
                    "column_default": row["dflt_value"],
                    "is_primary_key": bool(row["pk"]),
                }
                for row in result["rows"]
            ]
        return []

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def json_serializer(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Type {type(obj)} not serializable")


def format_results(result: dict, output_format: str = "table") -> str:
    """
    Format query results for display.

    Args:
        result: Execution result dictionary
        output_format: Output format - 'table', 'json', 'csv'

    Returns:
        Formatted string
    """
    if not result["success"]:
        return f"Error: {result['error']}"

    if output_format == "json":
        return json.dumps(result, indent=2, default=json_serializer)

    if output_format == "csv":
        if not result["rows"]:
            return ""
        lines = [",".join(result["columns"])]
        for row in result["rows"]:
            if isinstance(row, dict):
                values = [str(row.get(col, "")) for col in result["columns"]]
            else:
                values = [str(v) for v in row]
            # Escape quotes and wrap in quotes if contains comma
            escaped = []
            for v in values:
                if "," in v or '"' in v:
                    v = '"' + v.replace('"', '""') + '"'
                escaped.append(v)
            lines.append(",".join(escaped))
        return "\n".join(lines)

    # Default: table format
    if not result["rows"]:
        return f"Query OK, {result['row_count']} rows affected ({result['execution_time_ms']:.2f} ms)"

    columns = result["columns"]
    rows = result["rows"]

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in rows:
        if isinstance(row, dict):
            for col in columns:
                val_len = len(str(row.get(col, "")))
                widths[col] = max(widths[col], min(val_len, 50))
        else:
            for i, col in enumerate(columns):
                val_len = len(str(row[i]))
                widths[col] = max(widths[col], min(val_len, 50))

    # Build table
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)

    lines = [header, separator]
    for row in rows:
        if isinstance(row, dict):
            values = [str(row.get(col, ""))[:50].ljust(widths[col]) for col in columns]
        else:
            values = [str(v)[:50].ljust(widths[col]) for v in row]
        lines.append(" | ".join(values))

    lines.append(f"\n({result['row_count']} rows, {result['execution_time_ms']:.2f} ms)")
    return "\n".join(lines)


def save_results(result: dict, output_path: str, output_format: str = "json"):
    """Save results to file."""
    content = format_results(result, output_format)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Execute SQL queries against SQLite databases"
    )
    parser.add_argument(
        "database",
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "sql",
        help="SQL query string or path to .sql file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)"
    )

    args = parser.parse_args()

    # Validate database path
    if not Path(args.database).exists():
        print(f"Error: Database not found: {args.database}")
        exit(1)

    executor = SQLiteExecutor(args.database)

    # Check if input is a file
    sql_input = args.sql
    if Path(sql_input).exists() and sql_input.endswith(".sql"):
        with open(sql_input, "r", encoding="utf-8") as f:
            sql_input = f.read()

    try:
        with executor:
            result = executor.execute(sql_input)

        if args.output:
            save_results(result, args.output, args.format)
            print(f"Results saved to: {args.output}")
        else:
            print(format_results(result, args.format))

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
