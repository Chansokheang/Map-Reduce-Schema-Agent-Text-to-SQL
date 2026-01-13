import json
import os
from pathlib import Path
from anthropic import Anthropic

# Initialize Anthropic client
client = Anthropic()


def generate_column_description(
    db_name: str,
    table_name: str,
    column: dict,
    sample_values: list,
    is_pk: bool,
    fk_info: dict | None
) -> str:
    """Generate a description for a single column using Claude."""

    # Extract sample values for this column
    samples_str = ", ".join([repr(v) for v in sample_values[:5] if v is not None])
    if not samples_str:
        samples_str = "NULL values only"

    # Build context
    pk_str = " (PRIMARY KEY)" if is_pk else ""
    fk_str = ""
    if fk_info:
        fk_str = f" (FOREIGN KEY -> {fk_info['references_table']}.{fk_info['references_column']})"

    # Get readable name if different from original
    readable_name = column.get('readable_name', column['name'])
    readable_hint = ""
    if readable_name != column['name']:
        readable_hint = f"\n- Readable Name: {readable_name}"

    prompt = f"""Write a SHORT description for this database column.

Database: {db_name}
Table: {table_name}
Column: {column['name']}{pk_str}{fk_str}{readable_hint}
Sample Values: {samples_str}

Rules:
- ONE sentence only, 10-20 words max
- Explain WHAT the column means, not HOW it's stored
- DO NOT mention data types, formats, or storage details
- DO NOT describe sample values or give examples
- DO NOT start with "This column stores..." or "This column contains..."
- If column name is unclear (e.g., "A2", "k_symbol"), explain what it actually represents
- If values are in foreign language (e.g., Czech), briefly note the meaning

Good examples:
- "Unique identifier for each school combining county, district, and school codes."
- "Number of students eligible for free meals."
- "District name where the bank branch is located."
- "Transaction type code: POJISTNE=insurance, SIPO=household, UVER=loan."

Bad examples (too long/technical):
- "This column stores the academic year formatted as YYYY-YYYY text string..."
- "Integer values representing the count of students, typically ranging from..."

Return ONLY the description."""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip()


def process_schema(schema_path: Path, output_dir: Path):
    """Process a schema file and generate descriptions for all columns."""

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    db_name = schema["database"]
    print(f"\n{'='*50}")
    print(f"Processing: {db_name}")
    print(f"{'='*50}")

    result = {
        "database": db_name,
        "tables": {}
    }

    for table_name, table_info in schema["tables"].items():
        # Skip sqlite internal tables
        if table_name == "sqlite_sequence":
            continue

        table_readable_name = table_info.get("table_readable_name", table_name)
        print(f"\n  Table: {table_name} ({table_readable_name})")

        columns = table_info["columns"]
        primary_keys = table_info.get("primary_keys", [])
        foreign_keys = {fk["column"]: fk for fk in table_info.get("foreign_keys", [])}
        sample_rows = table_info.get("sample_rows", [])

        table_result = {
            "table_readable_name": table_readable_name,
            "columns": [],
            "primary_keys": primary_keys,
            "foreign_keys": table_info.get("foreign_keys", [])
        }

        for col_idx, column in enumerate(columns):
            col_name = column["name"]

            # Extract sample values for this column
            sample_values = []
            for row in sample_rows:
                if isinstance(row, list) and col_idx < len(row):
                    sample_values.append(row[col_idx])

            # Check if PK or FK
            is_pk = col_name in primary_keys
            fk_info = foreign_keys.get(col_name)

            # Generate description
            try:
                description = generate_column_description(
                    db_name, table_name, column, sample_values, is_pk, fk_info
                )
            except Exception as e:
                description = f"Error generating description: {e}"
                print(f"    Error for {col_name}: {e}")

            table_result["columns"].append({
                "name": col_name,
                "readable_name": column.get("readable_name", col_name),
                "type": column["type"],
                "description": description
            })

            print(f"    {col_name}: {description[:60]}...")

        result["tables"][table_name] = table_result

    # Save result
    output_path = output_dir / f"{db_name}_descriptions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved to: {output_path}")
    return result


def main():
    base_dir = Path(__file__).parent.parent.parent
    schema_dir = base_dir / "data" / "bird_data" / "schemas"
    output_dir = base_dir / "data" / "bird_data" / "descriptions"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    schema_files = list(schema_dir.glob("*_schema.json"))
    print(f"Found {len(schema_files)} schema files")

    for schema_path in schema_files:
        process_schema(schema_path, output_dir)

    print(f"\n{'='*50}")
    print(f"All descriptions saved to: {output_dir}")


if __name__ == "__main__":
    main()
