#!/usr/bin/env python3
"""
Update descriptions JSON files from BIRD benchmark CSV files.

This script reads the original column descriptions from:
    data/bird_data/dev_databases/{db_name}/database_description/*.csv

And updates the corresponding JSON files in:
    data/bird_data/descriptions/{db_name}_descriptions.json

CSV format: original_column_name, column_name, column_description, data_format, value_description
"""

import csv
import json
from pathlib import Path


def read_csv_descriptions(csv_path: Path) -> dict:
    """
    Read column descriptions from a CSV file.

    Returns:
        dict mapping column_name -> {readable_name, description}
    """
    columns = {}

    # Try multiple encodings
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    content = None

    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        print(f"    Warning: Could not decode {csv_path.name}")
        return columns

    import io
    reader = csv.DictReader(io.StringIO(content))

    for row in reader:
        original_name = row.get('original_column_name', '').strip()
        readable_name = row.get('column_name', '').strip()
        description = row.get('column_description', '').strip()
        value_desc = row.get('value_description', '').strip()

        if not original_name:
            continue

        # Use original_name as readable_name if column_name is empty
        if not readable_name:
            readable_name = original_name

        # Combine description and value_description if both exist
        full_description = description
        if value_desc and value_desc.upper() not in ['NOT USEFUL', 'NOT USEFUL.']:
            # Add value description as additional context if it's useful
            if full_description and not full_description.endswith('.'):
                full_description += '.'
            # Only add short value descriptions, skip very long ones
            if len(value_desc) < 200 and value_desc != description:
                if full_description:
                    full_description = f"{full_description}"
                else:
                    full_description = value_desc

        # If still no description, use readable_name
        if not full_description:
            full_description = readable_name if readable_name != original_name else original_name

        columns[original_name] = {
            'readable_name': readable_name,
            'description': full_description
        }

    return columns


def update_json_descriptions(json_path: Path, csv_dir: Path) -> bool:
    """
    Update a descriptions JSON file with data from CSV files.

    Args:
        json_path: Path to the descriptions JSON file
        csv_dir: Path to the directory containing CSV files

    Returns:
        True if updated, False otherwise
    """
    if not json_path.exists():
        print(f"  JSON file not found: {json_path}")
        return False

    if not csv_dir.exists():
        print(f"  CSV directory not found: {csv_dir}")
        return False

    # Load existing JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    updated = False
    tables = data.get('tables', {})

    # Process each CSV file (one per table)
    for csv_file in csv_dir.glob('*.csv'):
        table_name = csv_file.stem  # filename without extension

        if table_name not in tables:
            print(f"  Table '{table_name}' not found in JSON, skipping")
            continue

        print(f"  Processing table: {table_name}")

        # Read CSV descriptions
        csv_columns = read_csv_descriptions(csv_file)

        # Update JSON columns
        for col in tables[table_name].get('columns', []):
            col_name = col.get('name')

            if col_name in csv_columns:
                csv_info = csv_columns[col_name]

                # Update readable_name if CSV has a different one
                if csv_info['readable_name'] and csv_info['readable_name'] != col_name:
                    old_readable = col.get('readable_name', '')
                    if old_readable != csv_info['readable_name']:
                        col['readable_name'] = csv_info['readable_name']
                        updated = True

                # Update description from CSV
                if csv_info['description']:
                    old_desc = col.get('description', '')
                    new_desc = csv_info['description']

                    # Only update if CSV description is meaningful
                    if new_desc and new_desc.upper() not in ['NOT USEFUL', 'NOT USEFUL.']:
                        if old_desc != new_desc:
                            col['description'] = new_desc
                            updated = True

    # Save updated JSON
    if updated:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Updated: {json_path.name}")
    else:
        print(f"  No changes needed: {json_path.name}")

    return updated


def main():
    """Main function to update all description files."""
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    dev_databases_dir = project_root / 'data' / 'bird_data' / 'dev_databases'
    descriptions_dir = project_root / 'data' / 'bird_data' / 'descriptions'

    print(f"Dev databases: {dev_databases_dir}")
    print(f"Descriptions: {descriptions_dir}")
    print()

    if not dev_databases_dir.exists():
        print(f"Error: Dev databases directory not found: {dev_databases_dir}")
        return

    if not descriptions_dir.exists():
        print(f"Creating descriptions directory: {descriptions_dir}")
        descriptions_dir.mkdir(parents=True, exist_ok=True)

    # Process each database
    databases = sorted([d for d in dev_databases_dir.iterdir() if d.is_dir()])

    updated_count = 0
    for db_dir in databases:
        db_name = db_dir.name
        csv_dir = db_dir / 'database_description'
        json_path = descriptions_dir / f'{db_name}_descriptions.json'

        print(f"\nProcessing database: {db_name}")

        if not csv_dir.exists():
            print(f"  No database_description folder found, skipping")
            continue

        if not json_path.exists():
            print(f"  No JSON file found: {json_path.name}, skipping")
            continue

        if update_json_descriptions(json_path, csv_dir):
            updated_count += 1

    print(f"\n{'='*50}")
    print(f"Done! Updated {updated_count} description files.")


if __name__ == '__main__':
    main()
