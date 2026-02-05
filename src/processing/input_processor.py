"""
Input Processor Module

Handles the "Inputs" stage of the workflow:
- NL Query: Natural language query from user
- Database Schema: Table structures with columns, keys, relationships
- Database Profiling: Entity meaning, relationships, descriptions, sample values
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatabaseProfile:
    """Profile containing semantic information about the database."""
    entity_meanings: dict[str, str]  # table/column -> meaning
    relationships: list[dict[str, Any]]  # foreign key relationships
    short_descriptions: dict[str, str]  # table/column -> description
    sample_values: dict[str, list[Any]]  # column -> sample values


@dataclass
class ProcessedInput:
    """Container for all processed inputs."""
    nl_query: str
    database_name: str
    schema: dict[str, Any]
    profile: dict[str, Any]  # Raw profile dict for PromptBuilder compatibility


class InputProcessor:
    """Processes raw inputs into structured format for the pipeline."""

    def __init__(self, schema_dir: Path, profile_dir: Path):
        """
        Initialize the input processor.

        Args:
            schema_dir: Directory containing schema JSON files
            profile_dir: Directory containing profile/description JSON files
        """
        self.schema_dir = Path(schema_dir)
        self.profile_dir = Path(profile_dir)

    def load_schema(self, database_name: str) -> dict[str, Any]:
        """
        Load database schema from JSON file.

        Schema files are at: {schema_dir}/{database_name}_schema.json
        Format: {"database": "...", "tables": {table_name: {columns, primary_keys, foreign_keys, ...}}}

        Args:
            database_name: Name of the database (e.g., "california_schools")

        Returns:
            Schema dictionary with table_name -> table_info mapping
        """
        schema_path = self.schema_dir / f"{database_name}_schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get("tables", {})

    def load_profile(self, database_name: str) -> dict[str, Any]:
        """
        Load database profile (descriptions) from JSON file.

        Profile files are at: {profile_dir}/{database_name}_descriptions.json
        Format: {"database": "...", "tables": {table_name: {columns: [{name, description, ...}]}}}

        Args:
            database_name: Name of the database

        Returns:
            Profile dictionary compatible with PromptBuilder
        """
        profile_path = self.profile_dir / f"{database_name}_descriptions.json"
        if not profile_path.exists():
            return {"tables": {}}

        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def extract_entity_meanings(self, schema: dict, profile_data: dict) -> dict[str, str]:
        """
        Extract entity meanings from schema and profile data.

        Args:
            schema: Database schema (tables dict)
            profile_data: Raw profile data with descriptions

        Returns:
            Mapping of table/column names to their semantic meanings
        """
        meanings = {}
        profile_tables = profile_data.get("tables", {})

        for table_name, table_info in schema.items():
            # Table-level meaning from readable name
            readable = table_info.get("table_readable_name", table_name)
            if readable != table_name:
                meanings[table_name] = readable

            # Column-level meanings from profile descriptions
            profile_table = profile_tables.get(table_name, {})
            profile_cols = {}
            if profile_table.get("columns"):
                profile_cols = {c["name"]: c for c in profile_table["columns"] if isinstance(c, dict)}

            for col in table_info.get("columns", []):
                if isinstance(col, dict):
                    col_name = col.get("name", "")
                    key = f"{table_name}.{col_name}"

                    # Use profile description if available
                    profile_col = profile_cols.get(col_name, {})
                    if profile_col.get("description"):
                        meanings[key] = profile_col["description"]
                    elif col.get("readable_name") and col["readable_name"] != col_name:
                        meanings[key] = col["readable_name"]

        return meanings

    def extract_relationships(self, schema: dict) -> list[dict[str, Any]]:
        """
        Extract relationships between tables from foreign keys.

        Args:
            schema: Database schema (tables dict)

        Returns:
            List of relationship dictionaries
        """
        relationships = []

        for table_name, table_info in schema.items():
            for fk in table_info.get("foreign_keys", []):
                relationships.append({
                    "from_table": table_name,
                    "from_column": fk["column"],
                    "to_table": fk["references_table"],
                    "to_column": fk["references_column"]
                })

        return relationships

    def process(self, nl_query: str, database_name: str) -> ProcessedInput:
        """
        Process all inputs for a given query and database.

        Args:
            nl_query: Natural language query from user
            database_name: Target database name

        Returns:
            ProcessedInput containing all structured inputs
        """
        schema = self.load_schema(database_name)
        profile = self.load_profile(database_name)

        return ProcessedInput(
            nl_query=nl_query,
            database_name=database_name,
            schema=schema,
            profile=profile
        )
