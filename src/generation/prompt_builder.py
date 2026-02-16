"""
Prompt Builder Module

Builds prompts for SQL generation with 5 context-aware strategies:
1. Full Schema - Complete database schema
2. SME Metadata - Subject Matter Expert annotations
3. Minimal Profile - Basic column descriptions
4. Focused Schema - Relevant tables only (from Map-Reduce Agent)
5. Full Profile - Combined Minimal + SME

Prompts are loaded from src/prompt directory.
"""

from typing import Any
from enum import Enum

# Import prompts from src/prompt directory
from src.prompt import STRATEGY_PROMPTS


class ContextStrategy(Enum):
    """5 Context-Aware Generation Strategies."""
    FULL_SCHEMA = "full_schema"
    SME_METADATA = "sme_metadata"
    MINIMAL_PROFILE = "minimal_profile"
    FOCUSED_SCHEMA = "focused_schema"
    FULL_PROFILE = "full_profile"


class PromptBuilder:
    """
    Builds structured prompts for SQL generation.

    Supports 5 context-aware strategies as shown in the proposed method:
    - Full Schema: Complete database schema without descriptions
    - SME Metadata: Schema with Subject Matter Expert annotations
    - Minimal Profile: Schema with basic auto-generated descriptions
    - Focused Schema: Only relevant tables from Map-Reduce Agent
    - Full Profile: Combined minimal descriptions + SME annotations

    Prompts are loaded from src/prompt/ directory.
    """

    def __init__(self):
        """Initialize the Prompt Builder."""
        self.strategy_prompts = self._load_strategy_prompts()

    def _load_strategy_prompts(self) -> dict[ContextStrategy, dict[str, str]]:
        """Load strategy-specific prompts from src/prompt directory."""
        return {
            ContextStrategy.FULL_SCHEMA: STRATEGY_PROMPTS["full_schema"],
            ContextStrategy.SME_METADATA: STRATEGY_PROMPTS["sme_metadata"],
            ContextStrategy.MINIMAL_PROFILE: STRATEGY_PROMPTS["minimal_profile"],
            ContextStrategy.FOCUSED_SCHEMA: STRATEGY_PROMPTS["focused_schema"],
            ContextStrategy.FULL_PROFILE: STRATEGY_PROMPTS["full_profile"],
        }

    def format_table_schema(
        self,
        table_name: str,
        table_info: dict[str, Any],
        include_descriptions: bool = False,
        include_sme: bool = False
    ) -> str:
        """
        Format a single table as CREATE TABLE statement.

        Args:
            table_name: Name of the table
            table_info: Table schema information
            include_descriptions: Include column descriptions
            include_sme: Include SME annotations

        Returns:
            Formatted CREATE TABLE statement
        """
        lines = [f"CREATE TABLE {table_name} ("]

        columns = table_info.get("columns", [])
        column_lines = []

        for col in columns:
            if isinstance(col, dict):
                col_name = col.get("name", "")
                col_type = col.get("type", "TEXT")

                col_def = f"    {col_name} {col_type}"

                # Add description as comment if requested
                desc_parts = []
                if include_descriptions or include_sme:
                    if include_descriptions and col.get("description"):
                        desc_parts.append(col["description"])
                    if include_sme and col.get("sme_description"):
                        desc_parts.append(f"[SME: {col['sme_description']}]")
                    if col.get("readable_name") and col["readable_name"] != col_name:
                        desc_parts.insert(0, f"({col['readable_name']})")

                # Always add distinct values for categorical columns
                if col.get("distinct_values"):
                    values = col["distinct_values"]
                    if len(values) <= 10:
                        values_str = ", ".join(f"'{v}'" for v in values)
                    else:
                        shown = ", ".join(f"'{v}'" for v in values[:8])
                        values_str = f"{shown}, ... (+{len(values) - 8} more)"
                    desc_parts.append(f"[Values: {values_str}]")

                if desc_parts:
                    col_def += f"  -- {' '.join(desc_parts)}"

                column_lines.append(col_def)
            else:
                # Simple string column name
                column_lines.append(f"    {col} TEXT")

        lines.append(",\n".join(column_lines))
        lines.append(");")

        # Add table description if available
        table_desc = table_info.get("table_readable_name", "")
        if table_desc and table_desc != table_name:
            return f"-- Table: {table_desc}\n" + "\n".join(lines)

        return "\n".join(lines)

    def format_full_schema(self, schema: dict[str, Any]) -> str:
        """
        Format complete database schema (Strategy 1: Full Schema).

        No descriptions, just structure.

        Args:
            schema: Complete database schema

        Returns:
            Formatted schema string
        """
        tables = []
        for table_name, table_info in schema.items():
            tables.append(self.format_table_schema(
                table_name,
                table_info,
                include_descriptions=False,
                include_sme=False
            ))

        return "\n\n".join(tables)

    def format_sme_metadata(
        self,
        schema: dict[str, Any],
        sme_profile: dict[str, Any]
    ) -> str:
        """
        Format schema with SME annotations (Strategy 2: SME Metadata).

        Includes Subject Matter Expert knowledge about business logic.

        Args:
            schema: Database schema
            sme_profile: SME annotations and descriptions

        Returns:
            Formatted schema with SME annotations
        """
        tables = []
        sme_tables = sme_profile.get("tables", {}) if sme_profile else {}

        for table_name, table_info in schema.items():
            # Merge SME info into table_info
            merged_info = dict(table_info)
            sme_table = sme_tables.get(table_name, {})

            if sme_table.get("table_readable_name"):
                merged_info["table_readable_name"] = sme_table["table_readable_name"]

            # Merge column SME descriptions
            if sme_table.get("columns"):
                sme_cols = {c["name"]: c for c in sme_table["columns"] if isinstance(c, dict)}
                merged_columns = []
                for col in merged_info.get("columns", []):
                    if isinstance(col, dict):
                        col_copy = dict(col)
                        sme_col = sme_cols.get(col["name"], {})
                        if sme_col.get("description"):
                            col_copy["sme_description"] = sme_col["description"]
                        if sme_col.get("readable_name"):
                            col_copy["readable_name"] = sme_col["readable_name"]
                        if sme_col.get("distinct_values"):
                            col_copy["distinct_values"] = sme_col["distinct_values"]
                        merged_columns.append(col_copy)
                    else:
                        merged_columns.append(col)
                merged_info["columns"] = merged_columns

            tables.append(self.format_table_schema(
                table_name,
                merged_info,
                include_descriptions=False,
                include_sme=True
            ))

        return "\n\n".join(tables)

    def format_minimal_profile(
        self,
        schema: dict[str, Any],
        profile: dict[str, Any]
    ) -> str:
        """
        Format schema with minimal profile (Strategy 3: Minimal Profile).

        Includes basic auto-generated column descriptions.

        Args:
            schema: Database schema
            profile: Basic profile with column descriptions

        Returns:
            Formatted schema with minimal descriptions
        """
        tables = []
        profile_tables = profile.get("tables", {}) if profile else {}

        for table_name, table_info in schema.items():
            merged_info = dict(table_info)
            profile_table = profile_tables.get(table_name, {})

            if profile_table.get("table_readable_name"):
                merged_info["table_readable_name"] = profile_table["table_readable_name"]

            # Merge profile descriptions
            if profile_table.get("columns"):
                profile_cols = {c["name"]: c for c in profile_table["columns"] if isinstance(c, dict)}
                merged_columns = []
                for col in merged_info.get("columns", []):
                    if isinstance(col, dict):
                        col_copy = dict(col)
                        profile_col = profile_cols.get(col["name"], {})
                        if profile_col.get("description"):
                            col_copy["description"] = profile_col["description"]
                        if profile_col.get("readable_name"):
                            col_copy["readable_name"] = profile_col["readable_name"]
                        if profile_col.get("distinct_values"):
                            col_copy["distinct_values"] = profile_col["distinct_values"]
                        merged_columns.append(col_copy)
                    else:
                        merged_columns.append(col)
                merged_info["columns"] = merged_columns

            tables.append(self.format_table_schema(
                table_name,
                merged_info,
                include_descriptions=True,
                include_sme=False
            ))

        return "\n\n".join(tables)

    def format_focused_schema(
        self,
        focused_schema: dict[str, Any]
    ) -> str:
        """
        Format focused schema (Strategy 4: Focused Schema).

        Only includes relevant tables from Map-Reduce Schema Agent.

        Args:
            focused_schema: Filtered schema from Map-Reduce Agent

        Returns:
            Formatted focused schema
        """
        tables_data = focused_schema.get("tables", {})
        if not tables_data:
            return "-- No relevant tables identified"

        tables = []
        for table_name, table_info in tables_data.items():
            # Add relevance info as comment
            relevance = table_info.get("relevance_score", 0)
            reason = table_info.get("reason", "")

            header = f"-- Relevance: {relevance:.1f}"
            if reason:
                header += f" | {reason}"

            table_str = self.format_table_schema(
                table_name,
                table_info,
                include_descriptions=True,
                include_sme=False
            )
            tables.append(f"{header}\n{table_str}")

        return "\n\n".join(tables)

    def format_focused_schema_with_profile(
        self,
        focused_schema: dict[str, Any],
        profile: dict[str, Any]
    ) -> str:
        """
        Format focused schema with profile descriptions (Strategy 5: Full Profile).

        Combines:
        - Focused schema (relevant tables only)
        - Column descriptions from profile

        Args:
            focused_schema: Filtered schema from Map-Reduce Agent
            profile: Profile with column descriptions

        Returns:
            Formatted focused schema with descriptions
        """
        tables_data = focused_schema.get("tables", {})
        if not tables_data:
            return "-- No relevant tables identified"

        profile_tables = profile.get("tables", {}) if profile else {}
        tables = []

        for table_name, table_info in tables_data.items():
            # Add relevance info as comment
            relevance = table_info.get("relevance_score", 0)
            reason = table_info.get("reason", "")

            header = f"-- Relevance: {relevance:.1f}"
            if reason:
                header += f" | {reason}"

            # Merge profile descriptions into focused schema
            merged_info = dict(table_info)
            profile_table = profile_tables.get(table_name, {})

            if profile_table.get("table_readable_name"):
                merged_info["table_readable_name"] = profile_table["table_readable_name"]

            # Merge column descriptions from profile
            if profile_table.get("columns"):
                profile_cols = {c["name"]: c for c in profile_table["columns"] if isinstance(c, dict)}
                merged_columns = []
                for col in merged_info.get("columns", []):
                    if isinstance(col, dict):
                        col_copy = dict(col)
                        profile_col = profile_cols.get(col["name"], {})
                        if profile_col.get("description"):
                            col_copy["description"] = profile_col["description"]
                        if profile_col.get("readable_name"):
                            col_copy["readable_name"] = profile_col["readable_name"]
                        if profile_col.get("distinct_values"):
                            col_copy["distinct_values"] = profile_col["distinct_values"]
                        merged_columns.append(col_copy)
                    else:
                        merged_columns.append(col)
                merged_info["columns"] = merged_columns

            table_str = self.format_table_schema(
                table_name,
                merged_info,
                include_descriptions=True,
                include_sme=False
            )
            tables.append(f"{header}\n{table_str}")

        return "\n\n".join(tables)

    def format_full_profile(
        self,
        schema: dict[str, Any],
        profile: dict[str, Any],
        sme_profile: dict[str, Any]
    ) -> str:
        """
        Format schema with full profile (Strategy 5: Full Profile).

        Combines minimal descriptions + SME annotations.

        Args:
            schema: Database schema
            profile: Minimal profile with basic descriptions
            sme_profile: SME annotations

        Returns:
            Formatted schema with full profile
        """
        tables = []
        profile_tables = profile.get("tables", {}) if profile else {}
        sme_tables = sme_profile.get("tables", {}) if sme_profile else {}

        for table_name, table_info in schema.items():
            merged_info = dict(table_info)
            profile_table = profile_tables.get(table_name, {})
            sme_table = sme_tables.get(table_name, {})

            # Prefer SME table name, fallback to profile
            if sme_table.get("table_readable_name"):
                merged_info["table_readable_name"] = sme_table["table_readable_name"]
            elif profile_table.get("table_readable_name"):
                merged_info["table_readable_name"] = profile_table["table_readable_name"]

            # Merge both profile and SME column info
            profile_cols = {}
            if profile_table.get("columns"):
                profile_cols = {c["name"]: c for c in profile_table["columns"] if isinstance(c, dict)}

            sme_cols = {}
            if sme_table.get("columns"):
                sme_cols = {c["name"]: c for c in sme_table["columns"] if isinstance(c, dict)}

            merged_columns = []
            for col in merged_info.get("columns", []):
                if isinstance(col, dict):
                    col_copy = dict(col)
                    col_name = col["name"]

                    # Add profile description
                    profile_col = profile_cols.get(col_name, {})
                    if profile_col.get("description"):
                        col_copy["description"] = profile_col["description"]
                    if profile_col.get("readable_name"):
                        col_copy["readable_name"] = profile_col["readable_name"]
                    if profile_col.get("distinct_values"):
                        col_copy["distinct_values"] = profile_col["distinct_values"]

                    # Add SME description
                    sme_col = sme_cols.get(col_name, {})
                    if sme_col.get("description"):
                        col_copy["sme_description"] = sme_col["description"]
                    if sme_col.get("readable_name") and not col_copy.get("readable_name"):
                        col_copy["readable_name"] = sme_col["readable_name"]

                    merged_columns.append(col_copy)
                else:
                    merged_columns.append(col)
            merged_info["columns"] = merged_columns

            tables.append(self.format_table_schema(
                table_name,
                merged_info,
                include_descriptions=True,
                include_sme=True
            ))

        return "\n\n".join(tables)

    def build(
        self,
        nl_query: str,
        schema: dict[str, Any],
        strategy: ContextStrategy,
        focused_schema: dict[str, Any] = None,
        profile: dict[str, Any] = None,
        evidence: str = None
    ) -> dict[str, str]:
        """
        Build complete prompt for a given strategy.

        Each strategy has its own tailored system prompt and user prompt template.

        Args:
            nl_query: Natural language question
            schema: Full database schema
            strategy: Which context strategy to use
            focused_schema: Filtered schema (for FOCUSED_SCHEMA strategy)
            profile: Minimal profile with descriptions
            evidence: SME evidence/hints from BIRD dev.json (for SME_METADATA and FULL_PROFILE)

        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        # Get strategy-specific prompts
        strategy_config = self.strategy_prompts[strategy]
        system_prompt = strategy_config["system"]
        user_template = strategy_config["user_template"]

        # Format schema based on strategy
        if strategy == ContextStrategy.FULL_SCHEMA:
            schema_str = self.format_full_schema(schema)
            strategy_note = "Complete schema, focus on structure analysis."

        elif strategy == ContextStrategy.SME_METADATA:
            schema_str = self.format_full_schema(schema)
            strategy_note = "Schema with SME evidence/hints."

        elif strategy == ContextStrategy.MINIMAL_PROFILE:
            schema_str = self.format_minimal_profile(schema, profile)
            strategy_note = "Schema with basic column descriptions."

        elif strategy == ContextStrategy.FOCUSED_SCHEMA:
            if focused_schema:
                schema_str = self.format_focused_schema(focused_schema)
                strategy_note = "Pre-filtered relevant tables only."
            else:
                schema_str = self.format_full_schema(schema)
                strategy_note = "Fallback to full schema (focused not available)."

        elif strategy == ContextStrategy.FULL_PROFILE:
            if focused_schema:
                schema_str = self.format_focused_schema_with_profile(focused_schema, profile)
                strategy_note = "Focused schema + descriptions + SME evidence."
            else:
                schema_str = self.format_minimal_profile(schema, profile)
                strategy_note = "Fallback to full schema with descriptions (focused not available)."

        else:
            schema_str = self.format_full_schema(schema)
            strategy_note = "Default full schema."

        # Build user prompt using strategy-specific template
        # Handle evidence for SME_METADATA and FULL_PROFILE strategies
        evidence_str = evidence if evidence else "No additional hints provided."

        if strategy in [ContextStrategy.SME_METADATA, ContextStrategy.FULL_PROFILE]:
            user_prompt = user_template.format(
                schema=schema_str,
                question=nl_query,
                evidence=evidence_str
            )
        else:
            user_prompt = user_template.format(
                schema=schema_str,
                question=nl_query
            )

        return {
            "system": system_prompt,
            "user": user_prompt,
            "strategy": strategy.value,
            "strategy_note": strategy_note
        }

    def build_all_strategies(
        self,
        nl_query: str,
        schema: dict[str, Any],
        focused_schema: dict[str, Any] = None,
        profile: dict[str, Any] = None,
        evidence: str = None
    ) -> dict[ContextStrategy, dict[str, str]]:
        """
        Build prompts for all 5 strategies.

        Args:
            nl_query: Natural language question
            schema: Full database schema
            focused_schema: Filtered schema from Map-Reduce Agent
            profile: Minimal profile with descriptions
            evidence: SME evidence/hints from BIRD dev.json

        Returns:
            Dictionary mapping each strategy to its prompts
        """
        prompts = {}

        for strategy in ContextStrategy:
            prompts[strategy] = self.build(
                nl_query=nl_query,
                schema=schema,
                strategy=strategy,
                focused_schema=focused_schema,
                profile=profile,
                evidence=evidence
            )

        return prompts


def main():
    """Test the PromptBuilder with sample data."""
    # Sample schema
    schema = {
        "frpm": {
            "table_readable_name": "free and reduced-price meals",
            "columns": [
                {"name": "CDSCode", "type": "TEXT", "readable_name": "CDS Code"},
                {"name": "County Name", "type": "TEXT"},
                {"name": "Free Meal Count (K-12)", "type": "INTEGER", "readable_name": "Free Meal Count K-12"},
                {"name": "Enrollment (K-12)", "type": "INTEGER"}
            ]
        },
        "satscores": {
            "table_readable_name": "SAT scores",
            "columns": [
                {"name": "cds", "type": "TEXT", "readable_name": "CDS Code"},
                {"name": "AvgScrMath", "type": "INTEGER", "readable_name": "Average Math Score"},
                {"name": "NumTstTakr", "type": "INTEGER", "readable_name": "Number of Test Takers"}
            ]
        }
    }

    # Sample profile (descriptions)
    profile = {
        "tables": {
            "frpm": {
                "table_readable_name": "free and reduced-price meals",
                "columns": [
                    {"name": "CDSCode", "readable_name": "CDS Code", "description": "County-District-School code"},
                    {"name": "Free Meal Count (K-12)", "description": "Number of students eligible for free meals"}
                ]
            }
        }
    }

    # Sample focused schema
    focused_schema = {
        "tables": {
            "frpm": {
                "table_readable_name": "free and reduced-price meals",
                "relevance_score": 1.0,
                "reason": "Contains free meal rate data",
                "columns": [
                    {"name": "CDSCode", "type": "TEXT"},
                    {"name": "Free Meal Count (K-12)", "type": "INTEGER"}
                ]
            }
        }
    }

    builder = PromptBuilder()
    nl_query = "What is the highest eligible free rate for K-12 students?"

    print("=" * 70)
    print("Testing PromptBuilder with 5 Context-Aware Strategies")
    print("=" * 70)

    all_prompts = builder.build_all_strategies(
        nl_query=nl_query,
        schema=schema,
        focused_schema=focused_schema,
        profile=profile
    )

    for strategy, prompts in all_prompts.items():
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.value}")
        print(f"Note: {prompts['strategy_note']}")
        print("-" * 70)
        print("USER PROMPT:")
        print(prompts["user"][:500] + "..." if len(prompts["user"]) > 500 else prompts["user"])

    print("\n" + "=" * 70)
    print("PromptBuilder Test Complete!")


if __name__ == "__main__":
    main()
