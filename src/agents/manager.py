"""
Schema Manager (The Manager)

Implements Agentic Decomposition:
- Analyzes the natural language query
- Decomposes it into sub-components (e.g., "all members", "computer science")
- Coordinates parallel verification workers
- Aggregates results into focused schema
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ThreadPoolExecutor


@dataclass
class QueryComponent:
    """A decomposed component of the original query."""
    component_text: str  # e.g., "all members", "computer science"
    component_type: str  # e.g., "entity", "filter", "aggregation"
    relevant_tables: list[str] = field(default_factory=list)


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""
    original_query: str
    components: list[QueryComponent]
    relationships: list[tuple[str, str]] = field(default_factory=list)


class SchemaManager:
    """
    The Manager agent that performs agentic decomposition.

    Breaks down NL queries into sub-tasks and coordinates workers
    for parallel schema verification.
    """

    def __init__(self, llm_client: Any = None, max_workers: int = 4):
        """
        Initialize the Schema Manager.

        Args:
            llm_client: LLM client for query decomposition
            max_workers: Maximum number of parallel workers
        """
        self.llm_client = llm_client
        self.max_workers = max_workers

    def _parse_json_response(self, response: str) -> dict:
        """Extract and parse JSON from LLM response."""
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            response = json_match.group(1)

        # Try to find JSON object directly
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)

        return json.loads(response)

    def decompose_query(self, nl_query: str) -> DecomposedQuery:
        """
        Decompose a natural language query into semantic components.

        Example:
            Input: "List all members who are in Computer Science related majors"
            Output: Components like ["all members", "computer science", "majors"]

        Args:
            nl_query: Natural language query

        Returns:
            DecomposedQuery with identified components
        """
        # If no LLM client, use heuristic decomposition
        if not self.llm_client:
            return self._heuristic_decompose(nl_query)

        prompt = f"""Analyze this natural language query and decompose it into semantic components.

                    Query: "{nl_query}"

                    Decompose the query into components. Each component should be a meaningful phrase:
                    1. entity: References a data entity (e.g., "members", "students", "courses")
                    2. filter: A filtering condition (e.g., "computer science", "active", "after 2020")
                    3. aggregation: Aggregation operation (e.g., "count", "total", "average")
                    4. projection: Output fields (e.g., "names", "first and last name")

                    Return JSON format:
                    {{
                        "components": [
                            {{
                                "text": "the phrase from the query",
                                "type": "entity|filter|aggregation|projection"
                            }}
                        ]
                    }}

                    Rules:
                    - Extract meaningful phrases, not single words
                    - Identify ALL relevant components

                    Return ONLY valid JSON."""

        try:
            response = self.llm_client.complete(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.0
            )

            result = self._parse_json_response(response)

            components = []
            for comp in result.get("components", []):
                components.append(QueryComponent(
                    component_text=comp["text"],
                    component_type=comp["type"],
                    relevant_tables=[]
                ))

            return DecomposedQuery(
                original_query=nl_query,
                components=components
            )
        except (json.JSONDecodeError, KeyError, Exception):
            # Fallback to heuristic if LLM fails
            return self._heuristic_decompose(nl_query)

    def _heuristic_decompose(self, nl_query: str) -> DecomposedQuery:
        """
        Heuristic decomposition when LLM is not available.

        Uses keyword matching to identify components.
        """
        components = []
        query_lower = nl_query.lower()

        # Aggregation keywords
        aggregation_keywords = {
            "count": "count", "how many": "count", "number of": "count",
            "total": "sum", "sum": "sum",
            "average": "average", "avg": "average", "mean": "average",
            "maximum": "max", "max": "max", "highest": "max",
            "minimum": "min", "min": "min", "lowest": "min"
        }

        for keyword, agg_type in aggregation_keywords.items():
            if keyword in query_lower:
                components.append(QueryComponent(
                    component_text=agg_type,
                    component_type="aggregation",
                    relevant_tables=[]
                ))
                break

        # Projection keywords (what to show)
        projection_keywords = ["list", "show", "display", "get", "find", "select"]
        for keyword in projection_keywords:
            if keyword in query_lower:
                components.append(QueryComponent(
                    component_text=keyword,
                    component_type="projection",
                    relevant_tables=[]
                ))
                break

        # Extract potential entity/filter phrases using simple tokenization
        # Remove common words and extract meaningful phrases
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all",
            "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "who", "which", "what", "this", "that",
            "these", "those", "am", "i", "me", "my", "myself", "we",
            "our", "ours", "you", "your", "he", "him", "his", "she",
            "her", "it", "its", "they", "them", "their"
        }

        # Tokenize and filter
        words = re.findall(r'[a-zA-Z]+', nl_query.lower())
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]

        # Group consecutive meaningful words as potential entities/filters
        if meaningful_words:
            # First meaningful word is likely the entity
            components.append(QueryComponent(
                component_text=meaningful_words[0],
                component_type="entity",
                relevant_tables=[]
            ))

            # Remaining words are likely filters
            for word in meaningful_words[1:]:
                components.append(QueryComponent(
                    component_text=word,
                    component_type="filter",
                    relevant_tables=[]
                ))

        # If no components found, use entire query as entity
        if not components:
            components.append(QueryComponent(
                component_text=nl_query,
                component_type="entity",
                relevant_tables=[]
            ))

        return DecomposedQuery(
            original_query=nl_query,
            components=components
        )

    def identify_component_type(self, component: str, schema: dict[str, Any]) -> str:
        """
        Identify the type of a query component.

        Types:
        - entity: References a table/entity (e.g., "members", "students")
        - filter: A filtering condition (e.g., "computer science")
        - aggregation: Aggregation operation (e.g., "count", "average")
        - projection: Specific columns to select

        Args:
            component: Text component from query
            schema: Database schema

        Returns:
            Component type string
        """
        component_lower = component.lower()

        # Check for aggregation keywords
        aggregation_keywords = [
            "count", "total", "sum", "average", "avg", "maximum", "max",
            "minimum", "min", "how many", "number of"
        ]
        if any(kw in component_lower for kw in aggregation_keywords):
            return "aggregation"

        # Check for projection keywords
        projection_keywords = [
            "show", "list", "display", "name", "names", "email", "id",
            "select", "get", "return"
        ]
        if any(kw in component_lower for kw in projection_keywords):
            return "projection"

        # Check if component matches a table name (entity)
        table_names = [name.lower() for name in schema.keys()]
        for table_name in table_names:
            if table_name in component_lower or component_lower in table_name:
                return "entity"

        # Check if component matches column names (could be entity or filter)
        for table_schema in schema.values():
            columns = table_schema.get("columns", [])
            for col in columns:
                col_name = col.get("name", col) if isinstance(col, dict) else col
                if isinstance(col_name, str) and col_name.lower() in component_lower:
                    return "entity"

        # Default to filter for remaining components
        return "filter"

    def map_components_to_tables(
        self,
        components: list[QueryComponent],
        schema: dict[str, Any]
    ) -> dict[str, list[str]]:
        """
        Map each component to potentially relevant tables.

        Args:
            components: List of query components
            schema: Database schema

        Returns:
            Mapping of component text to list of relevant table names
        """
        component_to_tables: dict[str, list[str]] = {}

        for component in components:
            component_text = component.component_text.lower()
            relevant_tables: list[str] = []

            for table_name, table_schema in schema.items():
                table_name_lower = table_name.lower()
                table_readable = table_schema.get("table_readable_name", "").lower()

                # Check table name match
                if (table_name_lower in component_text or
                    component_text in table_name_lower or
                    table_readable in component_text or
                    component_text in table_readable):
                    relevant_tables.append(table_name)
                    continue

                # Check column names for match
                columns = table_schema.get("columns", [])
                for col in columns:
                    col_name = col.get("name", col) if isinstance(col, dict) else col
                    col_readable = col.get("readable_name", "") if isinstance(col, dict) else ""
                    col_desc = col.get("description", "") if isinstance(col, dict) else ""

                    col_text = f"{col_name} {col_readable} {col_desc}".lower()

                    # Tokenize component and check for matches
                    tokens = re.findall(r"[a-z0-9]+", component_text)
                    if any(token in col_text for token in tokens if len(token) > 2):
                        relevant_tables.append(table_name)
                        break

            component_to_tables[component.component_text] = list(set(relevant_tables))
            component.relevant_tables = list(set(relevant_tables))

        return component_to_tables

    def coordinate_workers(
        self,
        decomposed_query: DecomposedQuery,
        schema: dict[str, Any],
        profile: dict[str, Any] = None,
        evidence: str = None
    ) -> list[Any]:
        """
        Coordinate parallel verification workers.

        Each worker uses one mini LLM to score ONE table and its columns.
        All workers run in parallel (one worker per table).

        Args:
            decomposed_query: The decomposed query
            schema: Full database schema
            profile: Optional database profile
            evidence: Optional SME evidence/hints from BIRD dev.json

        Returns:
            List of VerificationResult from all workers
        """
        # Import worker here to avoid circular imports
        from .worker import SchemaWorker

        # Get all table names
        table_names = list(schema.keys())
        if not table_names:
            return []

        # Convert query components to dict format for workers
        query_components = [
            {"text": comp.component_text, "type": comp.component_type}
            for comp in decomposed_query.components
        ]

        # Get the original query for context
        original_query = decomposed_query.original_query

        # Create one worker per table, each with its own mini LLM
        # Execute all workers in parallel
        results = []
        with ThreadPoolExecutor(max_workers=len(table_names)) as executor:
            futures = []
            for idx, table_name in enumerate(table_names):
                # Each worker handles exactly ONE table
                worker = SchemaWorker(
                    worker_id=f"worker-{idx+1}",
                    llm_client=self.llm_client
                )
                future = executor.submit(
                    worker,
                    [table_name],  # Single table assignment
                    schema,
                    query_components,
                    profile,
                    original_query,  # Pass original query for better context
                    evidence  # Pass evidence for table relevance hints
                )
                futures.append((table_name, future))

            # Collect results from all workers
            for table_name, future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Worker error for table '{table_name}': {e}")

        return results

    def aggregate_results(
        self,
        verification_results: list[Any],
        schema: dict[str, Any],
        relevance_threshold: float = 0.50
    ) -> dict[str, Any]:
        """
        Aggregate results from all workers into focused schema.

        Implements the Reduce Function:
        - Collects all table relevances from workers
        - Filters tables with score >= threshold (default 0.50)
        - Sorts by relevance score descending
        - Builds focused schema with relevant tables/columns

        Args:
            verification_results: VerificationResult objects from each worker
            schema: Original full database schema
            relevance_threshold: Minimum score to include table (default 0.50)

        Returns:
            Focused schema containing only relevant tables/columns
        """
        # Collect all table relevances from all workers
        all_table_relevances = []
        for result in verification_results:
            if hasattr(result, 'table_relevances'):
                all_table_relevances.extend(result.table_relevances)

        # Filter by relevance threshold
        relevant_tables = [
            table for table in all_table_relevances
            if table.relevance_score >= relevance_threshold
        ]

        # Sort by relevance score descending
        relevant_tables.sort(key=lambda x: x.relevance_score, reverse=True)

        # Build focused schema
        focused_schema: dict[str, Any] = {
            "tables": {},
            "table_relevances": [],
            "metadata": {
                "total_tables_evaluated": len(all_table_relevances),
                "relevant_tables_count": len(relevant_tables),
                "relevance_threshold": relevance_threshold
            }
        }

        for table_rel in relevant_tables:
            table_name = table_rel.table_name
            original_table = schema.get(table_name, {})

            # Get relevant column names
            relevant_col_names = {
                col.column_name for col in table_rel.relevant_columns
            }

            # Build focused table schema
            focused_table: dict[str, Any] = {
                "table_name": table_name,
                "table_readable_name": original_table.get("table_readable_name", table_name),
                "relevance_score": table_rel.relevance_score,
                "reason": table_rel.reason,
                "columns": []
            }

            # Include relevant columns (or all if none specifically marked)
            original_columns = original_table.get("columns", [])
            for col in original_columns:
                col_name = col.get("name", col) if isinstance(col, dict) else col
                # Include column if it's marked relevant or if no specific columns marked
                if not relevant_col_names or col_name in relevant_col_names:
                    if isinstance(col, dict):
                        focused_table["columns"].append(col)
                    else:
                        focused_table["columns"].append({"name": col})

            focused_schema["tables"][table_name] = focused_table
            focused_schema["table_relevances"].append({
                "table_name": table_name,
                "relevance_score": table_rel.relevance_score,
                "reason": table_rel.reason,
                "relevant_columns": [col.column_name for col in table_rel.relevant_columns]
            })

        return focused_schema

    def run(
        self,
        nl_query: str,
        schema: dict[str, Any],
        profile: dict[str, Any] = None,
        evidence: str = None,
        relevance_threshold: float = 0.50
    ) -> dict[str, Any]:
        """
        Run the full Map-Reduce Schema Agent workflow.

        1. Decompose query into semantic components (entity, filter, aggregation, projection)
        2. Coordinate parallel workers (one worker per table, each with mini LLM)
        3. Aggregate results into focused schema (Reduce Function with threshold filtering)

        Args:
            nl_query: Natural language query
            schema: Full database schema
            profile: Optional database profile for enhanced matching
            evidence: Optional SME evidence/hints from BIRD dev.json
            relevance_threshold: Minimum relevance score for tables (default 0.50)

        Returns:
            Focused schema dictionary
        """
        # Step 1: Decompose query into semantic components
        # The Manager analyzes the query and extracts: [entity], [filter], [aggregation], [projection]
        decomposed_query = self.decompose_query(nl_query)

        # Step 2: Coordinate parallel workers (Mapping Function)
        # Each worker uses one mini LLM to score ONE table and its columns
        # All workers run in parallel
        verification_results = self.coordinate_workers(
            decomposed_query=decomposed_query,
            schema=schema,
            profile=profile,
            evidence=evidence
        )

        # Step 3: Aggregate results into focused schema (Reduce Function)
        # - Combines all RC scores from workers
        # - Filters tables with score >= threshold (default 0.50)
        # - Sorts by relevance score descending
        focused_schema = self.aggregate_results(
            verification_results=verification_results,
            schema=schema,
            relevance_threshold=relevance_threshold
        )

        # Add decomposition info to metadata
        focused_schema["metadata"]["original_query"] = nl_query
        focused_schema["metadata"]["decomposed_components"] = [
            {"text": comp.component_text, "type": comp.component_type}
            for comp in decomposed_query.components
        ]

        return focused_schema


def main():
    """Test the full Map-Reduce Schema Agent workflow."""
    import sys
    from pathlib import Path

    # Add src directory to path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))

    from utils.llm_client import LLMClient

    # Initialize
    llm_client = LLMClient()
    manager = SchemaManager(llm_client=llm_client, max_workers=4)

    # Sample schema (same as in worker.py for consistency)
    schema = {
        "members": {
            "table_readable_name": "members",
            "columns": [
                {"name": "id", "readable_name": "member id"},
                {"name": "name", "readable_name": "member name"},
                {"name": "major", "readable_name": "major"}
            ]
        },
        "subjects": {
            "table_readable_name": "subjects",
            "columns": [
                {"name": "id", "readable_name": "subject id"},
                {"name": "name", "readable_name": "subject name"}
            ]
        },
        "cafeteria": {
            "table_readable_name": "cafeteria menu",
            "columns": [
                {"name": "menu_id", "readable_name": "menu id"},
                {"name": "item_name", "readable_name": "item name"},
                {"name": "price", "readable_name": "price"}
            ]
        },
        "parking_lots": {
            "table_readable_name": "parking lots",
            "columns": [
                {"name": "lot_id", "readable_name": "lot id"},
                {"name": "capacity", "readable_name": "capacity"},
                {"name": "level", "readable_name": "level"}
            ]
        },
        "campus_events": {
            "table_readable_name": "campus events",
            "columns": [
                {"name": "event_id", "readable_name": "event id"},
                {"name": "title", "readable_name": "event title"},
                {"name": "event_date", "readable_name": "event date"}
            ]
        }
    }

    # Test query
    query = "List all members who are in Computer Science related majors"

    print("=" * 60)
    print("Testing Full Map-Reduce Schema Agent Workflow")
    print("=" * 60)

    print(f"\nInput Query: {query}")
    print("-" * 60)

    # Run the full workflow
    focused_schema = manager.run(
        nl_query=query,
        schema=schema,
        relevance_threshold=0.50
    )

    # Display results
    print("\n1. Decomposed Components:")
    for comp in focused_schema["metadata"]["decomposed_components"]:
        print(f"   - [{comp['type']}] {comp['text']}")

    print(f"\n2. Tables Evaluated: {focused_schema['metadata']['total_tables_evaluated']}")
    print(f"   Relevant Tables (score >= {focused_schema['metadata']['relevance_threshold']}): "
          f"{focused_schema['metadata']['relevant_tables_count']}")

    print("\n3. Focused Schema (Relevant Tables):")
    for table_rel in focused_schema["table_relevances"]:
        print(f"   - {table_rel['table_name']}: RC={table_rel['relevance_score']:.2f}")
        print(f"     Reason: {table_rel['reason']}")
        if table_rel['relevant_columns']:
            print(f"     Columns: {', '.join(table_rel['relevant_columns'])}")

    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
