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

        response = self.llm_client.complete(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.0
        )

        try:
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
        except (json.JSONDecodeError, KeyError):
            # Fallback: treat entire query as single entity component
            return DecomposedQuery(
                original_query=nl_query,
                components=[QueryComponent(
                    component_text=nl_query,
                    component_type="entity",
                    relevant_tables=[]
                )]
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
        # TODO: Implement component type identification
        pass

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
        # TODO: Implement table mapping
        pass

    def coordinate_workers(
        self,
        decomposed_query: DecomposedQuery,
        schema: dict[str, Any],
        workers: list['SchemaWorker']
    ) -> dict[str, Any]:
        """
        Coordinate parallel verification workers.

        Uses ThreadPoolExecutor to run workers in parallel,
        each verifying schema relevance for assigned tables.

        Args:
            decomposed_query: The decomposed query
            schema: Full database schema
            workers: List of worker instances

        Returns:
            Aggregated verification results
        """
        # TODO: Implement worker coordination with parallel execution
        pass

    def aggregate_results(
        self,
        verification_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Aggregate results from all workers into focused schema.

        Args:
            verification_results: Results from each worker

        Returns:
            Focused schema containing only relevant tables/columns
        """
        # TODO: Implement result aggregation
        pass

    def run(
        self,
        nl_query: str,
        schema: dict[str, Any],
        profile: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Run the full manager workflow.

        1. Decompose query into components
        2. Map components to tables
        3. Coordinate parallel workers
        4. Aggregate into focused schema

        Args:
            nl_query: Natural language query
            schema: Full database schema
            profile: Optional database profile for enhanced matching

        Returns:
            Focused schema dictionary
        """
        # TODO: Implement full workflow
        pass


def main():
    """Test the decompose_query function."""
    import sys
    from pathlib import Path

    # Add src directory to path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))

    from utils.llm_client import LLMClient

    # Initialize
    llm_client = LLMClient()
    manager = SchemaManager(llm_client=llm_client)

    # Test queries
    test_queries = [
        "List all members who are in Computer Science related majors",
        "How many students enrolled after 2020?",
        "Show the names and emails of active users",
        "What is the average salary of employees in the engineering department?",
    ]

    print("=" * 60)
    print("Testing decompose_query")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        result = manager.decompose_query(query)

        print(f"Components:")
        for comp in result.components:
            print(f"  - [{comp.component_type}] {comp.component_text}")

        print()


if __name__ == "__main__":
    main()
