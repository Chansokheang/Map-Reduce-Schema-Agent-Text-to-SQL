"""
Schema Worker (The Workers)

Implements Parallel Verification:
- Each worker is assigned specific tables to verify
- Verifies relevance of tables/columns to query components
- Reports back verification results to manager
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnRelevance:
    """Relevance assessment for a single column."""
    column_name: str
    relevance_score: float  # 0.0 to 1.0
    reason: str  # Why this column is/isn't relevant
    matched_component: str = None  # Which query component it matches


@dataclass
class TableRelevance:
    """Relevance assessment for a table."""
    table_name: str
    relevance_score: float  # 0.0 to 1.0
    relevant_columns: list[ColumnRelevance] = field(default_factory=list)
    reason: str = ""


@dataclass
class VerificationResult:
    """Result of a worker's verification task."""
    worker_id: str
    assigned_tables: list[str]
    table_relevances: list[TableRelevance]
    execution_time_ms: float = 0.0


class SchemaWorker:
    """
    A worker agent that verifies schema relevance in parallel.

    Each worker is assigned a subset of tables (e.g., members, subjects, others)
    and verifies their relevance to the query components.
    """

    def __init__(self, worker_id: str, llm_client: Any = None):
        """
        Initialize a Schema Worker.

        Args:
            worker_id: Unique identifier for this worker
            llm_client: LLM client for relevance assessment
        """
        self.worker_id = worker_id
        self.llm_client = llm_client

    def verify_table_relevance(
        self,
        table_name: str,
        table_schema: dict[str, Any],
        query_components: list[dict[str, str]],
        profile: dict[str, Any] = None,
        original_query: str = None,
        evidence: str = None
    ) -> TableRelevance:
        """
        Verify if a table is relevant to the query.

        Args:
            table_name: Name of the table
            table_schema: Schema of the table (columns, keys, etc.)
            query_components: Decomposed query components
            profile: Optional profile with semantic information
            original_query: Original natural language query for context
            evidence: Optional SME evidence/hints from BIRD dev.json

        Returns:
            TableRelevance with score and relevant columns
        """
        table_readable_name = table_schema.get("table_readable_name", table_name)
        table_profile = None
        if profile:
            table_profile = profile.get("tables", {}).get(table_name)
            if table_profile and table_profile.get("table_readable_name"):
                table_readable_name = table_profile["table_readable_name"]

        columns = self._collect_columns(table_schema, table_profile)
        component_texts = self._normalize_components(query_components)

        # Normalize query_components to ensure dict format
        normalized_components = []
        for comp in query_components:
            if isinstance(comp, dict):
                normalized_components.append(comp)
            else:
                normalized_components.append({
                    "text": getattr(comp, "component_text", str(comp)),
                    "type": getattr(comp, "component_type", "unknown")
                })

        if self.llm_client:
            try:
                llm_result = self._llm_table_relevance(
                    table_name,
                    table_readable_name,
                    columns,
                    normalized_components,
                    original_query,
                    evidence
                )
                return self._build_table_relevance(
                    table_name,
                    llm_result,
                    columns
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        heuristic_result = self._heuristic_table_relevance(
            table_name,
            table_readable_name,
            columns,
            component_texts
        )
        return self._build_table_relevance(table_name, heuristic_result, columns)

    def verify_column_relevance(
        self,
        column_name: str,
        column_info: dict[str, Any],
        table_name: str,
        query_components: list[dict[str, str]],
        profile: dict[str, Any] = None
    ) -> ColumnRelevance:
        """
        Verify if a column is relevant to the query.

        Args:
            column_name: Name of the column
            column_info: Column metadata (type, description, etc.)
            table_name: Parent table name
            query_components: Decomposed query components
            profile: Optional profile with semantic information

        Returns:
            ColumnRelevance with score and matched component
        """
        component_texts = self._normalize_components(query_components)
        col_text = " ".join(
            v for v in [
                column_name,
                column_info.get("readable_name", ""),
                column_info.get("description", "")
            ] if v
        )
        if self.llm_client:
            best_score = 0.0
            best_reason = ""
            best_component = ""
            for component in component_texts:
                score, reason = self.check_semantic_match(
                    col_text,
                    "",
                    component
                )
                if score > best_score:
                    best_score = score
                    best_reason = reason
                    best_component = component
            return ColumnRelevance(
                column_name=column_name,
                relevance_score=best_score,
                reason=best_reason or "LLM match evaluation",
                matched_component=best_component or None
            )

        score, reason, matched = self._heuristic_match(col_text, component_texts)
        return ColumnRelevance(
            column_name=column_name,
            relevance_score=score,
            reason=reason,
            matched_component=matched
        )

    def check_semantic_match(
        self,
        entity_name: str,
        entity_description: str,
        query_component: str
    ) -> tuple[float, str]:
        """
        Check semantic match between entity and query component.

        Uses LLM or embedding similarity to determine match.

        Args:
            entity_name: Name of table or column
            entity_description: Semantic description from profile
            query_component: Query component text

        Returns:
            Tuple of (score, reason)
        """
        if not self.llm_client:
            score, reason, _ = self._heuristic_match(
                f"{entity_name} {entity_description}".strip(),
                [query_component]
            )
            return score, reason

        prompt = f"""Evaluate semantic relevance between a schema entity and a query component.

                    Entity: {entity_name}
                    Description: {entity_description or "N/A"}
                    Query component: {query_component}

                    Return ONLY JSON:
                    {{
                    "score": 0.0,
                    "reason": "short reason"
                    }}
                    Rules:
                    - score is between 0 and 1
                    - use 0 if unrelated, 1 if clearly matches
                    """
        response = self.llm_client.complete(
            prompt=prompt,
            max_tokens=128,
            temperature=0.0
        )
        try:
            data = self._parse_json_response(response)
            score = float(data.get("score", 0.0))
            reason = data.get("reason", "").strip() or "LLM semantic match score"
            return score, reason
        except (json.JSONDecodeError, ValueError, TypeError):
            score, reason, _ = self._heuristic_match(
                f"{entity_name} {entity_description}".strip(),
                [query_component]
            )
            return score, reason

    def verify_tables(
        self,
        assigned_tables: list[str],
        schema: dict[str, Any],
        query_components: list[dict[str, str]],
        profile: dict[str, Any] = None,
        original_query: str = None,
        evidence: str = None
    ) -> VerificationResult:
        """
        Verify all assigned tables.

        Main entry point for worker execution.

        Args:
            assigned_tables: List of table names assigned to this worker
            schema: Full database schema
            query_components: Decomposed query components
            profile: Optional database profile
            original_query: Original natural language query for context
            evidence: Optional SME evidence/hints from BIRD dev.json

        Returns:
            VerificationResult with all table relevances
        """
        start_time = time.perf_counter()
        table_relevances = []

        for table_name in assigned_tables:
            table_schema = schema.get(table_name)
            if not table_schema:
                continue
            table_relevances.append(
                self.verify_table_relevance(
                    table_name,
                    table_schema,
                    query_components,
                    profile,
                    original_query,
                    evidence
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return VerificationResult(
            worker_id=self.worker_id,
            assigned_tables=assigned_tables,
            table_relevances=table_relevances,
            execution_time_ms=elapsed_ms
        )

    def __call__(
        self,
        assigned_tables: list[str],
        schema: dict[str, Any],
        query_components: list[dict[str, str]],
        profile: dict[str, Any] = None,
        original_query: str = None,
        evidence: str = None
    ) -> VerificationResult:
        """
        Callable interface for parallel execution.

        Allows worker to be used with ThreadPoolExecutor.
        """
        return self.verify_tables(assigned_tables, schema, query_components, profile, original_query, evidence)

    def _collect_columns(
        self,
        table_schema: dict[str, Any],
        table_profile: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        columns = []
        raw_columns = table_schema.get("columns", [])
        for col in raw_columns:
            if isinstance(col, str):
                columns.append({"name": col})
            elif isinstance(col, dict):
                columns.append({
                    "name": col.get("name", ""),
                    "readable_name": col.get("readable_name", ""),
                    "description": col.get("description", "")
                })

        if table_profile and table_profile.get("columns"):
            profile_map = {
                col["name"]: col
                for col in table_profile["columns"]
                if isinstance(col, dict) and col.get("name")
            }
            for col in columns:
                profile_col = profile_map.get(col.get("name"))
                if profile_col:
                    col["readable_name"] = profile_col.get(
                        "readable_name",
                        col.get("readable_name", "")
                    )
                    col["description"] = profile_col.get(
                        "description",
                        col.get("description", "")
                    )

        return columns

    def _normalize_components(
        self,
        query_components: list[Any]
    ) -> list[str]:
        component_texts = []
        for comp in query_components:
            if isinstance(comp, dict):
                component_texts.append(comp.get("text", ""))
            else:
                component_texts.append(getattr(comp, "component_text", str(comp)))
        return [text.strip() for text in component_texts if text and text.strip()]

    def _llm_table_relevance(
        self,
        table_name: str,
        table_readable_name: str,
        columns: list[dict[str, Any]],
        query_components: list[dict[str, str]],
        original_query: str = None,
        evidence: str = None
    ) -> dict[str, Any]:
        column_lines = []
        for col in columns:
            col_name = col.get("name", "")
            readable = col.get("readable_name", "")
            description = col.get("description", "")
            parts = [col_name]
            if readable and readable != col_name:
                parts.append(f"readable: {readable}")
            if description:
                parts.append(f"desc: {description}")
            column_lines.append(" - " + " | ".join(parts))

        # Format semantic components with their types
        component_lines = []
        for comp in query_components:
            comp_text = comp.get("text", "")
            comp_type = comp.get("type", "unknown")
            if comp_text:
                component_lines.append(f"  - [{comp_type}] {comp_text}")
        components_str = chr(10).join(component_lines) if component_lines else "  - (no components)"

        # Use original query as main question
        question_str = original_query if original_query else ", ".join(
            comp.get("text", "") for comp in query_components
        )

        # Format evidence section
        evidence_str = evidence if evidence else "No additional hints provided."

        prompt = f"""Determine if this table is REQUIRED to answer the following question.

Question: "{question_str}"

EVIDENCE (CRITICAL - contains hints about which tables/columns to use):
{evidence_str}

Semantic Components (extracted from the question):
{components_str}

Table: {table_name}
Table description: {table_readable_name}
Columns:
{chr(10).join(column_lines) if column_lines else " - (no columns provided)"}

Return ONLY JSON:
{{
  "relevant": true or false,
  "score": 0.0 to 1.0,
  "reason": "short reason",
  "relevant_columns": ["column_name"]
}}

STRICT Scoring Rules:
- 1.0: Table is DIRECTLY REQUIRED - the question asks for data that ONLY this table contains.
       Check if any [entity] or [filter] component matches this table's data.
       IMPORTANT: If the EVIDENCE mentions a column name that exists in THIS table, score 1.0!
- 0.5: Table MAY be needed - for JOINs or foreign key lookups, but not the main data source.
- 0.0: Table is NOT needed - the question does not require any data from this table.

EVIDENCE Guidelines (CRITICAL):
- If evidence mentions a column like "[Free Meal Count (K-12)]" and this table has that column → score 1.0
- If evidence mentions "table frpm" or similar table name hints → use that to determine relevance
- If evidence contains a formula like "rate = A / B", check if this table has columns A or B
- Evidence is from domain experts - trust it for column/table identification

Component Type Guidelines:
- [entity]: Look for tables that store this type of data
- [filter]: Look for columns that can filter by this condition
- [aggregation]: Consider if this table has data to aggregate
- [projection]: Check if this table has the requested output fields

Examples:
- Question about "free meal rates" → frpm table = 1.0, satscores table = 0.0
- Question about "SAT scores" → satscores table = 1.0, frpm table = 0.0
- Question needing school info for JOIN → schools table = 0.5
- Evidence says "[NumGE1500]" → satscores table = 1.0 (has that column)

Be strict: if the question doesn't need data from "{table_readable_name}", score 0.0."""

        response = self.llm_client.complete(
            prompt=prompt,
            max_tokens=256,
            temperature=0.0
        )
        return self._parse_json_response(response)

    def _heuristic_table_relevance(
        self,
        table_name: str,
        table_readable_name: str,
        columns: list[dict[str, Any]],
        component_texts: list[str]
    ) -> dict[str, Any]:
        haystack = " ".join([table_name, table_readable_name]).lower()
        relevant_columns = []
        best_score = 0.0
        matched_components = []

        for component in component_texts:
            score, _, _ = self._heuristic_match(haystack, [component])
            if score > 0:
                matched_components.append(component)
                best_score = max(best_score, score)

            tokens = self._tokenize(component)
            for col in columns:
                col_text = " ".join(
                    v for v in [
                        col.get("name", ""),
                        col.get("readable_name", ""),
                        col.get("description", "")
                    ] if v
                ).lower()
                if any(token in col_text for token in tokens):
                    relevant_columns.append(col.get("name", ""))
                    best_score = max(best_score, 0.4)

        relevant_columns = sorted({c for c in relevant_columns if c})
        return {
            "relevant": bool(matched_components or relevant_columns),
            "score": float(best_score),
            "reason": "Heuristic keyword match",
            "matched_components": matched_components,
            "relevant_columns": relevant_columns
        }

    def _discretize_score(self, score: float) -> float:
        """
        Discretize relevance score to [1.0, 0.5, 0.0].

        - score >= 0.7 → 1.0 (highly relevant)
        - score >= 0.3 → 0.5 (somewhat relevant)
        - score < 0.3 → 0.0 (not relevant)
        """
        if score >= 0.7:
            return 1.0
        elif score >= 0.3:
            return 0.5
        else:
            return 0.0

    def _build_table_relevance(
        self,
        table_name: str,
        result: dict[str, Any],
        columns: list[dict[str, Any]]
    ) -> TableRelevance:
        relevant = bool(result.get("relevant"))
        raw_score = float(result.get("score", 0.0))
        reason = result.get("reason", "").strip() or "No relevance detected"

        # Discretize score to [1.0, 0.5, 0.0]
        score = self._discretize_score(raw_score)

        relevant_columns = []
        col_map = {col.get("name", ""): col for col in columns if col.get("name")}
        for col_name in result.get("relevant_columns", []):
            if col_name in col_map:
                # Column score: 1.0 if table is relevant, else 0.5
                col_score = 1.0 if score >= 0.5 else 0.5
                relevant_columns.append(ColumnRelevance(
                    column_name=col_name,
                    relevance_score=col_score,
                    reason="Matches query component",
                    matched_component=None
                ))

        if not relevant:
            relevant_columns = []
            score = 0.0
            reason = reason or "No semantic match"

        return TableRelevance(
            table_name=table_name,
            relevance_score=score,
            relevant_columns=relevant_columns,
            reason=reason
        )

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _heuristic_match(
        self,
        haystack: str,
        components: list[str]
    ) -> tuple[float, str, str]:
        haystack = haystack.lower()
        best_score = 0.0
        best_component = ""
        for component in components:
            tokens = self._tokenize(component)
            if not tokens:
                continue
            matches = sum(1 for token in tokens if token in haystack)
            score = matches / len(tokens)
            if score > best_score:
                best_score = score
                best_component = component
        if best_score > 0:
            return best_score, "Token overlap match", best_component
        return 0.0, "No token overlap", ""

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            response = json_match.group(1)

        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)

        return json.loads(response)


def main() -> None:
    """Run a small integration test with the manager and a worker."""
    import sys
    from pathlib import Path

    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))

    from agents.manager import SchemaManager

    try:
        from utils.llm_client import LLMClient
        llm_client: Any = LLMClient()
    except Exception:
        llm_client = None

    manager = SchemaManager(llm_client=llm_client)
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

    query = "List all members who are in Computer Science related majors"
    decomposed = manager.decompose_query(query)
    components = [
        {"text": comp.component_text, "type": comp.component_type}
        for comp in decomposed.components
    ]

    table_names = list(schema.keys())
    max_workers = min(8, len(table_names))
    workers = [
        SchemaWorker(worker_id=f"worker-{idx + 1}", llm_client=llm_client)
        for idx in range(max_workers)
    ]
    assignments = [[] for _ in range(max_workers)]
    for idx, table_name in enumerate(table_names):
        assignments[idx % max_workers].append(table_name)

    results = []
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = [
            executor.submit(worker, assignment, schema, components)
            for worker, assignment in zip(workers, assignments)
        ]
        for future in futures:
            results.append(future.result())

    print("Integration test result:")
    for result in results:
        print(f"Worker: {result.worker_id}")
        for table in result.table_relevances:
            status = "relevant" if table.relevance_score > 0 else "not relevant"
            print(f"- {table.table_name}: {status} ({table.relevance_score:.2f})")


if __name__ == "__main__":
    main()
