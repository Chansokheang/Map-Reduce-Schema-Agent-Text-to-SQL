"""
SQL Judge Module

Implements "LLM As a Judge" pattern from the proposed method:
- ROLE: Senior SQL Reviewer
- TASK: Select the BEST query from successful candidates
- CONTEXT: Input query + SME evidence
- CANDIDATES: Successful SQL queries (Q1-Q5)
- EVALUATION CRITERIA: Correctness, completeness, accuracy, logic, efficiency

Prompt is loaded from src/prompt/judge.py
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from src.prompt import JUDGE_PROMPT
from src.utils.llm_client import AnthropicExhaustedError


@dataclass
class JudgmentResult:
    """Result of the judging process."""
    selected_id: int
    selected_sql: str
    confidence: float
    reasoning: str
    total_candidates: int
    successful_candidates: int


class SQLJudge:
    """
    LLM-based judge for SQL candidate selection.

    Takes only SUCCESSFUL candidates (those that passed execution),
    evaluates them, and selects the best one.

    Prompt is loaded from src/prompt/judge.py
    """

    def __init__(self, llm_client: Any = None):
        """
        Initialize the SQL Judge.

        Args:
            llm_client: LLM client for evaluation
        """
        self.llm_client = llm_client
        self.prompt_config = JUDGE_PROMPT

    def _format_candidates(
        self,
        candidates: list[dict[str, Any]]
    ) -> str:
        """
        Format candidates for the judge prompt.

        Each candidate is shown with its SQL AND its execution outcome
        (row count + up to 3 sample rows) so the judge can distinguish
        candidates that actually returned data from ones that returned
        empty after the retry loop.

        Args:
            candidates: List of candidate dicts with id, sql, strategy,
                        row_count, sample_rows

        Returns:
            Formatted candidates string
        """
        lines = []
        for c in candidates:
            candidate_id = c["candidate_id"]
            sql = c["sql"]
            strategy = c.get("strategy", "unknown")
            status = c.get("status", "success")
            iterations = c.get("iterations", 1)
            row_count = c.get("row_count", 0)
            sample_rows = c.get("sample_rows") or []

            header = f"Option {candidate_id} ({strategy})"
            if status == "refined":
                header += f" [refined after {iterations} attempts]"

            lines.append(f"{header}:")
            lines.append(f"{sql}")

            # Execution summary — critical for judge to compare outcomes
            if row_count == 0:
                lines.append(f"[Execution: 0 rows returned — EMPTY RESULT]")
            else:
                shown = min(len(sample_rows), 3)
                suffix = "" if row_count == shown else f", showing first {shown}"
                lines.append(f"[Execution: {row_count} row(s) returned{suffix}]")
                for i, row in enumerate(sample_rows[:3], 1):
                    # Truncate long row reprs to keep the prompt tight
                    row_str = repr(row)
                    if len(row_str) > 200:
                        row_str = row_str[:200] + "..."
                    lines.append(f"  Row {i}: {row_str}")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _extract_alias_to_table_map(sql: str) -> dict[str, str]:
        """
        Parse FROM/JOIN clauses to build {alias_or_tablename: real_table_name}.

        Handles:
          FROM trans AS T1                → T1 → trans, trans → trans
          INNER JOIN account AS T2 ON ... → T2 → account, account → account
          FROM frpm                       → frpm → frpm
          JOIN [order] o ON ...           → o → order, order → order

        Used to resolve `T1.amount` → `trans.amount` so the column-diff
        detector can tell apart `loan.amount` from `trans.amount`.
        """
        alias_map: dict[str, str] = {}
        # `FROM|JOIN <table>  [AS]?  <alias>?`  with optional bracket/backtick quoting
        pattern = re.compile(
            r"(?:FROM|JOIN)\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)[`\"\]]?"
            r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
            re.IGNORECASE,
        )
        for m in pattern.finditer(sql):
            table = m.group(1)
            alias = m.group(2)
            if table.upper() in {"SELECT", "WHERE", "GROUP", "ORDER",
                                 "HAVING", "LIMIT"}:
                continue
            alias_map[table.lower()] = table.lower()
            if alias and alias.upper() not in {
                "ON", "WHERE", "GROUP", "ORDER", "HAVING", "LIMIT",
                "INNER", "LEFT", "RIGHT", "OUTER", "CROSS", "JOIN",
            }:
                alias_map[alias.lower()] = table.lower()
        return alias_map

    @staticmethod
    def _extract_columns_set(sql: str) -> frozenset[str]:
        """
        Extract a set of normalized `table.column` references from a SQL.

        Crucially, alias prefixes are RESOLVED to their real table name
        using the FROM/JOIN map, so `T1.amount` (where T1 = trans)
        becomes `trans.amount` — distinct from `T2.amount` (where T2 = loan)
        which becomes `loan.amount`. Without this, candidates that pull
        the same column name from different tables look identical and the
        judge's column-diff detector misses the disagreement.

        Bare column refs without an alias prefix can't be table-resolved
        (we'd need a full schema-aware parser), so they're stored as just
        the column name.
        """
        if not sql:
            return frozenset()

        skip_kw = {
            "AS", "ON", "AND", "OR", "NOT", "IS", "NULL", "IN", "LIKE",
            "BETWEEN", "WHERE", "FROM", "JOIN", "INNER", "LEFT", "RIGHT",
            "OUTER", "CROSS", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
            "OFFSET", "SELECT", "DISTINCT", "UNION", "INTERSECT", "EXCEPT",
            "CASE", "WHEN", "THEN", "ELSE", "END", "ASC", "DESC", "ALL",
            "EXISTS", "IF", "TRUE", "FALSE",
        }

        alias_map = SQLJudge._extract_alias_to_table_map(sql)
        tokens: set[str] = set()

        def _qualify(alias: str, col: str) -> str:
            """Resolve `alias` → `<real_table>.<col>` when alias is known."""
            real = alias_map.get(alias.lower(), alias.lower())
            return f"{real}.{col.lower()}"

        # alias.bareColumn → real_table.column
        for m in re.finditer(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)", sql):
            alias, col = m.group(1), m.group(2)
            if col.upper() in skip_kw:
                continue
            tokens.add(_qualify(alias, col))

        # alias.[Quoted Column] / alias.`Backtick Column`
        for m in re.finditer(r"\b([A-Za-z_]\w*)\.\[([^\]\n]+)\]", sql):
            tokens.add(_qualify(m.group(1), m.group(2).strip()))
        for m in re.finditer(r"\b([A-Za-z_]\w*)\.`([^`\n]+)`", sql):
            tokens.add(_qualify(m.group(1), m.group(2).strip()))

        # Standalone bracketed / backtick identifiers (no alias prefix —
        # we can't resolve to a table, so just store the name).
        for m in re.finditer(r"(?<![A-Za-z_0-9.])\[([^\]\n]+)\]", sql):
            tokens.add(m.group(1).strip().lower())
        for m in re.finditer(r"(?<![A-Za-z_0-9.])`([^`\n]+)`", sql):
            tokens.add(m.group(1).strip().lower())

        # Bare column refs next to comparison operators (no alias) —
        # also unqualifiable, store as bare name.
        compare_pat = re.compile(
            r"\b([A-Za-z_]\w*)\s*"
            r"(?:=|>=|<=|<>|!=|>|<|"
            r"\bLIKE\b|\bIN\b\s*\(|\bIS\b\s+(?:NOT\s+)?\bNULL\b|\bBETWEEN\b)",
            re.IGNORECASE,
        )
        for m in compare_pat.finditer(sql):
            col = m.group(1)
            if col.upper() not in skip_kw:
                tokens.add(col.lower())

        # Bare column refs inside aggregates: COUNT(col), SUM(col), etc.
        agg_pat = re.compile(
            r"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?"
            r"([A-Za-z_]\w*)\s*\)",
            re.IGNORECASE,
        )
        for m in agg_pat.finditer(sql):
            col = m.group(1)
            if col.upper() not in skip_kw and col != "*":
                tokens.add(col.lower())

        return frozenset(tokens)

    @staticmethod
    def _candidates_differ_on_columns(successful: list[dict[str, Any]]) -> bool:
        """
        Return True when at least two successful candidates reference
        different column sets. Used to decide whether to inject the
        schema into the judge prompt.
        """
        if len(successful) < 2:
            return False
        col_sets = [SQLJudge._extract_columns_set(c.get("sql", "")) for c in successful]
        first = col_sets[0]
        return any(s != first for s in col_sets[1:])

    def _parse_judgment(self, response: str) -> dict[str, Any]:
        """
        Parse judge's JSON response.

        Args:
            response: Raw LLM response

        Returns:
            Parsed judgment dict
        """
        # Try to extract JSON from code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
        if json_match:
            response = json_match.group(1)

        # Try to find JSON object
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)

        return json.loads(response)

    def judge(
        self,
        candidates: list[Any],
        execution_results: list[Any],
        nl_query: str,
        evidence: str = None,
        schema_str: str = None,
    ) -> JudgmentResult:
        """
        Main entry point for judging.

        Evaluates all SUCCESSFUL candidates and selects the best one.

        Args:
            candidates: List of SQLCandidate objects
            execution_results: Results of executing each candidate
            nl_query: Original natural language question
            evidence: SME evidence/hints

        Returns:
            JudgmentResult with selected candidate and reasoning
        """
        # Build candidate-to-result map
        result_map = {r.candidate_id: r for r in execution_results}

        # Collect only successful candidates
        successful = []
        for candidate in candidates:
            result = result_map.get(candidate.candidate_id)
            if result and result.success:
                # Keep first 3 rows as samples so the judge can see the actual
                # output shape and compare empty vs non-empty candidates.
                sample_rows = (result.result or [])[:3]
                successful.append({
                    "candidate_id": candidate.candidate_id,
                    "sql": result.sql,  # Use executed SQL (may be refined)
                    "strategy": candidate.strategy_name,
                    "status": result.status,
                    "iterations": result.iterations,
                    "row_count": result.row_count,
                    "sample_rows": sample_rows,
                })

        total = len(candidates)
        num_successful = len(successful)

        # No successful candidates → return rejection
        if not successful:
            return JudgmentResult(
                selected_id=-1,
                selected_sql="",
                confidence=0.0,
                reasoning="All candidates failed execution after retry loops.",
                total_candidates=total,
                successful_candidates=0
            )

        # Only one successful candidate → select it directly
        if len(successful) == 1:
            c = successful[0]
            return JudgmentResult(
                selected_id=c["candidate_id"],
                selected_sql=c["sql"],
                confidence=0.8,
                reasoning=f"Only one candidate (Q{c['candidate_id']}, {c['strategy']}) passed execution.",
                total_candidates=total,
                successful_candidates=1
            )

        # Multiple successful candidates → use LLM judge
        if not self.llm_client:
            # No LLM → pick the first successful candidate
            c = successful[0]
            return JudgmentResult(
                selected_id=c["candidate_id"],
                selected_sql=c["sql"],
                confidence=0.5,
                reasoning=f"No LLM available. Selected first successful candidate (Q{c['candidate_id']}).",
                total_candidates=total,
                successful_candidates=num_successful
            )

        return self._llm_judge(
            successful=successful,
            nl_query=nl_query,
            evidence=evidence,
            total=total,
            schema_str=schema_str,
        )

    def _llm_judge(
        self,
        successful: list[dict[str, Any]],
        nl_query: str,
        evidence: str,
        total: int,
        schema_str: str = None,
    ) -> JudgmentResult:
        """
        Use LLM to judge and select the best candidate.

        Args:
            successful: List of successful candidate dicts
            nl_query: Original question
            evidence: SME evidence
            total: Total number of candidates

        Returns:
            JudgmentResult
        """
        # Format candidates for prompt
        candidates_str = self._format_candidates(successful)
        evidence_str = evidence if evidence else "No additional hints provided."

        # Conditionally inject schema when candidates differ on which columns
        # they use. This catches cases like q56 (schools.City vs schools.MailCity)
        # where the wrong column is more "internally consistent" than the right
        # one and would otherwise win on majority alone.
        schema_section = ""
        if schema_str and self._candidates_differ_on_columns(successful):
            schema_section = (
                "\nDatabase Schema (relevant tables — candidates DIFFER on "
                "column choice; use the descriptions below to pick the column "
                "whose meaning matches the question):\n"
                f"{schema_str}\n"
            )

        # Build prompt from template
        user_prompt = self.prompt_config["user_template"].format(
            question=nl_query,
            evidence=evidence_str,
            schema_section=schema_section,
            candidates=candidates_str,
        )

        try:
            response = self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=self.prompt_config["system"],
                max_tokens=512,
                temperature=0.0
            )

            result = self._parse_judgment(response)

            selected_id = int(result.get("selected_id", -1))
            selected_sql = result.get("selected_sql", "")
            confidence = float(result.get("confidence", 0.0))
            reasoning = result.get("reasoning", "")

            # Validate selected_id exists in successful candidates
            valid_ids = {c["candidate_id"] for c in successful}
            if selected_id not in valid_ids:
                # Fallback: pick the candidate whose SQL matches
                for c in successful:
                    if c["sql"].strip() == selected_sql.strip():
                        selected_id = c["candidate_id"]
                        break
                else:
                    # Last fallback: pick first successful
                    selected_id = successful[0]["candidate_id"]
                    selected_sql = successful[0]["sql"]
                    reasoning = f"LLM returned invalid ID. Fallback to Q{selected_id}. " + reasoning

            # Ensure selected_sql matches the candidate
            if not selected_sql:
                for c in successful:
                    if c["candidate_id"] == selected_id:
                        selected_sql = c["sql"]
                        break

            return JudgmentResult(
                selected_id=selected_id,
                selected_sql=selected_sql,
                confidence=confidence,
                reasoning=reasoning,
                total_candidates=total,
                successful_candidates=len(successful)
            )

        except AnthropicExhaustedError:
            # Terminal API failure — halt the batch rather than fall back to
            # "first successful candidate" which would ship a non-judged SQL.
            raise
        except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
            # LLM returned unparseable output → pick first successful candidate
            c = successful[0]
            return JudgmentResult(
                selected_id=c["candidate_id"],
                selected_sql=c["sql"],
                confidence=0.5,
                reasoning=f"LLM judge failed ({str(e)}). Selected first successful candidate.",
                total_candidates=total,
                successful_candidates=len(successful)
            )


def main():
    """Test the SQLJudge."""
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class MockCandidate:
        candidate_id: int
        sql: str
        strategy_name: str = ""

    @dataclass
    class MockResult:
        candidate_id: int
        sql: str
        success: bool
        status: str = "success"
        iterations: int = 1
        row_count: int = 0

    print("=" * 70)
    print("Testing SQLJudge (without LLM)")
    print("=" * 70)

    judge = SQLJudge(llm_client=None)

    # Test 1: No successful candidates
    print("\nTest 1: No successful candidates")
    candidates = [MockCandidate(1, "SELECT 1;", "full_schema")]
    results = [MockResult(1, "SELECT 1;", success=False, status="rejected")]
    judgment = judge.judge(candidates, results, "test query")
    print(f"  Selected: Q{judgment.selected_id}")
    print(f"  Reasoning: {judgment.reasoning}")

    # Test 2: One successful candidate
    print("\nTest 2: One successful candidate")
    candidates = [
        MockCandidate(1, "SELECT 1;", "full_schema"),
        MockCandidate(2, "SELECT 2;", "sme_metadata")
    ]
    results = [
        MockResult(1, "SELECT 1;", success=False, status="rejected"),
        MockResult(2, "SELECT 2;", success=True, status="success", row_count=5)
    ]
    judgment = judge.judge(candidates, results, "test query")
    print(f"  Selected: Q{judgment.selected_id}")
    print(f"  Confidence: {judgment.confidence}")
    print(f"  Reasoning: {judgment.reasoning}")

    # Test 3: Multiple successful candidates (no LLM → picks first)
    print("\nTest 3: Multiple successful candidates (no LLM)")
    candidates = [
        MockCandidate(1, "SELECT COUNT(*) FROM frpm;", "full_schema"),
        MockCandidate(2, "SELECT COUNT(*) FROM frpm WHERE x=1;", "sme_metadata"),
        MockCandidate(3, "SELECT COUNT(*) FROM frpm;", "minimal_profile"),
    ]
    results = [
        MockResult(1, "SELECT COUNT(*) FROM frpm;", success=True, row_count=1),
        MockResult(2, "SELECT COUNT(*) FROM frpm WHERE x=1;", success=True, row_count=1),
        MockResult(3, "SELECT COUNT(*) FROM frpm;", success=True, row_count=1),
    ]
    judgment = judge.judge(candidates, results, "How many records?")
    print(f"  Selected: Q{judgment.selected_id}")
    print(f"  Confidence: {judgment.confidence}")
    print(f"  Successful: {judgment.successful_candidates}/{judgment.total_candidates}")
    print(f"  Reasoning: {judgment.reasoning}")

    print("\n" + "=" * 70)
    print("With LLM, the judge evaluates all successful candidates and selects best.")
    print("=" * 70)


if __name__ == "__main__":
    main()
