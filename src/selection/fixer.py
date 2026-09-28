"""
SQL Fixer Module

Post-execution review stage. After a candidate's SQL has executed
successfully, the fixer LLM reviews the SQL + result sample and either
approves or proposes a refined SQL. The refined SQL is then re-executed.
The cycle repeats up to `max_iterations` times (default 3).

Catches:
- Evidence/hint not applied (e.g., "X refers to <Col>" but candidate didn't use <Col>)
- Wrong result shape (singular question but many rows; "list X and Y" but 1 column)
- Missing DISTINCT when result has obvious duplicates
- Padded date for year-only filter (use STRFTIME)
- Bare-column CAST in WHERE/ORDER BY
- GROUP_CONCAT / UNION misuse in SELECT
- NULL leaks in single-row answers
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.prompt import FIXER_PROMPT
from src.utils.llm_client import AnthropicExhaustedError


@dataclass
class FixerOutcome:
    """Result of fixing one candidate."""
    candidate_id: int
    is_acceptable: bool
    issues: list[str]
    final_sql: str
    final_rows: list[Any] | None
    final_row_count: int
    iterations: int  # how many fix-refine cycles ran
    refined: bool  # True if SQL was changed from the input


class SQLFixer:
    """
    Post-execution fixer. Operates on candidates that already passed
    `SQLExecutor.execute_with_retry` (so they ran successfully).
    """

    SAMPLE_LIMIT = 15

    def __init__(
        self,
        llm_client: Any = None,
        max_iterations: int = 3,
        query_timeout: float = 30.0,
    ):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.query_timeout = query_timeout
        self.prompt_config = FIXER_PROMPT

    @staticmethod
    def _has_duplicate_rows(rows: list[Any]) -> bool:
        if not rows:
            return False
        try:
            return len(rows) != len({tuple(r) for r in rows})
        except TypeError:
            return len(rows) != len({repr(r) for r in rows})

    @staticmethod
    def _has_null_values(rows: list[Any]) -> bool:
        if not rows:
            return False
        for r in rows:
            try:
                if any(v is None for v in r):
                    return True
            except TypeError:
                if r is None:
                    return True
        return False

    @staticmethod
    def _format_sample(rows: list[Any], limit: int) -> str:
        if not rows:
            return "(empty result)"
        sample = rows[:limit]
        lines = []
        for i, r in enumerate(sample, 1):
            row_str = repr(r)
            if len(row_str) > 200:
                row_str = row_str[:200] + "..."
            lines.append(f"  {i}. {row_str}")
        if len(rows) > limit:
            lines.append(f"  ... (+{len(rows) - limit} more rows not shown)")
        return "\n".join(lines)

    @staticmethod
    def _column_count(rows: list[Any]) -> int:
        if not rows:
            return 0
        try:
            return len(rows[0])
        except TypeError:
            return 1

    def _execute_sql(self, sql: str, db_path: Path) -> tuple[bool, list[Any] | None, str | None]:
        if not db_path or not Path(db_path).exists():
            return False, None, f"Database not found: {db_path}"
        conn = None
        try:
            conn = sqlite3.connect(str(db_path), timeout=self.query_timeout)
            cur = conn.cursor()
            cur.execute(sql)
            return True, cur.fetchall(), None
        except sqlite3.Error as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, f"Unexpected: {e}"
        finally:
            if conn:
                conn.close()

    @staticmethod
    def _extract_sql(response: str) -> str:
        if not response:
            return ""
        m = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", response, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"(SELECT\s+[\s\S]*?)(?:;|\Z)", response, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(";")
        return response.strip()

    def _parse_verdict(self, response: str) -> dict[str, Any]:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        if m:
            response = m.group(1)
        m = re.search(r"\{[\s\S]*\}", response)
        if m:
            response = m.group(0)
        return json.loads(response)

    def _llm_review(
        self,
        sql: str,
        rows: list[Any],
        nl_query: str,
        evidence: str,
        schema_str: str = "",
    ) -> dict[str, Any]:
        row_count = len(rows) if rows else 0
        col_count = self._column_count(rows)
        has_dups = self._has_duplicate_rows(rows or [])
        has_nulls = self._has_null_values(rows or [])
        sample_str = self._format_sample(rows or [], self.SAMPLE_LIMIT)

        user_prompt = self.prompt_config["user_template"].format(
            question=nl_query,
            evidence=evidence or "No additional hints provided.",
            schema=schema_str or "(schema not provided)",
            sql=sql,
            row_count=row_count,
            column_count=col_count,
            has_duplicates=has_dups,
            has_nulls=has_nulls,
            sample_rows=sample_str,
        )

        response = self.llm_client.complete(
            prompt=user_prompt,
            system_prompt=self.prompt_config["system"],
            max_tokens=1024,
            temperature=0.0,
        )
        return self._parse_verdict(response)

    def fix(
        self,
        candidate: Any,
        execution_result: Any,
        nl_query: str,
        evidence: str,
        db_path: Path,
        schema_str: str = "",
    ) -> FixerOutcome:
        """
        Run the review-and-refine loop on one candidate.

        On each iteration:
          1. Fixer reviews (sql, rows, question, evidence)
          2. If is_acceptable → STOP, return PASS
          3. Else: execute the refined SQL
             - If success: replace sql/rows, loop again (up to max_iterations)
             - If failure: stop with current sql/rows
        """
        if not self.llm_client:
            return FixerOutcome(
                candidate_id=candidate.candidate_id,
                is_acceptable=True,
                issues=[],
                final_sql=execution_result.sql,
                final_rows=execution_result.result,
                final_row_count=execution_result.row_count,
                iterations=0,
                refined=False,
            )

        current_sql = execution_result.sql
        current_rows = execution_result.result or []
        all_issues: list[str] = []
        refined = False

        for it in range(1, self.max_iterations + 1):
            try:
                verdict = self._llm_review(
                    sql=current_sql,
                    rows=current_rows,
                    nl_query=nl_query,
                    evidence=evidence,
                    schema_str=schema_str,
                )
            except AnthropicExhaustedError:
                raise
            except Exception as e:
                # Fixer LLM failed — accept candidate as-is, but record why.
                all_issues.append(f"fixer_error: {type(e).__name__}: {e}")
                return FixerOutcome(
                    candidate_id=candidate.candidate_id,
                    is_acceptable=True,
                    issues=all_issues,
                    final_sql=current_sql,
                    final_rows=current_rows,
                    final_row_count=len(current_rows),
                    iterations=it - 1,
                    refined=refined,
                )

            is_ok = bool(verdict.get("is_acceptable", True))
            issues = verdict.get("issues") or []
            if isinstance(issues, list):
                all_issues.extend(str(x) for x in issues)
            if is_ok:
                return FixerOutcome(
                    candidate_id=candidate.candidate_id,
                    is_acceptable=True,
                    issues=all_issues,
                    final_sql=current_sql,
                    final_rows=current_rows,
                    final_row_count=len(current_rows),
                    iterations=it,
                    refined=refined,
                )

            refined_sql_raw = verdict.get("refined_sql") or ""
            refined_sql = self._extract_sql(refined_sql_raw)
            if not refined_sql or refined_sql.strip() == current_sql.strip():
                return FixerOutcome(
                    candidate_id=candidate.candidate_id,
                    is_acceptable=False,
                    issues=all_issues,
                    final_sql=current_sql,
                    final_rows=current_rows,
                    final_row_count=len(current_rows),
                    iterations=it,
                    refined=refined,
                )

            success, new_rows, _err = self._execute_sql(refined_sql, db_path)
            if not success:
                # Refined SQL didn't run — keep the previous version.
                return FixerOutcome(
                    candidate_id=candidate.candidate_id,
                    is_acceptable=False,
                    issues=all_issues,
                    final_sql=current_sql,
                    final_rows=current_rows,
                    final_row_count=len(current_rows),
                    iterations=it,
                    refined=refined,
                )

            current_sql = refined_sql
            current_rows = new_rows or []
            refined = True
            # Loop continues to re-review the refined SQL.

        return FixerOutcome(
            candidate_id=candidate.candidate_id,
            is_acceptable=False,
            issues=all_issues,
            final_sql=current_sql,
            final_rows=current_rows,
            final_row_count=len(current_rows),
            iterations=self.max_iterations,
            refined=refined,
        )
