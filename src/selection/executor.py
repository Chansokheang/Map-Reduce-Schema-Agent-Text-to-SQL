"""
SQL Executor Module

Handles SQL Execution & Refinement with retry loop:
- Execute SQL candidates against the database
- If failed: LLM refines the SQL and retries
- Max 3 iterations per candidate
- If still failing after 3 attempts: rejection
- If ALL candidates fail: Last Resort generation using errors as context
"""

import re
import time
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.prompt import LAST_RESORT_PROMPT


@dataclass
class ExecutionResult:
    """Result of SQL execution."""
    candidate_id: int
    sql: str
    success: bool
    result: list[Any] = None
    error: str = None
    execution_time_ms: float = 0.0
    row_count: int = 0
    iterations: int = 1
    refinement_history: list[dict] = field(default_factory=list)
    status: str = "success"  # "success", "refined", "rejected"


class SQLExecutor:
    """
    Executes SQL candidates with retry loop.

    From the proposed method diagram:
    - Execute each candidate SQL against the database
    - If failed and iter < 3: LLM refines SQL → retry
    - If failed and iter >= 3: rejection
    - If correct: pass to Judge
    """

    def __init__(
        self,
        db_path: Path = None,
        llm_client: Any = None,
        max_iterations: int = 3,
        query_timeout: float = 30.0
    ):
        """
        Initialize the SQL Executor.

        Args:
            db_path: Path to SQLite database
            llm_client: LLM client for SQL refinement
            max_iterations: Max retry attempts per candidate (default: 3)
            query_timeout: Query timeout in seconds (default: 30)
        """
        self.db_path = db_path
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.query_timeout = query_timeout

    def execute_sql(
        self,
        sql: str,
        db_path: Path = None
    ) -> tuple[bool, list[Any] | None, str | None]:
        """
        Execute a single SQL statement against the database.

        Args:
            sql: SQL query to execute
            db_path: Optional database path override

        Returns:
            Tuple of (success, results, error_message)
        """
        path = db_path or self.db_path
        if not path or not Path(path).exists():
            return False, None, f"Database not found: {path}"

        conn = None
        try:
            conn = sqlite3.connect(str(path), timeout=self.query_timeout)
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return True, rows, None

        except sqlite3.Error as e:
            return False, None, str(e)

        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"

        finally:
            if conn:
                conn.close()

    def refine_sql(
        self,
        sql: str,
        error: str,
        nl_query: str,
        schema_str: str = None,
        iteration: int = 1
    ) -> str:
        """
        Use LLM to refine failed SQL.

        Args:
            sql: SQL that produced an error
            error: Error message from execution
            nl_query: Original natural language question
            schema_str: Optional schema context
            iteration: Current iteration number

        Returns:
            Refined SQL string
        """
        if not self.llm_client:
            return sql

        schema_context = ""
        if schema_str:
            schema_context = f"\nDatabase Schema:\n{schema_str}\n"

        prompt = f"""The following SQL query failed with an error. Fix the SQL query.

Question: {nl_query}
{schema_context}
Failed SQL (attempt {iteration}/{self.max_iterations}):
{sql}

Error: {error}

Fix the SQL to resolve this error. Return ONLY the corrected SQL query, nothing else."""

        try:
            response = self.llm_client.complete(
                prompt=prompt,
                system_prompt="You are an SQL debugging expert. Fix the SQL query based on the error. Return ONLY the corrected SQL.",
                max_tokens=1024,
                temperature=0.0
            )
            return self._extract_sql(response) or sql

        except Exception:
            return sql

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response."""
        if not response:
            return ""

        # Try code blocks first
        sql_block = re.search(r'```(?:sql)?\s*([\s\S]*?)\s*```', response, re.IGNORECASE)
        if sql_block:
            return sql_block.group(1).strip()

        # Try to find SELECT statement
        select_match = re.search(r'(SELECT\s+[\s\S]*?)(?:;|\Z)', response, re.IGNORECASE)
        if select_match:
            sql = select_match.group(1).strip()
            if not sql.endswith(';'):
                sql += ';'
            return sql

        return response.strip()

    def execute_with_retry(
        self,
        candidate: Any,
        nl_query: str,
        db_path: Path = None,
        schema_str: str = None
    ) -> ExecutionResult:
        """
        Execute a SQL candidate with retry loop (max 3 iterations).

        From the diagram:
        1. Execute SQL
        2. If correct → return success
        3. If failed and iter < 3 → LLM refines → retry
        4. If failed and iter >= 3 → rejection

        Args:
            candidate: SQLCandidate to execute
            nl_query: Original natural language question
            db_path: Optional database path override
            schema_str: Optional schema context for refinement

        Returns:
            ExecutionResult with status
        """
        current_sql = candidate.sql
        refinement_history = []
        start_time = time.perf_counter()

        for iteration in range(1, self.max_iterations + 1):
            # Execute SQL
            success, rows, error = self.execute_sql(current_sql, db_path)

            if success:
                # Correct → return result
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                status = "success" if iteration == 1 else "refined"

                return ExecutionResult(
                    candidate_id=candidate.candidate_id,
                    sql=current_sql,
                    success=True,
                    result=rows,
                    execution_time_ms=elapsed_ms,
                    row_count=len(rows) if rows else 0,
                    iterations=iteration,
                    refinement_history=refinement_history,
                    status=status
                )

            # Failed → record this attempt
            refinement_history.append({
                "iteration": iteration,
                "sql": current_sql,
                "error": error
            })

            # If not the last iteration → refine and retry
            if iteration < self.max_iterations:
                current_sql = self.refine_sql(
                    sql=current_sql,
                    error=error,
                    nl_query=nl_query,
                    schema_str=schema_str,
                    iteration=iteration
                )

        # All iterations failed → rejection
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ExecutionResult(
            candidate_id=candidate.candidate_id,
            sql=current_sql,
            success=False,
            error=error,
            execution_time_ms=elapsed_ms,
            iterations=self.max_iterations,
            refinement_history=refinement_history,
            status="rejected"
        )

    def execute_all_candidates(
        self,
        candidates: list[Any],
        nl_query: str,
        db_path: Path = None,
        schema_str: str = None
    ) -> list[ExecutionResult]:
        """
        Execute all SQL candidates with retry loops.

        Args:
            candidates: List of SQLCandidate objects
            nl_query: Original natural language question
            db_path: Database path
            schema_str: Schema context for refinement

        Returns:
            List of ExecutionResult objects
        """
        results = []
        for candidate in candidates:
            result = self.execute_with_retry(
                candidate=candidate,
                nl_query=nl_query,
                db_path=db_path,
                schema_str=schema_str
            )
            results.append(result)
        return results

    def filter_successful(
        self,
        results: list[ExecutionResult]
    ) -> list[ExecutionResult]:
        """
        Filter to only successful executions.

        Args:
            results: All execution results

        Returns:
            List of successful results only
        """
        return [r for r in results if r.success]

    def _format_failed_attempts(
        self,
        results: list[ExecutionResult]
    ) -> str:
        """
        Format failed execution results for the Last Resort prompt.
        Only includes the LAST attempt from each candidate (best effort).

        Args:
            results: List of failed ExecutionResult objects

        Returns:
            Formatted string with last attempt per candidate
        """
        lines = []
        for r in results:
            lines.append(f"--- Candidate Q{r.candidate_id} ---")
            # Only show the last attempt (final best effort)
            if r.refinement_history:
                last = r.refinement_history[-1]
                lines.append(f"SQL: {last['sql']}")
                lines.append(f"Error: {last['error']}")
            else:
                lines.append(f"SQL: {r.sql}")
                lines.append(f"Error: {r.error}")
            lines.append("")
        return "\n".join(lines)

    def last_resort(
        self,
        results: list[ExecutionResult],
        nl_query: str,
        schema_str: str,
        evidence: str = None,
        db_path: Path = None
    ) -> ExecutionResult | None:
        """
        Last Resort: when ALL candidates fail, use their errors as context
        to generate one final corrected SQL query.

        This is triggered ONLY when every candidate has been rejected
        after exhausting all retry iterations.

        Args:
            results: List of all failed ExecutionResult objects
            nl_query: Original natural language question
            schema_str: Database schema string
            evidence: SME evidence from dev.json
            db_path: Optional database path override

        Returns:
            ExecutionResult if generation succeeds, None if LLM unavailable
        """
        if not self.llm_client:
            return None

        failed_results = [r for r in results if not r.success]
        if not failed_results:
            return None

        # Format all failed attempts
        failed_attempts_str = self._format_failed_attempts(failed_results)
        evidence_str = evidence if evidence else "No additional hints provided."

        # Build prompt from LAST_RESORT_PROMPT template
        user_prompt = LAST_RESORT_PROMPT["user_template"].format(
            question=nl_query,
            evidence=evidence_str,
            schema=schema_str,
            failed_attempts=failed_attempts_str,
            num_failed=len(failed_results)
        )

        start_time = time.perf_counter()

        try:
            response = self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=LAST_RESORT_PROMPT["system"],
                max_tokens=1024,
                temperature=0.0
            )

            last_resort_sql = self._extract_sql(response)
            if not last_resort_sql:
                return None

            # Execute the last resort SQL
            success, rows, error = self.execute_sql(last_resort_sql, db_path)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ExecutionResult(
                candidate_id=0,  # Special ID for last resort
                sql=last_resort_sql,
                success=success,
                result=rows if success else None,
                error=error if not success else None,
                execution_time_ms=elapsed_ms,
                row_count=len(rows) if success and rows else 0,
                iterations=1,
                refinement_history=[{
                    "source": "last_resort",
                    "num_failed_candidates": len(failed_results)
                }],
                status="last_resort" if success else "rejected"
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                candidate_id=0,
                sql="",
                success=False,
                error=f"Last resort generation failed: {str(e)}",
                execution_time_ms=elapsed_ms,
                iterations=1,
                status="rejected"
            )


def main():
    """Test the SQLExecutor."""
    from dataclasses import dataclass

    @dataclass
    class MockCandidate:
        candidate_id: int
        sql: str

    print("=" * 70)
    print("Testing SQLExecutor")
    print("=" * 70)

    # Test without a real database
    executor = SQLExecutor(
        db_path=None,
        llm_client=None,
        max_iterations=3
    )

    # Test with mock candidates
    candidates = [
        MockCandidate(1, "SELECT COUNT(*) FROM frpm WHERE `Charter School (Y/N)` = 1;"),
        MockCandidate(2, "INVALID SQL QUERY;"),
        MockCandidate(3, "SELECT * FROM nonexistent_table;"),
    ]

    print("\nTest 1: execute_sql without database")
    success, rows, error = executor.execute_sql("SELECT 1;")
    print(f"  Success: {success}, Error: {error}")

    print("\nTest 2: execute_with_retry (no database)")
    for c in candidates:
        result = executor.execute_with_retry(
            candidate=c,
            nl_query="How many charter schools?"
        )
        print(f"  Q{result.candidate_id}: status={result.status}, "
              f"iterations={result.iterations}, error={result.error}")

    print("\nTest 3: last_resort (no LLM → returns None)")
    all_results = executor.execute_all_candidates(
        candidates=candidates,
        nl_query="How many charter schools?"
    )
    failed = [r for r in all_results if not r.success]
    print(f"  Failed candidates: {len(failed)}")
    last_resort_result = executor.last_resort(
        results=all_results,
        nl_query="How many charter schools?",
        schema_str="CREATE TABLE frpm (...);"
    )
    print(f"  Last resort result: {last_resort_result}")
    print("  (None because no LLM client is available)")

    print("\n" + "=" * 70)
    print("Pipeline flow:")
    print("  1. execute_all_candidates() → run all 5 with retry loops")
    print("  2. filter_successful() → check if any passed")
    print("  3. If ALL failed → last_resort() → one final attempt")
    print("  4. Pass successful results → Judge for selection")
    print("=" * 70)


if __name__ == "__main__":
    main()
