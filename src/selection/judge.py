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

        Args:
            candidates: List of candidate dicts with id, sql, strategy

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

            header = f"Option {candidate_id} ({strategy})"
            if status == "refined":
                header += f" [refined after {iterations} attempts]"

            lines.append(f"{header}:")
            lines.append(f"{sql}")
            lines.append("")

        return "\n".join(lines)

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
        evidence: str = None
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
                successful.append({
                    "candidate_id": candidate.candidate_id,
                    "sql": result.sql,  # Use executed SQL (may be refined)
                    "strategy": candidate.strategy_name,
                    "status": result.status,
                    "iterations": result.iterations,
                    "row_count": result.row_count
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
            total=total
        )

    def _llm_judge(
        self,
        successful: list[dict[str, Any]],
        nl_query: str,
        evidence: str,
        total: int
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

        # Build prompt from template
        user_prompt = self.prompt_config["user_template"].format(
            question=nl_query,
            evidence=evidence_str,
            candidates=candidates_str
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

        except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
            # LLM failed → pick first successful candidate
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
