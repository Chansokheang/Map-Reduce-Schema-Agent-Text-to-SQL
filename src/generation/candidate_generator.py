"""
Candidate Generator Module

Generates 5 SQL candidates using different context-aware strategies:
1. Full Schema - Complete database schema
2. SME Metadata - Schema with evidence/hints from domain experts
3. Minimal Profile - Schema with basic column descriptions
4. Focused Schema - Relevant tables only (from Map-Reduce Agent)
5. Full Profile - Focused schema + descriptions + evidence
"""

import re
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from .prompt_builder import PromptBuilder, ContextStrategy


@dataclass
class SQLCandidate:
    """A generated SQL candidate."""
    candidate_id: int
    sql: str
    strategy: ContextStrategy
    strategy_name: str = ""
    confidence_score: float = 0.0
    prompt_used: str = ""
    generation_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.strategy_name:
            self.strategy_name = self.strategy.value


class CandidateGenerator:
    """
    Generates SQL candidates using 5 context-aware strategies.

    Uses PromptBuilder to create strategy-specific prompts, then
    calls LLM to generate SQL for each strategy.
    """

    def __init__(self, llm_client: Any = None):
        """
        Initialize the Candidate Generator.

        Args:
            llm_client: LLM client for SQL generation
        """
        self.llm_client = llm_client
        self.prompt_builder = PromptBuilder()

    def _extract_sql(self, response: str) -> str:
        """
        Extract SQL from LLM response.

        Handles various response formats:
        - Raw SQL
        - SQL in code blocks (```sql ... ```)
        - SQL with explanations

        Args:
            response: Raw LLM response

        Returns:
            Cleaned SQL string
        """
        if not response:
            return ""

        # Try to extract from code blocks first
        sql_block = re.search(r'```(?:sql)?\s*([\s\S]*?)\s*```', response, re.IGNORECASE)
        if sql_block:
            return sql_block.group(1).strip()

        # Try to find SELECT statement
        select_match = re.search(
            r'(SELECT\s+[\s\S]*?)(?:;|\Z)',
            response,
            re.IGNORECASE
        )
        if select_match:
            sql = select_match.group(1).strip()
            # Add semicolon if missing
            if not sql.endswith(';'):
                sql += ';'
            return sql

        # Return cleaned response as fallback
        return response.strip()

    def generate_candidate(
        self,
        prompts: dict[str, str],
        strategy: ContextStrategy,
        candidate_id: int
    ) -> SQLCandidate:
        """
        Generate a single SQL candidate using the given prompts.

        Args:
            prompts: Dictionary with 'system' and 'user' prompts
            strategy: The generation strategy used
            candidate_id: Unique identifier for this candidate

        Returns:
            SQLCandidate with generated SQL
        """
        if not self.llm_client:
            return SQLCandidate(
                candidate_id=candidate_id,
                sql="-- Error: No LLM client available",
                strategy=strategy,
                confidence_score=0.0,
                prompt_used=prompts.get("user", ""),
                generation_metadata={"error": "No LLM client"}
            )

        try:
            # Call LLM with system and user prompts
            response = self.llm_client.complete(
                prompt=prompts["user"],
                system_prompt=prompts.get("system", ""),
                max_tokens=1024,
                temperature=0.0
            )

            # Extract SQL from response
            sql = self._extract_sql(response)

            return SQLCandidate(
                candidate_id=candidate_id,
                sql=sql,
                strategy=strategy,
                confidence_score=1.0 if sql and sql.upper().startswith("SELECT") else 0.5,
                prompt_used=prompts["user"],
                generation_metadata={
                    "strategy_note": prompts.get("strategy_note", ""),
                    "raw_response": response
                }
            )

        except Exception as e:
            return SQLCandidate(
                candidate_id=candidate_id,
                sql=f"-- Error: {str(e)}",
                strategy=strategy,
                confidence_score=0.0,
                prompt_used=prompts.get("user", ""),
                generation_metadata={"error": str(e)}
            )

    def generate_all_candidates(
        self,
        nl_query: str,
        schema: dict[str, Any],
        focused_schema: dict[str, Any] = None,
        profile: dict[str, Any] = None,
        evidence: str = None,
        parallel: bool = True
    ) -> list[SQLCandidate]:
        """
        Generate candidates using all 5 strategies.

        Args:
            nl_query: Natural language question
            schema: Full database schema
            focused_schema: Filtered schema from Map-Reduce Agent
            profile: Database profile with descriptions
            evidence: SME evidence/hints from BIRD dev.json
            parallel: Whether to generate candidates in parallel

        Returns:
            List of 5 SQLCandidate objects (Q1-Q5)
        """
        # Build prompts for all 5 strategies
        all_prompts = self.prompt_builder.build_all_strategies(
            nl_query=nl_query,
            schema=schema,
            focused_schema=focused_schema,
            profile=profile,
            evidence=evidence
        )

        # Strategy order matches the diagram: Q1-Q5
        strategy_order = [
            ContextStrategy.FULL_SCHEMA,      # Q1
            ContextStrategy.SME_METADATA,     # Q2
            ContextStrategy.MINIMAL_PROFILE,  # Q3
            ContextStrategy.FOCUSED_SCHEMA,   # Q4
            ContextStrategy.FULL_PROFILE,     # Q5
        ]

        candidates = []

        if parallel and self.llm_client:
            # Generate candidates in parallel
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for idx, strategy in enumerate(strategy_order):
                    prompts = all_prompts[strategy]
                    future = executor.submit(
                        self.generate_candidate,
                        prompts,
                        strategy,
                        idx + 1  # candidate_id: 1-5
                    )
                    futures.append(future)

                # Collect results in order
                for future in futures:
                    candidates.append(future.result())
        else:
            # Generate candidates sequentially
            for idx, strategy in enumerate(strategy_order):
                prompts = all_prompts[strategy]
                candidate = self.generate_candidate(
                    prompts=prompts,
                    strategy=strategy,
                    candidate_id=idx + 1
                )
                candidates.append(candidate)

        return candidates

    def generate(
        self,
        nl_query: str,
        schema: dict[str, Any],
        focused_schema: dict[str, Any] = None,
        profile: dict[str, Any] = None,
        evidence: str = None,
        strategies: list[ContextStrategy] = None
    ) -> list[SQLCandidate]:
        """
        Main entry point for candidate generation.

        Args:
            nl_query: Natural language question
            schema: Full database schema
            focused_schema: Optional filtered schema from Map-Reduce Agent
            profile: Optional database profile with descriptions
            evidence: Optional SME evidence/hints
            strategies: Optional list of specific strategies to use

        Returns:
            List of generated SQL candidates
        """
        if strategies is None:
            # Use all 5 strategies by default
            return self.generate_all_candidates(
                nl_query=nl_query,
                schema=schema,
                focused_schema=focused_schema,
                profile=profile,
                evidence=evidence
            )

        # Generate for specific strategies only
        candidates = []
        for idx, strategy in enumerate(strategies):
            prompts = self.prompt_builder.build(
                nl_query=nl_query,
                schema=schema,
                strategy=strategy,
                focused_schema=focused_schema,
                profile=profile,
                evidence=evidence
            )
            candidate = self.generate_candidate(
                prompts=prompts,
                strategy=strategy,
                candidate_id=idx + 1
            )
            candidates.append(candidate)

        return candidates


def main():
    """Test the CandidateGenerator."""
    # Sample data
    schema = {
        "frpm": {
            "table_readable_name": "free and reduced-price meals",
            "columns": [
                {"name": "CDSCode", "type": "TEXT"},
                {"name": "Charter School (Y/N)", "type": "INTEGER"},
                {"name": "District Name", "type": "TEXT"},
                {"name": "Free Meal Count (K-12)", "type": "INTEGER"},
                {"name": "Enrollment (K-12)", "type": "INTEGER"}
            ]
        },
        "schools": {
            "table_readable_name": "schools",
            "columns": [
                {"name": "CDSCode", "type": "TEXT"},
                {"name": "Zip", "type": "TEXT"},
                {"name": "School", "type": "TEXT"}
            ]
        }
    }

    focused_schema = {
        "tables": {
            "frpm": {
                "table_readable_name": "free and reduced-price meals",
                "relevance_score": 1.0,
                "reason": "Contains charter school and meal data",
                "columns": [
                    {"name": "CDSCode", "type": "TEXT"},
                    {"name": "Charter School (Y/N)", "type": "INTEGER"},
                    {"name": "Free Meal Count (K-12)", "type": "INTEGER"}
                ]
            }
        }
    }

    profile = {
        "tables": {
            "frpm": {
                "columns": [
                    {"name": "CDSCode", "description": "County-District-School code"},
                    {"name": "Charter School (Y/N)", "description": "1 if charter, 0 otherwise"},
                    {"name": "Free Meal Count (K-12)", "description": "Students eligible for free meals"}
                ]
            }
        }
    }

    evidence = "Charter schools refers to `Charter School (Y/N)` = 1"
    question = "How many charter schools are there?"

    print("=" * 70)
    print("Testing CandidateGenerator (without LLM)")
    print("=" * 70)

    # Test without LLM (will return error placeholders)
    generator = CandidateGenerator(llm_client=None)
    candidates = generator.generate_all_candidates(
        nl_query=question,
        schema=schema,
        focused_schema=focused_schema,
        profile=profile,
        evidence=evidence
    )

    for candidate in candidates:
        print(f"\nQ{candidate.candidate_id}: {candidate.strategy_name}")
        print(f"  SQL: {candidate.sql[:60]}...")
        print(f"  Confidence: {candidate.confidence_score}")

    print("\n" + "=" * 70)
    print("To test with LLM, set ANTHROPIC_API_KEY and use:")
    print("  from src.utils.llm_client import LLMClient")
    print("  generator = CandidateGenerator(llm_client=LLMClient())")
    print("=" * 70)


if __name__ == "__main__":
    main()
