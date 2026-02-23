"""
================================================================================
QA-SQL PIPELINE - USAGE INSTRUCTIONS
================================================================================


# EXAMPLE USAGE (after implementation):
#   python -m src.pipeline -q 0                           # Run question 0
#   python -m src.pipeline -q 5 -m claude-3-5-haiku-20241022  # Use Haiku model
#   python -m src.pipeline -b --range 0 10               # Batch: questions 0-9
#   python -m src.pipeline -d /path/to/data -o /path/to/output   # python -m src.pipeline -b --range 0 10 -d ./data/bird_data -o ./output/ver1
#   python -m src.pipeline --threshold 0.6 --timeout 60

================================================================================
Main Pipeline Orchestrator

Orchestrates the full QA-SQL workflow:
1. Input Processing - Load schema and profile for the target database
2. Schema Agent - Map-Reduce decomposition and table relevance verification
3. Candidate Generation - Generate 5 SQL candidates using context-aware strategies
4. SQL Execution - Execute candidates with retry loops, last resort if all fail
5. LLM Judge - Select the best SQL candidate
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .processing import InputProcessor, ProcessedInput
from .agents import SchemaManager
from .generation import CandidateGenerator, SQLCandidate, PromptBuilder
from .selection import SQLExecutor, SQLJudge, ExecutionResult, JudgmentResult
from .utils import LLMClient, Config


@dataclass
class PipelineResult:
    """Result of the full pipeline execution."""
    nl_query: str
    database_name: str
    generated_sql: str
    confidence: float
    all_candidates: list[SQLCandidate]
    execution_results: list[ExecutionResult]
    judgment: JudgmentResult
    focused_schema: dict[str, Any] = None
    evidence: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class QASQLPipeline:
    """
    Main pipeline for Query Augmentation to SQL.

    Orchestrates all components:
    NL Query → Schema Agent → 5 Candidates → Execute & Refine → Last Resort → Judge → Final SQL
    """

    def __init__(self, config: Config = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or Config()
        self.llm_client = None
        self.input_processor = None
        self.schema_manager = None
        self.candidate_generator = None
        self.prompt_builder = None
        self.sql_judge = None
        self._initialized = False
        self._run_index = 0  # Tracks question index for BIRD output format

    def initialize(self):
        """Initialize all pipeline components."""
        self.llm_client = LLMClient(model=self.config.llm_model)

        self.input_processor = InputProcessor(
            schema_dir=self.config.schema_dir,
            profile_dir=self.config.profile_dir
        )

        self.schema_manager = SchemaManager(
            llm_client=self.llm_client,
            max_workers=self.config.max_workers
        )

        self.candidate_generator = CandidateGenerator(
            llm_client=self.llm_client
        )

        self.prompt_builder = PromptBuilder()

        self.sql_judge = SQLJudge(
            llm_client=self.llm_client
        )

        self._initialized = True

    def _save_results(self, result: PipelineResult, run_idx: int = 0):
        """
        Save candidate SQL and selected SQL in BIRD benchmark format.

        Format per entry: "idx": "SQL\t----- bird -----\tdb_id"

        Creates 6 JSON files (each accumulates entries across runs):
        - candidate_full_schema.json (Q1)
        - candidate_sme_metadata.json (Q2)
        - candidate_minimal_profile.json (Q3)
        - candidate_focused_schema.json (Q4)
        - candidate_full_profile.json (Q5)
        - selected.json (judge's pick)
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        idx = str(run_idx)
        db_name = result.database_name

        # Map candidate_id → output filename
        strategy_files = {
            1: "candidate_full_schema.json",
            2: "candidate_sme_metadata.json",
            3: "candidate_minimal_profile.json",
            4: "candidate_focused_schema.json",
            5: "candidate_full_profile.json",
        }

        # Use executed SQL (may have been refined) if available
        result_map = {r.candidate_id: r for r in result.execution_results}

        for cid, filename in strategy_files.items():
            exec_result = result_map.get(cid)
            if exec_result:
                sql = exec_result.sql
            else:
                matching = [c for c in result.all_candidates if c.candidate_id == cid]
                sql = matching[0].sql if matching else ""

            sql = sql.strip()
            self._append_bird_entry(output_dir / filename, idx, sql, db_name)

        # Save selected SQL
        selected_sql = result.generated_sql.strip()
        self._append_bird_entry(output_dir / "selected.json", idx, selected_sql, db_name)

        # Save agent outputs
        self._save_agent_outputs(result)

        # Save timing data
        self._save_timings(result)

        self._run_index += 1

    def _append_bird_entry(self, filepath: Path, idx: str, sql: str, db_name: str):
        """
        Append one entry to a BIRD-format JSON file.

        Format: {"0": "SQL\\t----- bird -----\\tdb_id", "1": ...}
        """
        # Load existing data or start fresh
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        entry = f"{sql}\t----- bird -----\t{db_name}"
        data[idx] = entry

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _save_agent_outputs(self, result: PipelineResult):
        """
        Save Agentic Decomposition and Map-Reduce Schema Agent outputs.

        Creates 2 files (append mode, one JSON per line):
        - agentic_decomposition.jsonl — decomposed query components
        - schema_agent_output.jsonl — focused schema with table relevances
        """
        output_dir = Path(self.config.output_dir)
        focused = result.focused_schema or {}
        metadata = focused.get("metadata", {})

        # File 1: Agentic Decomposition output
        decomposition = {
            "question": result.nl_query,
            "database": result.database_name,
            "components": metadata.get("decomposed_components", [])
        }
        with open(output_dir / "agentic_decomposition.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(decomposition, ensure_ascii=False) + "\n")

        # File 2: Map-Reduce Schema Agent output
        schema_output = {
            "question": result.nl_query,
            "database": result.database_name,
            "total_tables_evaluated": metadata.get("total_tables_evaluated", 0),
            "relevant_tables_count": metadata.get("relevant_tables_count", 0),
            "relevance_threshold": metadata.get("relevance_threshold", 0.5),
            "table_relevances": focused.get("table_relevances", [])
        }
        with open(output_dir / "schema_agent_output.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(schema_output, ensure_ascii=False) + "\n")

    def _save_timings(self, result: PipelineResult):
        """
        Save per-run timing breakdown to a JSONL file.

        Each line contains timing data for one pipeline run:
        - question, database, and per-stage millisecond timings
        """
        output_dir = Path(self.config.output_dir)
        timings = result.metadata.get("timings", {})

        timing_entry = {
            "question": result.nl_query,
            "database": result.database_name,
            "input_processing_ms": round(timings.get("input_processing_ms", 0)),
            "schema_agent_ms": round(timings.get("schema_agent_ms", 0)),
            "candidate_generation_ms": round(timings.get("candidate_generation_ms", 0)),
            "execution_ms": round(timings.get("execution_ms", 0)),
            "last_resort_ms": round(timings.get("last_resort_ms", 0)),
            "judge_ms": round(timings.get("judge_ms", 0)),
            "total_ms": round(timings.get("total_ms", 0)),
        }

        with open(output_dir / "timings.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(timing_entry, ensure_ascii=False) + "\n")

    def _resolve_db_path(self, database_name: str, db_path: Path = None) -> Path:
        """
        Resolve the database file path.

        Args:
            database_name: Name of the database
            db_path: Optional explicit path

        Returns:
            Path to the SQLite database file
        """
        if db_path:
            return Path(db_path)

        # Auto-resolve from BIRD data directory
        return self.config.data_dir / "dev_databases" / database_name / f"{database_name}.sqlite"

    def run(
        self,
        nl_query: str,
        database_name: str,
        db_path: Path = None,
        evidence: str = "",
        question_index: int = None
    ) -> PipelineResult:
        """
        Run the full pipeline for a single query.

        Flow:
        1. Load schema + profile
        2. Map-Reduce Schema Agent → focused schema
        3. Generate 5 SQL candidates (parallel)
        4. Execute all candidates with retry loops
        5. If ALL failed → Last Resort generation
        6. LLM Judge → select best candidate

        Args:
            nl_query: Natural language query
            database_name: Target database name (e.g., "california_schools")
            db_path: Optional explicit path to database file
            evidence: SME evidence/hints from BIRD dev.json
            question_index: Explicit question index for BIRD output format.
                           If None, uses internal auto-increment counter.

        Returns:
            PipelineResult with generated SQL and metadata
        """
        if not self._initialized:
            self.initialize()

        pipeline_start = time.perf_counter()
        metadata = {"timings": {}}

        # --- Stage 1: Load inputs ---
        t0 = time.perf_counter()
        processed = self.input_processor.process(nl_query, database_name)
        schema = processed.schema
        profile = processed.profile
        metadata["timings"]["input_processing_ms"] = (time.perf_counter() - t0) * 1000

        # Resolve database path
        resolved_db_path = self._resolve_db_path(database_name, db_path)

        # --- Stage 2: Map-Reduce Schema Agent ---
        t0 = time.perf_counter()
        focused_schema = self.schema_manager.run(
            nl_query=nl_query,
            schema=schema,
            profile=profile,
            relevance_threshold=self.config.relevance_threshold
        )
        metadata["timings"]["schema_agent_ms"] = (time.perf_counter() - t0) * 1000
        metadata["relevant_tables"] = list(focused_schema.get("tables", {}).keys())

        # --- Stage 3: Generate 5 SQL candidates ---
        t0 = time.perf_counter()
        candidates = self.candidate_generator.generate_all_candidates(
            nl_query=nl_query,
            schema=schema,
            focused_schema=focused_schema,
            profile=profile,
            evidence=evidence,
            parallel=True
        )
        metadata["timings"]["candidate_generation_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 4: Execute all candidates with retry loops ---
        t0 = time.perf_counter()
        schema_str = self.prompt_builder.format_full_schema(schema)

        executor = SQLExecutor(
            db_path=resolved_db_path,
            llm_client=self.llm_client,
            max_iterations=3,
            query_timeout=self.config.query_timeout
        )

        execution_results = executor.execute_all_candidates(
            candidates=candidates,
            nl_query=nl_query,
            db_path=resolved_db_path,
            schema_str=schema_str
        )
        metadata["timings"]["execution_ms"] = (time.perf_counter() - t0) * 1000

        # Execution summary
        successful = executor.filter_successful(execution_results)
        metadata["execution_summary"] = {
            "total": len(execution_results),
            "successful": len(successful),
            "rejected": len(execution_results) - len(successful)
        }

        # --- Stage 4b: Last Resort if ALL failed ---
        if not successful:
            t0 = time.perf_counter()
            last_resort_result = executor.last_resort(
                results=execution_results,
                nl_query=nl_query,
                schema_str=schema_str,
                evidence=evidence,
                db_path=resolved_db_path
            )
            metadata["timings"]["last_resort_ms"] = (time.perf_counter() - t0) * 1000

            if last_resort_result:
                execution_results.append(last_resort_result)
                metadata["last_resort_used"] = True
                metadata["last_resort_success"] = last_resort_result.success

                # Create a synthetic candidate so Judge can process it
                if last_resort_result.success:
                    from .generation.prompt_builder import ContextStrategy
                    last_resort_candidate = SQLCandidate(
                        candidate_id=0,
                        sql=last_resort_result.sql,
                        strategy=ContextStrategy.FULL_SCHEMA,
                        strategy_name="last_resort"
                    )
                    candidates.append(last_resort_candidate)
        else:
            metadata["last_resort_used"] = False

        # --- Stage 5: LLM Judge → select best ---
        t0 = time.perf_counter()
        judgment = self.sql_judge.judge(
            candidates=candidates,
            execution_results=execution_results,
            nl_query=nl_query,
            evidence=evidence
        )
        metadata["timings"]["judge_ms"] = (time.perf_counter() - t0) * 1000

        # Total pipeline time
        metadata["timings"]["total_ms"] = (time.perf_counter() - pipeline_start) * 1000

        result = PipelineResult(
            nl_query=nl_query,
            database_name=database_name,
            generated_sql=judgment.selected_sql,
            confidence=judgment.confidence,
            all_candidates=candidates,
            execution_results=execution_results,
            judgment=judgment,
            focused_schema=focused_schema,
            evidence=evidence,
            metadata=metadata
        )

        # Resolve question index: explicit > auto-increment
        run_idx = question_index if question_index is not None else self._run_index

        # Save candidates and selected SQL to output files
        self._save_results(result, run_idx)

        return result

    def run_batch(
        self,
        queries: list[dict[str, str]],
        db_path: Path = None,
        start_index: int = 0
    ) -> list[PipelineResult]:
        """
        Run pipeline on multiple queries.

        Args:
            queries: List of dicts with keys: question, db_id, evidence
            db_path: Optional database path override
            start_index: Starting question index for output file naming (default: 0)

        Returns:
            List of PipelineResult objects
        """
        if not self._initialized:
            self.initialize()

        results = []
        total = len(queries)

        for idx, query in enumerate(queries):
            nl_query = query.get("question", "")
            database_name = query.get("db_id", "")
            evidence = query.get("evidence", "")

            # Use absolute question index (start_index + local idx)
            absolute_idx = start_index + idx

            print(f"[{idx + 1}/{total}] Processing Q{absolute_idx}: {nl_query[:70]}...")

            try:
                result = self.run(
                    nl_query=nl_query,
                    database_name=database_name,
                    db_path=db_path,
                    evidence=evidence,
                    question_index=absolute_idx
                )
                results.append(result)

                status = "OK" if result.generated_sql else "FAILED"
                print(f"  [{status}] Confidence: {result.confidence:.2f} "
                      f"| Candidates: {result.judgment.successful_candidates}/{result.judgment.total_candidates}")

            except Exception as e:
                print(f"  [ERROR] {str(e)}")
                # Create a failed result
                results.append(PipelineResult(
                    nl_query=nl_query,
                    database_name=database_name,
                    generated_sql="",
                    confidence=0.0,
                    all_candidates=[],
                    execution_results=[],
                    judgment=JudgmentResult(
                        selected_id=-1,
                        selected_sql="",
                        confidence=0.0,
                        reasoning=f"Pipeline error: {str(e)}",
                        total_candidates=0,
                        successful_candidates=0
                    ),
                    evidence=evidence,
                    metadata={"error": str(e)}
                ))

        return results


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="QA-SQL Pipeline - Convert natural language to SQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline -q 0                              # Run question 0
  python -m src.pipeline -q 5 -m claude-3-5-haiku-20241022 # Use Haiku model
  python -m src.pipeline -b --range 0 10                   # Batch: questions 0-9
  python -m src.pipeline -d ./data/bird_data -o ./output/ver1
  python -m src.pipeline --threshold 0.6 --timeout 60
        """
    )

    # Basic options
    parser.add_argument(
        "-q", "--question",
        type=int,
        default=0,
        help="Question index from dev.json (default: 0)"
    )
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        default=None,
        help="Path to BIRD data directory (default: data/bird_data)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Path to output directory (default: output/ver1)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Direct path to SQLite database file"
    )

    # Model options
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="LLM model to use (default: claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="Relevance threshold for schema agent (default: 0.5)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="SQL query timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel workers for schema agent (default: 4)"
    )

    # Batch mode options
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Enable batch mode (process multiple questions)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for batch mode (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for batch mode (default: all)"
    )
    parser.add_argument(
        "--range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Process questions in range [START, END)"
    )

    # Other options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args()


def main():
    """Run the pipeline on questions from BIRD dev.json."""
    import sys

    args = parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else base_dir / "data" / "bird_data"
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "output" / "ver1"
    dev_json_path = data_dir / "dev.json"

    # Load dev.json
    if not dev_json_path.exists():
        print(f"dev.json not found at: {dev_json_path}")
        sys.exit(1)

    with open(dev_json_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    # Create config
    config = Config(
        data_dir=data_dir,
        output_dir=output_dir,
        llm_model=args.model,
        relevance_threshold=args.threshold,
        query_timeout=args.timeout,
        max_workers=args.max_workers
    )

    pipeline = QASQLPipeline(config=config)

    # Determine which questions to process
    if args.batch:
        # Batch mode
        if args.range:
            start_idx, end_idx = args.range
        else:
            start_idx = args.start
            end_idx = args.end if args.end is not None else len(dev_data)

        # Validate range
        start_idx = max(0, start_idx)
        end_idx = min(len(dev_data), end_idx)

        if start_idx >= end_idx:
            print(f"Invalid range: [{start_idx}, {end_idx})")
            sys.exit(1)

        print("=" * 70)
        print("QA-SQL Pipeline - BATCH MODE")
        print("=" * 70)
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Model: {args.model}")
        print(f"Questions: [{start_idx}, {end_idx}) ({end_idx - start_idx} total)")
        print("=" * 70)

        # Run batch
        questions = dev_data[start_idx:end_idx]
        results = pipeline.run_batch(
            queries=questions,
            db_path=Path(args.db_path) if args.db_path else None,
            start_index=start_idx  # Pass absolute starting index for correct output naming
        )

        # Summary
        successful = sum(1 for r in results if r.generated_sql)
        print(f"\n{'=' * 70}")
        print("BATCH COMPLETE")
        print("=" * 70)
        print(f"Processed: {len(results)} questions")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Results saved to: {output_dir}")
        print("=" * 70)

    else:
        # Single question mode
        question_idx = args.question
        if question_idx >= len(dev_data):
            print(f"Question index {question_idx} out of range (max: {len(dev_data) - 1})")
            sys.exit(1)

        entry = dev_data[question_idx]

        print("=" * 70)
        print("QA-SQL Pipeline")
        print("=" * 70)
        print(f"Question #{entry['question_id']}: {entry['question']}")
        print(f"Database: {entry['db_id']}")
        print(f"Evidence: {entry.get('evidence', '(none)')}")
        print(f"Difficulty: {entry.get('difficulty', 'unknown')}")
        if args.verbose:
            print(f"Data dir: {data_dir}")
            print(f"Output dir: {output_dir}")
            print(f"Model: {args.model}")
        print("=" * 70)

        # Run pipeline
        result = pipeline.run(
            nl_query=entry["question"],
            database_name=entry["db_id"],
            db_path=Path(args.db_path) if args.db_path else None,
            evidence=entry.get("evidence", ""),
            question_index=question_idx
        )

        # Display results
        print(f"\n{'=' * 70}")
        print("RESULT")
        print("=" * 70)
        print(f"Generated SQL: {result.generated_sql}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.judgment.reasoning}")
        print(f"\nCandidates: {result.judgment.successful_candidates}/{result.judgment.total_candidates} successful")

        if result.metadata.get("last_resort_used"):
            print(f"Last Resort: used (success={result.metadata.get('last_resort_success')})")

        # Show timings
        timings = result.metadata.get("timings", {})
        print(f"\nTimings:")
        for stage, ms in timings.items():
            print(f"  {stage}: {ms:.0f}ms")

        # Compare with ground truth if available
        if "SQL" in entry:
            print(f"\nGround Truth SQL: {entry['SQL']}")

        print("=" * 70)


if __name__ == "__main__":
    main()
