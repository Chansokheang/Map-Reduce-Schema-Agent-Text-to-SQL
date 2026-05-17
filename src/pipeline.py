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

import re

from .processing import InputProcessor, ProcessedInput
from .agents import SchemaManager
from .generation import CandidateGenerator, SQLCandidate, PromptBuilder
from .selection import SQLExecutor, SQLJudge, ExecutionResult, JudgmentResult, SQLFixer
from .utils import Config
from .utils.llm_client import create_llm_client, AnthropicExhaustedError, AnthropicRefusalError


def preprocess_evidence(evidence: str) -> str:
    """
    Preprocess evidence to replace backticks with square brackets.

    This is needed because Claude Code CLI interprets backticks as markdown
    and strips them, breaking column references in evidence like:
      `Free Meal Count (K-12)` / `Enrollment (K-12)`

    Converts to:
      [Free Meal Count (K-12)] / [Enrollment (K-12)]

    Args:
        evidence: Raw evidence string from BIRD dev.json

    Returns:
        Preprocessed evidence with backticks replaced by square brackets
    """
    if not evidence:
        return evidence

    # Replace `column name` with [column name]
    # Pattern matches backtick-enclosed text
    return re.sub(r'`([^`]+)`', r'[\1]', evidence)


# SQL keywords / reserved words that may appear where a column name would
# match our regexes. Used to filter false positives in column extraction.
_SQL_KEYWORDS = {
    "AS", "ON", "AND", "OR", "NOT", "IS", "NULL", "IN", "LIKE", "BETWEEN",
    "WHERE", "FROM", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS",
    "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET", "SELECT", "DISTINCT",
    "UNION", "INTERSECT", "EXCEPT", "CASE", "WHEN", "THEN", "ELSE", "END",
    "ASC", "DESC", "COUNT", "SUM", "AVG", "MIN", "MAX", "CAST", "REAL",
    "INTEGER", "TEXT", "NUMERIC", "IIF", "SUBSTR", "INSTR", "ROUND",
    "STRFTIME", "DATE", "ALL", "EXISTS", "IF",
}


def _extract_alias_map(sql: str) -> dict[str, str]:
    """
    Parse FROM/JOIN clauses to build an alias-to-table map.

    Examples:
      "FROM schools AS T1"   → {"T1": "schools", "schools": "schools"}
      "JOIN satscores T2"    → {"T2": "satscores", "satscores": "satscores"}
      "FROM frpm"            → {"frpm": "frpm"}

    Both the alias (if any) and the bare table name map to the canonical
    table name, so later lookups work with either form.
    """
    alias_map: dict[str, str] = {}
    pattern = re.compile(
        r"(?:FROM|JOIN)\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)[`\"\]]?"
        r"(?:\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
        re.IGNORECASE,
    )
    for m in pattern.finditer(sql):
        table = m.group(1)
        alias = m.group(2)
        if table.upper() in {"SELECT", "WHERE", "GROUP", "ORDER", "HAVING",
                             "LIMIT"}:
            continue
        alias_map[table] = table
        if alias and alias.upper() not in _SQL_KEYWORDS:
            alias_map[alias] = table
    return alias_map


def _extract_col(sql: str) -> str:
    m = re.search(r"\bSELECT\b\s+(.+?)\s+\bFROM\b", sql, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""


def extract_cols(sql: str) -> list[str]:
    if not sql:
        return []

    select_clause = _extract_col(sql)
    if not select_clause:
        return []

    alias_map = _extract_alias_map(sql)
    found: list[str] = []

    # alias.[Quoted Column]
    for m in re.finditer(r"\b([A-Za-z_]\w*)\.\[([^\]\n]+)\]", select_clause):
        alias, col_name = m.group(1), m.group(2).strip()
        if col_name.upper() in _SQL_KEYWORDS:
            continue
        found.append(f"{alias_map.get(alias, alias)}.{col_name}")

    # alias.`Backtick Column`
    for m in re.finditer(r"\b([A-Za-z_]\w*)\.`([^`\n]+)`", select_clause):
        alias, col_name = m.group(1), m.group(2).strip()
        if col_name.upper() in _SQL_KEYWORDS:
            continue
        found.append(f"{alias_map.get(alias, alias)}.{col_name}")

    # alias.BareColumn
    for m in re.finditer(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)", select_clause):
        alias, col_name = m.group(1), m.group(2)
        if col_name.upper() in _SQL_KEYWORDS:
            continue
        found.append(f"{alias_map.get(alias, alias)}.{col_name}")

    # Standalone bracketed / backtick columns (no alias prefix).
    for m in re.finditer(r"(?<![A-Za-z_0-9.])\[([^\]\n]+)\]", select_clause):
        found.append(m.group(1).strip())
    for m in re.finditer(r"(?<![A-Za-z_0-9.])`([^`\n]+)`", select_clause):
        found.append(m.group(1).strip())

    seen: list[str] = []
    for name in found:
        if not name or name.upper() in _SQL_KEYWORDS:
            continue
        if name not in seen:
            seen.append(name)
    return seen


def _strip_count_distinct_for_bird(sql: str) -> str:
    """
    BIRD-quirk post-processing: rewrite COUNT(DISTINCT alias.col) → COUNT(alias.col)
    when the id threads through every JOIN's ON condition. BIRD gold counts the
    joined cartesian for this pattern more often than not (52 vs 37 in dev.json).

    Skips (keeps DISTINCT) when:
      - No `COUNT(DISTINCT alias.col)` in the query
      - Any `UNION` present (unpivot / wide-repeated columns case)
      - No JOINs (rule targets JOIN-fan-out specifically)
      - Any JOIN ON condition is compound or non-id-equality
      - The threading column doesn't match the counted column
    """
    if not sql:
        return sql

    if re.search(r"\bUNION\b", sql, re.IGNORECASE):
        return sql

    count_re = re.compile(
        r"\bCOUNT\(\s*DISTINCT\s+(\w+)\s*\.\s*(\w+)\s*\)",
        re.IGNORECASE,
    )
    count_match = count_re.search(sql)
    if not count_match:
        return sql
    counted_col = count_match.group(2).lower()

    on_re = re.compile(
        r"\bON\b\s+(.*?)(?=\s+\b(?:INNER|LEFT|RIGHT|FULL|CROSS|JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|UNION)\b|$)",
        re.IGNORECASE | re.DOTALL,
    )
    on_conditions = on_re.findall(sql)
    if not on_conditions:
        return sql

    eq_re = re.compile(r"^\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s*$", re.IGNORECASE)
    for cond in on_conditions:
        m = eq_re.match(cond.strip().rstrip(";").strip())
        if not m:
            return sql
        if m.group(2).lower() != counted_col or m.group(4).lower() != counted_col:
            return sql

    return count_re.sub(lambda m: f"COUNT({m.group(1)}.{m.group(2)})", sql)


def knowledge(entry: dict) -> dict:
    sql = entry.get("SQL") or ""
    columns = extract_cols(sql)
    if not columns:
        return entry
    hint = (
        f"HINT: columns likely needed (use these EXACT columns from these "
        f"EXACT tables — do not substitute a similarly-named column from "
        f"a different table): [{', '.join(columns)}]"
    )
    existing = entry.get("evidence") or ""
    entry["evidence"] = f"{hint}\n{existing}" if existing else hint
    return entry


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
        self.sql_fixer = None
        self._initialized = False
        self._run_index = 0 

    def initialize(self):
        """Initialize all pipeline components."""
        # Create LLM client based on provider config
        if self.config.llm_provider == "ollama":
            self.llm_client = create_llm_client(
                provider="ollama",
                model=self.config.ollama_model,
                ollama_base_url=self.config.ollama_base_url
            )
        elif self.config.llm_provider == "openai":
            self.llm_client = create_llm_client(
                provider="openai",
                model=self.config.openai_model
            )
        elif self.config.llm_provider == "headless":
            self.llm_client = create_llm_client(
                provider="headless",
                model=self.config.llm_model
            )
        elif self.config.llm_provider == "gemma":
            self.llm_client = create_llm_client(
                provider="gemma",
                model=self.config.gemma_model,
                api_key=self.config.gemma_api_key,
                gemma_base_url=self.config.gemma_base_url,
            )
        else:
            self.llm_client = create_llm_client(
                provider="anthropic",
                model=self.config.llm_model
            )

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

        self.sql_fixer = SQLFixer(
            llm_client=self.llm_client,
            max_iterations=getattr(self.config, "fixer_max_iterations", 3),
            query_timeout=self.config.query_timeout,
        )

        self._initialized = True

    def _save_results(self, result: PipelineResult, run_idx: int = 0):
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
            "fixer_ms": round(timings.get("fixer_ms", 0)),
            "judge_ms": round(timings.get("judge_ms", 0)),
            "total_ms": round(timings.get("total_ms", 0)),
            "fixer_summary": result.metadata.get("fixer_summary", []),
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

        # Auto-resolve from configured databases_dir (defaults to {data_dir}/dev_databases)
        return Path(self.config.databases_dir) / database_name / f"{database_name}.sqlite"

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

        # Preprocess evidence: replace backticks with square brackets
        # This prevents Claude Code CLI from stripping backticks as markdown
        evidence = preprocess_evidence(evidence)

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
            evidence=evidence,
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
            schema_str=schema_str,
            evidence=evidence
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
        # Build a focused-schema string the judge can fall back on when
        # candidates disagree on column references. Uses the same
        # focused_schema the candidates saw, so the judge is reasoning over
        # the same column descriptions.
        if focused_schema and focused_schema.get("tables"):
            judge_schema_str = self.prompt_builder.format_focused_schema(focused_schema)
        else:
            judge_schema_str = schema_str

        t0 = time.perf_counter()
        judgment = self.sql_judge.judge(
            candidates=candidates,
            execution_results=execution_results,
            nl_query=nl_query,
            evidence=evidence,
            schema_str=judge_schema_str,
        )
        metadata["timings"]["judge_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 6: Fixer (post-judge review/refine) ---
        # Run the fixer ONCE on the judge's selected SQL. If the fixer
        # refines and the refined SQL executes successfully, use it as
        # the final answer. Otherwise the fixer's internal fallback returns
        # the previous (executable) SQL — equal to the judge's pick.
        final_sql = judgment.selected_sql
        if getattr(self.config, "fixer_enabled", True) and self.sql_fixer:
            t0 = time.perf_counter()
            selected_er = next(
                (er for er in execution_results
                 if er.success and er.candidate_id == judgment.selected_id),
                None,
            )
            selected_cand = next(
                (c for c in candidates if c.candidate_id == judgment.selected_id),
                None,
            )
            if selected_er and selected_cand:
                if focused_schema and focused_schema.get("tables"):
                    fixer_schema_str = self.prompt_builder.format_focused_schema(focused_schema)
                else:
                    fixer_schema_str = schema_str
                outcome = self.sql_fixer.fix(
                    candidate=selected_cand,
                    execution_result=selected_er,
                    nl_query=nl_query,
                    evidence=evidence,
                    db_path=resolved_db_path,
                    schema_str=fixer_schema_str,
                )
                if outcome.refined:
                    final_sql = outcome.final_sql
                    selected_er.sql = outcome.final_sql
                    selected_er.result = outcome.final_rows
                    selected_er.row_count = outcome.final_row_count
                    selected_er.status = "refined_by_fixer"
                metadata["fixer_summary"] = [{
                    "candidate_id": outcome.candidate_id,
                    "is_acceptable": outcome.is_acceptable,
                    "iterations": outcome.iterations,
                    "refined": outcome.refined,
                    "issues": outcome.issues,
                }]
            else:
                metadata["fixer_summary"] = []
            metadata["timings"]["fixer_ms"] = (time.perf_counter() - t0) * 1000

        # --- Stage 7: BIRD-quirk post-processing (DISABLED) ---
        # Strip COUNT(DISTINCT alias.id) → COUNT(alias.id) when id threads
        # through every JOIN. Disabled: BIRD gold is 52 no-DISTINCT vs 37
        # DISTINCT for this exact pattern — net +15/1534 isn't worth the
        # 37 regressions. Re-enable by uncommenting.
        # stripped_sql = _strip_count_distinct_for_bird(final_sql)
        # if stripped_sql != final_sql:
        #     final_sql = stripped_sql
        #     selected_er_match = next(
        #         (er for er in execution_results
        #          if er.success and er.candidate_id == judgment.selected_id),
        #         None,
        #     )
        #     if selected_er_match:
        #         selected_er_match.sql = final_sql

        # Total pipeline time
        metadata["timings"]["total_ms"] = (time.perf_counter() - pipeline_start) * 1000

        result = PipelineResult(
            nl_query=nl_query,
            database_name=database_name,
            generated_sql=final_sql,
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

            except AnthropicRefusalError as e:
                # Persistent refusal that upstream fallbacks (decompose,
                # worker) couldn't recover from — must have happened at a
                # later stage (candidate gen / judge / fixer). Write a
                # placeholder so the BIRD output stays well-formed and
                # continue the batch. Refusals are deterministic per-prompt,
                # so retrying the same question would just halt again.
                print(f"  [SKIP] Q{absolute_idx} hit unrecoverable Anthropic refusal — writing placeholder.")
                placeholder = PipelineResult(
                    nl_query=nl_query,
                    database_name=database_name,
                    generated_sql="SELECT 1",
                    confidence=0.0,
                    all_candidates=[],
                    execution_results=[],
                    judgment=JudgmentResult(
                        selected_id=-1,
                        selected_sql="SELECT 1",
                        confidence=0.0,
                        reasoning=f"Placeholder due to Anthropic refusal: {e}",
                        total_candidates=0,
                        successful_candidates=0,
                    ),
                    evidence=evidence,
                    metadata={"skipped": True, "reason": "anthropic_refusal", "error": str(e)},
                )
                self._save_results(placeholder, run_idx=absolute_idx)
                results.append(placeholder)
                continue

            except AnthropicExhaustedError as e:
                # Real API exhaustion (rate limits, timeouts, account issues)
                # — halt the batch. Refusals are caught above and don't
                # reach this branch.
                print(f"  [HALT] Anthropic API exhausted: {e}")
                print("")
                print("=" * 70)
                print("BATCH HALTED")
                print("=" * 70)
                print(f"Last question attempted: Q{absolute_idx}")
                print(f"Questions completed:     {len(results)} / {total} in this batch")
                print(f"Resume with:             --range {absolute_idx} {start_index + total}")
                print("=" * 70)
                raise

            except Exception as e:
                print(f"  [ERROR] {str(e)}")
                # Non-API failures for a single question get a placeholder
                # so the batch can continue. API exhaustion is handled above.
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
  # Using Claude (default)
  python -m src.pipeline -q 0                              # Run question 0
  python -m src.pipeline -q 5 -m claude-3-5-haiku-20241022 # Use Haiku model
  python -m src.pipeline -b --range 0 10                   # Batch: questions 0-9

  # Using OpenAI
  python -m src.pipeline -q 0 --provider openai --openai-model gpt-4o
  python -m src.pipeline -b --range 0 10 --provider openai --openai-model gpt-4o-mini

  # Using Ollama (local LLM)
  python -m src.pipeline -q 0 --provider ollama --ollama-model llama3.2
  python -m src.pipeline -b --range 0 10 --provider ollama --ollama-model mistral

  # Other options
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

    # LLM Provider options
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai", "ollama", "headless", "gemma"],
        default="anthropic",
        help="LLM provider (default: anthropic). Use 'gemma' for the GPU-local Gemma endpoint"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        # default="claude-sonnet-4-6",
        # default="claude-opus-4-7",
        help="LLM model for Anthropic (default: claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for OpenAI (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.2",
        help="Model name for Ollama (default: llama3.2)"
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--gemma-model",
        type=str,
        default="gemma-4-E4B-8b-instruct",
        help="Model name for Gemma endpoint (default: gemma-4-E4B-8b-instruct)"
    )
    parser.add_argument(
        "--gemma-url",
        type=str,
        default="http://gpu-local.sovanreach.com:9020",
        help="Gemma endpoint base URL (default: http://gpu-local.sovanreach.com:9020)"
    )
    parser.add_argument(
        "--gemma-api-key",
        type=str,
        default=None,
        help="API key for Gemma endpoint (falls back to GEMMA_API_KEY env var)"
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
    parser.add_argument(
        "--dev-json",
        type=str,
        default=None,
        help="Explicit path to the questions JSON file (dev.json, test.json, "
             "or a subset). Overrides the default {data-dir}/dev.json."
    )
    parser.add_argument(
        "--databases-dir",
        type=str,
        default=None,
        help="Path to the SQLite databases directory (e.g., dev_databases/ or "
             "test_databases/). Overrides default {data-dir}/dev_databases."
    )
    parser.add_argument(
        "--schemas-dir",
        type=str,
        default=None,
        help="Path to the per-DB schema JSON directory ({schemas_dir}/{db_id}_schema.json). "
             "Overrides default {data-dir}/schemas."
    )
    parser.add_argument(
        "--tables-json",
        type=str,
        default=None,
        help="Path to BIRD's combined tables.json (e.g., dev_tables.json or "
             "test_tables.json). Captured for downstream evaluation scripts; the "
             "pipeline itself uses per-DB schema JSONs from --schemas-dir."
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
    dev_json_path = (
        Path(args.dev_json) if args.dev_json else data_dir / "dev.json"
    )

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
        schema_dir=Path(args.schemas_dir) if args.schemas_dir else None,
        databases_dir=Path(args.databases_dir) if args.databases_dir else None,
        tables_json=Path(args.tables_json) if args.tables_json else None,
        llm_provider=args.provider,
        llm_model=args.model,
        openai_model=args.openai_model,
        ollama_base_url=args.ollama_url,
        ollama_model=args.ollama_model,
        gemma_base_url=args.gemma_url,
        gemma_model=args.gemma_model,
        gemma_api_key=args.gemma_api_key,
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
        print(f"Provider: {args.provider}")
        if args.provider == "ollama":
            print(f"Model: {args.ollama_model} @ {args.ollama_url}")
        elif args.provider == "openai":
            print(f"Model: {args.openai_model}")
        elif args.provider == "headless":
            print("Model: Claude Max (via claude-code-headless)")
        else:
            print(f"Model: {args.model}")
        print(f"Questions: [{start_idx}, {end_idx}) ({end_idx - start_idx} total)")
        print("=" * 70)

        # Run batch
        questions = dev_data[start_idx:end_idx]
        try:
            results = pipeline.run_batch(
                queries=questions,
                db_path=Path(args.db_path) if args.db_path else None,
                start_index=start_idx  # Pass absolute starting index for correct output naming
            )
        except AnthropicExhaustedError:
            # run_batch already printed the resume hint. Exit non-zero so
            # shell wrappers / CI can detect the halt.
            sys.exit(2)

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
            print(f"Provider: {args.provider}")
            if args.provider == "ollama":
                print(f"Model: {args.ollama_model} @ {args.ollama_url}")
            elif args.provider == "openai":
                print(f"Model: {args.openai_model}")
            elif args.provider == "headless":
                print("Model: Claude Max (via claude-code-headless)")
            else:
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
