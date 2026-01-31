# QA-SQL: Query Augmentation to SQL

A multi-stage pipeline for converting natural language queries to SQL using a Map-Reduce Schema Agent pattern combined with an LLM-based SQL Selection Agent.

---

## Architecture Overview

The proposed method consists of two main agents working in sequence:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MAP-REDUCE SCHEMA AGENT                                 │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐                 │
│  │   Input     │───▶│ Agentic          │───▶│ Mapping Function│                 │
│  │   Query     │    │ Decomposition    │    │ (Parallel Workers)                │
│  └─────────────┘    │ (The Manager)    │    └────────┬────────┘                 │
│                     └──────────────────┘             │                          │
│                                                      ▼                          │
│                                           ┌─────────────────┐                   │
│                                           │ Reduce Function │                   │
│                                           │   (Ranking)     │                   │
│                                           └────────┬────────┘                   │
│                                                    │                            │
│                                                    ▼                            │
│                                           ┌─────────────────┐                   │
│                                           │ Focused Schema  │                   │
│                                           └────────┬────────┘                   │
└────────────────────────────────────────────────────┼────────────────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SQL SELECTION AGENT                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │ Context-Aware Generation (5 Strategies)                              │        │
│  │  • Full Schema    • SME Metadata    • Minimal Profile                │        │
│  │  • Focused Schema • Full Profile                                     │        │
│  └─────────────────────────────────┬───────────────────────────────────┘        │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │                     LLM As a Judge                                   │        │
│  │  Role: Senior SQL Reviewer | Evaluates & Selects BEST Query          │        │
│  └─────────────────────────────────┬───────────────────────────────────┘        │
│                                    │                                            │
│                                    ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │             Iteration Loop (Max 3 Times)                             │        │
│  │  Execute → Validate → If Failed & iter<3 → Retry with Feedback       │        │
│  │                     → If iter=3 → Reject                             │        │
│  │                     → If Correct → Return Final SQL                  │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Instructions

### Phase 1: Map-Reduce Schema Agent

#### 1.1 Agentic Decomposition (The Manager)

**File:** `src/agents/manager.py`

**Purpose:** Decompose natural language queries into semantic components.

**Implementation Steps:**

1. **Parse Input Query** - Extract semantic components:
   - `[entity]` - Main entity being queried (e.g., "members")
   - `[filter]` - Filtering conditions (e.g., "Computer Science")
   - `[aggregation]` - Aggregation operations (e.g., "list", "count", "sum")
   - `[projection]` - Fields to return (e.g., "all members", "name only")

2. **LLM Prompt for Decomposition:**
   ```
   You are a query decomposition expert. Given a natural language query,
   extract the following components:

   - entity: The main table/entity being queried
   - filter: Any filtering conditions
   - aggregation: Aggregation operations (list, count, sum, avg, etc.)
   - projection: What columns/fields to return

   Query: "{query}"

   Return JSON: {"entity": "", "filter": "", "aggregation": "", "projection": ""}
   ```

3. **Output:** `DecomposedQuery` dataclass with components and relationships

---

#### 1.2 Mapping Function (Parallel Workers)

**File:** `src/agents/worker.py`

**Purpose:** Score relevance of each table to the decomposed query components.

**Implementation Steps:**

1. **Worker Assignment:**
   - Each worker processes one or more tables
   - Workers run in parallel using `ThreadPoolExecutor`

2. **Relevance Scoring (RC - Relevant Coefficient):**
   ```python
   def calculate_relevance_score(table_name, columns, decomposed_query):
       # Method 1: Keyword matching (fast heuristic)
       keyword_score = match_keywords(table_name, columns, decomposed_query)

       # Method 2: LLM-based semantic matching (accurate)
       llm_score = llm_semantic_match(table_name, columns, decomposed_query)

       # Combine scores (weighted average or max)
       return max(keyword_score, llm_score)
   ```

3. **LLM Prompt for Relevance:**
   ```
   Given the query components and table information, rate the relevance (0.0-1.0):

   Query Components:
   - Entity: {entity}
   - Filter: {filter}
   - Aggregation: {aggregation}
   - Projection: {projection}

   Table: {table_name}
   Columns: {columns}

   Return JSON: {"relevance_score": 0.0-1.0, "reason": "explanation"}
   ```

4. **Output per Worker:** `TableRelevance` with score and relevant columns

---

#### 1.3 Reduce Function (Ranking)

**File:** `src/agents/manager.py` (aggregate_results method)

**Purpose:** Combine worker results and filter to focused schema.

**Implementation Steps:**

1. **Collect All Results:**
   ```python
   all_results = []
   for worker_result in worker_results:
       all_results.extend(worker_result.table_relevances)
   ```

2. **Apply Threshold Filter:**
   ```python
   RELEVANCE_THRESHOLD = 0.50  # Configurable

   focused_tables = [
       table for table in all_results
       if table.relevance_score >= RELEVANCE_THRESHOLD
   ]
   ```

3. **Sort by Relevance:**
   ```python
   focused_tables.sort(key=lambda x: x.relevance_score, reverse=True)
   ```

4. **Output:** `FocusedSchema` containing only relevant tables and columns

---

### Phase 2: SQL Selection Agent

#### 2.1 Context-Aware SQL Generation (5 Strategies)

**File:** `src/generation/candidate_generator.py`

**Purpose:** Generate diverse SQL candidates using different context strategies.

| Strategy | Description | Input |
|----------|-------------|-------|
| **Full Schema** | Complete database schema | All tables, all columns |
| **SME Metadata** | Subject Matter Expert annotations | Schema + domain descriptions |
| **Minimal Profile** | Basic column descriptions | Schema + auto-generated descriptions |
| **Focused Schema** | Relevant tables only | Output from Map-Reduce Agent |
| **Full Profile** | Combined all metadata | Minimal + SME + relationships |

**Implementation Steps:**

1. **Build Strategy-Specific Prompts** (`src/generation/prompt_builder.py`):
   ```python
   def build_prompt(strategy, query, schema, profile):
       system_prompt = """You are an expert SQL query generator.
       Generate a valid SQL query for the given natural language question."""

       if strategy == "FOCUSED_SCHEMA":
           context = format_focused_schema(schema)
       elif strategy == "FULL_SCHEMA":
           context = format_full_schema(schema)
       elif strategy == "MINIMAL_PROFILE":
           context = format_minimal_profile(schema, profile)
       elif strategy == "SME_METADATA":
           context = format_sme_metadata(schema, profile)
       elif strategy == "FULL_PROFILE":
           context = format_full_profile(schema, profile)

       return system_prompt, f"Schema:\n{context}\n\nQuestion: {query}"
   ```

2. **Generate Candidates:**
   ```python
   candidates = []
   for strategy in strategies:
       prompt = build_prompt(strategy, query, schema, profile)
       sql = llm_client.complete(prompt)
       candidates.append(SQLCandidate(
           id=f"Q{len(candidates)+1}",
           sql=sql,
           strategy=strategy
       ))
   ```

---

#### 2.2 LLM As a Judge

**File:** `src/selection/judge.py`

**Purpose:** Evaluate and select the best SQL candidate.

**Judge System Prompt:**
```
ROLE:
You are a Senior SQL Reviewer.

TASK:
Your job is to select the BEST query.

CONTEXT:
- Input query: {natural_language_query}
- SME definitions: {schema_descriptions}

CANDIDATES:
Option 1 (Q1): {sql_1}
Option 2 (Q2): {sql_2}
Option 3 (Q3): {sql_3}
...

EVALUATION CRITERIA:
1. Correctness: Does the SQL accurately answer the question?
2. Efficiency: Is the query optimized (proper JOINs, indexes)?
3. Readability: Is the SQL clean and maintainable?
4. Completeness: Does it return all required information?

Return JSON:
{
  "selected_option": "Q1",
  "confidence": 0.95,
  "reasoning": "Explanation of why this query is best"
}
```

**Implementation:**
```python
class SQLJudge:
    def judge(self, query, candidates, schema_profile):
        prompt = self.build_judge_prompt(query, candidates, schema_profile)
        result = self.llm_client.complete(prompt)
        return JudgmentResult.from_json(result)
```

---

#### 2.3 Iteration Loop (Max 3 Retries)

**File:** `src/selection/executor.py`

**Purpose:** Execute, validate, and refine SQL with retry logic.

**Flow:**
```
┌─────────────────┐
│ Execute SQL     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     Yes    ┌─────────────────┐
│ Query Failed?   │───────────▶│  iter < 3?      │
└────────┬────────┘            └────────┬────────┘
         │ No                           │
         ▼                     Yes      │      No
┌─────────────────┐            │        ▼
│ Return Correct  │◀───────────┘  ┌─────────────┐
│ Result          │               │ REJECTION   │
└─────────────────┘               │ (Give up)   │
         ▲                        └─────────────┘
         │
┌─────────────────┐
│ Refine with LLM │
│ (Error Feedback)│
└─────────────────┘
```

**Implementation:**
```python
MAX_ITERATIONS = 3

def execute_with_retry(self, sql, database_path):
    for iteration in range(MAX_ITERATIONS):
        try:
            result = self.execute_sql(sql, database_path)
            if self.validate_result(result):
                return {"status": "correct", "result": result}
        except Exception as e:
            if iteration < MAX_ITERATIONS - 1:
                # Refine SQL with error feedback
                sql = self.refine_sql(sql, error=str(e))
            else:
                return {"status": "rejected", "error": str(e)}

    return {"status": "rejected", "error": "Max iterations reached"}
```

**Refinement Prompt:**
```
The following SQL query failed with an error:

SQL: {failed_sql}
Error: {error_message}

Schema:
{schema}

Original Question: {query}

Please fix the SQL query to resolve the error.
Return only the corrected SQL, no explanation.
```

---

### Phase 3: Pipeline Integration

**File:** `src/pipeline.py`

**Complete Pipeline Flow:**

```python
class QASQLPipeline:
    def run(self, query: str, database_name: str) -> PipelineResult:
        # Stage 1: Load inputs
        processed_input = self.input_processor.process(query, database_name)

        # Stage 2: Map-Reduce Schema Agent
        decomposed = self.schema_manager.decompose_query(query)
        worker_results = self.schema_manager.coordinate_workers(
            decomposed, processed_input.schema
        )
        focused_schema = self.schema_manager.aggregate_results(worker_results)

        # Stage 3: Generate SQL Candidates (5 strategies)
        candidates = self.candidate_generator.generate_all_candidates(
            query=query,
            full_schema=processed_input.schema,
            focused_schema=focused_schema,
            profile=processed_input.profile
        )

        # Stage 4: Execute and Refine (Max 3 iterations)
        execution_results = []
        for candidate in candidates:
            result = self.executor.execute_with_retry(
                candidate.sql,
                database_path=processed_input.database_path
            )
            execution_results.append(result)

        # Stage 5: LLM Judge selects best
        successful_candidates = self.executor.filter_successful(execution_results)

        if not successful_candidates:
            return PipelineResult(status="rejected")

        judgment = self.judge.judge(
            query=query,
            candidates=successful_candidates,
            schema_profile=processed_input.profile
        )

        return PipelineResult(
            generated_sql=judgment.selected_sql,
            confidence=judgment.confidence,
            candidates=candidates,
            execution_results=execution_results
        )
```

---

## Project Structure

```
src/
├── agents/
│   ├── manager.py         # Agentic Decomposition + Reduce Function
│   └── worker.py          # Parallel relevance verification workers
├── generation/
│   ├── candidate_generator.py  # 5-strategy SQL generation
│   └── prompt_builder.py       # Strategy-specific prompt building
├── selection/
│   ├── executor.py        # SQL execution + retry loop
│   └── judge.py           # LLM-as-a-Judge evaluation
├── processing/
│   ├── input_processor.py      # Load schema and profiles
│   ├── database_profiling.py   # Generate column descriptions
│   └── extract_schema.py       # Extract schema from SQLite
├── utils/
│   ├── llm_client.py      # Anthropic API wrapper
│   └── config.py          # Pipeline configuration
└── pipeline.py            # Main orchestrator
```

---

## Configuration

**File:** `config/default.json`

```json
{
  "llm": {
    "model": "claude-3-5-haiku-20241022",
    "max_tokens": 2048,
    "temperature": 0.0
  },
  "agent": {
    "max_workers": 4,
    "relevance_threshold": 0.50
  },
  "generation": {
    "strategies": [
      "FULL_SCHEMA",
      "SME_METADATA",
      "MINIMAL_PROFILE",
      "FOCUSED_SCHEMA",
      "FULL_PROFILE"
    ]
  },
  "execution": {
    "query_timeout": 30,
    "max_iterations": 3
  }
}
```

---

## Setup and Running

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY='your-key'
```

### Database Profiling

```bash
python src/processing/database_profiling.py
```

### Extract Schema

```bash
python src/processing/extract_schema.py
```

### Run Pipeline

```python
from src.pipeline import QASQLPipeline
from src.utils.config import Config

config = Config.from_json("config/default.json")
pipeline = QASQLPipeline(config)

result = pipeline.run(
    query="List all members who are in Computer Science related majors",
    database_name="student_club"
)

print(f"Generated SQL: {result.generated_sql}")
print(f"Confidence: {result.confidence}")
```

---

## Key Data Structures

### QueryComponent
```python
@dataclass
class QueryComponent:
    text: str           # Component text
    type: str           # entity|filter|aggregation|projection
    relevant_tables: List[str]
```

### TableRelevance
```python
@dataclass
class TableRelevance:
    table_name: str
    relevance_score: float  # 0.0 - 1.0 (RC score)
    relevant_columns: List[ColumnRelevance]
```

### SQLCandidate
```python
@dataclass
class SQLCandidate:
    id: str             # Q1, Q2, Q3...
    sql: str            # Generated SQL
    strategy: str       # Generation strategy used
    confidence: float   # LLM confidence score
```

### JudgmentResult
```python
@dataclass
class JudgmentResult:
    selected_id: str
    selected_sql: str
    confidence: float
    reasoning: str
    all_evaluations: List[CandidateEvaluation]
```

---

## Implementation Checklist

### Map-Reduce Schema Agent
- [ ] Implement `decompose_query()` with LLM parsing
- [ ] Implement `identify_component_type()`
- [ ] Implement `map_components_to_tables()`
- [ ] Implement `coordinate_workers()` with ThreadPoolExecutor
- [ ] Implement `verify_table_relevance()` with RC scoring
- [ ] Implement `aggregate_results()` with threshold filtering

### SQL Selection Agent
- [ ] Implement all 5 generation strategies
- [ ] Implement `build_prompt()` for each strategy
- [ ] Implement `judge()` with evaluation criteria
- [ ] Implement `execute_with_retry()` with 3-iteration loop
- [ ] Implement `refine_sql()` with error feedback

### Pipeline Integration
- [ ] Implement `process_inputs()`
- [ ] Implement `run()` method orchestration
- [ ] Implement `run_batch()` for multiple queries
- [ ] Add logging and metrics collection
- [ ] Add result caching for efficiency

---

## Evaluation Metrics

The system can be evaluated on the BIRD benchmark using:
- **Execution Accuracy (EX)**: Does the generated SQL execute without errors?
- **Valid Efficiency Score (VES)**: How efficient is the generated query?
- **Result Match**: Does the output match the ground truth?