"""
Test script for Map-Reduce Schema Agent with BIRD benchmark databases.

Questions are loaded from: data/bird_data/dev.json (1534 questions)

Run with:
    python test_schema_agent.py                          # Test student_club (default)
    python test_schema_agent.py --db california_schools  # Test specific database
    python test_schema_agent.py --question-id 0          # Test specific question by ID
    python test_schema_agent.py --questions 5            # Test first 5 questions
    python test_schema_agent.py --use-llm                # Use LLM (requires API key)
    python test_schema_agent.py --list-dbs               # List available databases
    python test_schema_agent.py --all                    # Test all databases
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.manager import SchemaManager
from agents.worker import SchemaWorker

# Paths
DATA_DIR = Path(__file__).parent / "data" / "bird_data"
SCHEMAS_DIR = DATA_DIR / "schemas"
DESCRIPTIONS_DIR = DATA_DIR / "descriptions"
DATABASES_DIR = DATA_DIR / "dev_databases"
DEV_JSON = DATA_DIR / "dev.json"


def list_databases():
    """List all available BIRD databases."""
    print("Available BIRD databases:")
    print("-" * 40)

    for schema_file in sorted(SCHEMAS_DIR.glob("*_schema.json")):
        db_name = schema_file.stem.replace("_schema", "")
        db_path = DATABASES_DIR / db_name / f"{db_name}.sqlite"
        status = "✓" if db_path.exists() else "✗"
        print(f"  {status} {db_name}")

    print("-" * 40)


def load_schema(db_name: str) -> dict:
    """Load schema for a database."""
    schema_path = SCHEMAS_DIR / f"{db_name}_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("tables", {})


def load_descriptions(db_name: str) -> dict:
    """Load descriptions/profile for a database (SME metadata)."""
    desc_path = DESCRIPTIONS_DIR / f"{db_name}_descriptions.json"
    if not desc_path.exists():
        return None

    with open(desc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("tables", {})


def load_questions(db_name: str = None, limit: int = 5, question_id: int = None) -> list:
    """Load questions from dev.json, optionally filtered by database or question ID."""
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Filter by specific question ID
    if question_id is not None:
        questions = [q for q in questions if q["question_id"] == question_id]
        return questions

    # Filter by database
    if db_name:
        questions = [q for q in questions if q["db_id"] == db_name]

    return questions[:limit]


def get_question_by_id(question_id: int) -> dict:
    """Get a specific question by ID."""
    with open(DEV_JSON, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        if q["question_id"] == question_id:
            return q
    return None


def test_schema_agent(db_name: str, use_llm: bool = False, num_questions: int = 3):
    """Test the Map-Reduce Schema Agent on a BIRD database."""
    print("=" * 70)
    print(f"Testing Map-Reduce Schema Agent on: {db_name}")
    print(f"Mode: {'LLM' if use_llm else 'Heuristic'}")
    print("=" * 70)

    # Load schema
    try:
        schema = load_schema(db_name)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Load descriptions (SME metadata)
    profile = load_descriptions(db_name)
    if profile:
        print(f"Using descriptions from: {db_name}_descriptions.json")
    else:
        print("No descriptions found (using schema only)")

    print(f"\nDatabase: {db_name}")
    print(f"Tables: {len(schema)}")
    for table_name in schema.keys():
        num_cols = len(schema[table_name].get("columns", []))
        readable_name = ""
        if profile and table_name in profile:
            readable_name = f" ({profile[table_name].get('table_readable_name', '')})"
        print(f"  - {table_name}{readable_name} ({num_cols} columns)")

    # Initialize manager
    llm_client = None
    if use_llm:
        from utils.llm_client import LLMClient
        llm_client = LLMClient()

    manager = SchemaManager(llm_client=llm_client)

    # Load questions for this database
    questions = load_questions(db_name, limit=num_questions)

    if not questions:
        print(f"\nNo questions found for database: {db_name}")
        # Use a default test query
        questions = [{
            "question_id": -1,
            "question": f"List all records from {list(schema.keys())[0]}",
            "difficulty": "test"
        }]

    print(f"\nTesting with {len(questions)} question(s)...")
    print("-" * 70)

    for q in questions:
        print(f"\n[Question {q['question_id']}] ({q.get('difficulty', 'N/A')})")
        print(f"Q: {q['question']}")

        if q.get('evidence'):
            print(f"Evidence: {q['evidence']}")

        # Run the schema agent with profile (descriptions)
        focused_schema = manager.run(
            nl_query=q['question'],
            schema=schema,
            profile={"tables": profile} if profile else None,
            relevance_threshold=0.50
        )

        # Display results
        print(f"\nDecomposed Components:")
        for comp in focused_schema["metadata"]["decomposed_components"]:
            print(f"  - [{comp['type']}] {comp['text']}")

        print(f"\nTable Relevance Scores (RC):")
        total_tables = focused_schema["metadata"]["total_tables_evaluated"]
        relevant_tables = focused_schema["metadata"]["relevant_tables_count"]

        # Show all tables with their scores
        for table_rel in focused_schema.get("table_relevances", []):
            status = "✓" if table_rel["relevance_score"] >= 0.5 else "✗"
            print(f"  {status} {table_rel['table_name']}: RC={table_rel['relevance_score']:.1f}")

        # Show tables that were filtered out (RC < 0.5)
        relevant_table_names = {t["table_name"] for t in focused_schema.get("table_relevances", [])}
        filtered_out = [t for t in schema.keys() if t not in relevant_table_names]
        for table_name in filtered_out:
            print(f"  ✗ {table_name}: RC=0.0")

        print(f"\nFocused Schema: {relevant_tables}/{total_tables} tables selected")

        if q.get("SQL"):
            print(f"\nGround Truth SQL:")
            print(f"  {q['SQL']}")

        print("-" * 70)

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)


def test_all_databases(use_llm: bool = False):
    """Test schema agent on all available databases."""
    print("Testing all BIRD databases...")
    print("=" * 70)

    for schema_file in sorted(SCHEMAS_DIR.glob("*_schema.json")):
        db_name = schema_file.stem.replace("_schema", "")
        print(f"\n>>> Testing: {db_name}")

        try:
            test_schema_agent(db_name, use_llm=use_llm, num_questions=1)
        except Exception as e:
            print(f"Error testing {db_name}: {e}")


def test_single_question(question_id: int, use_llm: bool = False):
    """Test schema agent on a single question by ID."""
    question = get_question_by_id(question_id)

    if not question:
        print(f"Error: Question ID {question_id} not found in dev.json")
        return

    db_name = question["db_id"]
    print("=" * 70)
    print(f"Testing Question ID: {question_id}")
    print(f"Database: {db_name}")
    print(f"Mode: {'LLM' if use_llm else 'Heuristic'}")
    print("=" * 70)

    # Load schema and descriptions
    schema = load_schema(db_name)
    profile = load_descriptions(db_name)

    if profile:
        print(f"Using descriptions from: {db_name}_descriptions.json")

    print(f"\nTables in {db_name}: {len(schema)}")
    for table_name in schema.keys():
        num_cols = len(schema[table_name].get("columns", []))
        readable_name = ""
        if profile and table_name in profile:
            readable_name = f" ({profile[table_name].get('table_readable_name', '')})"
        print(f"  - {table_name}{readable_name} ({num_cols} columns)")

    # Initialize manager
    llm_client = None
    if use_llm:
        from utils.llm_client import LLMClient
        llm_client = LLMClient()

    manager = SchemaManager(llm_client=llm_client)

    print(f"\n" + "-" * 70)
    print(f"[Question {question['question_id']}] ({question.get('difficulty', 'N/A')})")
    print(f"Q: {question['question']}")

    if question.get('evidence'):
        print(f"Evidence: {question['evidence']}")

    # Run the schema agent with profile (descriptions)
    focused_schema = manager.run(
        nl_query=question['question'],
        schema=schema,
        profile={"tables": profile} if profile else None,
        relevance_threshold=0.50
    )

    # Display results
    print(f"\nDecomposed Components:")
    for comp in focused_schema["metadata"]["decomposed_components"]:
        print(f"  - [{comp['type']}] {comp['text']}")

    print(f"\nTable Relevance Scores (RC):")

    # Show all tables with their scores
    for table_rel in focused_schema.get("table_relevances", []):
        print(f"  ✓ {table_rel['table_name']}: RC={table_rel['relevance_score']:.1f}")

    # Show tables that were filtered out
    relevant_table_names = {t["table_name"] for t in focused_schema.get("table_relevances", [])}
    for table_name in schema.keys():
        if table_name not in relevant_table_names:
            print(f"  ✗ {table_name}: RC=0.0")

    total = focused_schema["metadata"]["total_tables_evaluated"]
    selected = focused_schema["metadata"]["relevant_tables_count"]
    print(f"\nFocused Schema: {selected}/{total} tables selected")

    if question.get("SQL"):
        print(f"\nGround Truth SQL:")
        print(f"  {question['SQL']}")

    print("-" * 70)
    print("Test Complete!")


def main():
    parser = argparse.ArgumentParser(description="Test Map-Reduce Schema Agent with BIRD benchmark")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--db", type=str, default="student_club", help="Database to test (default: student_club)")
    parser.add_argument("--question-id", type=int, help="Test specific question by ID (from dev.json)")
    parser.add_argument("--list-dbs", action="store_true", help="List available databases")
    parser.add_argument("--all", action="store_true", help="Test all databases")
    parser.add_argument("--questions", type=int, default=3, help="Number of questions to test (default: 3)")

    args = parser.parse_args()

    if args.list_dbs:
        list_databases()
        return

    if args.question_id is not None:
        test_single_question(args.question_id, use_llm=args.use_llm)
        return

    if args.all:
        test_all_databases(use_llm=args.use_llm)
        return

    test_schema_agent(args.db, use_llm=args.use_llm, num_questions=args.questions)

    if not args.use_llm:
        print("\n💡 Tip: Run with --use-llm flag for better results:")
        print("   export ANTHROPIC_API_KEY='your-key'")
        print("   python test_schema_agent.py --use-llm")


if __name__ == "__main__":
    main()
