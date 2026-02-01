"""
Test script for Map-Reduce Schema Agent.

Run with: python test_schema_agent.py

To test with LLM:
    export ANTHROPIC_API_KEY='your-key'
    python test_schema_agent.py --use-llm
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.manager import SchemaManager
from agents.worker import SchemaWorker


def test_without_llm():
    """Test using heuristic matching only (no API key needed)."""
    print("=" * 60)
    print("Testing Map-Reduce Schema Agent (Heuristic Mode)")
    print("=" * 60)

    # No LLM client - will use heuristic fallback
    manager = SchemaManager(llm_client=None)

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
                {"name": "subject_name", "readable_name": "subject name"},
                {"name": "department", "readable_name": "department", "description": "computer science, math, etc."}
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
                {"name": "capacity", "readable_name": "capacity"}
            ]
        }
    }

    query = "List all members who are in Computer Science related majors"

    print(f"\nInput Query: {query}")
    print("-" * 60)

    # Test decompose_query (will fail without LLM, uses fallback)
    print("\n1. Testing decompose_query (fallback mode):")
    decomposed = manager.decompose_query(query)
    print(f"   Original: {decomposed.original_query}")
    print(f"   Components: {len(decomposed.components)}")
    for comp in decomposed.components:
        print(f"     - [{comp.component_type}] {comp.component_text}")

    # Test worker directly with heuristic
    print("\n2. Testing single worker (heuristic mode):")
    worker = SchemaWorker(worker_id="test-worker", llm_client=None)

    query_components = [
        {"text": "members", "type": "entity"},
        {"text": "Computer Science", "type": "filter"},
        {"text": "majors", "type": "filter"}
    ]

    for table_name in schema.keys():
        result = worker.verify_table_relevance(
            table_name=table_name,
            table_schema=schema[table_name],
            query_components=query_components
        )
        status = "✓ relevant" if result.relevance_score >= 0.5 else "✗ not relevant"
        print(f"   {table_name}: RC={result.relevance_score:.2f} ({status})")

    print("\n" + "=" * 60)
    print("Heuristic Test Complete!")
    print("=" * 60)


def test_with_llm():
    """Test using actual LLM (requires ANTHROPIC_API_KEY)."""
    print("=" * 60)
    print("Testing Map-Reduce Schema Agent (LLM Mode)")
    print("=" * 60)

    from utils.llm_client import LLMClient

    llm_client = LLMClient()
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
                {"name": "subject_name", "readable_name": "subject name"},
                {"name": "department", "readable_name": "department"}
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
                {"name": "capacity", "readable_name": "capacity"}
            ]
        }
    }

    query = "List all members who are in Computer Science related majors"

    print(f"\nInput Query: {query}")
    print("-" * 60)

    # Run full workflow
    focused_schema = manager.run(
        nl_query=query,
        schema=schema,
        relevance_threshold=0.50
    )

    # Display results
    print("\n1. Decomposed Components:")
    for comp in focused_schema["metadata"]["decomposed_components"]:
        print(f"   - [{comp['type']}] {comp['text']}")

    print(f"\n2. Tables Evaluated: {focused_schema['metadata']['total_tables_evaluated']}")
    print(f"   Relevant Tables (RC >= {focused_schema['metadata']['relevance_threshold']}): "
          f"{focused_schema['metadata']['relevant_tables_count']}")

    print("\n3. All Table Scores:")
    # We need to show all tables, not just relevant ones
    # For this, let's re-run coordinate_workers to see all scores

    print("\n4. Focused Schema (Relevant Tables Only):")
    if focused_schema["table_relevances"]:
        for table_rel in focused_schema["table_relevances"]:
            print(f"   ✓ {table_rel['table_name']}: RC={table_rel['relevance_score']:.2f}")
            print(f"     Reason: {table_rel['reason']}")
            if table_rel['relevant_columns']:
                print(f"     Columns: {', '.join(table_rel['relevant_columns'])}")
    else:
        print("   No tables passed the relevance threshold.")

    print("\n" + "=" * 60)
    print("LLM Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    if "--use-llm" in sys.argv:
        test_with_llm()
    else:
        test_without_llm()
        print("\n💡 Tip: Run with --use-llm flag to test with actual LLM:")
        print("   export ANTHROPIC_API_KEY='your-key'")
        print("   python test_schema_agent.py --use-llm")
