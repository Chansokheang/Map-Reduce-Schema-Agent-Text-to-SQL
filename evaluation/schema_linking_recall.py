"""
Schema Linking Recall Evaluation

Evaluates how well the Map-Reduce Schema Agent identifies the correct tables
needed for each query by comparing agent-selected tables against tables
referenced in the BIRD benchmark ground truth SQL.

Metrics:
  - Recall:    |predicted ∩ gold| / |gold|
  - Precision: |predicted ∩ gold| / |predicted|
  - F1:        harmonic mean of precision and recall
"""

import sys
import os
import re
import json
import argparse
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def extract_tables_from_sql(sql: str) -> set[str]:
    """
    Extract table names from SQL by matching FROM and JOIN clauses.
    Handles subqueries, CTEs, aliases, and backtick-quoted identifiers.
    Returns a set of lowercase table names.
    """
    # Remove string literals to avoid false matches
    cleaned = re.sub(r"'[^']*'", "''", sql)
    cleaned = re.sub(r'"[^"]*"', '""', cleaned)

    tables = set()

    # Match: FROM table, JOIN table, FROM `table`, JOIN `table`
    # Captures table name after FROM/JOIN, before optional AS/alias/ON/WHERE etc.
    pattern = r'(?:FROM|JOIN)\s+`?(\w+)`?'
    for match in re.finditer(pattern, cleaned, re.IGNORECASE):
        table = match.group(1)
        # Skip CTE names and SQL keywords that aren't real tables
        if table.upper() in ('SELECT', 'WHERE', 'GROUP', 'ORDER', 'HAVING',
                             'UNION', 'EXCEPT', 'INTERSECT', 'VALUES', 'SET'):
            continue
        tables.add(table.lower())

    # Remove CTE names defined in WITH clauses
    cte_pattern = r'WITH\s+(\w+)\s+AS\s*\('
    for match in re.finditer(cte_pattern, cleaned, re.IGNORECASE):
        cte_name = match.group(1).lower()
        tables.discard(cte_name)

    # Also handle chained CTEs: ), cte_name AS (
    chained_cte_pattern = r'\)\s*,\s*(\w+)\s+AS\s*\('
    for match in re.finditer(chained_cte_pattern, cleaned, re.IGNORECASE):
        cte_name = match.group(1).lower()
        tables.discard(cte_name)

    # Remove subquery aliases (FROM (subquery) AS alias)
    subquery_alias_pattern = r'\)\s+AS\s+(\w+)'
    for match in re.finditer(subquery_alias_pattern, cleaned, re.IGNORECASE):
        alias = match.group(1).lower()
        tables.discard(alias)

    return tables


def load_gold_tables(dev_json_path: str) -> list[dict]:
    """Load BIRD dev.json and extract gold tables for each question."""
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for item in data:
        gold_tables = extract_tables_from_sql(item['SQL'])
        results.append({
            'question_id': item.get('question_id', len(results)),
            'db_id': item['db_id'],
            'question': item['question'],
            'gold_sql': item['SQL'],
            'gold_tables': gold_tables,
            'difficulty': item.get('difficulty', 'unknown'),
        })
    return results


def load_predicted_tables(schema_agent_path: str) -> list[set[str]]:
    """Load schema agent output and extract predicted tables per question."""
    predicted = []
    with open(schema_agent_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tables = set()
            for tr in record.get('table_relevances', []):
                tables.add(tr['table_name'].strip().lower())
            predicted.append(tables)
    return predicted


def compute_metrics(predicted: set, gold: set) -> dict:
    """Compute precision, recall, F1 for a single question."""
    if not gold:
        return {'recall': 1.0, 'precision': 1.0, 'f1': 1.0}

    tp = len(predicted & gold)
    recall = tp / len(gold) if gold else 0.0
    precision = tp / len(predicted) if predicted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'recall': recall, 'precision': precision, 'f1': f1}


def evaluate_schema_linking(dev_json_path: str,
                            schema_agent_path: str,
                            show_errors: bool = False,
                            top_k_errors: int = 20) -> dict:
    """Run full schema linking evaluation."""
    gold_data = load_gold_tables(dev_json_path)
    predicted_tables = load_predicted_tables(schema_agent_path)

    n_gold = len(gold_data)
    n_pred = len(predicted_tables)
    n_eval = min(n_gold, n_pred)

    if n_gold != n_pred:
        print(f"Warning: gold has {n_gold} questions, predictions have {n_pred}. "
              f"Evaluating first {n_eval}.")

    # Per-question metrics
    all_metrics = []
    by_difficulty = defaultdict(list)
    by_database = defaultdict(list)
    missed_examples = []

    for i in range(n_eval):
        gold = gold_data[i]['gold_tables']
        pred = predicted_tables[i]
        metrics = compute_metrics(pred, gold)
        metrics['idx'] = i
        metrics['difficulty'] = gold_data[i]['difficulty']
        metrics['db_id'] = gold_data[i]['db_id']

        all_metrics.append(metrics)
        by_difficulty[gold_data[i]['difficulty']].append(metrics)
        by_database[gold_data[i]['db_id']].append(metrics)

        # Track missed tables
        missed = gold - pred
        if missed:
            missed_examples.append({
                'idx': i,
                'question': gold_data[i]['question'],
                'db_id': gold_data[i]['db_id'],
                'difficulty': gold_data[i]['difficulty'],
                'gold_tables': sorted(gold),
                'predicted_tables': sorted(pred),
                'missed_tables': sorted(missed),
                'gold_sql': gold_data[i]['gold_sql'],
            })

    # Aggregate
    def avg(lst, key):
        return sum(m[key] for m in lst) / len(lst) if lst else 0.0

    summary = {
        'total_questions': n_eval,
        'overall': {
            'recall': avg(all_metrics, 'recall') * 100,
            'precision': avg(all_metrics, 'precision') * 100,
            'f1': avg(all_metrics, 'f1') * 100,
        },
        'perfect_recall_count': sum(1 for m in all_metrics if m['recall'] == 1.0),
        'perfect_recall_pct': sum(1 for m in all_metrics if m['recall'] == 1.0) / n_eval * 100,
        'by_difficulty': {},
        'missed_count': len(missed_examples),
    }

    # By difficulty
    for diff in ['simple', 'moderate', 'challenging']:
        metrics_list = by_difficulty.get(diff, [])
        summary['by_difficulty'][diff] = {
            'count': len(metrics_list),
            'recall': avg(metrics_list, 'recall') * 100,
            'precision': avg(metrics_list, 'precision') * 100,
            'f1': avg(metrics_list, 'f1') * 100,
            'perfect_recall_pct': (
                sum(1 for m in metrics_list if m['recall'] == 1.0) / len(metrics_list) * 100
                if metrics_list else 0.0
            ),
        }

    return summary, missed_examples


def print_summary(summary: dict):
    """Print evaluation summary."""
    print()
    print("=" * 80)
    print("  SCHEMA LINKING EVALUATION")
    print("=" * 80)
    print(f"  Total questions evaluated: {summary['total_questions']}")
    print(f"  Perfect recall (all gold tables found): "
          f"{summary['perfect_recall_count']}/{summary['total_questions']} "
          f"({summary['perfect_recall_pct']:.1f}%)")
    print()

    # Overall
    o = summary['overall']
    print(f"  {'':20s} {'Recall':>10s} {'Precision':>10s} {'F1':>10s} {'Count':>8s} {'Perfect%':>10s}")
    print(f"  {'-'*68}")
    print(f"  {'Overall':20s} {o['recall']:10.2f} {o['precision']:10.2f} {o['f1']:10.2f} "
          f"{summary['total_questions']:8d} {summary['perfect_recall_pct']:10.1f}")

    # By difficulty
    for diff in ['simple', 'moderate', 'challenging']:
        d = summary['by_difficulty'].get(diff, {})
        if d.get('count', 0) > 0:
            print(f"  {diff:20s} {d['recall']:10.2f} {d['precision']:10.2f} {d['f1']:10.2f} "
                  f"{d['count']:8d} {d['perfect_recall_pct']:10.1f}")

    print("=" * 80)


def print_missed_examples(missed_examples: list, top_k: int = 20):
    """Print examples where recall < 1.0."""
    if not missed_examples:
        print("\n  All gold tables were correctly identified!")
        return

    print(f"\n  MISSED TABLE EXAMPLES (showing {min(top_k, len(missed_examples))} "
          f"of {len(missed_examples)} total)")
    print(f"  {'-'*70}")

    for ex in missed_examples[:top_k]:
        print(f"\n  Q{ex['idx']} [{ex['db_id']}] ({ex['difficulty']})")
        print(f"    Question:  {ex['question'][:100]}")
        print(f"    Gold:      {ex['gold_tables']}")
        print(f"    Predicted: {ex['predicted_tables']}")
        print(f"    Missed:    {ex['missed_tables']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate schema linking recall against BIRD ground truth"
    )
    parser.add_argument('--dev_json', type=str, required=True,
                        help="Path to BIRD dev.json with gold SQL")
    parser.add_argument('--schema_agent_output', type=str, required=True,
                        help="Path to schema_agent_output.jsonl")
    parser.add_argument('--show_errors', action='store_true',
                        help="Show examples where recall < 1.0")
    parser.add_argument('--top_k_errors', type=int, default=20,
                        help="Number of error examples to show (default: 20)")
    parser.add_argument('--output_json', type=str, default=None,
                        help="Save detailed results to JSON file")
    args = parser.parse_args()

    summary, missed_examples = evaluate_schema_linking(
        args.dev_json,
        args.schema_agent_output,
        show_errors=args.show_errors,
        top_k_errors=args.top_k_errors,
    )

    print_summary(summary)

    if args.show_errors:
        print_missed_examples(missed_examples, args.top_k_errors)

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': summary,
                'missed_examples': missed_examples,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n  Detailed results saved to: {args.output_json}")
