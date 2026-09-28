"""Describe BIRD dev output projections; never modify predictions or inference prompts."""
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sys
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/column_projection_analysis"


def projection(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        expressions = list(tree.selects)
        return {"expressions":[e.sql(dialect="sqlite") for e in expressions],
                "width":len(expressions),
                "star":any(isinstance(e, exp.Star) or (isinstance(e, exp.Column) and e.is_star) for e in expressions),
                "names":[e.alias_or_name for e in expressions],
                "types":[type(e.unalias()).__name__ for e in expressions],
                "error":None}
    except Exception as error:
        return {"expressions":[], "width":None, "star":False, "names":[], "types":[], "error":str(error)}


def main():
    OUT.mkdir(exist_ok=True)
    gold = json.loads((ROOT / "data/bird_data/dev.json").read_text(encoding="utf-8"))
    predictions = json.loads((ROOT / "output/claude_headless_v6/selected.json").read_text(encoding="utf-8"))
    with (ROOT / "analysis/v6_accuracy_audit/cases.csv").open(encoding="utf-8-sig", newline="") as handle:
        audit = {int(r["question_id"]):r for r in csv.DictReader(handle)}
    rows = []
    for entry in gold:
        qid = entry["question_id"]
        selected, db = predictions[str(qid)].rsplit("\t----- bird -----\t", 1)
        assert db == entry["db_id"]
        g, p = projection(entry["SQL"]), projection(selected)
        rows.append({**entry, "gold_projection":g, "selected_sql":selected, "selected_projection":p,
                     "diagnostic_category":audit[qid]["category"],
                     "projection_mapping":audit[qid]["projection_mapping"]})
    patterns = {"how_many":r"\bhow many\b", "full_name":r"\bfull names?\b",
                "ranking":r"\brank(?:ing|ed|s)?\b", "percentage_or_ratio":r"\b(?:percent(?:age)?|ratio|proportion)\b",
                "starts_yes_no":r"^(?:is|are|does|do|did|was|were|has|have|can)\b"}
    summary = {"n":len(rows), "parser_version":sqlglot.__version__,
               "gold_parse_errors":[r['question_id'] for r in rows if r['gold_projection']['error']],
               "prediction_parse_errors":[r['question_id'] for r in rows if r['selected_projection']['error']],
               "gold_star_ids":[r['question_id'] for r in rows if r['gold_projection']['star']],
               "gold_widths":dict(sorted(Counter(r['gold_projection']['width'] for r in rows if not r['gold_projection']['error']).items())),
               "patterns":{}, "projection_error_ids":{}}
    for name, pattern in patterns.items():
        subset = [r for r in rows if re.search(pattern,r['question'],re.I)]
        summary['patterns'][name] = {"n":len(subset), "ids":[r['question_id'] for r in subset],
            "widths":dict(Counter(r['gold_projection']['width'] for r in subset))}
    for category in ["extra_columns", "missing_columns", "column_order"]:
        summary['projection_error_ids'][category] = [r['question_id'] for r in rows if r['diagnostic_category']==category]
    for filename, data in [('all_projections.json',rows),('summary.json',summary)]:
        (OUT / filename).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    with (OUT / 'all_projections.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields=['question_id','db_id','question','evidence','gold_width','gold_columns','predicted_width','predicted_columns','category','gold_sql','predicted_sql']
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(zip(fields,[r['question_id'],r['db_id'],r['question'],r['evidence'],
                r['gold_projection']['width'],json.dumps(r['gold_projection']['expressions'],ensure_ascii=False),
                r['selected_projection']['width'],json.dumps(r['selected_projection']['expressions'],ensure_ascii=False),
                r['diagnostic_category'],r['SQL'],r['selected_sql']])))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
