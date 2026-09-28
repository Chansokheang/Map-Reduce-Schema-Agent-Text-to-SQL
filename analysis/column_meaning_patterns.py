"""Offline analysis of data/column_meaning.json (BIRD-supplied dev column meanings).

Questions answered:
1. What kind of information does it add beyond the supplied description CSVs?
2. Leakage check: does it contain values that only gold SQL (not question/evidence) uses?
3. Would it have separated the right column or value in the failures of selected.json?
Reads gold; never used at inference. Writes analysis/column_meaning_patterns/summary.json.
"""
import csv, io, json, re, sys, collections
from pathlib import Path
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.output_width_patterns import words, analyse_query
OUT = ROOT / "analysis/column_meaning_patterns"
DELIM = "\t----- bird -----\t"
GENERIC = {"column", "table", "database", "represent", "store", "text", "integer", "real", "value", "data", "used",
           "contain", "indicate", "type", "can", "the", "each", "which", "specific", "information", "field", "record",
           "row", "entry", "possible", "include", "such", "example", "e.g", "format", "refer"}

CATEGORIES = {
    "value_list": re.compile(r"possible values|can take (on )?(one of )?the following|values? (are|include|such as|like)|"
                             r"such as|e\.g\.|for example|either|one of the following", re.I),
    "identifier": re.compile(r"identif|primary key|unique|foreign key|references", re.I),
    "format": re.compile(r"format|yyyy|mm/dd|dd/mm|hh:mm|timestamp|decimal|percentage|in (seconds|minutes|milliseconds|kg|cm)", re.I),
    "null_or_empty_meaning": re.compile(r"\bnull\b|empty|missing|not available|none", re.I),
    "abbreviation_or_code": re.compile(r"abbreviat|stands for|short for|code for|\bcode\b|acronym|refers to", re.I),
}


def load_csv_descriptions(schemas):
    out = {}
    for db, tables in schemas.items():
        for table, info in tables.items():
            text = info.get("supplied_description_csv")
            if not text:
                continue
            for row in csv.DictReader(io.StringIO(text)):
                col = (row.get("original_column_name") or "").strip()
                if col:
                    out[(db, table.casefold(), col.casefold())] = " ".join(
                        (row.get(k) or "") for k in ("column_name", "column_description", "value_description"))
    return out


def string_literals(sql):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    return {l.this for l in tree.find_all(exp.Literal) if l.is_string and l.this.strip("%_ ")}


def meaning_words(text, db):
    return words(text) - GENERIC - words(db.replace("_", " "))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    cm = json.load(open(ROOT / "data/column_meaning.json", encoding="utf-8"))
    gold = {e["question_id"]: e for e in json.load(open(ROOT / "data/bird_data/dev.json", encoding="utf-8"))}
    pred = {int(k): v.split(DELIM)[0] for k, v in json.load(open(ROOT / "output/claude_headless_v6/selected.json", encoding="utf-8")).items()}
    schemas = json.load(open(ROOT / "output/selection_experiment/v2/schemas.json", encoding="utf-8"))
    csv_desc = load_csv_descriptions(schemas)
    category = {}
    for line in open(ROOT / "analysis/v6_accuracy_audit/details.jsonl", encoding="utf-8"):
        d = json.loads(line)
        category[d["question_id"]] = d["results"]["selected"]["category"]

    by_col = collections.defaultdict(list)          # (db, column) -> [meaning text]
    cm_by_db = collections.defaultdict(str)
    for key, text in cm.items():
        db, table, col = key.split("|")
        by_col[(db, col.casefold())].append(text)
        cm_by_db[db] += "\n" + text
    csv_by_col = collections.defaultdict(list)
    csv_by_db = collections.defaultdict(str)
    for (db, table, col), text in csv_desc.items():
        csv_by_col[(db, col)].append(text)
        csv_by_db[db] += "\n" + text

    summary = {"entries": len(cm), "csv_description_entries": len(csv_desc)}

    # 1. Content categories, and whether the CSV already carries a value description.
    cats = collections.Counter()
    adds_values = 0
    for key, text in cm.items():
        db, table, col = key.split("|")
        for name, rx in CATEGORIES.items():
            if rx.search(text):
                cats[name] += 1
        csv_text = csv_desc.get((db, table.casefold(), col.casefold()), "")
        if CATEGORIES["value_list"].search(text) and not CATEGORIES["value_list"].search(csv_text):
            adds_values += 1
    summary["content_categories"] = dict(cats)
    summary["value_lists_not_in_csv"] = adds_values
    summary["median_chars"] = {"column_meaning": sorted(len(v) for v in cm.values())[len(cm) // 2],
                               "csv": sorted(len(v) for v in csv_desc.values())[len(csv_desc) // 2]}

    # 2. Leakage check on string literals used by gold SQL.
    lit = collections.Counter()
    only_gold_examples = []
    for qid, g in gold.items():
        text = (g["question"] + " " + (g.get("evidence") or "")).casefold()
        for v in string_literals(g["SQL"]):
            core = v.strip("%_ ").casefold()
            lit["gold_string_literals"] += 1
            in_text = core in text
            in_cm = core in cm_by_db[g["db_id"]].casefold()
            in_csv = core in csv_by_db[g["db_id"]].casefold()
            lit["in_question_or_evidence"] += in_text
            lit["in_column_meaning"] += in_cm
            lit["in_csv"] += in_csv
            if not in_text:
                lit["not_in_text"] += 1
                lit["not_in_text_but_in_column_meaning"] += in_cm
                lit["not_in_text_but_in_csv"] += in_csv
                if in_cm and len(only_gold_examples) < 25:
                    only_gold_examples.append((qid, v))
    summary["gold_literal_sources"] = dict(lit)
    summary["gold_literals_not_in_text_found_in_column_meaning_examples"] = only_gold_examples

    # 3a. Literal failures: gold uses a value the prediction lacks. Does column_meaning name it?
    fail = [q for q in gold if category.get(q) not in ("strict", "gold_error", None)]
    lit_fail = {"failures_with_missing_gold_literal": 0, "missing_literal_in_question_or_evidence": 0,
                "missing_literal_in_column_meaning": 0, "missing_literal_only_in_column_meaning": 0}
    lit_cases = []
    for q in fail:
        g, p = gold[q], pred[q]
        missing = {v for v in string_literals(g["SQL"])} - string_literals(p)
        missing = {v for v in missing if v.strip("%_ ")}
        if not missing:
            continue
        lit_fail["failures_with_missing_gold_literal"] += 1
        text = (g["question"] + " " + (g.get("evidence") or "")).casefold()
        cmt, csvt = cm_by_db[g["db_id"]].casefold(), csv_by_db[g["db_id"]].casefold()
        a = any(v.strip("%_ ").casefold() in text for v in missing)
        b = any(v.strip("%_ ").casefold() in cmt for v in missing)
        lit_fail["missing_literal_in_question_or_evidence"] += a
        lit_fail["missing_literal_in_column_meaning"] += b
        if b and not a and not any(v.strip("%_ ").casefold() in csvt for v in missing):
            lit_fail["missing_literal_only_in_column_meaning"] += 1
            lit_cases.append((q, sorted(missing), sorted(string_literals(p) - string_literals(g["SQL"]))))
    summary["literal_failures"] = lit_fail
    summary["literal_failures_only_column_meaning_has_value"] = lit_cases

    # 3b. Column swaps in the output: prediction projects P where gold projects G.
    swap = collections.Counter()
    swap_cases = []
    for q in fail:
        g = gold[q]
        try:
            G, P = analyse_query(g["SQL"]), analyse_query(pred[q])
        except Exception:
            continue
        gonly, ponly = G["out_cols"] - P["out_cols"], P["out_cols"] - G["out_cols"]
        if not gonly or not ponly:
            continue
        qw = words(g["question"] + " " + (g.get("evidence") or ""))
        for source, table in (("column_meaning", by_col), ("csv", csv_by_col)):
            gs = max((len(meaning_words(t, g["db_id"]) & qw) for c in gonly for t in table.get((g["db_id"], c), [])), default=None)
            ps = max((len(meaning_words(t, g["db_id"]) & qw) for c in ponly for t in table.get((g["db_id"], c), [])), default=None)
            if gs is None or ps is None:
                swap[f"{source}:no_description"] += 1
                continue
            swap[f"{source}:gold_column_closer"] += gs > ps
            swap[f"{source}:pred_column_closer"] += ps > gs
            swap[f"{source}:tie"] += gs == ps
            if source == "column_meaning":
                swap_cases.append((q, sorted(gonly), gs, sorted(ponly), ps))
    swap["questions"] = len({c[0] for c in swap_cases})
    summary["output_column_swaps"] = dict(swap)
    summary["output_column_swap_cases"] = swap_cases

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
