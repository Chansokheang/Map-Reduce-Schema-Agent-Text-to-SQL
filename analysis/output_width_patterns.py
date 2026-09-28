"""Offline analysis: why do predictions return extra or missing output columns?

For every output column that differs between selected SQL and gold, record the role the
column plays in its own query (ORDER BY key, filter, group key, identifier, name part)
and whether the question or evidence mentions it. Then test question-text cues against
gold behaviour on all 1534 dev questions. Reads gold; never used at inference.
Writes analysis/output_width_patterns/{cases.csv, summary.json}.
"""
import csv, io, json, re, sys, collections
from pathlib import Path
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/output_width_patterns"
DELIM = "\t----- bird -----\t"
STOP = {"the", "of", "a", "an", "in", "for", "to", "is", "are", "and", "or", "on", "at", "by", "with",
        "this", "that", "which", "who", "what", "its", "their", "his", "her", "be", "was", "were", "it"}
NAME_PARTS = {"forename", "surname", "first_name", "last_name", "firstname", "lastname", "first", "last"}
SUPERLATIVE = re.compile(r"\b(highest|lowest|most|least|max(imum)?|min(imum)?|top|best|worst|fastest|slowest|largest|"
                         r"smallest|biggest|oldest|youngest|earliest|latest|longest|shortest|greatest|heaviest|tallest|"
                         r"first|last|higher|lower|more|less|better)\b", re.I)
VALUE_Q = re.compile(r"^\s*(what\s+(is|was|are|were)|what's|how\s+(much|many|long|old|tall|heavy|fast))\b", re.I)
ENTITY_Q = re.compile(r"^\s*(which|who|whose|name|list|please list|identify|find)\b", re.I)
DIRECTIVE = re.compile(r"(?:[.?;,]|\band\b)\s*(please\s+)?(indicate|state|list|give|provide|include|show|mention|tell|"
                       r"name|identify|describe|write down|calculate)\b|\b(along with|as well as|together with|"
                       r"and (?:its|their|his|her)|with (?:its|their|his|her))\b", re.I)


def words(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    out = set()
    for w in re.split(r"[^A-Za-z0-9]+", text.lower()):
        if not w or w in STOP:
            continue
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def load_aliases(schemas):
    """db -> column name (casefold) -> list of alias word-sets from name and supplied descriptions."""
    aliases = {}
    for db, tables in schemas.items():
        cols = collections.defaultdict(list)
        for info in tables.values():
            text = info.get("supplied_description_csv")
            if not text:
                continue
            for row in csv.DictReader(io.StringIO(text)):
                raw = (row.get("original_column_name") or "").strip()
                if not raw:
                    continue
                key = raw.casefold()
                for alias in (raw, row.get("column_name") or "", row.get("column_description") or ""):
                    ws = words(alias)
                    if ws and len(ws) <= 5:
                        cols[key].append(ws)
        aliases[db] = cols
    return aliases


def mentioned(col, text, db_aliases):
    tw = words(text)
    cands = db_aliases.get(col, []) or [words(col)]
    return any(ws and ws <= tw for ws in cands)


def outer_select(tree):
    return tree if isinstance(tree, exp.Select) else tree.find(exp.Select)


def cols_in(node):
    return {c.name.casefold() for c in node.find_all(exp.Column)} if node is not None else set()


def analyse_query(sql):
    tree = sqlglot.parse_one(sql, read="sqlite")
    sel = outer_select(tree)
    outputs = [cols_in(e) for e in sel.expressions]
    return {"width": len(sel.expressions), "outputs": outputs, "out_cols": set().union(*outputs) if outputs else set(),
            "order": cols_in(sel.args.get("order")), "where": cols_in(sel.args.get("where")) | cols_in(sel.args.get("having")),
            "group": cols_in(sel.args.get("group")), "limit": sel.args.get("limit") is not None,
            "join": set().union(*[cols_in(j.args.get("on")) for j in sel.args.get("joins") or []]) if sel.args.get("joins") else set(),
            "agg_output": any(e.find(exp.AggFunc) for e in sel.expressions)}


def id_like(c):
    return c == "id" or c.endswith("_id") or c.endswith("id") or c in {"code", "uuid", "cds", "cdscode", "setcode"}


def roles(col, q):
    r = []
    if col in q["order"]: r.append("order_key")
    if col in q["where"]: r.append("filter")
    if col in q["group"]: r.append("group_key")
    if col in q["join"]: r.append("join_key")
    if id_like(col): r.append("id_like")
    if col in NAME_PARTS or "name" in col: r.append("name_like")
    return r


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    gold = {e["question_id"]: e for e in json.load(open(ROOT / "data/bird_data/dev.json", encoding="utf-8"))}
    pred = {int(k): v.split(DELIM)[0] for k, v in json.load(open(ROOT / "output/claude_headless_v6/selected.json", encoding="utf-8")).items()}
    schemas = json.load(open(ROOT / "output/selection_experiment/v2/schemas.json", encoding="utf-8"))
    aliases = load_aliases(schemas)
    category = {}
    for line in open(ROOT / "analysis/v6_accuracy_audit/details.jsonl", encoding="utf-8"):
        d = json.loads(line)
        category[d["question_id"]] = d["results"]["selected"]["category"]

    rows, cases = [], []
    for qid, g in gold.items():
        try:
            G, P = analyse_query(g["SQL"]), analyse_query(pred[qid])
        except Exception:
            continue
        q, ev, db = g["question"], g.get("evidence") or "", g["db_id"]
        rec = {"qid": qid, "db": db, "cat": category.get(qid), "ok": category.get(qid) == "strict", "G": G, "P": P,
               "question": q, "evidence": ev,
               "value_q": bool(VALUE_Q.search(q)), "entity_q": bool(ENTITY_Q.search(q)),
               "superlative": bool(SUPERLATIVE.search(q)), "directive": bool(DIRECTIVE.search(q))}
        rows.append(rec)
        if G["width"] == P["width"] and G["out_cols"] == P["out_cols"]:
            continue
        for kind, cols, owner in (("extra", P["out_cols"] - G["out_cols"], P), ("missing", G["out_cols"] - P["out_cols"], G)):
            for c in sorted(cols):
                cases.append({"qid": qid, "db": db, "category": rec["cat"], "kind": kind, "column": c,
                              "roles": "|".join(roles(c, owner)) or "plain",
                              "in_question": mentioned(c, q, aliases[db]), "in_evidence": c in ev.casefold() or mentioned(c, ev, aliases[db]),
                              "gold_width": G["width"], "pred_width": P["width"], "question": q, "evidence": ev})

    summary = {"n": len(rows)}
    # 1. The pure width failures labelled by the execution audit.
    for kind, cat in (("extra", "extra_columns"), ("missing", "missing_columns")):
        sub = [c for c in cases if c["category"] == cat and c["kind"] == kind]
        role_counts = collections.Counter(r for c in sub for r in c["roles"].split("|"))
        summary[f"pure_{cat}"] = {
            "questions": len({c["qid"] for c in sub}), "columns": len(sub),
            "roles": dict(role_counts.most_common()),
            "mentioned_in_question_pct": pct(sum(c["in_question"] for c in sub), len(sub)),
            "mentioned_in_evidence_pct": pct(sum(c["in_evidence"] for c in sub), len(sub)),
            "examples": [(c["qid"], c["column"], c["roles"], c["in_question"]) for c in sub]}

    # 2. Gold behaviour for ORDER BY keys, by question form (all 1534).
    def sort_key_table(subset_name, pick):
        out = {}
        for label, cond in (("value question (what is / how much ...)", lambda r: r["value_q"]),
                            ("entity question (which / who / list ...)", lambda r: r["entity_q"] and not r["value_q"]),
                            ("other form", lambda r: not r["entity_q"] and not r["value_q"])):
            for mention_label, mcond in (("key mentioned in question", True), ("key not mentioned", False)):
                n = gp = pp = err = 0
                for r in rows:
                    Q = pick(r)
                    keys = Q["order"]
                    if not (Q["limit"] and keys) or not cond(r):
                        continue
                    key = sorted(keys)[0]
                    if mentioned(key, r["question"], aliases[r["db"]]) != mcond:
                        continue
                    n += 1
                    gp += bool(keys & r["G"]["out_cols"])
                    pp += bool(keys & r["P"]["out_cols"])
                    err += bool(keys & r["G"]["out_cols"]) != bool(keys & r["P"]["out_cols"])
                out[f"{label} / {mention_label}"] = {"n": n, "gold_projects_sort_key_pct": pct(gp, n),
                                                      "pred_projects_sort_key_pct": pct(pp, n), "disagreements": err}
        summary[subset_name] = out
    sort_key_table("sort_key_projection_gold_superlatives", lambda r: r["G"])

    # 3. Directive clauses ("Indicate ...", "along with ...") and output width.
    tbl = {}
    for label, cond in (("directive clause present", lambda r: r["directive"]), ("no directive clause", lambda r: not r["directive"])):
        sub = [r for r in rows if cond(r)]
        tbl[label] = {"n": len(sub), "gold_width_ge2_pct": pct(sum(r["G"]["width"] >= 2 for r in sub), len(sub)),
                      "pred_width_ge2_pct": pct(sum(r["P"]["width"] >= 2 for r in sub), len(sub)),
                      "pred_wider_than_gold": sum(r["P"]["width"] > r["G"]["width"] for r in sub),
                      "pred_narrower_than_gold": sum(r["P"]["width"] < r["G"]["width"] for r in sub)}
    summary["directive_clause_vs_width"] = tbl

    # 4. Aggregate questions: does gold add the grouping entity next to the aggregate?
    agg = [r for r in rows if r["G"]["agg_output"] or r["P"]["agg_output"]]
    summary["aggregate_outputs"] = {
        "n": len(agg), "pred_adds_non_aggregate_column": sum(r["P"]["width"] > r["G"]["width"] for r in agg),
        "pred_drops_column": sum(r["P"]["width"] < r["G"]["width"] for r in agg)}

    # 5. Width errors overall, split by the text features.
    width_err = [r for r in rows if r["G"]["width"] != r["P"]["width"]]
    summary["width_errors"] = {"total": len(width_err),
        "pred_wider": sum(r["P"]["width"] > r["G"]["width"] for r in width_err),
        "pred_narrower": sum(r["P"]["width"] < r["G"]["width"] for r in width_err),
        "rate_by_feature": {f: {"with": pct(sum(r[f] for r in width_err), sum(r[f] for r in rows)),
                                "without": pct(sum(not r[f] for r in width_err), sum(not r[f] for r in rows))}
                            for f in ("value_q", "entity_q", "superlative", "directive")}}

    with open(OUT / "cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        w.writeheader(); w.writerows(cases)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
