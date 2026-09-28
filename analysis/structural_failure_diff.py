"""Offline: structural diff of original selected SQL vs gold, by execution outcome."""
import json, re, sys, collections
from pathlib import Path
import sqlglot
from sqlglot import exp
ROOT = Path(r"C:\Users\user\OneDrive\Documents\02. Master Degree\03. Lab\10. Experiment\03. QA-SQL-Query Augmentation to SQL")
sys.stdout.reconfigure(encoding="utf-8")
DELIM = "\t----- bird -----\t"
gold = {e["question_id"]: e for e in json.load(open(ROOT / "data/bird_data/dev.json", encoding="utf-8"))}
pred = {int(k): v.split(DELIM)[0] for k, v in json.load(open(ROOT / "output/claude_headless_v6/selected.json", encoding="utf-8")).items()}
pq = {r["question_id"]: r for r in json.load(open(ROOT / "output/focused_selection/v1/full_results/evaluation/per_question.json", encoding="utf-8"))}
schema_cols = {}
for db in {e["db_id"] for e in gold.values()}:
    import sqlite3
    con = sqlite3.connect(str(ROOT / "data/bird_data/dev_databases" / db / f"{db}.sqlite"))
    for (t,) in con.execute("select name from sqlite_master where type='table'"):
        for row in con.execute(f'pragma table_info("{t}")'):
            schema_cols.setdefault(db, {}).setdefault(t.lower(), set()).add(row[1].lower())
    con.close()

AGG = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)

def norm_col(c):
    return c.name.lower()

def features(sql, db):
    tree = sqlglot.parse_one(sql, read="sqlite")
    f = {}
    f["tables"] = {t.name.lower() for t in tree.find_all(exp.Table)}
    sel = tree if isinstance(tree, exp.Select) else next(tree.find_all(exp.Select))
    outs = sel.expressions
    f["width"] = len(outs)
    f["proj_cols"] = {norm_col(c) for e in outs for c in e.find_all(exp.Column)}
    f["proj_aggs"] = {type(a).__name__ for e in outs for a in e.find_all(*AGG)}
    f["distinct"] = bool(sel.args.get("distinct")) or any(isinstance(a.this, exp.Distinct) for e in outs for a in e.find_all(*AGG))
    where_cols, lits, notnull, cast, like_ = set(), set(), False, False, False
    for node in list(tree.find_all(exp.Where)) + list(tree.find_all(exp.Having)):
        for c in node.find_all(exp.Column): where_cols.add(norm_col(c))
        for l in node.find_all(exp.Literal): lits.add(str(l.this).lower().strip())
        if node.find(exp.Like): like_ = True
        for n in node.find_all(exp.Is):
            notnull = True
    f["where_cols"], f["literals"], f["notnull"], f["like"] = where_cols, lits, notnull, like_
    f["order"] = bool(tree.find(exp.Order)); f["limit"] = bool(tree.find(exp.Limit))
    f["group"] = bool(tree.find(exp.Group)); f["subq"] = len(list(tree.find_all(exp.Select))) > 1
    f["joins"] = len(list(tree.find_all(exp.Join)))
    f["cast"] = bool(tree.find(exp.Cast)); f["iif_case"] = bool(tree.find(exp.Case) or tree.find(exp.If))
    f["window"] = bool(tree.find(exp.Window))
    return f

rows = []
parse_fail = []
for qid, g in gold.items():
    try:
        fp, fg = features(pred[qid], g["db_id"]), features(g["SQL"], g["db_id"])
    except Exception as e:
        parse_fail.append(qid); continue
    tags = set()
    if fp["tables"] != fg["tables"]:
        tags.add("table_missing" if fg["tables"] - fp["tables"] else "table_extra")
    if fp["width"] != fg["width"]: tags.add("width_" + ("more" if fp["width"] > fg["width"] else "fewer"))
    if fp["proj_cols"] != fg["proj_cols"]: tags.add("proj_cols")
    if fp["proj_aggs"] != fg["proj_aggs"]: tags.add("agg_func")
    if fp["where_cols"] != fg["where_cols"]:
        tags.add("filter_cols_missing" if fg["where_cols"] - fp["where_cols"] else "filter_cols_extra")
    if fp["literals"] != fg["literals"]:
        tags.add("literal_missing" if fg["literals"] - fp["literals"] else "literal_extra")
    if fp["distinct"] != fg["distinct"]: tags.add("distinct_" + ("pred" if fp["distinct"] else "gold"))
    if fp["notnull"] != fg["notnull"]: tags.add("nullpred_" + ("pred" if fp["notnull"] else "gold"))
    if fp["limit"] != fg["limit"]: tags.add("limit_" + ("pred" if fp["limit"] else "gold"))
    if fp["group"] != fg["group"]: tags.add("group_" + ("pred" if fp["group"] else "gold"))
    if fp["subq"] != fg["subq"]: tags.add("subquery_diff")
    if fp["cast"] != fg["cast"]: tags.add("cast_diff")
    if fp["iif_case"] != fg["iif_case"]: tags.add("case_diff")
    if fp["window"] != fg["window"]: tags.add("window_diff")
    ev = (g.get("evidence") or "").lower(); q = g["question"].lower()
    gold_lits_missing = fg["literals"] - fp["literals"]
    lit_in_text = {l for l in gold_lits_missing if l and (l in ev or l in q)}
    rows.append({"qid": qid, "db": g["db_id"], "ok": pq[qid]["scores"]["original"], "tags": tags,
                 "passing": pq[qid]["passing_candidate_count"], "gold_lits_missing": gold_lits_missing,
                 "gold_lit_in_text": lit_in_text, "pred_lits_extra": fp["literals"] - fg["literals"],
                 "gold_proj_missing": fg["proj_cols"] - fp["proj_cols"], "pred_proj_extra": fp["proj_cols"] - fg["proj_cols"],
                 "gold_filter_missing": fg["where_cols"] - fp["where_cols"], "pred_filter_extra": fp["where_cols"] - fg["where_cols"],
                 "difficulty": g.get("difficulty")})
print("parse failures:", parse_fail)
fails = [r for r in rows if not r["ok"]]; oks = [r for r in rows if r["ok"]]
print(f"rows {len(rows)} correct {len(oks)} failures {len(fails)}")
tagc_f = collections.Counter(t for r in fails for t in r["tags"]); tagc_o = collections.Counter(t for r in oks for t in r["tags"])
print("\n### tag frequency: failures vs correct (rate = share of group carrying the tag)")
print(f"{'tag':22s} {'fail n':>6s} {'fail%':>6s} {'ok n':>6s} {'ok%':>6s} {'lift':>5s}")
for t, n in tagc_f.most_common():
    fr, orate = n / len(fails), tagc_o[t] / len(oks)
    print(f"{t:22s} {n:6d} {100*fr:6.1f} {tagc_o[t]:6d} {100*orate:6.1f} {fr/max(orate,1e-9):5.1f}")
print("\n### failures with no structural tag at all (same shape, different semantics):", sum(1 for r in fails if not r["tags"]))
print("### failures whose ONLY tags are in a given family")
fam = {"projection only": {"proj_cols", "width_more", "width_fewer"}, "filter cols only": {"filter_cols_missing", "filter_cols_extra"},
       "literal only": {"literal_missing", "literal_extra"}, "distinct only": {"distinct_pred", "distinct_gold"},
       "null predicate only": {"nullpred_pred", "nullpred_gold"}, "table only": {"table_missing", "table_extra"},
       "agg/group only": {"agg_func", "group_pred", "group_gold"}, "cast/case only": {"cast_diff", "case_diff"},
       "limit only": {"limit_pred", "limit_gold"}}
for name, s in fam.items():
    only = [r["qid"] for r in fails if r["tags"] and r["tags"] <= s]
    print(f"  {name:20s} {len(only):4d}  e.g. {only[:12]}")
print("\n### by difficulty")
for d in ("simple", "moderate", "challenging"):
    n = sum(1 for r in rows if r["difficulty"] == d); k = sum(1 for r in oks if r["difficulty"] == d)
    print(f"  {d:12s} {k}/{n} = {100*k/max(n,1):.1f}%")
print("\n### literal analysis on failures")
lm = [r for r in fails if r["gold_lits_missing"]]
print("failures where gold uses a literal the prediction lacks:", len(lm))
print("  of which the missing gold literal appears verbatim in question/evidence:", sum(1 for r in lm if r["gold_lit_in_text"]))
print("  of which NOT in text (needs DB value lookup or transformation):", sum(1 for r in lm if not r["gold_lit_in_text"]))
print("  sample not-in-text:", [(r["qid"], sorted(r["gold_lits_missing"])[:3], sorted(r["pred_lits_extra"])[:3]) for r in lm if not r["gold_lit_in_text"]][:15])
print("\n### filter-column analysis on failures")
fm = [r for r in fails if r["gold_filter_missing"]]
print("failures where gold filters on a column the prediction does not:", len(fm))
cc = collections.Counter(c for r in fm for c in r["gold_filter_missing"]); print("  most common missing filter columns:", cc.most_common(25))
pc = collections.Counter(c for r in fails for c in r["pred_filter_extra"]); print("  most common extra pred filter columns:", pc.most_common(20))
print("\n### projection-column analysis on failures")
pm = [r for r in fails if r["gold_proj_missing"] or r["pred_proj_extra"]]
print("failures with different projected base columns:", len(pm))
print("  gold projects but pred lacks:", collections.Counter(c for r in fails for c in r["gold_proj_missing"]).most_common(20))
print("  pred projects but gold lacks:", collections.Counter(c for r in fails for c in r["pred_proj_extra"]).most_common(20))
print("\n### tag frequency among the 344 non-recoverable vs 88 recoverable failures")
for label, grp in (("recoverable", [r for r in fails if r["passing"]]), ("non-recoverable", [r for r in fails if not r["passing"]])):
    c = collections.Counter(t for r in grp for t in r["tags"])
    print(f"  {label} n={len(grp)}:", [(t, n, f"{100*n/len(grp):.0f}%") for t, n in c.most_common(10)])
json.dump([{**r, "tags": sorted(r["tags"]), **{k: sorted(r[k]) for k in ("gold_lits_missing","gold_lit_in_text","pred_lits_extra","gold_proj_missing","pred_proj_extra","gold_filter_missing","pred_filter_extra")}} for r in rows],
          open(Path(sys.argv[1]), "w", encoding="utf-8"), indent=1)
