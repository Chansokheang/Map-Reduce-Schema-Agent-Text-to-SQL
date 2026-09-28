"""Check whether conventions observed in dev gold also hold in public BIRD train gold.

Gold-only, offline. Reads no predictions and writes only under output/bird_train_audit.
Usage: python experiments/train_conventions/validate_conventions.py
"""
import json, re, sys, collections
from pathlib import Path
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "output/bird_train_audit/20260916/train_filtered.jsonl"
DEV = ROOT / "data/bird_data/dev.json"
OUT = ROOT / "output/bird_train_audit/20260916/conventions.json"

NAME_WORDS = re.compile(r"\b(name|names|named|title|titles|full name|display ?name|called)\b", re.I)
PER_GROUP = re.compile(r"\b(each|every|per|respective|respectively|for all|by (?:year|month|country|type|category|gender|city|state|school|district|team|user|player|customer|molecule|element))\b", re.I)
HOW_MANY = re.compile(r"\b(how many|number of|count of|total number)\b", re.I)


def id_like(c):
    c = c.lower()
    return c == "id" or c.endswith("_id") or c.endswith("id") and len(c) <= 14 or c in ("code", "uuid", "cds", "cdscode", "setcode")


def name_like(c):
    c = c.lower()
    return "name" in c or c in ("title", "forename", "surname", "first_name", "last_name", "school", "displayname")


def load(path):
    if path.suffix == ".jsonl":
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.load(open(path, encoding="utf-8"))


def analyse(records, label):
    stats = collections.Counter()
    ident = collections.Counter()
    lit = collections.Counter()
    grp = collections.Counter()
    parse_fail = 0
    for r in records:
        try:
            tree = sqlglot.parse_one(r["SQL"], read="sqlite")
        except Exception:
            parse_fail += 1
            continue
        sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        if sel is None:
            parse_fail += 1
            continue
        q, ev = r["question"], r.get("evidence") or ""
        text = q + " " + ev
        outs = sel.expressions
        stats["n"] += 1
        stats["distinct"] += bool(sel.args.get("distinct"))
        stats["group_by"] += bool(tree.find(exp.Group))
        stats["width_" + str(min(len(outs), 4))] += 1
        # Identifier convention: single plain column output, entity-type question
        if len(outs) == 1 and isinstance(outs[0], exp.Column):
            col = outs[0].name
            asks_name = bool(NAME_WORDS.search(q))
            if id_like(col) or name_like(col):
                key = ("asks_name" if asks_name else "no_name_word") + "/" + ("id" if id_like(col) else "name")
                ident[key] += 1
        # Extra identifier: projecting both id-like and name-like base columns
        cols = [c.name for e in outs for c in e.find_all(exp.Column)]
        if any(id_like(c) for c in cols) and any(name_like(c) for c in cols):
            stats["projects_id_and_name"] += 1
        # Literal fidelity
        for node in list(tree.find_all(exp.Where)) + list(tree.find_all(exp.Having)):
            for l in node.find_all(exp.Literal):
                if not l.is_string:
                    continue
                v = l.this
                lit["literals"] += 1
                if "%" in v or "_" in v and "like" in ev.lower():
                    lit["wildcard"] += 1
                    core = v.strip("%")
                    lit["wildcard_evidence_has_like_or_pct"] += bool(re.search(r"like|%", ev, re.I))
                    lit["wildcard_core_in_text"] += bool(core and core in text)
                    continue
                if v in text:
                    lit["verbatim"] += 1
                elif v.lower() in text.lower():
                    lit["case_differs"] += 1
                elif v.replace(" ", "") in text.replace(" ", ""):
                    lit["spacing_differs"] += 1
                else:
                    lit["not_in_text"] += 1
        # Grouping grain
        has_group = bool(tree.find(exp.Group))
        cue = bool(PER_GROUP.search(q))
        grp[("group" if has_group else "nogroup") + "/" + ("cue" if cue else "nocue")] += 1
        if HOW_MANY.search(q) and not cue:
            grp["howmany_nocue"] += 1
            grp["howmany_nocue_with_group"] += has_group
    n = stats["n"]
    def pct(a, b):
        return round(100 * a / b, 1) if b else None
    ident_total_noname = ident["no_name_word/id"] + ident["no_name_word/name"]
    ident_total_name = ident["asks_name/id"] + ident["asks_name/name"]
    report = {
        "label": label, "n_parsed": n, "parse_failures": parse_fail,
        "distinct_pct": pct(stats["distinct"], n), "group_by_pct": pct(stats["group_by"], n),
        "width_distribution": {k: v for k, v in stats.items() if k.startswith("width_")},
        "projects_id_and_name_pct": pct(stats["projects_id_and_name"], n),
        "identifier_convention": {
            "single_column_entity_answers_without_name_word": ident_total_noname,
            "of_which_id_like_pct": pct(ident["no_name_word/id"], ident_total_noname),
            "single_column_answers_with_name_word": ident_total_name,
            "of_which_name_like_pct": pct(ident["asks_name/name"], ident_total_name)},
        "literal_fidelity": {
            "string_literals": lit["literals"],
            "verbatim_pct": pct(lit["verbatim"], lit["literals"]),
            "case_differs_pct": pct(lit["case_differs"], lit["literals"]),
            "spacing_differs_pct": pct(lit["spacing_differs"], lit["literals"]),
            "not_in_text_pct": pct(lit["not_in_text"], lit["literals"]),
            "wildcard_pct": pct(lit["wildcard"], lit["literals"]),
            "wildcard_when_evidence_says_like_or_pct": pct(lit["wildcard_evidence_has_like_or_pct"], lit["wildcard"])},
        "grouping_grain": {
            "group_by_with_per_group_cue_pct": pct(grp["group/cue"], grp["group/cue"] + grp["group/nocue"]),
            "per_group_cue_questions_using_group_by_pct": pct(grp["group/cue"], grp["group/cue"] + grp["nogroup/cue"]),
            "how_many_questions_without_cue": grp["howmany_nocue"],
            "of_which_gold_uses_group_by_pct": pct(grp["howmany_nocue_with_group"], grp["howmany_nocue"])},
    }
    return report


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    reports = {"train_filtered": analyse(load(TRAIN), "train_filtered"), "dev": analyse(load(DEV), "dev")}
    OUT.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
