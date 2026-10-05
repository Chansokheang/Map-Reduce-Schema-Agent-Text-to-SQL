"""Pairwise tournament selection (Agentar-Scale-SQL stage 2), without the trained selector.

Candidates are grouped by execution result; one representative per group enters a round-robin
where each pair is compared head to head and the winner scores a point. The comparison prompt is
deliberately minimal - question, evidence, the two queries and what they return - rather than the
shipped judge prompt with its 11 rules and 18 criteria.

Each pair is asked twice with the order swapped. Agreement means a real preference; disagreement
means position decided it, which is worth knowing before trusting any tournament.
"""
import json
import re

SYSTEM = """You compare two SQL queries that answer the same question and decide which one is right.

You are given the question, any evidence, and for each query the SQL and the rows it returns.
Judge only which query answers the question exactly as asked:
- the columns asked for, in the order the question asks for them, and nothing extra
- the rows the question asks for, no more and no less
- the filters and the entity the question names, read literally

Pick the query whose RESULT is what the question asks for. If both results look the same, prefer
the simpler query. Answer with JSON only: {"winner": 1, "reason": "<one sentence>"}"""

USER = """Question: {question}
Evidence: {evidence}

Query 1:
{sql1}
Returns {n1} row(s): {rows1}

Query 2:
{sql2}
Returns {n2} row(s): {rows2}

Which query answers the question? JSON only."""


def build(question, evidence, a, b):
    return USER.format(question=question, evidence=evidence or "(none)",
                       sql1=a["sql"], n1=a["row_count"], rows1=str(a["sample_rows"])[:300],
                       sql2=b["sql"], n2=b["row_count"], rows2=str(b["sample_rows"])[:300])


def parse_winner(text):
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None, ""
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    w = d.get("winner")
    try:
        w = int(w)
    except (TypeError, ValueError):
        return None, str(d.get("reason", ""))
    return (w if w in (1, 2) else None), str(d.get("reason", ""))


def compare(client, question, evidence, a, b):
    """One head-to-head. Returns the winning entry, or None when the answer is unusable."""
    out = client.complete(prompt=build(question, evidence, a, b), system_prompt=SYSTEM,
                          max_tokens=400, temperature=0.0)
    w, reason = parse_winner(out)
    if w is None:
        return None, reason
    return (a if w == 1 else b), reason


def tournament(client, question, evidence, reps, both_orders=True):
    """Round-robin over group representatives. Returns (winner, scores, notes)."""
    scores = {r["candidate_id"]: 0 for r in reps}
    notes = []
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = reps[i], reps[j]
            w1, r1 = compare(client, question, evidence, a, b)
            if w1:
                scores[w1["candidate_id"]] += 1
            if both_orders:
                w2, r2 = compare(client, question, evidence, b, a)
                if w2:
                    scores[w2["candidate_id"]] += 1
                notes.append({"pair": (a["candidate_id"], b["candidate_id"]),
                              "forward": w1["candidate_id"] if w1 else None,
                              "reversed": w2["candidate_id"] if w2 else None,
                              "consistent": bool(w1 and w2 and w1["candidate_id"] == w2["candidate_id"]),
                              "reason": r1[:160]})
    best = max(scores, key=lambda cid: scores[cid])
    return next(r for r in reps if r["candidate_id"] == best), scores, notes
