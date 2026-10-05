"""Phrase-to-column alignment as the selection task.

The shipped judge is asked "which candidate is best", and it answers with a global quality
argument. Two failures that survived every other intervention are not quality judgements at all,
they are scoping questions about individual phrases:

  Q56 "schools with a MAILING STATE address ... active in SAN JOAQUIN CITY"
      -> MailState for the first phrase, City for the second. We used MailCity for both.
  Q15 "the highest AVERAGE SCORE in Reading", where AvgScrRead is already a per-school average
      -> take the max of that column, do not average it again.

So this prompt asks a different question: for each phrase in the question that names data, which
column does each candidate use, and does that match what the phrase says? The verdict follows from
the mapping rather than from an overall impression.
"""
import json
import re

SYSTEM = """You check whether a SQL query uses the right column for each phrase in a question.

Work phrase by phrase, not overall:
1. List the phrases in the question that refer to data - a filter, a value, a thing to display.
2. For each phrase, say which column the query uses for it.
3. Decide whether that column matches the phrase EXACTLY.

Two things to watch:
- A modifier applies only to the words it attaches to. In "schools with a mailing state address in
  California ... active in San Joaquin city", "mailing" qualifies the STATE, not the CITY: the state
  filter uses a mailing-state column, the city filter uses the plain city column.
- If a column already holds an aggregate (a name or description saying average, total, count), the
  question asking for "the highest average X" means the MAXIMUM of that column, not AVG() of it.

Answer with JSON only:
{"mappings": [{"phrase": "...", "column": "...", "matches": true}], "verdict": "ok" or "mismatch",
 "reason": "<one sentence>"}"""

USER = """Question: {question}
Evidence: {evidence}

Relevant columns available:
{columns}

Query:
{sql}

Check each phrase. JSON only."""


def build(question, evidence, columns, sql):
    listed = "\n".join(f"  - {c}" for c in columns)
    return USER.format(question=question, evidence=evidence or "(none)", columns=listed, sql=sql)


def parse(text):
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return {"verdict": str(d.get("verdict", "")).lower(),
            "reason": str(d.get("reason", ""))[:200],
            "mappings": d.get("mappings", [])}


def check(client, question, evidence, columns, sql):
    """True when the query's phrase-to-column mapping is judged sound."""
    out = client.complete(prompt=build(question, evidence, columns, sql), system_prompt=SYSTEM,
                          max_tokens=700, temperature=0.0)
    got = parse(out)
    if not got:
        return None, ""
    bad = any(m.get("matches") is False for m in got["mappings"] if isinstance(m, dict))
    return (got["verdict"] == "ok" and not bad), got["reason"]
