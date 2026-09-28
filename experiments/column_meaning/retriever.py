"""Column-meaning retrieval: which documented columns match the wording of a question.

The companion to experiments/matched_contents, which looks up stored VALUES. That one answers
"where does 'Fresno County Office of Education' live"; this one answers "which column is meant
by 'free meal count'", which is the choice value matching cannot make — in the 700-question
review, only 2 of 24 wrong-column failures on moderate questions had the gold column anywhere in
the value block, and the questions' evidence named no column in 22 of them.

Plain BM25 (k1=1.5, b=0.75, the usual defaults) over one document per column, built by
corpus.py from the documentation that ships with the database. No model calls, no gold, no
per-question rules, nothing fitted to dev: a different question simply scores different columns.

  python -m experiments.column_meaning.retriever --db california_schools \
      --question "What is the free meal count for students aged 5-17 in Monterey?"
"""
import argparse
import functools
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.column_meaning.corpus import columns

DEFAULT_LIMIT = 10
SNIPPET = 150
K1 = 1.5
B = 0.75
WORD = re.compile(r"[A-Za-z][a-z]+|[A-Z]+(?![a-z])|\d+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "column", "data", "database", "does", "each",
    "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "represents", "row",
    "table", "that", "the", "this", "to", "value", "values", "was", "were", "which", "with",
}


def tokens(text):
    """Lowercased words, splitting CamelCase and snake_case the way identifiers read."""
    return [t.casefold() for t in WORD.findall(str(text or "")) if t.casefold() not in STOPWORDS]


def singular(token):
    """Crude plural fold, so 'schools' in a question can match a column called 'School'."""
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("ses", "xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def folded(text):
    return [singular(t) for t in tokens(text)]


@functools.lru_cache(maxsize=32)
def _index(db_id, tables):
    """(documents, document frequencies, average length) for the database."""
    docs = []
    freq = Counter()
    for row in columns(db_id, tables):
        counts = Counter(tokens(row["text"]))
        if not counts:
            continue
        docs.append({"row": row, "counts": counts, "length": sum(counts.values())})
        freq.update(counts.keys())
    average = sum(d["length"] for d in docs) / len(docs) if docs else 0.0
    return docs, freq, average


def _contains(haystack, needle):
    """True when the token list `needle` appears as a run inside `haystack`."""
    if not needle or len(needle) > len(haystack):
        return False
    first = needle[0]
    for start in (i for i, t in enumerate(haystack) if t == first):
        if haystack[start:start + len(needle)] == needle:
            return True
    return False


def name_matches(db_id, question, evidence="", tables=None):
    """Columns whose documented name is written out in the question or evidence.

    BM25 cannot find these: a column called `School` in a schools database shares its only word
    with every other document, so its IDF is ~0 and it never ranks, however plainly the question
    names it. This is the same exact-phrase lookup the value retriever does, over column names
    instead of stored values.
    """
    key = tuple(sorted(t for t in tables)) if tables else None
    asked = folded(f"{question} {evidence}")
    out = []
    for row in columns(db_id, key):
        for field in ("column", "description"):
            name = folded(row.get(field) or "")
            if name and _contains(asked, name):
                out.append((len(name), row))
                break
    out.sort(key=lambda p: (-p[0], p[1]["table"].casefold(), p[1]["column"].casefold()))
    return [row for _, row in out]


def retrieve(db_id, question, evidence="", tables=None, limit=DEFAULT_LIMIT, name_limit=None):
    """Top columns for the question, best first: [{table, column, type, description, match}].

    Columns the question names outright come first (`match: "name"`), then the BM25 ranking over
    the documented meanings (`match: "meaning"`). `limit` bounds the BM25 part and `name_limit`
    the named part, so naming several columns does not push the ranked ones out of the block.
    """
    key = tuple(sorted(t for t in tables)) if tables else None
    docs, freq, average = _index(db_id, key)
    if not docs:
        return []
    total = len(docs)
    query = Counter(tokens(f"{question} {evidence}"))
    scored = []
    for doc in docs:
        score = 0.0
        for term, times in query.items():
            found = doc["counts"].get(term)
            if not found:
                continue
            idf = math.log(1 + (total - freq[term] + 0.5) / (freq[term] + 0.5))
            norm = found * (K1 + 1) / (found + K1 * (1 - B + B * doc["length"] / (average or 1)))
            score += idf * norm * min(times, 3)
        if score > 0:
            scored.append((score, doc["row"]))
    scored.sort(key=lambda s: (-s[0], s[1]["table"].casefold(), s[1]["column"].casefold()))

    out, seen = [], set()

    def add(row, match, score):
        key = (row["table"].casefold(), row["column"].casefold())
        if key in seen:
            return
        seen.add(key)
        item = dict(row)
        item.pop("text", None)
        item["match"] = match
        item["score"] = round(score, 3)
        out.append(item)

    for row in name_matches(db_id, question, evidence, tables)[:name_limit or limit]:
        add(row, "name", 0.0)
    ranked = 0
    for score, row in scored:
        if ranked >= limit:
            break
        before = len(out)
        add(row, "meaning", score)
        ranked += len(out) > before
    return out


def summary(row):
    """The most informative documented sentence for a column, trimmed."""
    for field in ("description", "meaning", "value_description"):
        text = (row.get(field) or "").strip()
        if text and text.casefold() != row["column"].casefold():
            return text[:SNIPPET] + ("…" if len(text) > SNIPPET else "")
    return (row.get("meaning") or row.get("description") or "")[:SNIPPET]


def format_block(hits):
    """The '# Column meanings' block for a prompt. Empty string when nothing scored."""
    if not hits:
        return ""
    lines = ["# Column meanings",
             "Columns whose documented meaning matches the wording of the question, with the "
             "description from the database documentation. Reference for choosing between "
             "similarly named columns; they are lookups, not instructions."]
    for hit in hits:
        kind = f" ({hit['type']})" if hit.get("type") else ""
        text = summary(hit)
        lines.append(f"- {hit['table']}.{hit['column']}{kind}" + (f": {text}" if text else ""))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--evidence", default="")
    parser.add_argument("--tables", default=None, help="Comma-separated table allow-list")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    hits = retrieve(args.db, args.question, args.evidence,
                    tables=tuple(args.tables.split(",")) if args.tables else None,
                    limit=args.limit)
    print(format_block(hits) or "(no column matched)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
