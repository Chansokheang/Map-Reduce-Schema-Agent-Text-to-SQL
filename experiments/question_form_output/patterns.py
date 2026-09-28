"""Question-form classification from question text only (no SQL, no gold).

Five forms, taken from analysis/output_width_patterns/README.md. Each question gets at
most one form, checked in this order: rank, count_then_list, entity_then_attribute /
value_then_additive (two-part questions), list_entity (single sentence).
"""
import re

ATTR = (r"(name|id|title|code|number|date|type|address|amount|value|score|rate|percentage|"
        r"count|total|description|url|website|time|year|location|email|phone|zip|city|countr(y|ie)|"
        r"label|element|status|uuid|api|birthday|age|sex|gender|height|weight|salary|income|"
        r"language|text|power|rank|point|position|nationality|colou?r|body|dob|link|ratio|average|"
        r"reference|nickname|surname|forename|detail|record|length|duration|speed|rating|index)(s|es)?")
ATTRIBUTE_WORDS = re.compile(r"\b" + ATTR + r"\b", re.I)
ADDITIVE = re.compile(r"\b(also|include|including|along with|as well as|together with|in addition)\b", re.I)
ENTITY_START = re.compile(r"^\s*((in|from|on|at|for|of|to|by|under)\s+)?(which|who|whose|whom)\b", re.I)
SECOND_QUESTION = re.compile(r"(\band|,)\s+(what|which|who|how|where|when)\b", re.I)
HOW_MANY = re.compile(r"^\s*how many\b", re.I)
RANK = re.compile(r"^\s*rank\b", re.I)
DIRECTIVE_SENTENCE = re.compile(r"^\s*(please\s+)?(also,?\s+)?(indicate|state|list|give|provide|tell|show|mention|"
                                r"name|identify|find|include)\b", re.I)
LIST_VERB = re.compile(r"^\s*(please\s+)?(list|name)\s+(?P<rest>.*)$", re.I)
LEAD = re.compile(r"^\s*(out|down|all|the|of|top|\d+|one|two|three|four|five|six|seven|eight|nine|ten|at|least)\s+", re.I)
BOUNDARY = re.compile(r"\b(whose|who|that|which|with|from|in|on|at|by|for|of|where|when|and|having|has|have|"
                      r"were|was|are|is)\b|[,?.]", re.I)
AND_ATTRIBUTE = re.compile(r"\band\s+(the\s+|its\s+|their\s+|his\s+|her\s+)?([a-z'-]+\s+){0,3}" + ATTR + r"\b", re.I)
COMMA_ATTRIBUTE = re.compile(r",\s*(the\s+)?([a-z'-]+\s+){0,2}" + ATTR + r"\b", re.I)
STARTS_WITH_OF = re.compile(r"^\s*of\b", re.I)

FORMS = ("rank", "count_then_list", "entity_then_attribute", "value_then_additive", "list_entity")


def sentences(question):
    parts = [p.strip() for p in re.split(r"(?<=[?.])\s+", question.strip()) if p.strip()]
    return parts or [question.strip()]


def split_camel(text):
    return re.sub(r"([a-z])([A-Z])", lambda m: m.group(1) + " " + m.group(2), text)


def classify(question):
    s = sentences(question)
    first, later = s[0], s[1:]
    directive_later = [p for p in later if DIRECTIVE_SENTENCE.search(p)]
    if RANK.search(first):
        return "rank"
    if HOW_MANY.search(first) and any(re.match(r"^\s*(please\s+)?(list|name|state|give|show|provide|indicate|identify)\b",
                                               p, re.I) for p in later):
        return "count_then_list"
    if ENTITY_START.search(first) and directive_later:
        if any(ADDITIVE.search(p) for p in directive_later) or SECOND_QUESTION.search(first):
            return "value_then_additive"
        return "entity_then_attribute"
    if directive_later or (later and any(ADDITIVE.search(p) for p in later)):
        return "value_then_additive"
    if len(s) == 1:
        m = LIST_VERB.match(first)
        if m and not ADDITIVE.search(first) and not SECOND_QUESTION.search(first):
            rest = m.group("rest")
            while LEAD.match(rest):
                rest = LEAD.sub("", rest, count=1)
            b = BOUNDARY.search(rest)
            phrase, tail = (rest[:b.start()], rest[b.start():]) if b else (rest, "")
            if (phrase.strip() and not ATTRIBUTE_WORDS.search(split_camel(phrase)) and "'" not in phrase
                    and not STARTS_WITH_OF.match(tail) and not AND_ATTRIBUTE.search(tail)
                    and not COMMA_ATTRIBUTE.search(tail)):
                return "list_entity"
    return None
