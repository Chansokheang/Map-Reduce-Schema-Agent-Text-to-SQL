# Question-form output experiment v1: results (2026-09-17)

Output: output/question_form_output/v1. Run once on the full set at the user's request,
evaluated once with the unmodified repository evaluator and original dev.json gold.
Base: original selected.json. Forms come from analysis/output_width_patterns (dev gold),
so this is a dev-derived method, not an independent validation.

Classifier (patterns.py) matched 168 questions by text alone; before freezing, it was
corrected three times for implementation bugs judged from question text only. 168 model
calls (claude-sonnet-5), CLI list cost $3.74 on the Max subscription.

| Method | Correct / 1534 | EX | Recovered | Regressed |
|---|---:|---:|---:|---:|
| Original | 1102 | 71.84% | - | - |
| Projection alignment v1 | 1104 | 71.97% | 7 | 5 |
| **Question form, all five forms (pre-registered primary)** | **1109** | **72.29%** | **17** | **10** |
| Question form without list_entity (pre-registered split) | 1114 | 72.62% | 14 | 2 |

list_entity was flagged as risky before the run because id-over-name did not hold on
train gold. It was the only harmful form.

| Form | Matched | Rewritten | Recovered | Regressed |
|---|---:|---:|---|---|
| rank | 3 | 3 | 17, 726, 728 | none |
| count_then_list | 7 | 4 | 435, 436, 978 | none |
| entity_then_attribute | 46 | 8 | 986, 989 | 773, 1000 |
| value_then_additive | 73 | 10 | 58, 80, 403, 431, 894, 1235 | none |
| list_entity | 39 | 17 | 159, 165, 1366 | 730, 899, 929, 1038, 1044, 1051, 1082, 1147 |

Regressions outside list_entity: Q773 dropped the superhero whose publisher was asked
("Which superhero ...? Indicate the publisher"), gold keeps both. Q1000 added country
following the evidence "full location refers to location+country", gold returns only
location. list_entity replaced names with ids in 8 questions where gold keeps the name.

Caveat on the numbers: the forms and the per-form split were read from the same dev gold
they are scored on, so 1109 and 1114 overstate what to expect on unseen data.
