# Judge grouping A/B: does telling the judge which candidates agree help? (2026-09-18)

User's proposal: run a voting component before the judge, show the judge which candidates
produced the same result, and instruct it not to favour the majority.

Output: output/judge_grouping/v1. Frozen v6 candidates (nothing regenerated), production
judge prompt in both arms, answers restricted to a candidate ID or abstention, 870 calls,
no failures, CLI list cost $25.92 on the Max subscription. Evaluated once on the full set.

| File | Correct / 1534 | EX | Recovered | Regressed |
|---|---:|---:|---:|---:|
| Original selected.json | 1102 | 71.84% | - | - |
| Arm A: judge without grouping (live-judge view) | 1107 | 72.16% | 27 | 22 |
| Arm B: judge with result groups + neutrality line | 1101 | 71.77% | 27 | 28 |

Arm B is 6 behind arm A. The arms chose differently on 62 of 435 triggered questions;
arm B was better on 6 (83, 169, 239, 323, 1032, 1190) and worse on 12 (63, 120, 273, 300,
436, 449, 529, 861, 987, 992, 1282, 1360).

The anti-bias instruction held: on those 62 questions arm B chose the smaller result group
18 times and the larger one 12 times, so it was not pulled toward the majority. The grouping
information changed decisions without improving them.

Arm A's +5 is not a win either: 27 recovered against 22 regressed is the churn a rerun of
the same judge produces, and this harness omits the production fixer, so read it as
reproducing 1102 rather than beating it.

## Context: everything tried on this candidate pool

| Method | Correct |
|---|---:|
| Oracle (best candidate per question) | 1190 |
| Arm A here (production prompt, no grouping) | 1107 |
| Original / v2 control judge (prompt + grouping fields) | 1102 |
| Arm B here (grouping + neutrality) | 1101 |
| Focused requirement verification | 1101 |
| Majority voting variants | 1067-1084 |
| Mechanical override rules (strip repairs / revert outliers) | 1091 / 1071 |
| Disagreement-focused judge | 1092 |

Six methods now re-read the same five candidates; none beats 1102 by a defensible margin.
Agreement counting cannot help on the 206 questions where all five candidates agree on a
wrong answer, and the judge already outperforms every voting rule where they split.
