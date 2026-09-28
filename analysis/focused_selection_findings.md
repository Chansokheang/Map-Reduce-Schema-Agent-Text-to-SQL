# Completed focused-selection experiment

Original: 1102/1534 (71.84%).
Focused verification: 1101/1534 (71.77%).
Recovered 4 original failures and regressed 5 successes; net -1.

## Actual execution

- 435 questions triggered review; all 1534 were exported and scored.
- 786 logged model responses; 50 failed question reviews retained originals.
- 317 alignments marked ambiguity; 199 database probes, 0 probe errors.
- Statuses: {"not_triggered": 1099, "abstained": 371, "selected": 14, "failed": 50}.
- Model usage: {"logged_responses": 786, "reported_total_cost_usd": 45.6734224, "models": ["claude-haiku-4-5-20251001", "claude-sonnet-5"], "note": "CLI reported list cost, not necessarily subscription billing"}.

## Interpretation

This experiment checks independent answer requirements against frozen candidate SQL. It does not
regenerate queries, train a selector or provide gold at inference. Model judgments remain fallible.
Recovery/regression counts refer to strict local execution matching, not semantic proof.
The original-input gold was used; revised dev_20251106 inputs/gold were not substituted.
Original sources were checked at export. Production behavior was not changed.

## Why recoverable failures remained

```json
{
  "Unresolved input ambiguity": 60,
  "No supported requirement clearly fails the original": 15,
  "Retained original after failed verification": 7,
  "No alternative passes every requirement": 1,
  "Original fails a supported requirement; exactly one alternative result group passes every check": 1
}
```

## Changed execution outcomes

Recovered IDs: [48, 453, 856, 887]

Regressed IDs: [10, 220, 405, 529, 805]

## By database

| Database | Original correct | Focused correct | N |
|---|---:|---:|---:|
| california_schools | 62 | 62 | 89 |
| card_games | 128 | 127 | 191 |
| codebase_community | 130 | 130 | 186 |
| debit_card_specializing | 44 | 44 | 64 |
| european_football_2 | 101 | 101 | 129 |
| financial | 75 | 75 | 106 |
| formula_1 | 112 | 114 | 174 |
| student_club | 133 | 133 | 158 |
| superhero | 117 | 116 | 129 |
| thrombosis_prediction | 99 | 99 | 163 |
| toxicology | 101 | 100 | 145 |

<!-- manual-analysis -->

## Manual analysis (2026-09-15, after evaluation)

Evaluated with the frozen original dev.json gold (sha256 630272f2...), the unmodified
repository `execute_model`, 5574 unique SQL/reference pairs. Accuracy denominator 1534.

| Method | Correct | EX | Recoveries | Regressions | Net |
|---|---:|---:|---:|---:|---:|
| Original | 1102 | 71.84% | - | - | - |
| v2 control judge | 1102 | 71.84% | 30 | 30 | 0 |
| v2 broad disagreement judge | 1092 | 71.19% | 24 | 34 | -10 |
| **Focused verification (this run)** | **1101** | **71.77%** | **4** | **5** | **-1** |

The focused protocol did not improve accuracy. It was far more conservative than the
v2 judges (14 switches), so its loss is small, but its four recoveries (48, 453, 856, 887)
are a strict subset of what the v2 control judge already recovered, and two of its five
regressions (405, 805) also regressed under the v2 control judge. Per database: formula_1 +2,
card_games -1, superhero -1, toxicology -1, california_schools 0 (+48, -10).

### The 14 switches, one by one

| Q | DB | Original -> focused | Outcome | Diagnosis |
|---|---|---|---|---|
| 48 | california_schools | County filter on numerator only -> on both terms | recovery | Genuine logic fix; R1/R2 correctly failed the original. |
| 453 | card_games | convertedManaCost -> manaCost | recovery | Schema description says manaCost is the unconverted cost. |
| 856 | formula_1 | DISTINCT time with IS NOT NULL -> plain time | recovery | Gold keeps NULL race times; the original's NULL filter was the error. |
| 887 | formula_1 | circuitId NOT IN -> name NOT IN | recovery | Evidence "not hosted means not in" resolved by name. |
| 10 | california_schools | ORDER BY DESC LIMIT 1 -> equals MAX(AvgScrRead) | **regression** | Top school (653) has no frpm row; winner returns **0 rows**. The verifier saw this in the packet and probe and still passed it, calling the empty result "a downstream join issue". |
| 529 | card_games | Korean AND NOT LIKE Japanese on the same row -> NOT IN (Japanese sets) | **regression** | Gold literally uses the contradictory same-row filter; the semantically sensible winner returns **0 rows** and the verifier passed it as "consistent with the probe". |
| 405 | card_games | 'Portuguese (Brazil)' -> 'Portuguese (Brasil)' | **regression** | Evidence spells "Brasil", gold and the data use "Brazil". The probe on "Brasil" returned **no rows**; the verifier still failed the original for not matching the evidence literal. Count 5 -> 0. |
| 220 | toxicology | DISTINCT element LIMIT 3 -> no DISTINCT | **regression** | The alignment invented R4 "without deduplicating"; gold uses DISTINCT. LIMIT makes the difference visible. |
| 805 | superhero | DISTINCT full_name with IS NOT NULL -> no filters | **regression** | Gold has the same NULL filter as the original. R2 failed the original for an "extra" predicate. |
| 1039 | european_football_2 | join on player_api_id -> player_fifa_api_id | neutral (1->1) | Same result group as the original: an execution no-op. |
| 267 | toxicology | DISTINCT bond_type -> bond_type | neutral (0->0) | Same result group; gold also returns molecule_id. |
| 465 | card_games | list Korean set_translations -> Yes/No via foreign_data | neutral (0->0) | Gold uses set_translations with IIF; no candidate passes. |
| 602 | codebase_community | OwnerUserId IS NOT NULL -> no filter | neutral (0->0) | Gold answers from postHistory; no candidate passes. |
| 1458 | student_club | x100 formula -> evidence formula without x100 | neutral (0->0), wrong pick | C2 passes gold but was failed on R1 and R3; the checks preferred the evidence-literal formula. |

Three of the five regressions (10, 529, 405) switched to a candidate whose packet result was
empty or a zero count while the original returned data. Two (220, 805) dropped DISTINCT/NULL
filters that gold happens to keep. Two of the 14 switches (267, 1039) stayed inside the
original's result group, so the "exactly one alternative result group" rule did not exclude
the original's own group.

### How good are the checks themselves?

Every candidate of the 435 triggered questions was scored offline against gold
(`full_results/evaluation/candidate_scores_offline_only.json`, never an inference input).
106 questions reached the verification stage (not ambiguous, quotes valid). Over their
supported checks:

| Measure | Value |
|---|---:|
| Verdict "fail" on a gold-passing candidate (false fail) | 51 / 523 = 9.8% |
| Verdict "pass" on a gold-failing candidate (false pass) | 662 / 806 = 82.1% |
| Candidate passes every check and is gold-correct | 119 |
| Candidate passes every check and is gold-wrong | 144 |
| Precision of "all checks pass" as a correctness predictor | 45.2% |
| Recall | 74.4% |

The checks are lenient, not discriminating: most wrong candidates pass every requirement.
Among the 15 recoverable failures where the checks ran but nothing failed the original,
five (58, 866, 898, 1032, 1177) had every candidate passing every check, and ten (72, 214,
436, 486, 689, 873, 943, 1108, 1510, 1527) had the gold-correct candidate marked "fail" on at
least one requirement while the original passed. In Q648 and Q1458 the gold-correct candidate
was likewise failed. So when requirements are specific enough to discriminate, they tend to
encode the same interpretation the original already made, and to reject the gold reading.

### Why 84 of the 88 recoverable failures were not recovered

| Reason | Count |
|---|---:|
| Alignment declared whole-question ambiguity; no checks ran | 60 |
| Checks ran but no supported requirement failed the original | 15 |
| Quote validation failed; review discarded (5 of these also marked ambiguous) | 7 |
| Checks failed the original but no alternative passed everything (Q648) | 1 |
| Switched to the wrong candidate (Q1458) | 1 |

The ambiguity gate is the dominant blocker. Sixteen of the 60 ambiguity-blocked recoverables
had three or more of the five candidates passing gold (56, 83, 239, 580, 618, 637, 794, 839,
842, 851, 967, 987, 1195, 1238, 1282, 1525). The stated reasons are almost always generic
conventions the prompt told the model not to assume: DISTINCT/duplicates (99 of 279 ambiguity
reasons mention it), ties (103), NULL/missing handling (33), unspecified ordering or LIMIT (62),
which of two similar columns to return (73). BIRD gold resolves these by convention rather than
by stating them, so declining to decide forfeits most recoveries, and it did not prevent the
regressions above (all five passed the gate).

The gate did protect originally correct answers: of the 195 originally correct triggered
questions, 115 were retained by ambiguity, 26 by failed quote validation, 45 passed every
check, 3 had a failing verdict on the original but were saved by the one-group rule
(4, 257, 287), and 6 were switched (5 regressed, 1 neutral).

### Exact-quote validation

All 50 failed reviews were `Requirement quote does not occur in permitted source`. 38 of the
50 alignments had also set ambiguous=true, so only 12 reviews were actually lost to the check.
Causes: ellipsis-abbreviated quotes ("top 5 schools ... with the highest"), quotes spanning
CSV/DDL whitespace (11 of 50 would match after whitespace normalisation), and paraphrases
("'Chinese Simplified' is the language"). The validation works as an anti-hallucination gate;
a whitespace-normalised comparison would be a safe change for a future version.

### Assessment

The independent-requirements idea produced clean requirement lists and the probes worked
(199 probes, 0 errors), but the verification step trusts requirement text over the execution
evidence it was given: it passed empty results twice and an evidence-literal filter that the
probe had already shown matches nothing. Combined with an ambiguity gate that abstains on
exactly the conventions BIRD gold silently applies, and checks that fail the gold-correct
candidate more often than they fail the original, the method is net-neutral to slightly
negative on this candidate pool. Selection over these five candidates has now been tried three
ways (general judge, disagreement judge, requirement checks) without beating 1102; the 344
failures with no passing candidate remain untouched by any of them.

Post-hoc observations, **not** applied to the finished run and derived after seeing gold:
an execution-sanity guard (never switch to a candidate whose result is empty or a lone zero
when the original's is not) would have blocked regressions 10, 529 and 405 and none of the four
recoveries; per-requirement ambiguity instead of whole-question ambiguity would have let the
checks run on the 60 blocked recoverables, with unknown regression risk given the 82% false-pass
rate. Both are hypotheses for a separate experiment directory, not tuned results.
