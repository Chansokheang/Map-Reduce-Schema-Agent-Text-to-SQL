# Projection alignment v1: results (2026-09-16)

Output: output/projection_alignment/v1. Evaluated once, full set, unmodified repository
evaluator, original dev.json gold. 1534 model responses (claude-sonnet-5), CLI-reported
list cost $24.91 on a Max subscription login (not an API bill).

| Method | Correct / 1534 | EX | Recovered | Regressed | Net |
|---|---:|---:|---:|---:|---:|
| Original | 1102 | 71.84% | - | - | - |
| v2 control judge | 1102 | 71.84% | 30 | 30 | 0 |
| v2 disagreement judge | 1092 | 71.19% | 24 | 34 | -10 |
| Focused verification | 1101 | 71.77% | 4 | 5 | -1 |
| **Projection alignment** | **1104** | **71.97%** | **7** | **5** | **+2** |

Statuses: 1493 retained, 21 rewritten, 14 rejected by the mechanical validator,
6 failed quote validation. Per database: california_schools +1, card_games +2,
formula_1 +1, thrombosis_prediction +1, european_football_2 -1, superhero -1, toxicology -1.

## The 21 rewrites by kind of change

| Change | Recovered | Regressed | Neutral |
|---|---|---|---|
| Reorder only | 80, 430, 431, 894 | none | 17, 81, 1021 |
| Drop a column | 907, 1216 | 264, 1085 | 23, 728, 958, 985 |
| Substitute a column | 453 | 812 | none |
| Add a column | none | 426, 1000 | 342, 520 |

- **Reorders are the reliable part**: 4 recoveries, 0 regressions. Gold orders output by
  the question's mention order.
- **Drops of sort-only columns split evenly**. 907 and 1216 recover because gold omits the
  ORDER BY column; 1085 regresses because gold keeps `crossing` for "which player performs
  best". 264 regresses because the model dropped `molecule_id` on the grounds that the ids
  were given in the question, and gold keeps it.
- **Substitution**: 453 follows the schema description (manaCost is the unconverted cost).
  812 overrode the evidence, which maps "name of superheroes" to superhero_name, in favour
  of the question word "full names".
- **Additions never helped**. 426 added `code` next to the requested name, violating the
  prompt's own no-extra-identifier rule; 1000 added name and country. Gold for 1000 returns
  only location although the evidence says full location = location + country.

Three regressions (812, 1000, 1085) were also regressed by the v2 judges; they sit on
questions where gold departs from the evidence or the natural reading.

## Assessment

A small, real, net gain (+2) with the regression side measured. It is the first selection-
or repair-style experiment in this project to beat the original, but +2 of 1534 is within
what a different sampling of the same prompt could move, so it is not strong evidence on
its own.

Post-hoc only (derived after seeing gold; do not treat as validated): restricting the
rewrite to reordering existing output expressions would have given 1106 (+4, 0 regressions)
on this run. Additions gave 0 recoveries and 2 regressions. A v2 in a new directory could
test "reorder always, drop only with evidence support, never add, never override an
evidence mapping", and should be judged on a fresh full run.
