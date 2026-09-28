# Full pipeline: original pipeline + question-form output post-process

`src/` is not modified. The wrapper runs the original pipeline through
`scripts/run_pipeline.sh`, then applies the post-process tested in
`experiments/question_form_output` to the pipeline's `selected.json`.

```text
scripts/run_pipeline.sh (unchanged)          writes <output-dir>/selected.json
experiments/full_pipeline/postprocess.py     reads it, writes <pp-out>/selected_postprocessed.json
scripts/run_evaluation.sh (unchanged)        scores either file
```

## Run

Defaults at the top of `run_full_pipeline.sh`: `-o ./output/mc_v1/`,
`--dev-json ./data/bird_data/dev.json`, `--databases-dir ./data/bird_data/dev_databases`.

```bash
# Matched-contents run into the default folder (./output/mc_v1/)
bash experiments/full_pipeline/run_full_pipeline.sh --headless --matched-contents -b 0 200

# Full dev run with Claude Max (headless), then post-process
bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/full_v1/ --headless -b 0 1534

# Only post-process an existing pipeline output
bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/full_v1/ --skip-generation

# Generation with the matched-contents block (build the value index first, see
# experiments/matched_contents/README.md)
bash experiments/full_pipeline/run_full_pipeline.sh -o ./output/mc_v1/ --headless --matched-contents -b 0 200

# Evaluate selected.json and selected_postprocessed.json side by side (dev only).
# Defaults at the top of the script: -o ./output/full_v2/, --start 0, --end 10.
bash experiments/full_pipeline/evaluate_full_pipeline.sh
bash experiments/full_pipeline/evaluate_full_pipeline.sh --start 14 --end 28
bash experiments/full_pipeline/evaluate_full_pipeline.sh -o ./output/full_v3/ --all
```

`evaluate_full_pipeline.sh` matches predictions to gold by question key, so partial runs
are scored correctly. `scripts/run_evaluation.sh` applies `--start/--end` to list positions
in the prediction file, which is only correct when the file starts at question 0 with no
gaps; it also defaults to questions 80-81. Both use the same `execute_model` scoring. The
new script prints accuracy by difficulty for both files, the questions recovered and
regressed, a row per question that matched a post-process form, and writes
`<output-dir>/evaluation/evaluation_<start>_<end>.json`. Checked on the full v6 run: 1102
and 1114, identical to the experiment's evaluation.

For the test set, pass the same `--dev-json` and `--databases-dir` you give the pipeline;
the wrapper forwards them to both steps. Submit
`<pp-out>/selected_postprocessed.json`, which has the same key and
`SQL\t----- bird -----\tdb_id` format as `selected.json`.

## Post-process behaviour

- Classifies each question by wording alone into rank, count_then_list,
  entity_then_attribute, value_then_additive or list_entity.
- Default forms: the first four. `list_entity` is off because it regressed 8 of 11 changed
  questions and had been flagged as risky before the run. Enable it with
  `--pp-forms rank,count_then_list,entity_then_attribute,value_then_additive,list_entity`.
- One model call per matched question with the question, evidence, relevant table DDL
  and supplied description CSVs, the pipeline's SQL and its execution result. No gold.
- A rewrite is kept only if nothing but the SELECT list / DISTINCT changed (GROUP BY too for
  count_then_list), it executes, and it is not newly empty. Otherwise the pipeline's SQL stays.
- Client `cli` (default): Claude Code CLI with tools and MCP disabled, run from the temp
  folder, model `sonnet`, as in the tested experiment. Client `api`: any provider in
  `src/utils/llm_client.py` (`--pp-client api --pp-provider anthropic --pp-model claude-sonnet-5`).
- Resumable: per-question results are checkpointed in `<pp-out>/results`; a rerun makes no
  new calls for finished questions. Prompts and responses are logged in `<pp-out>/calls`.
- `<pp-out>/report.json` lists counts per form and status and the changed keys.

## Verified on 2026-09-17

- Unit tests: 15 pass under the pipeline's system Python (sqlglot 30.11) and the
  experiment venv (30.18), including exact validator and schema parity with the tested run.
- Post-process only, on output/claude_headless_v6 (to output/full_pipeline_check/v6_postprocess):
  27 of 1534 changed; exactly the same 14 recoveries and 2 regressions as the tested
  four-form run, i.e. 1102 to 1114 by per-question execute_model. `scripts/run_evaluation.sh`
  reports 72.43% (1111); that script scores the original selected.json at 1099, so the
  gain is +12 on both scorers. The v6 selected.json hash was unchanged.
- Full wrapper with generation on question 80 (`--headless -b 80 81`): the wrapper and
  post-process ran, but **generation failed on this Windows machine**. Every headless call
  returned `[WinError 206] The filename or extension is too long`, because the
  claude-code-headless package passes the whole prompt as a command-line argument and the
  prompts exceed Windows' 32,767-character limit. The post-process correctly kept the error
  text instead of adopting a model-written replacement query. Run full generation from
  Linux or inside WSL, or with `--anthropic` and an API key.
- Note: the headless generation client runs the Claude CLI with its default tools from the
  project folder, where dev.json contains gold SQL. The post-process client disables tools
  and runs from the temp folder. Consider the same isolation for generation before a
  submission run.

## Windows headless fix (2026-09-17)

On Windows, claude-code-headless runs `wsl bash -c '<prompt>'`. The outer shell expands
backticks in prompts (`relevant_columns: command not found`) and long prompts fail with
WinError 206, so every candidate and selected SQL became error text while the pipeline still
reported "Successful". With `--headless` the wrapper now loads
`headless_patch/sitecustomize.py` (via PYTHONPATH, `src/` unchanged), which calls the local
`claude` executable without a shell: prompt on stdin, system prompt via
`--system-prompt-file`, tools and MCP off, temp working directory, same model and rate limit.
Opt out with `--no-native-headless`.

Verified: a 40,000-character prompt with backticks and `$HOME` round-trips unchanged; the
full wrapper on question 80 generated 5/5 candidates, and the post-process reordered the
output columns, turning a wrong answer into a correct one. About 90 seconds per question.

Use a fresh `-o` folder for each run. The pipeline appends to its .jsonl logs and updates
selected.json by key; the post-process reuses a cached result only when the pipeline SQL for
that key is unchanged.

## Expected effect on dev

Measured once on dev in output/question_form_output/v1: 1102 original, 1109 with all five
forms, 1114 without list_entity. These forms were derived from dev gold, so the dev gain
is an optimistic estimate for the test set.
