#!/usr/bin/env bash
# R-VES over a prediction file, using BIRD's own reward buckets.
#
#   bash experiments/rves/run_rves.sh -p ./output/rules_on_v1/projection_review/selected_projection_reviewed.json
#   bash experiments/rves/run_rves.sh -p <file> -s 0 -e 100 -i 20        # quick, NOT comparable
#
# Options
#   -p FILE   predictions, {idx: "SQL\t----- bird -----\tdb_id"}        (required)
#   -s N      start index, inclusive                                     (default 0)
#   -e N      end index, exclusive                                       (default: all)
#   -i N      iterations per query; BIRD's default is 100                (default 100)
#   -t SEC    per-query timeout, multiplied by the iteration count       (default 30)
#   -o FILE   per-question rewards; resumable, re-running skips finished work
#   -q FILE   questions json                        (default data/bird_data/dev.json)
#   -d DIR    databases dir                         (default data/bird_data/dev_databases)
#
# R-VES is a WALL-CLOCK measurement. Run it on an otherwise idle machine and do not run two
# copies at once - the timings would measure contention, not SQL. It is single-process on
# purpose, which is why it is slow: 1534 questions x 100 iterations x 2 queries is an overnight
# job. Use -o so an interrupted run resumes instead of starting over.
set -u

PRED=""; START=0; END=""; ITER=100; TMO=30
OUT=""; QJSON="./data/bird_data/dev.json"; DBDIR="./data/bird_data/dev_databases"

while getopts "p:s:e:i:t:o:q:d:h" opt; do
    case "$opt" in
        p) PRED="$OPTARG" ;;
        s) START="$OPTARG" ;;
        e) END="$OPTARG" ;;
        i) ITER="$OPTARG" ;;
        t) TMO="$OPTARG" ;;
        o) OUT="$OPTARG" ;;
        q) QJSON="$OPTARG" ;;
        d) DBDIR="$OPTARG" ;;
        h) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "see: bash $0 -h" >&2; exit 1 ;;
    esac
done

if [[ -z "$PRED" ]]; then
    echo "Error: -p <predictions.json> is required.  bash $0 -h" >&2
    exit 1
fi
if [[ ! -f "$PRED" ]]; then
    echo "Error: no such prediction file: $PRED" >&2
    exit 1
fi

cd "$(dirname "$0")/../.." || exit 1

ARGS=(--predictions "$PRED" --questions "$QJSON" --databases-dir "$DBDIR"
      --start "$START" --iterate-num "$ITER" --meta-time-out "$TMO")
if [[ -n "$END" ]]; then ARGS+=(--end "$END"); fi
if [[ -z "$OUT" ]]; then
    # default cache next to the predictions, so a re-run resumes by default
    OUT="${PRED%.json}_rves.json"
fi
ARGS+=(--out "$OUT")

echo "predictions : $PRED"
echo "rewards file: $OUT  (resumable)"
echo "iterations  : $ITER per query$([[ "$ITER" -lt 100 ]] && echo '   <-- below BIRD default, not comparable')"
echo ""

python -m experiments.rves.rves "${ARGS[@]}"
