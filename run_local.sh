#!/bin/bash
set -e
trap 'echo "Interrupted. Killing all jobs..."; kill 0; exit 1' INT TERM

NUM_SEEDS=10
BENCHMARK=${1:?"Usage: bash run_local.sh <benchmark>  (w or s)"}
PROJECT_PATH="./data"

echo "Running benchmark='${BENCHMARK}' for seeds 1..${NUM_SEEDS}"

for SEED in $(seq 1 $NUM_SEEDS); do
    echo "--- Launching seed ${SEED}/${NUM_SEEDS} ---"
    uv run python benchmark.py \
        --project_path="${PROJECT_PATH}" \
        --benchmark="${BENCHMARK}" \
        --seeds="${NUM_SEEDS}" \
        --seed="${SEED}" &

    # Keep at most 4 jobs running in parallel
    while [ "$(jobs -r | wc -l)" -ge 5 ]; do
        wait -n 2>/dev/null || sleep 1
    done
done

wait
echo "All seeds finished."

echo "Done."
