#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

# List of object name
# These have the same length
beliefs=(
    0
    1
    0
    1
    0
    1
    0
    1
    0
    1
)
algos=(
    "sst"
    "sst"
    "aorrt"
    "aorrt"
    "aorrt"
    "aorrt"
    "aorrt"
    "aorrt"
    "aorrt"
    "aorrt"
)
terminal_weights=(
    0.0
    0.0
    0.0
    0.0
    5.0
    5.0
    20.0
    20.0
    50.0
    50.0
)
n_reps=5

# Split by data usage
for i in "${!algos[@]}"; do
    belief="${beliefs[$i]}"
    algo="${algos[$i]}"
    terminal_weight="${terminal_weights[$i]}"

    echo "=== Planning Car: ${algo} belief: ${belief} tw: ${terminal_weight} with ${n_reps} reps ==="
    python "$script_dir/planning_car.py" \
        "$belief" "$algo" "$terminal_weight" "$n_reps" &
    echo
done
wait

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All planning jobs completed in %02d:%02d:%02d ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
