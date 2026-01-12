#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

planners=(
    "sst"
    "aorrt"
    "aorrt"
)
terminal_weights=(
    0.0
    0.0
    1.0
)
n_reps=5

# Split by data usage
for i in "${!planners[@]}"; do
    planner="${planners[$i]}"
    terminal_weight="${terminal_weights[$i]}"

    # Run planning.py in the background
    echo "=== Planning with $planner $terminal_weight with $n_reps reps ==="
    python "$script_dir/planning_flappy.py" "$planner" "$terminal_weight" "$n_reps" &
    echo
done

wait
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All planning jobs completed in %02d:%02d:%02d ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
