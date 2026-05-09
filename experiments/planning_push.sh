#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

# List of object name
objs=(
    mustard_bottle_flipped
    master_chef_can_flipped
    real_cracker_box_flipped
    real_trash_truck
)
model_type="mlp"
n_datas=(
    # 200
    # 400
    # 600
    # 800
    1000
)

# These have the same length
use_vars=(
    0
    0
    1
    1
    0
    0
    0
    1
)
algos=(
    "aorrt"
    "sst"
    "aorrt"
    "sst"
    "aorrt"
    "sst"
    "aorrt"
    "aorrt"
)
active_samplings=(
    0
    0
    0
    0
    1
    1
    0
    0
)
terminal_weights=(
    0.0
    0.0
    0.0
    0.0
    0.0
    0.0
    20.0
    20.0
)
n_reps=5

# Split by data usage
for obj in "${objs[@]}"; do
    for n_data in "${n_datas[@]}"; do

        for i in "${!algos[@]}"; do
            use_var="${use_vars[$i]}"
            algo="${algos[$i]}"
            active_sampling="${active_samplings[$i]}"
            terminal_weight="${terminal_weights[$i]}"

            echo "=== Planning: ${obj}; ${model_type} with ${n_data} and var ${use_var};"
            echo "Planner ${algo} active: ${active_sampling} tw: ${terminal_weight} with ${n_reps} reps ==="
            python "$script_dir/planning_push.py" \
                "$obj" "$model_type" "$use_var" "$n_data" \
                "$algo" "$active_sampling" "$terminal_weight" "$n_reps" &
            echo
        done
        wait
    done
done
wait

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All planning jobs completed in %02d:%02d:%02d ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
