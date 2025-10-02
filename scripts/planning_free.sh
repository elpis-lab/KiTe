#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

# List of object name
objs=(
    cracker_box_flipped
    # mustard_bottle_flipped
    # banana
    # letter_t
    master_chef_can_flipped
    # school_bus
    # trash_truck
    # real_cracker_box_flipped
    # real_school_bus
    # real_trash_truck
)
model_type="mlp"
use_var=1
n_datas=(
    # 100
    # 200
    # 300
    # 400
    # 500
    # 600
    # 700
    # 800
    # 900
    1000
)
active_sampling=(
    0
    1
)
active_selection=(
    0
    1
)
predef_controls=1
n_reps=3

# Split by data usage
for obj in "${objs[@]}"; do
    for n_data in "${n_datas[@]}"; do

        # Plan for active pusher
        echo "=== Planning $obj with $model_type 0 $n_data 0 1 0 ==="
        python scripts/planning_free.py "$obj" "$model_type" 0 "$n_data" 0 1 0 "$predef_controls" "$n_reps" &
        echo

        # Plan for non-belief space
        echo "=== Planning $obj with $model_type $use_var $n_data 0 0 0 ==="
        python scripts/planning_free.py "$obj" "$model_type" $use_var "$n_data" 0 0 0 "$predef_controls" "$n_reps" &
        echo

        # Plan for belief space
        for sampling in "${active_sampling[@]}"; do
            for selection in "${active_selection[@]}"; do
                # Run plnning.py in the background
                echo "=== Planning $obj with $model_type $use_var $n_data 1 $sampling $selection ==="
                python scripts/planning_free.py "$obj" "$model_type" $use_var "$n_data" 1 $sampling $selection "$predef_controls" "$n_reps" &
                echo
            done
        done
        wait

    done
done

wait
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All planning jobs completed in %02d:%02d:%02d ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
