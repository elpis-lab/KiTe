#!/usr/bin/env bash
set -euo pipefail

# List of object name
objs=(
    cracker_box_flipped
    master_chef_can_flipped
)
model_type="mlp"
use_var=2.0
data_usages=(
    100x1x10
    200x1x10
    300x1x10
    400x1x10
    500x1x10
    600x1x10
    700x1x10
    800x1x10
    900x1x10
    1000x1x10
)
active_sampling=(
    0
    1
)
active_selection=(
    0
    1
)

# Split by data usage
for data_usage in "${data_usages[@]}"; do

    for obj in "${objs[@]}"; do
        # Plan for non-belief space
        echo "=== Planning $obj with $model_type $use_var $data_usage 0 0 0 ==="
        python planning.py "$obj" "$model_type" $use_var "$data_usage" 0 0 0 &
        echo

        # Plan for belief space
        for sampling in "${active_sampling[@]}"; do
            for selection in "${active_selection[@]}"; do
                # Run plnning.py in the background
                echo "=== Planning $obj with $model_type $use_var $data_usage 1 $sampling $selection ==="
                python planning.py "$obj" "$model_type" $use_var "$data_usage" 1 $sampling $selection &
                echo
            done
        done
    done

    wait
done

wait
echo "=== All planning completed ==="
