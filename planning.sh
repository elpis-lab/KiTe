#!/usr/bin/env bash
set -euo pipefail

# List of object names
# mustard_bottle_flipped
# letter_t
# vehicle
objs=(
    master_chef_can_flipped
    cracker_box_flipped
)

model_type="mlp"
active_sampling=(
    0
    1
)
active_selection=(
    0
    1
)

for obj in "${objs[@]}"; do
    # Plan for non-belief space
    echo "=== Planning $obj with $model_type, 0, 0, 0 ==="
    python planning.py "$obj" "$model_type" 0 0 0 &
    echo

    # Plan for belief space
    for sampling in "${active_sampling[@]}"; do
        for selection in "${active_selection[@]}"; do
            # Run plnning.py in the background
            echo "=== Planning $obj with $model_type, 1, $sampling, $selection ==="
            python planning.py "$obj" "$model_type" 1 $sampling $selection &
            echo
        done
    done
done

wait
echo "=== All planning completed ==="
