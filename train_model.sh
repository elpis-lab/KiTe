#!/usr/bin/env bash
set -euo pipefail

# List of object names
# mustard_bottle_flipped
# letter_t
# vehicle
objs=(
    cracker_box_flipped
    master_chef_can_flipped
)
model_types=(
    mlp
)
use_vars=(
    2
)
data_usages=(
    1000x1
    750x1
    500x1
    250x1
)

for obj in "${objs[@]}"; do
    for model_type in "${model_types[@]}"; do
        for use_var in "${use_vars[@]}"; do
            for data_usage in "${data_usages[@]}"; do
                echo "=== Training $obj with $model_type, $use_var, $data_usage ==="

                # Run train_model.py in the background
                python train_model.py "$obj" "$model_type" "$use_var" "$data_usage" &
                train_pid=$!
                wait $train_pid

                echo "=== Done with $obj with $model_type, $use_var, $data_usage ==="
                echo
            done
        done
    done
done

wait
echo "=== All training jobs completed ==="
