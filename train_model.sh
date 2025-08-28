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
    0
    1
    2
)
data_usages=(
    100x1
    200x1
    300x1
    400x1
    500x1
    600x1
    700x1
    800x1
    900x1
    1000x1
)

for seed in {1..10}; do
    for obj in "${objs[@]}"; do
        for model_type in "${model_types[@]}"; do
            for use_var in "${use_vars[@]}"; do
                for data_usage in "${data_usages[@]}"; do
                    data_usage_seed="${data_usage}x${seed}"

                    echo "=== Training $obj with $model_type, $use_var, $data_usage, $seed ==="

                    # Run train_model.py in the background
                    python train_model.py "$obj" "$model_type" "$use_var" "$data_usage_seed" "$seed" &
                    train_pid=$!
                    wait $train_pid

                    echo "=== Done with $obj with $model_type, $use_var, $data_usage, $seed ==="
                    echo
                done
            done
        done
    done
done

wait
echo "=== All training jobs completed ==="
