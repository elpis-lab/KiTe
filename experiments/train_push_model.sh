#!/usr/bin/env bash
set -euo pipefail
script_dir="$(dirname "$(realpath "$0")")"
# Start timer
start_time=$(date +%s)

object_names=(
    # cracker_box_flipped
    # mustard_bottle_flipped
    # banana
    # letter_t
    # master_chef_can_flipped
    # trash_truck
    real_cracker_box_flipped
    real_trash_truck
)
use_vars=(
    1
    0
)
n_datas=(
    100
    200
    300
    400
    500
    600
    700
    800
    900
    1000
)
devices=(
    cuda:0
    # cuda:3
)
model_classes=(
    mlp
)
n_experiments=5

# Repeat experiment
model_idx=0
device_idx=0
for experiment_idx in $(seq 0 $((n_experiments - 1))); do
    for object_name in ${object_names[@]}; do
        for use_var in ${use_vars[@]}; do
            for n_data in ${n_datas[@]}; do
                model=${model_classes[$model_idx]}
                device=${devices[$device_idx]}

                echo "Experiment $experiment_idx - Training $object_name with $use_var var and $n_data data"
                python "$script_dir/train_push_model.py" $object_name $model $use_var $n_data $experiment_idx &
                wait
                echo
            done
        done
    done
done

wait
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
printf "=== All training jobs completed in %02d:%02d:%02d ===\n" \
    $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
