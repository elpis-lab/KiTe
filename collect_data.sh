#!/usr/bin/env bash
set -euo pipefail

# List of object names
# vehicle
objs=(
    cracker_box_flipped
    letter_t
    master_chef_can_flipped
    mustard_bottle_flipped
    banana
)

for obj in "${objs[@]}"; do
    echo "=== Processing $obj ==="

    # Start sim_network.py in the background
    python sim_network.py "$obj" &
    sim_pid=$!
    trap 'kill '"$sim_pid"' 2>/dev/null || true' EXIT

    # Wait 3 second
    sleep 3

    # Start collect_data.py in the background
    python collect_data.py "$obj" &
    col_pid=$!
    wait $col_pid

    # Kill sim_network.py now that data collection is done
    kill "$sim_pid" 2>/dev/null || true
    trap - EXIT

    echo "=== Done with $obj ==="
    echo
done