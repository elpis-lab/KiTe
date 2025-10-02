import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from itertools import product

from geometry.pose import SE2Pose
from planning.planning_utils import out_of_bounds, in_collision_with_circles
from run_plans import run_plans


def run_plans_pool(
    obj_name,
    plan_states,
    plan_controls,
    obj_shape,
    circle_poses,
    circle_rads,
    dataset,
):
    """
    Run the plans in a simplified manner

    If planned with pre-defined control list,
    a very large batch of control parameters from the dataset,
    which is NOT used for training the model.

    In this case, we can assume that the actual effect of these controls
    are the same as in dataset. We can therefore evaluate the plan using
    only the corresponding dataset values without running the simulation again.
    """
    # load potential control list and delta list
    control_list = dataset["x_pool"].astype(np.float64)
    delta_list = dataset["y_pool"].astype(np.float64)

    # Execute the plans
    exec_states = []
    exec_delta_states = []
    states_invalids = []
    # for each plan
    for i, (states, controls) in enumerate(zip(plan_states, plan_controls)):
        # Start state
        pose = SE2Pose([states[0][0], states[0][1]], states[0][2])
        e_states = [[*pose.pr()[0], pose.pr()[1]]]
        e_delta_states = []
        invalid = [False]

        # Find corresponding effect of plan controls
        controls = np.asarray(controls, dtype=np.float64)
        dists = np.linalg.norm(
            controls[:, None, :] - control_list[None, :, :], axis=2
        )
        idxs = np.argmin(dists, axis=1)
        assert np.allclose(control_list[idxs], controls)
        deltas = np.asarray(delta_list)[idxs]

        # Execute the plan
        for delta in deltas:
            # update pose
            delta_pose = SE2Pose(np.array([delta[0], delta[1]]), delta[2])
            pose = pose @ delta_pose
            e_delta_states.append([*delta_pose.pr()[0], delta_pose.pr()[1]])
            e_state = [*pose.pr()[0], pose.pr()[1]]
            e_states.append(e_state)

            # check validity
            invalid.append(
                in_collision_with_circles(
                    e_state, obj_shape[:2], circle_poses, circle_rads
                )
                or out_of_bounds(
                    e_state[:2], pos_range=((-0.76, 0.76), (-1.1, -0.3))
                )
            )

        exec_states.append(e_states)
        exec_delta_states.append(e_delta_states)
        states_invalids.append(invalid)

    return exec_states, exec_delta_states, states_invalids


if __name__ == "__main__":
    obj_names = [
        "cracker_box_flipped",
        # "mustard_bottle_flipped",
        # "banana",
        # "letter_t",
        "master_chef_can_flipped",
        "school_bus",
        # "trash_truck",
        # "real_cracker_box_flipped",
        # "real_mustard_bottle_flipped",
        # "real_master_chef_can_flipped",
        # "real_school_bus",
        # "real_trash_truck",
    ]
    model = "mlp"
    n_datas = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    envs = ["free", "obs"]

    # n_datas = [1000]
    obj_names = ["cracker_box_flipped", "master_chef_can_flipped"]
    envs = ["obs"]

    # Configs (use_var, belief_space, active_sampling, active_selection)
    configs = [
        # (1.0, False, False, False),  # Baseline SST
        # (0.0, False, True, False),  # Active Pusher
        # (1.0, True, False, False),  # Belief Random Tree
        # (1.0, True, True, False),  # AS
        # (1.0, True, False, True),  # AP
        # (1.0, True, True, True),  # Active Puna
        (1.0, "regular", "random", "cost"),  # Baseline SST
        (0.0, "regular", "active", "cost"),  # Active Pusher
        (1.0, "belief", "random", "cost"),  # Belief Random Tree
        (1.0, "belief", "active", "cost"),  # AS
        (1.0, "belief", "random", "prob"),  # AP
        (1.0, "belief", "active", "prob"),  # Active Puna
    ]

    for obj_name in obj_names:
        for env in envs:
            for n_data in n_datas:
                for config in configs:
                    use_var, belief, active_sampling, active_selection = config

                    name = (
                        f"{obj_name}_{model}_{use_var}_{n_data}"
                        + f"_{belief}_{active_sampling}_{active_selection}"
                    )
                    folder = f"results/planning_{env}"
                    poses_file = f"{folder}/{name}_plan_states.npy"
                    controls_file = f"{folder}/{name}_controls.npy"
                    plan_states = np.load(poses_file, allow_pickle=True)
                    plan_controls = np.load(controls_file, allow_pickle=True)
                    results, success, invalid, exec_states = run_plans(
                        run_plans_pool,
                        plan_states,
                        plan_controls,
                        obj_name,
                        env,
                        reps_in_states=3,
                    )
                    print(f"{name}: {np.mean(results[:, 0])}")
                    print(f"SE2 Error: {np.mean(results[:, 2])}")
                    np.save(f"{folder}/{name}_results.npy", results)
                    np.save(
                        f"{folder}/{name}_exec_states.npy",
                        np.array(exec_states, dtype=object),
                    )
