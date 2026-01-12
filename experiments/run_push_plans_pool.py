import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from geometry.pose import SE2Pose
from geometry.object_model import get_obj_shape
from experiments.run_plans import run_plans
from experiments.utils import DataLoader, get_names
from planning.planning_utils import (
    points_out_of_bound,
    rects_circles_in_collision,
    se2_points_in_region,
)
from planning.push import POS_RANGES, OBSTACLES
from planning.push import visualize_push_env


def get_eval_fn(obj_name):
    """
    Return a function that runs the plans in a simplified manner
    """
    # Load data and model
    model_name, data_name, rep_data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")[:2]
    data_loader = DataLoader(data_name)
    dataset = data_loader.load_data()

    # Load potential control list and delta list
    control_list = dataset["x_pool"].astype(np.float64)
    delta_list = dataset["y_pool"].astype(np.float64)

    # Goals are common for all plans
    goal_state = np.array([0, -0.7, 0])
    goal_size = 0.06

    def run_plans_pool(plan_states, plan_controls):
        """
        Run the plans in a simplified manner

        If planned with pre-defined control list,
        a very large batch of control parameters from the dataset,
        which is NOT used for training the model.

        In this case, we can assume that the actual effect of these controls
        are the same as in dataset. We can therefore evaluate the plan using
        only the corresponding dataset values without running the simulation again.
        """
        # Containers
        exec_states = []
        hits = []
        reacheds = []

        # Execute each plan
        for p_states, controls in zip(plan_states, plan_controls):
            # Find corresponding effect of plan controls
            controls = np.asarray(controls, dtype=np.float64)
            dists = np.linalg.norm(
                controls[:, None, :] - control_list[None, :, :], axis=2
            )
            idxs = np.argmin(dists, axis=1)
            assert np.allclose(control_list[idxs], controls)
            deltas = np.asarray(delta_list)[idxs]

            # Start state
            pose = SE2Pose([p_states[0][0], p_states[0][1]], p_states[0][2])
            states = [pose.flat()]

            # Execute the plan
            for delta in deltas:
                # update pose
                delta_pose = SE2Pose(np.array([delta[0], delta[1]]), delta[2])
                pose = pose @ delta_pose
                states.append(pose.flat())
            states = np.array(states)

            # Check collision
            hit = rects_circles_in_collision(
                states, obj_shape, OBSTACLES
            ) | points_out_of_bound(states[:, :2], POS_RANGES)
            # Check goal
            reached = se2_points_in_region(states, goal_state, goal_size)

            exec_states.append(states)
            hits.append(hit)
            reacheds.append(reached)
        return exec_states, hits, reacheds

    return run_plans_pool


if __name__ == "__main__":
    obj_names = [
        "cracker_box_flipped",
        # "banana",
        "master_chef_can_flipped",
        "trash_truck",
        # "real_cracker_box_flipped",
        # "real_trash_truck",
    ]
    model_type = "mlp"
    # n_datas = [100, 500, 1000]
    # n_datas = [100, 1000]
    n_datas = [1000]

    # Configs (algo, use_var, active_sampling, terminal_weight)
    configs = [
        ("aorrt", "l2", "active", 0.0),  # Active Pusher
        ("sst", "l2", "active", 0.0),  # Active Pusher
        ("aorrt", "w2", "random", 0.0),  # Belief Random Tree
        ("sst", "w2", "random", 0.0),  # Belief Random Tree
        ("aorrt", "l2", "random", 2.0),  # Proposed
        ("aorrt", "w2", "random", 2.0),  # Proposed
    ]

    # Execute plans
    for obj_name in obj_names:
        for n_data in n_datas:
            for i in range(len(configs)):
                algo = configs[i][0]
                belief = configs[i][1]
                use_var = 1 if belief == "w2" else 0
                active_sampling = configs[i][2]
                terminal_weight = configs[i][3]

                # Load
                name = (
                    f"{obj_name}_{model_type}_{use_var}_{n_data}"
                    + f"_{algo}_{belief}_{active_sampling}_{terminal_weight}"
                )
                folder = f"results/planning_push"
                poses_file = f"{folder}/{name}_plan_states.npy"
                controls_file = f"{folder}/{name}_controls.npy"
                costs_file = f"{folder}/{name}_costs.npy"
                all_states = np.load(poses_file, allow_pickle=True)
                all_controls = np.load(controls_file, allow_pickle=True)
                all_costs = np.load(costs_file, allow_pickle=True)

                # # Temp
                # case = 19
                # run_plans_pool = get_eval_fn(obj_name)
                # results, exec_states = run_plans(
                #     run_plans_pool,
                #     all_states[0, case, :].reshape(-1),
                #     all_controls[0, case, :].reshape(-1),
                #     reps_in_states=1,
                # )
                # print(all_states[0, case, -1])
                # # Visualize
                # envs = np.load(
                #     f"data/planning_push_envs.npy", allow_pickle=True
                # )
                # model_name, _, _ = get_names(obj_name)
                # obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
                # for j in range(len(all_states[0, case, :])):
                #     if (
                #         j > 0
                #         and all_controls[0, case, j][-1]
                #         == all_controls[0, case, j - 1][-1]
                #     ):
                #         continue
                #     print(f"Visualizing {j}th plan")
                #     print(all_controls[0, case, j][-1])
                #     visualize_push_env(
                #         envs[case],
                #         all_states[0, case, j],
                #         exec_states[j],
                #         obj_shape[:2],
                #     )
                #     plt.show()

                # Run
                run_plans_pool = get_eval_fn(obj_name)
                results, exec_states = run_plans(
                    run_plans_pool,
                    all_states[:, :, -1].reshape(-1),
                    all_controls[:, :, -1].reshape(-1),
                    reps_in_states=5,
                )
                np.save(f"{folder}/{name}_results.npy", results)
                np.save(
                    f"{folder}/{name}_exec_states.npy",
                    np.array(exec_states, dtype=object),
                )

                print(f"{name}:")
                print(f"Success:\t{np.mean(results[:, :, 0])}")
                print(f"SE2 Error:\t{np.mean(results[:, :, 3])}")
                print(f"Hits Obs:\t{results[:, :, 1]}")
                print(f"Reached:\t{results[:, :, 2]}")

                # Visualize
                envs = np.load(
                    f"data/planning_push_envs.npy", allow_pickle=True
                )
                model_name, _, _ = get_names(obj_name)
                obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
                from experiments.train_push_model import load_model

                model = load_model(model_type, obj_shape, use_var)
                name = f"{obj_name}_{model_type}_{use_var}_{n_data}_{0}"
                model.load(f"results/models/{name}.pth")
                for j in range(len(envs)):
                    cs = all_controls[:, :, -1].reshape(-1)[j]
                    std = np.exp(model(cs)[:, 3:])
                    for s in std:
                        print(s)
                    for s in all_states[:, :, -1].reshape(-1)[j]:
                        print(s)
                    visualize_push_env(
                        envs[j],
                        all_states[:, :, -1].reshape(-1)[j],
                        exec_states[j],
                        # obj_shape[:2],
                    )
                    plt.show()
