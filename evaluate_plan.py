import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from utils import DataLoader
from geometry.pose import Pose
from planning_utils import is_state_success
from lie_group.lie_se2 import log_se2


def is_plan_successful(state):
    goal_state = np.array([0, -0.7, 0])
    goal_ranges = np.array(
        [[-0.03, 0.03], [-0.03, 0.03], [-np.pi / 8, np.pi / 8]]
    )
    return is_state_success(state, goal_state, goal_ranges)[0]


def evaluate_plan(plan_states, plan_controls, control_list, control_effects):
    """Evaluate if the plan is successful and compute the path error"""
    control_list = np.asarray(control_list)
    plan_controls = np.asarray(plan_controls)
    # # TODO Debug
    # a = np.mean((plan_controls[:, 0] == 0.25) | (plan_controls[:, 0] == 0.75))

    # Find the delta pose of the plan controls
    dist = np.sum(
        (plan_controls[:, None, :] - control_list[None, :, :]) ** 2, axis=2
    )
    idx = np.argmin(dist, axis=1)
    delta_states = np.asarray(control_effects)[idx]

    # Find the actual delta pose
    plan_delta_poses = []
    for i in range(len(plan_states) - 1):
        plan_pose1 = Pose(
            (plan_states[i][0], plan_states[i][1], 0),
            (0, 0, plan_states[i][2]),
        )
        plan_pose2 = Pose(
            (plan_states[i + 1][0], plan_states[i + 1][1], 0),
            (0, 0, plan_states[i + 1][2]),
        )
        plan_delta_poses.append(plan_pose1.invert @ plan_pose2)

    # Evaluate the plan
    path_error = 0
    state = plan_states[0]
    pose = Pose((state[0], state[1], 0), (0, 0, state[2]))
    for plan_delta_pose, delta_state in zip(plan_delta_poses, delta_states):
        delta_pose = Pose(
            (delta_state[0], delta_state[1], 0), (0, 0, delta_state[2])
        )
        pose = pose @ delta_pose
        path_error += np.linalg.norm(
            log_se2((plan_delta_pose.invert @ delta_pose).matrix)
        )

    success = is_plan_successful(
        (pose.position[0], pose.position[1], pose.euler[2])
    )
    return success, path_error / len(plan_controls), len(plan_controls)


def main():
    # Model
    model_type = "mlp"
    use_var = 2.0  # 1.0
    seed = 10  # 5
    data_usages = (
        "100x1",
        "200x1",
        "300x1",
        "400x1",
        "500x1",
        "600x1",
        "700x1",
        "800x1",
        "900x1",
        "1000x1",
    )
    data_usages = [d + f"x{seed}" for d in data_usages]

    # Planning
    n_reps = 10
    n_problems = 100
    objs = (
        "cracker_box_flipped",
        "master_chef_can_flipped",
    )
    belief_space = [False, True]
    active_sampling = [False, True]
    active_selection = [False, True]
    configs = []
    for selection, sampling, belief in product(
        active_selection, active_sampling, belief_space
    ):
        if not belief and (sampling or selection):
            continue
        configs.append((belief, sampling, selection))

    for obj_name in objs:
        print("Object: ", obj_name)

        # Load the control list
        data_loader = DataLoader(obj_name + "_10000", "data", val_size=0)
        datasets = data_loader.load_data()
        control_list = datasets["x_pool"]
        control_effects = datasets["y_pool"]

        # Results
        success = np.zeros(
            (n_reps, len(configs), len(data_usages), n_problems)
        )
        path_error = np.zeros(
            (n_reps, len(configs), len(data_usages), n_problems)
        )
        path_length = np.zeros(
            (n_reps, len(configs), len(data_usages), n_problems)
        )

        for k, data_usage in enumerate(data_usages):
            for j, config in enumerate(configs):
                belief, sampling, selection = config

                # Load the plan
                belief = "belief" if belief else "regular"
                sampling = "active" if sampling else "random"
                selection = "prob" if selection else "cost"
                name = (
                    f"{obj_name}_{model_type}_{use_var}_{data_usage}"
                    + f"_{belief}_{sampling}_{selection}"
                )
                folder = f"results/planning"
                states = np.load(
                    f"{folder}/{name}_states.npy", allow_pickle=True
                )
                controls = np.load(
                    f"{folder}/{name}_controls.npy", allow_pickle=True
                )

                # Evaluate the plan
                for idx, (plan_states, plan_controls) in enumerate(
                    zip(states, controls)
                ):
                    i, l = idx // n_problems, idx % n_problems
                    (
                        success[i, j, k, l],
                        path_error[i, j, k, l],
                        path_length[i, j, k, l],
                    ) = evaluate_plan(
                        plan_states,
                        plan_controls,
                        control_list,
                        control_effects,
                    )

                print(
                    f"\nBelief: {belief}; Sampling: {sampling}; Selection: {selection}"
                )
                print(f"Success rate: {np.mean(success)}")
                print(f"Average path error: {np.mean(path_error)}")
                print(f"Average number of steps: {np.mean(path_length)}")

        # Save Results
        np.save(f"results/planning/{obj_name}_success.npy", success)
        np.save(f"results/planning/{obj_name}_path_error.npy", path_error)
        np.save(f"results/planning/{obj_name}_path_length.npy", path_length)

        # Average over the number of problems
        success = np.mean(success, axis=3)
        path_error = np.mean(path_error, axis=3)
        path_length = np.mean(path_length, axis=3)

        # Average over the number of reps
        success_mean = np.mean(success, axis=0)
        success_std = np.std(success, axis=0)
        path_error_mean = np.mean(path_error, axis=0)
        path_error_std = np.std(path_error, axis=0)
        path_length_mean = np.mean(path_length, axis=0)
        path_length_std = np.std(path_length, axis=0)

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
        x = np.arange(len(data_usages))

        for j, config in enumerate(configs):
            belief = "belief" if config[0] else "regular"
            sampling = "active" if config[1] else "random"
            selection = "prob" if config[2] else "cost"
            y1 = success_mean[j, :]
            y1_std = success_std[j, :]
            y2 = path_error_mean[j, :]
            y2_std = path_error_std[j, :]
            ax1.plot(x, y1, label=f"{belief}_{sampling}_{selection}")
            ax1.fill_between(x, y1 - y1_std, y1 + y1_std, alpha=0.2)
            ax2.plot(x, y2, label=f"{belief}_{sampling}_{selection}")
            ax2.fill_between(x, y2 - y2_std, y2 + y2_std, alpha=0.2)
        for ax, ylabel in zip((ax1, ax2), ("Success Rate", "Path Error")):
            ax.set_xticks(x)
            ax.set_xticklabels(data_usages, rotation=45)
            ax.set_ylabel(ylabel)
            ax.legend()
        ax2.set_xlabel("Data")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
