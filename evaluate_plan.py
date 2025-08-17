import numpy as np
from itertools import product

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

    a = np.mean((plan_controls[:, 0] == 0.25) | (plan_controls[:, 0] == 0.75))

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
    return success, path_error / len(plan_controls), a  # len(plan_controls)


def main():
    objs = (
        "cracker_box_flipped",
        "master_chef_can_flipped",
        # "letter_t",
        # "mustard_bottle_flipped",
        # "banana",
    )
    # model_types = ("mlp", "residual")
    # use_vars = (0, 1, 2)
    # data_usages = ("1000x1", "500x2", "333x3", "250x4", "200x5", "100x10")
    model_type = "mlp"
    belief_space = [False, True]
    active_sampling = [False, True]
    active_selection = [False, True]

    for obj_name in objs:
        print("Object: ", obj_name)

        # Load the control list
        data_loader = DataLoader(obj_name + "_10000", "data", val_size=0)
        datasets = data_loader.load_data()
        control_list = datasets["x_pool"]
        control_effects = datasets["y_pool"]

        for selection, sampling, belief in product(
            active_selection, active_sampling, belief_space
        ):
            if not belief and (sampling or selection):
                continue

            # Load the plan
            belief = "belief" if belief else "regular"
            sampling = "active" if sampling else "random"
            selection = "prob" if selection else "cost"
            name = f"{obj_name}_{model_type}_{belief}_{sampling}_{selection}"
            folder = f"results/planning"
            states = np.load(f"{folder}/{name}_states.npy", allow_pickle=True)
            controls = np.load(
                f"{folder}/{name}_controls.npy", allow_pickle=True
            )

            success = np.zeros(len(states))
            path_error = np.zeros(len(controls))
            path_length = np.zeros(len(controls))
            for i, (plan_states, plan_controls) in enumerate(
                zip(states, controls)
            ):
                success[i], path_error[i], path_length[i] = evaluate_plan(
                    plan_states, plan_controls, control_list, control_effects
                )

            print(
                f"\nBelief: {belief}; Sampling: {sampling}; Selection: {selection}"
            )
            print(f"Success rate: {np.mean(success)}")
            print(f"Average path error: {np.mean(path_error)}")
            print(f"Average number of steps: {np.mean(path_length)}")


if __name__ == "__main__":
    main()
