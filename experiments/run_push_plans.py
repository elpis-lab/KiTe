import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

from experiments.run_plans import RunPlans
from experiments.utils import DataLoader, get_names
from experiments.train_push_model import load_model
from geometry.pose import SE2Pose
from geometry.object_model import get_obj_shape
from planning.planning_utils import points_out_of_bound, se2_points_in_region
from planning.planning_utils import circles_in_collision
from planning.planning_utils import vec_to_cov, cov_to_vec

from lie_group.propagation import propagate_cov
from lie_group.lie_se2 import log_se2, to_se2_transform, to_se2_vec
from planning.push import POS_RANGES, OBSTACLES
from planning.push import SE2PushPlanner, SE2PushOptimizationObjective


class RunPushPlansPool(RunPlans):
    """Run push plans when using a predefined control pool."""

    def __init__(self, obj_name, goal, goal_size, obstacles=OBSTACLES):
        super().__init__()

        # Load control and delta list
        model_name, data_name, _ = get_names(obj_name)
        data_loader = DataLoader(data_name)
        dataset = data_loader.load_data()
        self.control_list = dataset["x_pool"].astype(np.float64)
        self.delta_list = dataset["y_pool"].astype(np.float64)
        obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")

        # Load models
        model_type = "mlp"
        use_var = 1
        n_data = 1000
        m_id = 0
        name = f"{obj_name}_{model_type}_{use_var}_{n_data}_{m_id}"
        self.model = load_model(model_type, obj_shape, use_var)
        self.model.load(f"results/models/{name}.pth")

        # Common setup
        self.planner = SE2PushPlanner(obstacles, obj_shape, None)
        self.goal = goal
        self.goal_size = goal_size
        self.obstacles = obstacles

    def run_plans(self, plan_states, plan_controls, idx_flat_to_grid=None):
        """
        Run the plans by looking up deltas from the dataset.

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
        costs = []

        for p_states, controls in zip(plan_states, plan_controls):
            # Look up corresponding deltas from dataset
            if len(controls) > 0:
                controls = np.asarray(controls, dtype=np.float64)
                dists = np.linalg.norm(
                    controls[:, None, :] - self.control_list[None, :, :],
                    axis=2,
                )
                idxs = np.argmin(dists, axis=1)
                assert np.allclose(
                    self.control_list[idxs], controls
                ), "Pre-defined control list and plan controls do not match!"
                deltas = np.asarray(self.delta_list)[idxs]
            else:
                deltas = []

            # Start state
            pose = SE2Pose([p_states[0][0], p_states[0][1]], p_states[0][2])
            states = [pose.flat()]

            # Apply deltas to get executed trajectory
            for delta in deltas:
                delta_pose = SE2Pose(np.array([delta[0], delta[1]]), delta[2])
                pose = pose @ delta_pose
                states.append(pose.flat())
            states = np.array(states)

            # Check collision (boundary only in simplified eval)
            hit = points_out_of_bound(states[:, :2], POS_RANGES)
            # hit = hit | rects_circles_in_collision(
            #     states[:, :3], self.planner.obj_shape, self.planner.obstacles
            # )
            for j in range(len(states)):
                hit[j] = hit[j] | circles_in_collision(
                    states[j], self.planner.obj_circles, self.planner.obstacles
                )

            # Check goal
            reached = se2_points_in_region(states, self.goal, self.goal_size)

            # Compute execution costs
            running_cost, terminal_cost = self.compute_costs(states)

            exec_states.append(states)
            hits.append(hit)
            reacheds.append(reached)
            costs.append([running_cost, terminal_cost])
        costs = np.array(costs, dtype=float)

        # Compute unified costs in belief space
        unified_costs = []
        for i in range(len(plan_states)):
            states = np.asarray(plan_states[i])
            controls = np.asarray(plan_controls[i])
            # Look up corresponding deltas from dataset
            if len(controls) > 0:
                deltas = self.model.predict(controls)
            else:
                deltas = []

            # initial state
            unified_states = np.zeros((len(states), 9))
            unified_states[0, :3] = states[0, :3]
            unified_states[0, 3:] = [1e-6, 0, 0, 1e-6, 0, 1e-6]
            # propagate states
            for j, delta in enumerate(deltas):
                delta_s = delta[:3]
                var = np.exp(delta[3:])
                delta_cov = vec_to_cov([var[0], 0, 0, var[1], 0, var[2]])

                # get delta
                curr_state = unified_states[j, :3]
                curr_cov = vec_to_cov(unified_states[j, 3:])
                # state
                t_delta = to_se2_transform(delta_s)
                state = to_se2_transform(curr_state) @ t_delta
                # covariance
                delta = log_se2(t_delta)
                cov = propagate_cov(delta, curr_cov, delta_cov)
                unified_states[j + 1, :3] = to_se2_vec(state)
                unified_states[j + 1, 3:] = cov_to_vec(cov)

            # Compute costs in belief space
            running_cost, terminal_cost = self.compute_costs(
                unified_states, belief=True
            )
            if running_cost == 0:
                running_cost = -1
                terminal_cost = -1
            unified_costs.append([running_cost, terminal_cost])
        unified_costs = np.array(unified_costs, dtype=float)

        return exec_states, hits, reacheds, costs, unified_costs

    def compute_costs(self, states, belief=False):
        """Compute running cost and terminal cost"""
        covs = [None] * len(states)
        if belief:
            for i in range(len(states)):
                covs[i] = vec_to_cov(states[i, 3:])
        states = np.asarray(states[:, :3])
        rot_weight = 0.2
        weight = np.diag([1.0, 1.0, rot_weight])

        # Running cost: accumulated SE2 motion cost
        running_cost = 0.0
        for i in range(len(states) - 1):
            # Compute SE2 distance
            r_cost = SE2PushOptimizationObjective.se2_distance(
                states[i], states[i + 1], covs[i], covs[i + 1], weight=weight
            )
            running_cost += r_cost

        # Terminal cost: distance to goal
        terminal_cost = SE2PushOptimizationObjective.se2_distance(
            states[-1], self.goal, covs[-1], weight=weight
        )
        return running_cost, terminal_cost


if __name__ == "__main__":
    obj_names = [
        "mustard_bottle_flipped",
        "master_chef_can_flipped",
    ]
    model_type = "mlp"
    # n_datas = [200, 400, 600, 800, 1000]
    n_datas = [1000]

    configs = [
        ("aorrt", "l2", "random", 0.0),  # Base
        ("sst", "l2", "random", 0.0),  # Base
        ("aorrt", "w2", "random", 0.0),  # Gaussian Belief Trees
        ("sst", "w2", "random", 0.0),  # Gaussian Belief Trees
        ("aorrt", "l2", "active", 0.0),  # Active Pusher
        ("sst", "l2", "active", 0.0),  # Active Pusher
        ("aorrt", "l2", "random", 20.0),  # KiTe
        ("aorrt", "w2", "random", 20.0),  # KiTe
    ]

    # Goals are common for all envs
    goal_state = np.array([0, -0.7, 0])
    goal_size = 0.05

    for obj_name, n_data in product(obj_names, n_datas):
        for i in range(len(configs)):
            algo = configs[i][0]
            belief = configs[i][1]
            use_var = 1 if belief == "w2" else 0
            active_sampling = configs[i][2]
            terminal_weight = configs[i][3]

            # Load plan states, controls and costs
            name = (
                f"{obj_name}_{model_type}_{use_var}_{n_data}"
                + f"_{algo}_{belief}_{active_sampling}_{terminal_weight}"
            )
            folder = "results/planning_push"
            poses_file = f"{folder}/{name}_plan_states.npy"
            controls_file = f"{folder}/{name}_controls.npy"
            costs_file = f"{folder}/{name}_costs.npy"

            all_states = np.load(poses_file, allow_pickle=True)
            all_controls = np.load(controls_file, allow_pickle=True)
            all_costs = np.load(costs_file, allow_pickle=True)

            runner = RunPushPlansPool(obj_name, goal_state, goal_size)
            results, exec_states = runner.evaluate(
                all_states[:, :, :], all_controls[:, :, :], all_costs[:, :, :]
            )
            results = np.asarray(results, dtype=float)
            exec_states = np.asarray(exec_states, dtype=object)
            np.save(f"{folder}/{name}_results.npy", results)
            np.save(f"{folder}/{name}_exec_states.npy", exec_states)

            # Get rid of the ones with zero errors (failed plans)
            mask = results[:, :, -1, 3] > 0
            print(f"\n{name}:")
            print(f"Failed plans count:\t{np.sum(~mask)}")
            print(f"Success:\t{np.mean(results[:, :, -1, 0])}")
            print(f"SE2 Error:\t{np.mean(results[:, :, -1, 3][mask])}")
            print(f"Hits count:\t{np.sum(results[:, :, -1, 1])}")
            print(f"Reached count:\t{np.sum(results[:, :, -1, 2])}")
            print(f"Running Cost:\t{np.mean(results[:, :, -1, 9][mask])}")
            print(f"Terminal Cost:\t{np.mean(results[:, :, -1, 10][mask])}")
            print()
