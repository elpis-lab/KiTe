import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm

from experiments.run_plans import RunPlans
from geometry.pose import wrap_to_pi
from planning.car import POS_RANGES, OBSTACLES, CAR_SIZE, SE2CarPlanner
from planning.car import SE2CarOptimizationObjective
from planning.planning_utils import (
    circles_in_collision,
    rects_rects_in_collision,
)
from planning.planning_utils import points_out_of_bound, se2_points_in_region
from simulation.car_sim import Sim


class RunCarPlans(RunPlans):
    """Run car plans"""

    def __init__(
        self,
        goals,
        goal_size,
        desired_goal=0,
        obstacles=OBSTACLES,
        load_saved_exec=False,
        saved_exec_file="",
    ):
        """Initialize car plan runner"""
        super().__init__()

        # Initialize simulation
        par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        xml = open(os.path.join(par_dir, "simulation/car_sim.xml")).read()
        self.sim = Sim(xml, 10, 0.01)

        # Common setup
        self.planner = SE2CarPlanner(obstacles)
        self.goals = goals
        self.goal_size = goal_size
        self.desired_goal = desired_goal
        self.obstacles = obstacles
        self.load_saved_exec = load_saved_exec
        self.saved_exec_file = saved_exec_file

    def run_plans(self, plan_states, plan_controls, idx_flat_to_grid=None):
        """
        Run the plans and collect the results

        Expected to return (list of list of result):
        For each state in each plan:
        exec_states: list of executed states
        hits: list of boolean indicating if hit obstacles at each state
        reacheds: list of boolean indicating if reached the goal at each states
        exec_costs: list of execution costs
        """
        # Check if we should load saved execution states
        if self.load_saved_exec:
            exec_states_grid = np.load(self.saved_exec_file, allow_pickle=True)
            exec_states = []
            for idx, (r, p, t) in idx_flat_to_grid.items():
                exec_states.append(exec_states_grid[r, p, t])
        else:
            # Run simulation to get execution states
            exec_states = []
            n_plans = len(plan_states)
            batch_size = self.sim.n_envs
            for batch_start in tqdm(range(0, n_plans, batch_size)):
                batch_end = min(batch_start + batch_size, n_plans)

                # Prepare controls and durations for parallel execution
                controls_batch = []
                durations_batch = []
                init_states_batch = []
                for p_states, p_controls in zip(
                    plan_states[batch_start:batch_end],
                    plan_controls[batch_start:batch_end],
                ):
                    p_states = np.asarray(p_states)
                    p_controls = np.asarray(p_controls)

                    # Extract initial state (first keyframe + v = 0)
                    init_state = np.append(p_states[0, :3], 0.0)
                    init_states_batch.append(init_state)
                    # Prepare controls and durations
                    controls_batch.append(p_controls)
                    durations_batch.append(np.ones(len(p_controls)) * 1.0)

                # Set initial states for the batch
                self.sim.set_car_init_states(np.array(init_states_batch))
                self.sim.reset(wait_time=0.0)
                # Execute controls in parallel
                batch_exec_states = self.sim.execute_controls(
                    controls_batch, durations_batch
                )
                exec_states.extend(batch_exec_states)

        # Compute hits, reacheds, and costs from loaded states
        hits = []
        reacheds = []
        costs = []
        for i, states in enumerate(exec_states):
            # Extract only SE2 states (x, y, yaw)
            states = np.asarray(states)[:, :3]

            # Check collision for each state
            hit = points_out_of_bound(states[:, :2], POS_RANGES)
            # hit = hit | rects_rects_in_collision(
            #     states[:, :3], CAR_SIZE, self.obstacles
            # )
            for j in range(len(states)):
                hit[j] = hit[j] | circles_in_collision(
                    states[j],
                    self.planner.car_circles,
                    self.planner.obstacles_circles,
                )

            # Check goal for each state
            reached = np.zeros(len(states), dtype=bool)
            for g_i, goal in enumerate(self.goals):
                reached = reached | se2_points_in_region(
                    states, goal, self.goal_size
                )

            # Compute costs from loaded states
            running_cost, terminal_cost = self.compute_costs(states)

            hits.append(hit)
            reacheds.append(reached)
            costs.append([running_cost, terminal_cost])
        return exec_states, hits, reacheds, np.array(costs)

    def compute_costs(self, states):
        """
        Compute running cost (accumulated SE2 motion cost) and terminal cost
        (Wasserstein distance to Dirac measure of desired goal).

        Args:
            states: (N, 3) array of keyframe states [x, y, yaw]

        Returns:
            running_cost: accumulated SE2 motion cost
            terminal_cost: Wasserstein distance to desired goal
        """
        states = np.asarray(states)
        rot_weight = 0.2
        weight = np.diag([1.0, 1.0, rot_weight])

        # Running cost: accumulated SE2 motion cost
        running_cost = 0.0
        for i in range(len(states) - 1):
            # Compute SE2 distance
            r_cost = SE2CarOptimizationObjective.se2_distance(
                states[i], states[i + 1], weight=weight
            )
            running_cost += r_cost

        # Terminal cost: distance to desired goal
        desired_goal = self.goals[self.desired_goal]
        terminal_cost = SE2CarOptimizationObjective.se2_distance(
            states[-1], desired_goal, weight=weight
        )

        return running_cost, terminal_cost

    def close(self):
        self.sim.close()


if __name__ == "__main__":
    # Configs (algo, use_var, active_sampling, terminal_weight)
    configs = [
        ("sst", "l2", 0.0),  # Vanilla
        ("sst", "w2", 0.0),  # Gaussian Belief Trees
        # ("aorrt", "l2", 0.0),  # Vanilla
        # ("aorrt", "w2", 0.0),  # Gaussian Belief Trees
        # ("aorrt", "l2", 5.0),  # Proposed
        # ("aorrt", "w2", 5.0),  # Proposed
        # ("aorrt", "l2", 20.0),  # Proposed
        # ("aorrt", "w2", 20.0),  # Proposed
        # ("aorrt", "l2", 50.0),  # Proposed
        # ("aorrt", "w2", 50.0),  # Proposed
    ]

    # Execute plans
    for i in range(len(configs)):
        algo = configs[i][0]
        belief = configs[i][1]
        terminal_weight = configs[i][2]
        print(f"Running {algo} with {belief} and weight {terminal_weight}")

        # Goals, goal_size and obstacles are the same for all envs
        envs = np.load("data/planning_car_envs.npy", allow_pickle=True)
        env = envs[0]
        goals = env["goals"]
        goal_size = env["goal_size"]
        obstacles = env["obstacles"]

        # Load
        name = f"car_{algo}_{belief}_{terminal_weight}"
        folder = f"results/planning_car"
        poses_file = f"{folder}/{name}_plan_states.npy"
        controls_file = f"{folder}/{name}_controls.npy"
        costs_file = f"{folder}/{name}_costs.npy"
        all_states = np.load(poses_file, allow_pickle=True)
        all_controls = np.load(controls_file, allow_pickle=True)
        all_costs = np.load(costs_file, allow_pickle=True)

        # Run
        runner = RunCarPlans(
            goals,
            goal_size,
            0,
            obstacles,
            load_saved_exec=False,
            saved_exec_file=f"{folder}/{name}_exec_states.npy",
        )
        results, exec_states = runner.evaluate(
            all_states[:, :, :], all_controls[:, :, :], all_costs[:, :, :]
        )
        results = np.asarray(results, dtype=float)
        exec_states = np.asarray(exec_states, dtype=object)
        np.save(f"{folder}/{name}_results.npy", results)
        np.save(f"{folder}/{name}_exec_states.npy", exec_states)
        runner.close()

        # Get rid of the ones with zero errors (failed plans)
        mask = results[:, :, -1, 3] > 0
        print(f"\n{name}:")
        print(f"Failed plans count:\t{np.sum(~mask)}")
        print(f"Success:\t{np.mean(results[:, :, -1, 0])}")
        print(f"SE2 Error:\t{np.mean(results[:, :, -1, 3][mask])}")
        # print(f"Hits Obs:\t{results[:, :, -1, 1]}")
        print(f"Hits count:\t{np.sum(results[:, :, -1, 1])}")
        # print(f"Reached:\t{results[:, :, -1, 2]}")
        print(f"Reached count:\t{np.sum(results[:, :, -1, 2])}")
        print(f"Running Cost:\t{np.mean(results[:, :, -1, 9][mask])}")
        print(f"Terminal Cost:\t{np.mean(results[:, :, -1, 10][mask])}")

        # # TODO
        # from geometry.pose import vec_to_cov
        # # Compute average planned running costs from all_states
        # l2_running_costs = []
        # w2_running_costs = []
        # # Iterate through all plans (reps, problems, times)
        # for rep_idx in range(all_states.shape[0]):
        #     for prob_idx in range(all_states.shape[1]):
        #         plan_path = all_states[rep_idx, prob_idx, -1]
        #         if plan_path is None or len(plan_path) < 2:
        #             continue
        #         plan_path = np.asarray(plan_path)

        #         # Compute running cost for this plan
        #         l2_cost = 0.0
        #         w2_cost = 0.0
        #         for i in range(len(plan_path) - 1):
        #             s1 = plan_path[i]
        #             s2 = plan_path[i + 1]
        #             # Extract state and covariance
        #             state1 = s1[:3]
        #             state2 = s2[:3]
        #             cov1 = vec_to_cov(s1[3:])
        #             cov2 = vec_to_cov(s2[3:])

        #             # L2 distance (no covariance)
        #             l2_dist = SE2CarOptimizationObjective.se2_distance(
        #                 state1, state2
        #             )
        #             l2_cost += l2_dist
        #             # W2 distance (with covariance)
        #             w2_dist = SE2CarOptimizationObjective.se2_distance(
        #                 state1, state2, cov1, cov2
        #             )
        #             w2_cost += w2_dist
        #         # print(len(plan_path))
        #         # input()
        #         l2_running_costs.append(l2_cost)
        #         w2_running_costs.append(w2_cost)
        # avg_l2_running_cost = np.mean(l2_running_costs)
        # avg_w2_running_cost = np.mean(w2_running_costs)
        # print(f"Planned Running Cost (L2):\t{avg_l2_running_cost:.4f}")
        # print(f"Planned Running Cost (W2):\t{avg_w2_running_cost:.4f}")

        # # TODO
        # # Compute average planned terminal costs from all_states
        # desired_goal = goals[0]  # desired_goal = 0 from line 219
        # l2_terminal_costs = []
        # w2_terminal_costs = []
        # # Iterate through all plans (reps, problems, times)
        # for rep_idx in range(all_states.shape[0]):
        #     for prob_idx in range(all_states.shape[1]):
        #         plan_path = all_states[rep_idx, prob_idx, -1]
        #         if plan_path is None or len(plan_path) < 2:
        #             continue
        #         plan_path = np.asarray(plan_path)

        #         # Get the last state
        #         last_state_vec = plan_path[-1]
        #         last_state = last_state_vec[:3]

        #         # L2 terminal cost (no covariance)
        #         l2_terminal = SE2CarOptimizationObjective.se2_distance(
        #             last_state, desired_goal
        #         )
        #         l2_terminal_costs.append(l2_terminal)

        #         # W2 terminal cost (with covariance, Wasserstein to Dirac at goal)
        #         cov_last = vec_to_cov(last_state_vec[3:])
        #         w2_terminal = SE2CarOptimizationObjective.se2_distance(
        #             last_state, desired_goal, cov_last
        #         )
        #         w2_terminal_costs.append(w2_terminal)

        # avg_l2_terminal_cost = np.mean(l2_terminal_costs)
        # avg_w2_terminal_cost = np.mean(w2_terminal_costs)
        # print(f"Planned Terminal Cost (L2):\t{avg_l2_terminal_cost:.4f}")
        # print(f"Planned Terminal Cost (W2):\t{avg_w2_terminal_cost:.4f}")

        # # Cholesky whitening: check if planned covariance matches executed states
        # # Use last trajectory of each (trial, problem); state cov in global frame.
        # n_reps, n_problems = all_states.shape[0], all_states.shape[1]
        # all_whitened = []
        # all_d2 = []
        # for rep in range(n_reps):
        #     for prob in range(n_problems):
        #         plan_path = all_states[rep, prob, -1]
        #         if plan_path is None or len(plan_path) < 2:
        #             continue
        #         plan_path = np.asarray(plan_path)
        #         # Executed trajectory for this (rep, prob)
        #         plan_flat_idx = rep * n_problems + prob
        #         exec_traj = np.asarray(exec_states[plan_flat_idx])[:, :3]

        #         # compute cholesky whitening
        #         for i in range(len(plan_path)):
        #             planned_vec = plan_path[i]
        #             if len(planned_vec) <= 3:
        #                 continue
        #             mu = np.asarray(planned_vec[:3], dtype=float)
        #             cov = vec_to_cov(planned_vec[3:])

        #             exec_state = exec_traj[i]
        #             residual = exec_state - mu
        #             residual[2] = wrap_to_pi(residual[2])

        #             chol = np.linalg.cholesky(cov)
        #             w = np.linalg.solve(chol, residual)
        #             all_whitened.append(w)
        #             d2 = float(w @ w)
        #             all_d2.append(d2)

        # if all_whitened:
        #     all_whitened = np.array(all_whitened)
        #     all_d2 = np.array(all_d2)
        #     # Chi-squared 3 dof quantiles: fraction of exec states inside that cov region
        #     chi2 = {"50%": 2.366, "90%": 6.251, "95%": 7.815, "99%": 11.345}
        #     cover = {k: np.mean(all_d2 <= v) for k, v in chi2.items()}

        #     print(
        #         "Cholesky whitened (planned cov vs exec; ideal mean~0, std~1):"
        #     )
        #     for j, name in enumerate(["x", "y", "yaw"]):
        #         print(
        #             f"  {name:>4s}: mean={all_whitened[:, j].mean():+.3f}, "
        #             f"std={all_whitened[:, j].std(ddof=1):.3f}"
        #         )
        #     print(f"  d^2: mean={all_d2.mean():.3f} (ideal ~3.0)")
        #     print("Coverage:", cover)
        # else:
        #     print(
        #         "Cholesky whitening: no valid (rep,prob,keyframe) with covariance."
        #     )

        # # TODO
        # # Visualize
        # from planning.car import visualize_car_env
        # import matplotlib.pyplot as plt

        # envs = np.load(f"data/planning_car_envs.npy", allow_pickle=True)
        # # for j in range(len(envs)):
        # for j in range(1):
        #     print(f"Problem {j}:")
        #     mask = results[:, j, 3] > 0
        #     print(f"Terminal Cost:\t{np.mean(results[:, j, 10][mask])}")
        #     print(f"Success:\t{np.mean(results[:, j, 0])}")
        #     for i in range(len(all_states)):
        #         visualize_car_env(
        #             envs[j],
        #             all_states[i, j, -1],
        #             exec_states[i * len(envs) + j],
        #             draw_car_shape=True,
        #         )
        #         plt.show()
