import os, sys

from sympy.logic.boolalg import false

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm

from experiments.run_plans import RunPlans
from geometry.pose import wrap_to_pi
from planning.car import POS_RANGES, OBSTACLES, CAR_SIZE, SE2CarPlanner
from planning.car import SE2CarOptimizationObjective
from planning.planning_utils import circles_in_collision
from planning.planning_utils import rects_rects_in_collision
from planning.planning_utils import points_out_of_bound, se2_points_in_region
from planning.planning_utils import vec_to_cov, cov_to_vec
from geometry.car_dynamics import propagate_analytical, propagate_cov_linear
from geometry.car_dynamics import process_cov_body
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

            # Compute costs from execution states
            running_cost, terminal_cost = self.compute_costs(states)

            hits.append(hit)
            reacheds.append(reached)
            costs.append([running_cost, terminal_cost])
        costs = np.array(costs, dtype=float)

        # Compute unified costs in belief space
        unified_costs = []
        for i in range(len(plan_states)):
            states = np.asarray(plan_states[i])
            controls = np.asarray(plan_controls[i])
            # initial state
            unified_states = np.zeros((len(states), 9))
            unified_states[0, :3] = states[0, :3]
            unified_states[0, 3:] = [1e-6, 0, 0, 1e-6, 0, 1e-6]
            # propagate states
            for j, control in enumerate(controls):
                curr_state = unified_states[j, :3]
                curr_cov = vec_to_cov(unified_states[j, 3:])
                unified_states[j + 1, :3] = propagate_analytical(
                    control, 1.0, curr_state
                )
                process_cov = process_cov_body(control, 1.0)
                cov = propagate_cov_linear(
                    control, 1.0, curr_state, curr_cov, process_cov
                )
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
            r_cost = SE2CarOptimizationObjective.se2_distance(
                states[i], states[i + 1], covs[i], covs[i + 1], weight=weight
            )
            running_cost += r_cost

        # Terminal cost: distance to desired goal
        desired_goal = self.goals[self.desired_goal]
        terminal_cost = SE2CarOptimizationObjective.se2_distance(
            states[-1], desired_goal, covs[-1], weight=weight
        )
        return running_cost, terminal_cost

    def close(self):
        self.sim.close()


if __name__ == "__main__":
    # Configs (algo, use_var, terminal_weight)
    configs = [
        ("aorrt", "l2", 0.0),  # Base
        ("sst", "l2", 0.0),  # Base
        ("aorrt", "w2", 0.0),  # Gaussian Belief Trees
        ("sst", "w2", 0.0),  # Gaussian Belief Trees
        ("aorrt", "l2", 50.0),  # KiTe
        ("aorrt", "w2", 50.0),  # KiTe
        ("aorrt", "l2", 20.0),  # KiTe
        ("aorrt", "w2", 20.0),  # KiTe
        ("aorrt", "l2", 5.0),  # KiTe
        ("aorrt", "w2", 5.0),  # KiTe
        ("aorrt", "l2", 10.0),  # KiTe
        ("aorrt", "w2", 10.0),  # KiTe
        ("aorrt", "l2", 100.0),  # KiTe
        ("aorrt", "w2", 100.0),  # KiTe
        # ("aorrt", "l2", 200.0),  # KiTe
        ("aorrt", "w2", 200.0),  # KiTe
    ]

    # Execute plans
    for i in range(len(configs)):
        algo = configs[i][0]
        belief = configs[i][1]
        terminal_weight = configs[i][2]
        print(f"\nRunning {algo} with {belief} and weight {terminal_weight}")

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
        print()
