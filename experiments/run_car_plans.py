import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from experiments.run_plans import RunPlans
from planning.car import POS_RANGES, CAR_SIZE, OBSTACLES
from planning.car import SE2CarOptimizationObjective
from planning.planning_utils import rects_rects_in_collision
from planning.planning_utils import points_out_of_bound, se2_points_in_region
from simulation.car_sim import Sim
from lie_group.lie_se2 import to_se2_transform, inv_se2_transform, log_se2


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
        self.sim = Sim(xml, n_envs=10, dt=0.01, visualize=True)

        # Common setup
        self.goals = goals
        self.goal_size = goal_size
        self.desired_goal = desired_goal
        self.obstacles = obstacles
        self.load_saved_exec = load_saved_exec
        self.saved_exec_file = saved_exec_file

    def run_plans(self, plan_states, plan_controls):
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
            exec_states = np.load(self.saved_exec_file, allow_pickle=True)
            exec_states = exec_states.tolist()
        # Execute plans in parallel using simulation
        else:
            # Run simulation to get execution states
            exec_states = []
            n_plans = len(plan_states)
            batch_size = self.sim.n_envs
            for batch_start in range(0, n_plans, batch_size):
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
        for states in exec_states:
            # Extract only SE2 states (x, y, yaw)
            states = np.asarray(states)[:, :3]

            # Check collision for each state
            hit = points_out_of_bound(states[:, :2], POS_RANGES)

            # Check goal for each state
            reached = np.zeros(len(states), dtype=bool)
            for goal in self.goals:
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
        # ("sst", "l2", 0.0),  # Vanilla
        # ("aorrt", "l2", 0.0),  # Vanilla
        # ("sst", "w2", 0.0),  # Gaussian Belief Trees
        # ("aorrt", "w2", 0.0),  # Gaussian Belief Trees
        # ("aorrt", "l2", 2.0),  # Proposed
        # ("aorrt", "w2", 2.0),  # Proposed
        # ("aorrt", "l2", 5.0),  # Proposed
        # ("aorrt", "w2", 5.0),  # Proposed
        # ("aorrt", "l2", 10.0),  # Proposed
        # ("aorrt", "w2", 10.0),  # Proposed
        ("aorrt", "l2", 50.0),  # Proposed
        ("aorrt", "w2", 50.0),  # Proposed
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
            True,
            f"{folder}/{name}_exec_states.npy",
        )
        results, exec_states = runner.evaluate(
            all_states[:, :, -1].reshape(-1),  # object
            all_controls[:, :, -1].reshape(-1),  # object
            all_costs[:, :, -1, :].reshape(-1, 2),  # float
            reps_in_states=5,
        )
        runner.close()
        np.save(f"{folder}/{name}_results.npy", results)
        np.save(
            f"{folder}/{name}_exec_states.npy",
            np.array(exec_states, dtype=object),
        )

        # Get rid of the ones with zero errors (failed plans)
        mask = results[:, :, 3] > 0
        print(f"\n{name}:")
        print(f"Success:\t{np.mean(results[:, :, 0])}")
        print(f"SE2 Error:\t{np.mean(results[:, :, 3][mask])}")
        print(f"Hits Obs:\t{results[:, :, 1]}")
        print(f"Reached:\t{results[:, :, 2]}")
        print(f"Running Cost:\t{np.mean(results[:, :, 9][mask])}")
        print(f"Terminal Cost:\t{np.mean(results[:, :, 10][mask])}")

        # Visualize
        from planning.car import visualize_car_env
        import matplotlib.pyplot as plt

        envs = np.load(f"data/planning_car_envs.npy", allow_pickle=True)
        # for j in range(len(envs)):
        for j in range(5):
            print(f"Problem {j}:")
            mask = results[:, j, 3] > 0
            print(f"Terminal Cost:\t{np.mean(results[:, j, 10][mask])}")
            print(f"Success:\t{np.mean(results[:, j, 0])}")
            # for i in range(5):
            #     visualize_car_env(
            #         envs[j],
            #         all_states[i, j, -1],
            #         exec_states[i * len(envs) + j],
            #         draw_car_shape=True,
            #     )
            #     plt.show()
