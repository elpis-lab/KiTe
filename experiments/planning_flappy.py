import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from experiments.utils import set_seed, parse_args
from planning.flappy import (
    FlappyPlanner,
    generate_flappy_env,
    visualize_flappy_env,
)


def generate_envs(n_states, visualize=False, overwrite=False):
    """Generate initial states"""
    if not overwrite and os.path.exists("data/planning_flappy_envs.npy"):
        return

    envs = []
    for _ in range(n_states):
        env = generate_flappy_env()
        envs.append(env)
        if visualize:
            visualize_flappy_env(env)
            plt.show()
    envs = np.array(envs, dtype=object)
    os.makedirs("data", exist_ok=True)
    np.save("data/planning_flappy_envs.npy", envs)


def run_planning(
    algo,
    terminal_weight,
    envs,
    planning_times,
    n_reps,
):
    n_problems = len(envs)
    all_states = [[None for _ in range(n_problems)] for _ in range(n_reps)]
    all_controls = [[None for _ in range(n_problems)] for _ in range(n_reps)]
    all_costs = [[None for _ in range(n_problems)] for _ in range(n_reps)]

    for rep in range(n_reps):
        for prob, env in enumerate(envs):
            planner = FlappyPlanner(env["obstacles"], algo, terminal_weight)
            plan_states, plan_controls, plan_costs = planner.plan(
                env["start"], env["goal"], env["goal_size"], planning_times
            )
            all_states[rep][prob] = plan_states
            all_controls[rep][prob] = plan_controls
            all_costs[rep][prob] = plan_costs

    all_states = np.array(all_states, dtype=object)
    all_controls = np.array(all_controls, dtype=object)
    all_costs = np.array(all_costs)
    return all_states, all_controls, all_costs


if __name__ == "__main__":
    set_seed(42)
    generate_envs(20, visualize=True)

    args = parse_args(
        [
            ("algo", "aorrt"),
            ("terminal_weight", 1.0, float),
            ("n_reps", 5, int),
        ]
    )

    # Load problem set
    envs = np.load("data/planning_flappy_envs.npy", allow_pickle=True)
    # Planning time: 0.1, 0.2, ..., 10.0
    planning_times = list(np.linspace(0.1, 10.0, 100))

    # Run planning
    all_states, all_controls, all_costs = run_planning(
        args.algo, args.terminal_weight, envs, planning_times, args.n_reps
    )

    # Save the results
    name = f"{args.algo}_{args.terminal_weight}"
    os.makedirs("results/planning_flappy", exist_ok=True)
    np.save(f"results/planning_flappy/{name}_plan_states.npy", all_states)
    np.save(f"results/planning_flappy/{name}_controls.npy", all_controls)
    np.save(f"results/planning_flappy/{name}_costs.npy", all_costs)
