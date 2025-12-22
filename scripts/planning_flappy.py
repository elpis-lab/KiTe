import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from utils import set_seed, parse_args
from planning.planning_utils import set_ompl_seed
from planning.flappy import (
    FlappyPlanner,
    generate_flappy_env,
    visualize_flappy_env,
)


def generate_envs(n_states, visualize=False):
    """Generate initial states"""
    obstacles = []
    for _ in range(n_states):
        env = generate_flappy_env()
        obstacles.append(env["obstacles"])
        if visualize:
            visualize_flappy_env(env)
    obstacles = np.array(obstacles)
    os.makedirs("data", exist_ok=True)
    np.save("data/planning_flappy_obstacles.npy", obstacles)


def run_planning(
    planner,
    terminal_weight,
    obstacles,
    start,
    goal,
    goal_size,
    planning_times,
    n_reps,
):
    n_problems = len(obstacles)
    all_states = [[None for _ in range(n_problems)] for _ in range(n_reps)]
    all_controls = [[None for _ in range(n_problems)] for _ in range(n_reps)]
    all_costs = [[None for _ in range(n_problems)] for _ in range(n_reps)]

    for rep in range(n_reps):
        for prob, obstacle in enumerate(obstacles):
            flappy = FlappyPlanner(obstacle, planner, terminal_weight)
            plan_states, plan_controls, plan_costs = flappy.plan(
                start, goal, goal_size, planning_times
            )
            all_states[rep][prob] = plan_states
            all_controls[rep][prob] = plan_controls
            all_costs[rep][prob] = plan_costs

    all_states = np.array(all_states, dtype=object)
    all_controls = np.array(all_controls, dtype=object)
    all_costs = np.array(all_costs, dtype=object)
    return all_states, all_controls, all_costs


if __name__ == "__main__":
    set_seed(42)
    set_ompl_seed(42)
    # generate_envs(10, visualize=True)

    args = parse_args(
        [
            ("planner", "aorrt"),
            ("terminal_weight", 2.0, float),
            ("n_reps", 10, int),
        ]
    )

    # Load problem set
    obstacles = np.load("data/planning_flappy_obstacles.npy")
    env = generate_flappy_env()
    start, goal, goal_size = env["start"], env["goal_center"], env["goal_size"]
    # Planning time
    planning_times = list(np.linspace(0.1, 10.0, 100))  # 0.1, 0.2, ..., 10.0

    # Run planning
    all_states, all_controls, all_costs = run_planning(
        args.planner,
        args.terminal_weight,
        obstacles,
        start,
        goal,
        goal_size,
        planning_times,
        args.n_reps,
    )

    # Save the results
    name = f"{args.planner}_{args.terminal_weight}"
    os.makedirs("results/planning_flappy", exist_ok=True)
    np.save(f"results/planning_flappy/{name}_plan_states.npy", all_states)
    np.save(f"results/planning_flappy/{name}_controls.npy", all_controls)
    np.save(f"results/planning_flappy/{name}_costs.npy", all_costs)
