import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from experiments.utils import set_seed, parse_args
from planning.car import SE2CarPlanner, generate_car_env, visualize_car_env
from planning.car import CAR_SIZE


def generate_envs(n_states, visualize=False):
    """Generate initial states"""
    envs = []
    for _ in range(n_states):
        env = generate_car_env()
        envs.append(env)
        if visualize:
            visualize_car_env(env, [env["start"]], draw_car_shape=True)
            plt.show()
    envs = np.array(envs, dtype=object)
    os.makedirs("data", exist_ok=True)
    np.save("data/planning_car_envs.npy", envs)


def run_planning(belief, algo, terminal_weight, envs, planning_times, n_reps):
    """Main function to do planning"""
    # Planning
    n_problems = len(envs)
    all_states = [[None for _ in range(n_problems)] for _ in range(n_reps)]
    all_controls = [[None for _ in range(n_problems)] for _ in range(n_reps)]
    all_costs = [[None for _ in range(n_problems)] for _ in range(n_reps)]

    for rep in range(n_reps):
        for prob, env in enumerate(envs):
            print(f"\nPlanning: {rep + 1}th repeat, {prob + 1}th problem")

            planner = SE2CarPlanner(
                env["obstacles"], CAR_SIZE, belief, algo, terminal_weight
            )
            plan_states, plan_controls, plan_costs = planner.plan(
                env["start"], env["goals"], env["goal_size"], 0, planning_times
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
    # generate_envs(20, visualize=False)

    args = parse_args(
        [
            ("belief", 0, int),
            ("algo", "aorrt", str),
            ("terminal_weight", 5.0, float),
            ("n_reps", 1, int),
        ]
    )

    # Load problem set
    envs = np.load("data/planning_car_envs.npy", allow_pickle=True)
    # Planning time: 1.0, 2.0 ..., 30.0
    planning_times = list(np.linspace(1.0, 30.0, 30))

    # Run planning
    all_states, all_controls, all_costs = run_planning(
        args.belief,
        args.algo,
        args.terminal_weight,
        envs,
        planning_times,
        args.n_reps,
    )

    # Save the results
    belief = "w2" if args.belief == 1 else "l2"
    name = f"car_{args.algo}_{belief}_{args.terminal_weight}"
    os.makedirs("results/planning_car", exist_ok=True)
    np.save(f"results/planning_car/{name}_plan_states.npy", all_states)
    np.save(f"results/planning_car/{name}_controls.npy", all_controls)
    np.save(f"results/planning_car/{name}_costs.npy", all_costs)
