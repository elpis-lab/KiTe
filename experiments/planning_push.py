import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from geometry.object_model import get_obj_shape
from experiments.train_push_model import load_model
from experiments.utils import DataLoader, set_seed, parse_args, get_names
from planning.planning_utils import vec_to_cov
from planning.push import SE2PushPlanner, generate_push_env, visualize_push_env


def generate_envs(n_states, visualize=False):
    """Generate initial states"""
    envs = []
    for _ in range(n_states):
        env = generate_push_env()
        envs.append(env)
        if visualize:
            visualize_push_env(env)
            plt.show()
    envs = np.array(envs, dtype=object)
    os.makedirs("data", exist_ok=True)
    np.save("data/planning_push_envs.npy", envs)


def run_planning(
    obj_name,
    model_type,
    use_var,
    n_data,
    algo,
    active_sampling,
    terminal_weight,
    predefined_controls,
    envs,
    planning_times,
    n_reps,
):
    """Main function to do planning"""
    # Load dynamics model
    m_id = 0
    model_name, data_name, rep_data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    # Load trained model (always use var=1 model for fair comparison)
    model = load_model(model_type, obj_shape, 1)
    name = f"{obj_name}_{model_type}_{1}_{n_data}_{m_id}"
    model.load(f"results/models/{name}.pth")
    # Load data
    data_loader = DataLoader(data_name)
    dataset = data_loader.load_data()
    used_indices = np.load(f"results/learning/idx_used_{name}.npy")

    # Active sampling
    if active_sampling:
        x_train = dataset["x_pool"][used_indices]
    else:
        x_train = None
    # Planning control lists
    if predefined_controls:
        # samplable controls, exclude training data from control pool
        mask = np.ones(len(dataset["x_pool"]), dtype=bool)
        mask[used_indices] = False
        control_list = dataset["x_pool"][mask].astype(np.float64)
    else:
        control_list = None

    # Planning
    n_problems = len(envs)
    all_states = [[None for _ in range(n_problems)] for _ in range(n_reps)]
    all_controls = [[None for _ in range(n_problems)] for _ in range(n_reps)]
    all_costs = [[None for _ in range(n_problems)] for _ in range(n_reps)]

    for rep in range(n_reps):
        for prob, env in enumerate(envs):
            print(f"Planning: {rep}th repeat, {prob}th problem")
            planner = SE2PushPlanner(
                env["obstacles"],
                obj_shape,
                model,
                x_train,
                use_var == 1,
                algo,
                terminal_weight,
                control_list,
            )
            plan_states, plan_controls, plan_costs = planner.plan(
                env["start"], env["goal"], env["goal_size"], planning_times
            )
            all_states[rep][prob] = plan_states
            all_controls[rep][prob] = plan_controls
            all_costs[rep][prob] = plan_costs
    planner.close()

    all_states = np.array(all_states, dtype=object)
    all_controls = np.array(all_controls, dtype=object)
    all_costs = np.array(all_costs)
    return all_states, all_controls, all_costs


if __name__ == "__main__":
    set_seed(42)
    # generate_envs(20, visualize=False)

    args = parse_args(
        [
            # Model
            ("obj_name", "cracker_box_flipped"),
            ("model_type", "mlp"),
            ("use_var", 1, int),
            ("n_data", 1000, int),
            # Planning
            ("algo", "aorrt", str),
            ("active_sampling", 0, int),
            ("terminal_weight", 2.0, float),
            ("n_reps", 1, int),
            ("predefined_controls", 1, int),
        ]
    )

    # Load problem set
    envs = np.load("data/planning_push_envs.npy", allow_pickle=True)
    # Planning time: 1.0, 2.0 ..., 10.0
    planning_times = list(np.linspace(1.0, 10.0, 10))

    # Run planning
    all_states, all_controls, all_costs = run_planning(
        args.obj_name,
        args.model_type,
        args.use_var,
        args.n_data,
        args.algo,
        args.active_sampling,
        args.terminal_weight,
        args.predefined_controls,
        envs,
        planning_times,
        args.n_reps,
    )

    # Save the results
    belief = "w2" if args.use_var == 1 else "l2"
    active_sampling = "active" if args.active_sampling == 1 else "random"
    name = (
        f"{args.obj_name}_{args.model_type}_{args.use_var}_{args.n_data}"
        + f"_{args.algo}_{belief}_{active_sampling}_{args.terminal_weight}"
    )
    os.makedirs("results/planning_push", exist_ok=True)
    np.save(f"results/planning_push/{name}_plan_states.npy", all_states)
    np.save(f"results/planning_push/{name}_controls.npy", all_controls)
    np.save(f"results/planning_push/{name}_costs.npy", all_costs)
