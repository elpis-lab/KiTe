import numpy as np

from geometry.object_model import get_obj_shape
from train_model import load_model, get_push_physics
from utils import DataLoader, set_seed, parse_args

from planning_utils import set_ompl_seed, get_random_se2_states
from planning_utils import SE2ControlPlanner


def generate_initial_states(n_states):
    """Generate initial states"""
    states = get_random_se2_states(n_states)
    np.save("data/planning_initial_states.npy", states)


def run_planning(
    obj_name,
    model_type,
    belief,
    sampling,
    selection,
    controls,
    start_states,
    planning_time,
):
    """Main function to do planning"""
    # Planning parameters
    # state bounds
    bounds = ((-0.76, 0.76), (-1.0, -0.4))
    # control bounds
    control_bounds = ((0, 4), (-0.4, 0.4), (0.0, 0.3))

    # Load dynamics model
    use_var = 2
    data_usage = "1000x1"
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")
    equation = get_push_physics(model_type, obj_shape)
    model = load_model(model_type, equation, use_var)
    name = f"{obj_name}_{model_type}_{use_var}_{data_usage}"
    model.load(f"results/models/{name}.pt")

    # Define commom goal
    goal_state = np.array([0, -0.7, 0])
    goal_ranges = np.array(
        [[-0.03, 0.03], [-0.03, 0.03], [-np.pi / 8, np.pi / 8]]
    )

    # Instantiate the planner
    planner = SE2ControlPlanner(
        bounds,
        control_bounds,
        model,
        belief,
        sampling,
        selection,
        controls,
    )

    # Define the start and goal
    all_controls = []  # [n, n_steps]
    all_states = []  # [n, n_steps]
    n_reps = 10
    n_problems = 100
    for _ in range(n_reps):
        for start_state in start_states[:n_problems]:
            # Plan
            plan_states, plan_controls = planner.plan(
                start_state, goal_state, goal_ranges, planning_time
            )
            if not plan_controls:
                plan_states = [start_state, start_state]
                plan_controls = [[0, 0, -0.05]]
            all_states.append(plan_states)
            all_controls.append(plan_controls)
    all_states = np.array(all_states, dtype=object)
    all_controls = np.array(all_controls, dtype=object)

    return all_states, all_controls


if __name__ == "__main__":
    set_seed(42)
    set_ompl_seed(42)
    # generate_initial_states(100)

    args = parse_args(
        [
            ("obj_name", "master_chef_can_flipped"),
            ("model_type", "mlp"),
            ("belief_space", 1, int),
            ("active_sampling", 1, int),
            ("active_selection", 1, int),
            ("predefined_controls", 1, int),
        ]
    )
    # Load problem set
    start_states = np.load("data/planning_initial_states.npy")
    # Load control list
    if args.predefined_controls:
        data_loader = DataLoader(args.obj_name + "_10000", "data", val_size=0)
        datasets = data_loader.load_data()
        controls = datasets["x_pool"].astype(np.float64)
    else:
        controls = None
    # Planning time
    planning_time = 3

    all_states, all_controls = run_planning(
        args.obj_name,
        args.model_type,
        args.belief_space,
        args.active_sampling,
        args.active_selection,
        controls,
        start_states,
        planning_time,
    )

    # Save the results
    belief = "belief" if args.belief_space else "regular"
    sampling = "active" if args.active_sampling else "random"
    selection = "prob" if args.active_selection else "cost"
    name = f"{args.obj_name}_{args.model_type}_{belief}_{sampling}_{selection}"
    np.save(f"results/planning/{name}_states.npy", all_states)
    np.save(f"results/planning/{name}_controls.npy", all_controls)
