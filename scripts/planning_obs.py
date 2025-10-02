import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
from geometry.object_model import get_obj_shape
from train_model import load_model, get_push_physics
from utils import DataLoader, set_seed, parse_args, get_names

from planning.ompl_utils import SE2ControlPlanner, set_ompl_seed
from planning.planning_utils import get_random_se2_states


def generate_initial_states_with_obstacles(n_states, obstacles):
    """Generate initial states"""
    safe_range = 0.15
    goal = np.array([0, -0.7, 0])
    obstacles = np.asarray(obstacles)  # assume they are circles

    valid_states = []
    while len(valid_states) < n_states:
        # Generate random state
        state = get_random_se2_states(1)[0]
        # Check clearance
        dists = np.linalg.norm(state[:2] - obstacles[:, :2], axis=-1)
        clearance = dists - (obstacles[:, 2] + safe_range)
        if clearance.min() <= 0.0:
            continue
        valid_states.append(state)

    os.makedirs("data", exist_ok=True)
    np.save("data/planning_obs_initial_states.npy", valid_states)


def run_planning(
    obj_name,
    model_type,
    use_var,
    n_data,
    datasets,
    belief,
    sampling,
    selection,
    predefined_controls,
    start_states,
    obstacles,
    planning_time,
    n_reps,
    accept_approximate=False,
):
    """Main function to do planning"""
    # Planning parameters
    # state bounds
    bounds = ((-0.6, 0.6), (-1.0, -0.4))
    # control bounds
    control_bounds = ((0, 4), (-0.4, 0.4), (0.0, 0.3))

    # Load dynamics model
    m_id = 0
    model_name, data_name, rep_data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    model = load_model(model_type, obj_shape, use_var)
    name = f"{obj_name}_{model_type}_{use_var}_{n_data}_{m_id}"
    model.load(f"results/models/{name}.pth")
    # Load data that is used for training, for active planning
    used_indices = np.load(f"results/learning/idx_used_{name}.npy")
    x_train = datasets["x_pool"][used_indices]
    # Planning control lists
    if predefined_controls:
        # samplable controls, exclude training data from control pool
        mask = np.ones(len(datasets["x_pool"]), dtype=bool)
        mask[used_indices] = False
        controls = datasets["x_pool"][mask].astype(np.float64)
    else:
        controls = None

    # Define commom goal
    goal_state = np.array([0, -0.7, 0])
    goal_ranges = np.array(
        [[-0.02, 0.02], [-0.02, 0.02], [-np.pi / 8, np.pi / 8]]
    )

    # Instantiate the planner
    planner = SE2ControlPlanner(
        bounds,
        control_bounds,
        obj_shape,
        obstacles,
        model,
        x_train,
        belief,
        sampling,
        selection,
        controls,
    )

    # Define the start and goal
    all_controls = []  # [n, n_steps]
    all_states = []  # [n, n_steps]
    for _ in range(n_reps):
        for start_state in start_states[:]:
            # Plan
            plan_states, plan_controls = planner.plan(
                start_state,
                goal_state,
                goal_ranges,
                planning_time,
                accept_approximate,
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
    # obstacles = np.array(
    #     [[0, -0.5, 0.05], [-0.25, -0.8, 0.05], [0.25, -0.8, 0.05]]
    # )
    # generate_initial_states_with_obstacles(100, obstacles)

    args = parse_args(
        [
            # Model
            ("obj_name", "cracker_box_flipped"),
            ("model_type", "mlp"),
            ("use_var", 1.0, float),
            ("n_data", 1000, int),
            # Planning
            ("belief_space", 1, int),
            ("active_sampling", 1, int),
            ("active_selection", 1, int),
            ("predefined_controls", 1, int),
            ("n_reps", 1, int),
        ]
    )

    # Load dataset
    model_name, data_name, rep_data_name = get_names(args.obj_name)
    data_loader = DataLoader(data_name)
    datasets = data_loader.load_data()  # Get all data (all in pool)
    # Load problem set
    start_states = np.load("data/planning_obs_initial_states.npy")
    # Planning environment
    obstacles = np.array(
        [[0, -0.5, 0.05], [-0.25, -0.8, 0.05], [0.25, -0.8, 0.05]]
    )
    # Planning time
    planning_time = 3

    all_states, all_controls = run_planning(
        args.obj_name,
        args.model_type,
        args.use_var,
        args.n_data,
        datasets,
        args.belief_space,
        args.active_sampling,
        args.active_selection,
        args.predefined_controls,
        start_states,
        obstacles,
        planning_time,
        args.n_reps,
    )

    # Save the results
    belief = "belief" if args.belief_space else "regular"
    sampling = "active" if args.active_sampling else "random"
    selection = "prob" if args.active_selection else "cost"
    name = (
        f"{args.obj_name}_{args.model_type}_{args.use_var}_{args.n_data}"
        + f"_{belief}_{sampling}_{selection}"
    )
    os.makedirs("results/planning_obs", exist_ok=True)
    np.save(f"results/planning_obs/{name}_plan_states.npy", all_states)
    np.save(f"results/planning_obs/{name}_controls.npy", all_controls)
