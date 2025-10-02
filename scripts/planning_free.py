import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
from geometry.object_model import get_obj_shape
from train_model import load_model, get_push_physics
from utils import DataLoader, set_seed, parse_args, get_names

from planning.ompl_utils import SE2ControlPlanner, set_ompl_seed
from planning.planning_utils import get_random_se2_states
from planning_obs import run_planning


def generate_initial_states(n_states):
    """Generate initial states"""
    states = get_random_se2_states(n_states)
    os.makedirs("data", exist_ok=True)
    np.save("data/planning_free_initial_states.npy", states)


if __name__ == "__main__":
    set_seed(42)
    set_ompl_seed(42)
    # generate_initial_states(100)

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
    start_states = np.load("data/planning_free_initial_states.npy")
    # Planning environment
    obstacles = np.array([])
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
    os.makedirs("results/planning_free", exist_ok=True)
    np.save(f"results/planning_free/{name}_plan_states.npy", all_states)
    np.save(f"results/planning_free/{name}_controls.npy", all_controls)
