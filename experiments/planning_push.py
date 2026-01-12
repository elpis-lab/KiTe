import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from geometry.object_model import get_obj_shape
from experiments.train_push_model import load_model
from experiments.utils import DataLoader, set_seed, parse_args, get_names
from planning.push import SE2PushPlanner, generate_push_env, visualize_push_env


def generate_envs(n_states, visualize=False):
    """Generate initial states"""
    envs = []
    for _ in range(n_states):
        env = generate_push_env()
        envs.append(env)
        if visualize:
            visualize_push_env(env)
    envs = np.array(envs, dtype=object)
    os.makedirs("data", exist_ok=True)
    np.save("data/planning_push_envs.npy", envs)


# State Utils
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

    all_states = np.array(all_states, dtype=object)
    all_controls = np.array(all_controls, dtype=object)
    all_costs = np.array(all_costs)
    return all_states, all_controls, all_costs


if __name__ == "__main__":
    set_seed(42)
    # generate_envs(20, visualize=True)

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
    # Planning time: 1.0, 2.0 ..., 30.0
    planning_times = list(np.linspace(1.0, 30.0, 30))

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

    # from planning.planning_utils import plot_belief_states
    # from planning.ompl_utils import SE2BeliefOptimizationObjective
    # from lie_group.lie_se2 import to_se2_transform, to_se2_vec

    # # beliefs, obstacles=np.array([]), goal=None, goal_size=None, conf=0.95
    # plan_states = np.array(all_states[0], dtype=np.float32)
    # controls = np.array(all_controls[0], dtype=np.float32)
    # obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    # model = load_model(args.model_type, obj_shape, args.use_var)
    # name = f"{args.obj_name}_{args.model_type}_{args.use_var}_{args.n_data}_0"
    # model.load(f"results/models/{name}.pth")

    # from lie_group.propagation import Propagation_SE2
    # from lie_group.lie_se2 import exp_se2, log_se2, inv_se2_transform
    # import matplotlib.pyplot as plt

    # pred = model.predict(controls)

    # # Use Monte Carlo
    # n_samples = 1000

    # motions = []
    # for p in pred:
    #     delta = log_se2(to_se2_transform(p[:3]))
    #     delta_cov = np.diag(np.exp(p[3:]))
    #     motion = np.random.multivariate_normal(
    #         delta, delta_cov, size=(n_samples)
    #     )
    #     motions.append(motion)
    # motions = np.array(motions, dtype=np.float32)
    # t0_delta = log_se2(to_se2_transform(plan_states[0, :3]))
    # t0_delta_cov = SE2BeliefOptimizationObjective.vec_to_cov(
    #     plan_states[0, 3:]
    # )
    # t0s = np.random.multivariate_normal(
    #     t0_delta, t0_delta_cov, size=(n_samples)
    # )
    # plt.figure()
    # transform_samples = [exp_se2(t0) for t0 in t0s]
    # for n in range(len(controls)):
    #     # xy = np.array([to_se2_vec(T)[:2] for T in transform_samples])
    #     # plt.plot(xy[:, 0], xy[:, 1], "b.", alpha=0.2)
    #     transform_samples = [
    #         transform_samples[i] @ exp_se2(motions[n, i])
    #         for i in range(n_samples)
    #     ]
    # xyyaw = np.array([to_se2_vec(T) for T in transform_samples])
    # plt.plot(xyyaw[:, 0], xyyaw[:, 1], "b.", alpha=0.2)

    # # count xyyaw in region
    # s_count = 0
    # goal = np.array([0, -0.7, 0])
    # goal_ranges = np.array(
    #     [[-0.02, 0.02], [-0.02, 0.02], [-np.pi / 8, np.pi / 8]]
    # )
    # region_inv = inv_se2_transform(to_se2_transform(goal))
    # rel_transforms = [region_inv @ tr for tr in transform_samples]
    # rel_vecs = np.array([to_se2_vec(tr) for tr in rel_transforms])
    # se2_lower, se2_upper = goal_ranges[:, 0], goal_ranges[:, 1]
    # for rel_vec in rel_vecs:
    #     if (
    #         se2_lower[0] < rel_vec[0] < se2_upper[0]
    #         and se2_lower[1] < rel_vec[1] < se2_upper[1]
    #         and se2_lower[2] < rel_vec[2] < se2_upper[2]
    #     ):
    #         s_count += 1
    # print("In region: ", s_count / n_samples)
    # # print(plan_states)
    # # print(model.predict(controls))
    # print("----------")

    # beliefs = []
    # for state in plan_states:
    #     T = to_se2_transform(state[:3])
    #     Sigma = SE2BeliefOptimizationObjective.vec_to_cov(state[3:])
    #     # Sigma = np.diag([0.001, 0.003, 0.001])
    #     beliefs.append([T, Sigma])
    # goal = np.array([0, -0.7, 0])
    # goal_size = np.array([0.04, 0.04, np.pi / 4])
    # obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    # plot_belief_states(beliefs, obstacles, obj_shape, goal, goal_size)
