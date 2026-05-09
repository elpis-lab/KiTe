import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from geometry.pose import flat_to_matrix, matrix_to_flat
from geometry.random_push import generate_path_from_params
from geometry.object_model import get_obj_shape
from real_world.physical_robot import PhysicalUR10

from utils import DataLoader, set_seed, parse_args, get_names
from experiments.collect_push_data import project_se3_pose
from experiments.collect_push_data_real import get_object_pose
from experiments.collect_push_data_real import two_step_detect, execute_push

from planning.push import visualize_push_env, SE2PushPlanner, generate_push_env
from experiments.planning_push import run_planning
from experiments.train_push_model import load_model


def main(
    obj_name,
    model_type,
    n_data,
    algorithm,
    belief,
    terminal_weight,
    active_sampling,
    predefined_controls,
    detection_wait,
    push_offset,
    planning_time,
):
    """Execute in real world"""
    # Initialization
    robot = PhysicalUR10()
    execution_dt = 0.008  # UR10 default dt
    tool_offset = np.array([0, 0, -push_offset, 1, 0, 0, 0])
    # Object
    model_name, data_name, rep_data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    # Robot
    robot_home = np.array([1.3, -1.571, 1.2, -1.2, -1.571, -0.271])
    robot.move_joint(robot_home)

    # First object pose detection
    rough_detect_pose = np.array([-0.05, -0.65, 0.6, 0, np.pi, 0])
    obj_pose, _ = two_step_detect(robot, rough_detect_pose, height=0.12)
    input("Center of The Object?")
    init_state = project_se3_pose(matrix_to_flat(obj_pose))

    # Planning
    env = generate_push_env()
    env["start"] = init_state
    envs = [env]
    planning_times = [planning_time]
    n_reps = 1
    use_var = True
    all_states, all_controls, all_costs = run_planning(
        obj_name,
        model_type,
        use_var,
        n_data,
        algorithm,
        active_sampling,
        terminal_weight,
        predefined_controls,
        envs,
        planning_times,
        n_reps,
    )
    states = np.array(all_states[0][0][-1], dtype=float)
    controls = np.array(all_controls[0][0][-1], dtype=float)
    costs = np.array(all_costs[0][0][-1], dtype=float)

    # # Load path
    # kite_path = np.load(
    #     f"results/planning_push/{obj_name}_mlp_1_1000_aorrt_w2_random_20.0_plan_states.npy",
    #     allow_pickle=True,
    # )
    # kite_controls = np.load(
    #     f"results/planning_push/{obj_name}_mlp_1_1000_aorrt_w2_random_20.0_controls.npy",
    #     allow_pickle=True,
    # )
    # states = np.array(kite_path[4][14][-1], dtype=float)
    # controls = np.array(kite_controls[4][14][-1], dtype=float)
    # ap_path = np.load(
    #     f"results/planning_push/{obj_name}_mlp_0_1000_aorrt_l2_active_0.0_plan_states.npy",
    #     allow_pickle=True,
    # )
    # ap_controls = np.load(
    #     f"results/planning_push/{obj_name}_mlp_0_1000_aorrt_l2_active_0.0_controls.npy",
    #     allow_pickle=True,
    # )
    # states = np.array(ap_path[3][14][-1], dtype=float)
    # controls = np.array(ap_controls[3][14][-1], dtype=float)

    # Visualize the plan
    visualize_push_env(env, path=states, obj_shape=obj_shape)
    plt.show()

    # Execute the plan (Open-loop)
    input("Execute the plan?")
    exec_states = np.zeros((len(controls) + 1, 3))
    exec_states[0] = init_state.copy()
    for i, control in enumerate(controls):
        # Collect one interaction data
        t_paths, ws_paths = generate_path_from_params(
            matrix_to_flat(obj_pose)[None, :],
            obj_shape,
            control[None, :],
            tool_offset=tool_offset,
            pre_push_offset=0.03,
            dt=execution_dt,
        )
        execute_push(robot, ws_paths[0], control)

        # Calculate the relative pose
        time.sleep(detection_wait)
        new_obj_pose, _ = get_object_pose(
            robot, 5, rough_detect_pose, debug_img_id=i
        )
        # delta_pose = np.linalg.inv(obj_pose) @ new_obj_pose
        # se2_delta = project_se3_pose(matrix_to_flat(delta_pose))
        obj_pose = new_obj_pose.copy()

        current_state = project_se3_pose(matrix_to_flat(obj_pose))
        exec_states[i + 1] = current_state.copy()

    return exec_states


if __name__ == "__main__":
    # obj_names = [
    #     "real_cracker_box_flipped",
    #     "real_trash_truck",
    # ]
    # model_type = "mlp"
    # # n_datas = [200, 400, 600, 800, 1000]
    # n_datas = [1000]

    # configs = [
    #     ("aorrt", "l2", "random", 0.0),  # Base
    #     ("sst", "l2", "random", 0.0),  # Base
    #     ("aorrt", "w2", "random", 0.0),  # Gaussian Belief Trees
    #     ("sst", "w2", "random", 0.0),  # Gaussian Belief Trees
    #     ("aorrt", "l2", "active", 0.0),  # Active Pusher
    #     ("sst", "l2", "active", 0.0),  # Active Pusher
    #     ("aorrt", "l2", "random", 20.0),  # KiTe
    #     ("aorrt", "w2", "random", 20.0),  # KiTe
    # ]

    args = parse_args(
        [
            # Model
            # ("obj_name", "real_trash_truck"),
            ("obj_name", "real_cracker_box_flipped"),
            ("model_type", "mlp"),  # residual
            ("n_data", "1000"),
            # Planning
            ("algorithm", "aorrt"),
            ("belief", 1, int),
            ("active_sampling", 0, int),
            ("terminal_weight", 20.0, float),
            ("predefined_controls", 1, int),
            # Execution
            ("detection_wait", 0.0, float),
            ("push_offset", 0.01, float),
        ]
    )

    # Planning time
    planning_time = 10

    main(
        args.obj_name,
        args.model_type,
        args.n_data,
        args.algorithm,
        args.belief,
        args.terminal_weight,
        args.active_sampling,
        args.predefined_controls,
        args.detection_wait,
        args.push_offset,
        planning_time,
    )
