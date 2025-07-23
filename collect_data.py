import numpy as np
from tqdm import tqdm
from geometry.pose import Pose, euler_to_quat
from ik import IK
from geometry.random_push import (
    get_random_push,
    generate_push_params,
    generate_path_form_params,
)
from geometry.model_object import get_obj_shape

from sim_network import SimClient


def get_random_init_states(
    n_envs, z, pos_range=(-0.2, 0.2, -0.5, -0.9), euler_range=(-np.pi, np.pi)
):
    """
    Get random initial states for n_envs
    Also return eulers for rotate_scene
    """
    pos_x = np.random.uniform(pos_range[0], pos_range[1], (n_envs, 1))
    pos_y = np.random.uniform(pos_range[2], pos_range[3], (n_envs, 1))
    pos_z = z * np.ones((n_envs, 1))
    pos = np.concatenate([pos_x, pos_y, pos_z], axis=-1)

    euler = np.random.uniform(euler_range[0], euler_range[1], n_envs)
    quat = euler_to_quat([(0, 0, e) for e in euler])

    return np.concatenate([pos, quat], axis=-1)


def collect_data(obj_name, n_data, random_init=True):
    """Collect n_data for obj_name"""
    # Need to run sim_network.py first
    # Sim class
    client = SimClient()
    # IK solver - Expansion GRR
    ik = IK("ur10_rod")
    tool_offset = Pose([0, 0, -0.02])

    # Initial state parameters
    n_envs, dt = client.execute("get_sim_info")
    client.execute(
        "set_robot_init_joints",
        [[np.pi / 2, -1.7, 2, -1.87, -np.pi / 2, np.pi]],
    )
    # assume all objects start the same
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured_vhacd.stl")

    # Data container
    assert n_data > 0 and n_data % n_envs == 0
    n_rounds = int(n_data // n_envs)
    data_x = np.zeros((n_data, 3))
    data_y = np.zeros((n_data, 3))

    # Start collecting
    for i in tqdm(range(n_rounds)):
        params = []
        pos_waypoints = []

        # Randomized initial states
        if random_init:
            init_states = get_random_init_states(n_envs, obj_shape[2] / 2)
        else:
            init_states = np.tile(
                [0, -0.7, obj_shape[2] / 2, 1, 0, 0, 0], (n_envs, 1)
            )
        client.execute("set_obj_init_poses", [init_states, 0])
        client.execute("reset")

        # Compute push trajectories
        for j in range(n_envs):
            push_params, times, ws_path = get_random_push(
                Pose(init_states[j][:3], init_states[j][3:]),
                obj_shape,
                tool_offset,
            )
            traj = ik.ws_path_to_traj(Pose(), times, ws_path)
            waypoints = traj.to_step_waypoints(dt)

            params.append(push_params)
            pos_waypoints.append(waypoints)

        # Stack (num_time_step, trials, robot_dof) and send it to Sim to execute
        params = np.array(params)
        pos_waypoints = np.stack(pos_waypoints, axis=1)
        pose = client.execute("execute_waypoints", pos_waypoints)

        # TODO
        # Convert pose to 3D SE2 pose
        # Save data
        data_x[i * n_envs : (i + 1) * n_envs] = params
        data_y[i * n_envs : (i + 1) * n_envs] = pose

    np.save(f"data/x_{obj_name}.npy", data_x)
    np.save(f"data/y_{obj_name}.npy", data_y)
    client.close()


def collect_repetitive_data(obj_name, n_data, n_reps, random_init=True):
    """Collect n_reps * n_data for obj_name"""
    # Need to run sim_network.py first
    # Sim class
    client = SimClient()
    # IK solver - Expansion GRR
    ik = IK("ur10_rod")
    tool_offset = Pose([0, 0, -0.02])

    # Initial state parameters
    n_envs, dt = client.execute("get_sim_info")
    client.execute(
        "set_robot_init_joints",
        [[np.pi / 2, -1.7, 2, -1.87, -np.pi / 2, np.pi]],
    )
    # assume all objects start the same
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured_vhacd.stl")

    # Data container
    assert n_data > 0 and n_data == n_envs
    data_x = np.zeros((n_data, 3))
    data_y = np.zeros((n_reps, n_data, 3))

    # Generate random push waypoints to repeat
    params = []
    for j in range(n_data):
        push_params = generate_push_params(obj_shape)
        params.append(push_params)
    params = np.array(params)
    data_x = params

    # Start collecting
    for i in tqdm(range(n_reps)):
        pos_waypoints = []

        # Randomized initial states
        if random_init:
            init_states = get_random_init_states(n_envs, obj_shape[2] / 2)
        else:
            init_states = np.tile(
                [0, -0.7, obj_shape[2] / 2, 1, 0, 0, 0], (n_envs, 1)
            )
        client.execute("set_obj_init_poses", [init_states, 0])
        client.execute("reset")

        # Compute push trajectories
        for j in range(n_data):
            times, ws_path = generate_path_form_params(
                Pose(init_states[j][:3], init_states[j][3:]),
                obj_shape,
                params[j],
                tool_offset,
            )
            traj = ik.ws_path_to_traj(Pose(), times, ws_path)
            waypoints = traj.to_step_waypoints(dt)
            pos_waypoints.append(waypoints)

        # Stack (num_time_step, trials, robot_dof) and send it to Sim to execute
        pos_waypoints = np.stack(pos_waypoints, axis=1)
        pose = client.execute("execute_waypoints", pos_waypoints)

        # TODO
        # Convert pose to 3D SE2 pose
        # Save data
        # data_y[i] = pose

    np.save(f"data/x_{obj_name}_rep{n_reps}.npy", data_x)
    np.save(f"data/y_{obj_name}_rep{n_reps}.npy", data_y)
    client.close()


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    np.set_printoptions(precision=4, suppress=True)

    obj_name = "cracker_box_flipped"
    # collect_data(obj_name, 10)
    collect_repetitive_data(obj_name, 10, 3)
