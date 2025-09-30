import numpy as np

from geometry.pose import matrix_to_quat, flat_to_matrix
from geometry.pose import euler_to_quat, quat_to_matrix


def get_random_push(
    n_params: int,
    obj_states: np.ndarray,
    obj_pcd: np.ndarray,
    obj_normals: np.ndarray,
    tool_offset: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 0]),
    distance_range: tuple[float, float] = (0, 0.3),  # push distance
    pre_push_offset: float = 0.02,  # pre-push offset at the beginning
    duration: float = 2,  # total time to complete the push
    dt: float = 0.1,  # time step of the path
    max_speed: float = 0.5,  # assume it will never exceed this speed
    max_acc: float = 1,  # assume it will never exceed this acceleration
):
    """Get a random push parameter and the corresponding path"""
    push_params = generate_push_params(
        n_params, obj_pcd, obj_normals, distance_range
    )
    times, ws_paths = generate_path_from_params(
        obj_states,
        push_params,
        tool_offset,
        pre_push_offset,
        duration,
        dt,
        max_speed,
        max_acc,
    )
    return push_params, times, ws_paths


def generate_push_params(
    n_params: int,
    # Object point cloud and normals
    obj_pcd: np.ndarray,
    obj_normals: np.ndarray,
    # Push distance
    distance_range: tuple[float, float] = (0, 0.3),
):
    """Generate a random push parameter"""
    assert (
        obj_pcd.shape[0] == obj_normals.shape[0]
    ), "Points and normals mismatch"
    n_points = obj_pcd.shape[0]

    # Choose n_params random points and normals
    idx = np.random.choice(n_points, size=n_params, replace=True)
    points = obj_pcd[idx]
    normals = obj_normals[idx]
    # Get n_params random distances
    distances = np.random.uniform(*distance_range, size=(n_params, 1))

    # Stack all parameters (n_params, 5)
    push_params = np.hstack([points, normals, distances])
    return push_params


def generate_path_from_params(
    obj_states: np.ndarray,
    push_params: np.ndarray,  # (N, 5): x, y, nx, ny, dist
    tool_offset: np.ndarray = np.array([0, 0, 0, 1, 0, 0, 0]),
    pre_push_offset: float = 0.02,
    duration: float = 2,
    dt: float = 0.1,
    max_speed: float = 0.5,
    max_acc: float = 1,
):
    """Generate a workspace path from the push parameters"""
    # Unpack parameters
    assert push_params.ndim == 2 and push_params.shape[1] == 5
    assert obj_states.shape[0] == push_params.shape[0]
    n_data = push_params.shape[0]
    points = push_params[:, :2]
    normals = push_params[:, 2:4]
    distances = push_params[:, 4]

    # Get local path (x, y) w.r.t. the object
    # direction vectors (N, 2)
    dir_vecs = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    # add pre-push offset to avoid hitting object at the beginning
    distances = distances + pre_push_offset  # (N, 1)
    # start point (N, 2)
    starts = points + pre_push_offset * dir_vecs

    # Check constraints before generating path
    peak_speed, peak_acc = get_sin_velocity_profile_peak(distances, duration)
    assert np.all(peak_speed <= max_speed) and np.all(peak_acc <= max_acc)
    # Generate the waypoint distance from start
    n_steps = int(duration / dt)
    t_paths = np.tile(np.linspace(0, duration, n_steps), (n_data, 1))  # (N, T)
    distances = distances[:, None]  # (N, 1)
    dists = sin_velocity_profile(t_paths, distances, duration)  # (N, T)

    # Generate path in local object frame
    # Position part
    # local = start - dist * dir_vec, (N, T, 2)
    local_xy = starts[:, None, :] - dists[:, :, None] * dir_vecs[:, None, :]
    # combine height (N, T, 3)
    local_z = np.zeros((n_data, n_steps, 1))
    local_pos = np.concatenate([local_xy, local_z], axis=2)

    # Rotation part
    # Batch operation for the following operations
    # for:
    #     pose_delta = rotate_z @ reflect_z @ tool_offset
    #     global_pose = obj_pose @ Pose(local_pos[i]) @ pose_delta

    # Transform matrix will be size (n_data, n_steps, 4, 4)
    # add reflection and offset - ensure ee pointing down
    t_rotate_z = np.tile(np.eye(4)[None, None, :, :], (n_data, 1, 1, 1))
    rotations = np.arctan2(dir_vecs[:, 1], dir_vecs[:, 0])
    t_rotate_z[:, 0, :3, :3] = euler_to_matrix("z", rotations + np.pi)
    t_reflect_z = np.eye(4)[None, None, :, :]
    t_reflect_z[0, 0, :3, :3] = euler_to_matrix("x", np.pi)
    t_tool_offset = flat_to_matrix(tool_offset)[None, None, :, :]
    t_delta = t_rotate_z @ t_reflect_z @ t_tool_offset

    # Combine as workspace path
    t_local_pos = np.tile(np.eye(4)[None, None, :, :], (n_data, n_steps, 1, 1))
    t_local_pos[:, :, :3, 3] = local_pos
    # get local poses
    t_local = t_local_pos @ t_delta
    # convert to world frame given the object world pose
    t_obj = flat_to_matrix(obj_states)
    t_global = t_obj[:, None, :, :] @ t_local

    # Flatten to 7D
    ws_pos = t_global[:, :, :3, 3]
    ws_quat = matrix_to_quat(t_global[:, :, :3, :3].reshape(-1, 3, 3))
    ws_quat = ws_quat.reshape(n_data, n_steps, 4)
    ws_paths = np.concatenate([ws_pos, ws_quat], axis=-1)
    return t_paths, ws_paths


########## Sinusoidal functions ##########
def get_sin_velocity_profile_peak(dist, duration):
    """Get the peak speed and acceleration for a sinusoidal velocity profile"""
    # v_max = 2 * dist / duration
    # a_max = v_max * np.pi / duration
    peak_speed = 2 * dist / duration
    peak_acc = peak_speed * np.pi / duration
    return peak_speed, peak_acc


def sin_velocity_profile(t, dist, duration):
    """Get the velocity at time t for a given push distance d and duration."""
    v_max, acc_max = get_sin_velocity_profile_peak(dist, duration)  # (N, 1)
    # Compute the velocity at time t for each sample given a sin velocity
    # v = (v_max / 2) * (np.sin(2 * np.pi * t / duration - np.pi / 2) + 1)
    # a = acc_max * np.sin(2 * np.pi * t / duration)
    scale = -v_max * duration / 4 / np.pi  # (N,)
    # get the position d at given time t (N, T)
    d = scale * np.sin(2 * np.pi * t / duration) + (v_max * t / 2)
    return d


########## Geometry functions ##########
def euler_to_matrix(seq: str, euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to rotation matrix"""
    quat = euler_to_quat(euler, seq)
    matrix = quat_to_matrix(quat)
    return matrix
