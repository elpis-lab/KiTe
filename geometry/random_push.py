import numpy as np
from geometry.pose import Pose, matrix_to_quat


def get_random_push(
    obj_pose: Pose,
    obj_shape: tuple[float, float, float],
    tool_offset: Pose = Pose(),
    rotation_range: tuple[float, float] = (0, 4),  # rotation to push from
    side_range: tuple[float, float] = (-0.4, 0.4),  # relative side offset
    distance_range: tuple[float, float] = (0, 0.3),  # push distance
    total_time: float = 3,  # total time to complete the push
    dt: float = 0.1,  # time step of the path
    max_speed: float = 0.5,  # assume it will never exceed this speed
    max_acc: float = 1,  # assume it will never exceed this acceleration
):
    """Get a random push parameter and the corresponding path"""
    push_params = generate_push_params(
        obj_shape,
        rotation_range=rotation_range,
        side_range=side_range,
        distance_range=distance_range,
    )

    times, ws_path = generate_path_form_params(
        obj_pose,
        obj_shape,
        push_params,
        tool_offset=tool_offset,
        total_time=total_time,
        dt=dt,
        max_speed=max_speed,
        max_acc=max_acc,
    )

    return push_params, times, ws_path


def generate_push_params(
    obj_shape: tuple[float, float, float],
    rotation_range: tuple[float, float] = (0, 4),  # rotation to push from
    side_range: tuple[float, float] = (-0.4, 0.4),  # relative side offset
    distance_range: tuple[float, float] = (0, 0.3),  # push distance
):
    """Generate a random push parameter"""
    edge = np.random.randint(*rotation_range)
    rotation = edge * np.pi / 2

    w, l, h = obj_shape
    if edge % 2 == 1:
        side_size = w
    else:
        side_size = l
    side = np.random.uniform(*side_range) * side_size

    distance = np.random.uniform(*distance_range)

    push_params = (rotation, side, distance)
    return push_params


def generate_path_form_params(
    obj_pose: Pose,
    obj_shape: tuple[float, float, float],
    push_params: tuple[float, float, float],
    tool_offset: Pose = Pose(),
    total_time: float = 3,
    dt: float = 0.1,
    max_speed: float = 0.5,
    max_acc: float = 1,
    relative_push_offset: bool = False,
):
    """Generate a workspace path from the push parameters"""
    rotation, side, distance = push_params
    # make sure rotation is multiples of pi / 2
    push_side = np.round(rotation / (np.pi / 2))
    rotation = push_side * (np.pi / 2)

    # The object is approximated as an AABB
    w, l, h = obj_shape
    if push_side % 2 == 1:
        size = l
        side = w * side if relative_push_offset else side
    else:
        size = w
        side = l * side if relative_push_offset else side

    # Get local path (x, y) w.r.t. the object
    # direction vectors
    dir_vec = np.array([np.cos(rotation), np.sin(rotation)])
    side_offset_vec = np.array([-dir_vec[1], dir_vec[0]])
    # small offset to avoid hitting object at the beginning
    pre_push_offset = 0.04
    # start point
    start = (dir_vec * (size / 2 + pre_push_offset)) + (side * side_offset_vec)
    distance += pre_push_offset

    # TODO - Move all these stuff to physics.py
    # Check constraints before generating path
    peak_speed = 2 * distance / total_time
    peak_speed = np.clip(peak_speed, 0, max_speed)
    peak_acc = peak_speed * np.pi / total_time
    peak_acc = np.clip(peak_acc, 0, max_acc)
    # Generate path
    # Sin velocity to complete this path from start to end
    # v(t) = -peak_speed / 2 * (cos(2 * pi * t / T) - 1)
    # d(t) = peak_speed * t / 2
    #      - peak_speed * total_time / (4 * pi) * sin(2 * pi * t / T)
    times = np.linspace(0, total_time, int(total_time / dt))
    scale = -peak_speed * total_time / 4 / np.pi
    dists = scale * np.sin(2 * np.pi * times / total_time) + (
        peak_speed * times / 2
    )

    # Generate path in local frame
    # local = start - dist * dir_vec
    local_xy = start[None, :] - np.outer(dists, dir_vec)
    local_z = -h / 2 * np.ones((len(times),))  # assume making z to be 0
    local_pos = np.stack([local_xy[:, 0], local_xy[:, 1], local_z], axis=1)

    # Add rotation and offset - ensure ee pointing down
    reflect_z = Pose([0, 0, 0], [np.pi, 0, 0])
    rotate_z = Pose([0, 0, 0], [0, 0, rotation])
    pose_delta = rotate_z @ reflect_z @ tool_offset

    # To speed up, use transform matrix directly instead of Pose computation
    # for i in range(len(times)):
    #     global_pose = obj_pose @ Pose(local_pos[i]) @ pose_delta
    t_delta = pose_delta.matrix
    t_obj = obj_pose.matrix
    t_local_pos = np.tile(np.eye(4)[None], (len(times), 1, 1))
    t_local_pos[:, :3, 3] = local_pos
    # Get local poses
    t_local = t_local_pos @ t_delta[None, :, :]
    # Convert to world frame given the object world pose
    t_global = t_obj[None, :, :] @ t_local

    ws_pos = t_global[:, :3, 3]
    ws_quat = matrix_to_quat(t_global[:, :3, :3])
    ws_path = np.concatenate([ws_pos, ws_quat], axis=1)
    return times, ws_path
