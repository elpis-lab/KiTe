import os
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from geometry.pose import flat_to_matrix, matrix_to_flat
from geometry.random_push import (
    generate_push_params,
    generate_path_form_params,
)
from geometry.object_model import get_obj_shape
from models.physics import push_physics
from real_world.physical_robot import PhysicalUR10
from utils import parse_args
from collect_data import project_se3_pose


########## Hardware Helper Functions ##########
def object_in_bounds(obj_pos, center, bounds):
    """Check if object is within safe bounds."""
    return (
        abs(obj_pos[0] - center[0]) < bounds[0]
        and abs(obj_pos[1] - center[1]) < bounds[1]
    )


def vision_check(robot: PhysicalUR10, rough_detect_pose, height=0.12):
    """Check if the vision system detects the object well"""
    adjust_joint_limit(robot)

    # Go to rough detection pose to check the object first
    robot.move_tool(rough_detect_pose)
    obj_pose, bounding_box = get_object_pose(robot)
    robot.move_tool(list(obj_pose[:2, 3]) + [0.35, 0.0, np.pi, 0.0])

    # Approach and check the object again
    obj_pose, bounding_box = get_object_pose(robot)
    robot.move_tool(list(obj_pose[:2, 3]) + [height, 0.0, np.pi, 0.0])

    return obj_pose, bounding_box


def get_object_pose(robot: PhysicalUR10):
    """Detect the object and check the pose"""
    pose = robot.get_object_pose_hand()
    obj_pose = pose["pose"]
    bounding_box = pose["bounding_box"]
    # Debug
    img = pose["result_image"]
    img.save(f"debug/result_{time.time()}.jpg")

    # Convert to robot base frame
    tcp_base_pose = robot.get_ee_transform()
    tcp_ee_pose = np.eye(4)
    tcp_ee_pose[2, 3] = 0.260
    obj_pose = tcp_base_pose @ np.linalg.inv(tcp_ee_pose) @ obj_pose

    # Project to SE2
    obj_se2 = project_se3_pose(matrix_to_flat(obj_pose))
    # back to 3d pose
    obj_pose = to_se3_matrix(obj_se2)

    return obj_pose, bounding_box


def reset_object(robot: PhysicalUR10, obj_pos, center):
    """Reset the object to the safe position."""
    # Get the direction vector
    dir = center - obj_pos
    dist = np.linalg.norm(dir)
    dir_unit = dir / dist
    angle = np.arctan2(dir[1], dir[0])

    # Pointing downwards, facing pushing direction
    r = R.from_euler("xyz", [-np.pi, 0, angle - np.pi / 2], degrees=False)
    r = r.as_rotvec()

    # Get the position of the pre-push and post-push
    pre_push_dist = 0.15
    pre_push_coord = obj_pos - dir_unit * pre_push_dist
    post_push_coord = obj_pos + dir_unit * dist
    x_d, y_d = pre_push_coord
    x_f, y_f = post_push_coord

    # Move to the recovery push
    print(f"\n(RESET) | Trying to push the object to {x_f:.5f}, {y_f:.5f}")
    adjust_joint_limit(robot)
    robot.move_tool([x_d, y_d, 0.10, *r], speed=1.0, acceleration=2.0)
    robot.move_tool([x_d, y_d, 0.03, *r])
    robot.move_tool([x_f, y_f, 0.03, *r])
    robot.move_tool([x_f, y_f, 0.35, *r], speed=1.0, acceleration=2.0)


def adjust_joint_limit(robot: PhysicalUR10):
    """Adjust the last joint to be within the limit"""
    joint_val = robot.get_q_values()
    if joint_val[5] >= np.pi or joint_val[5] <= -np.pi:
        last_joint_reset = np.array([*joint_val[:5], 0.0])
        robot.move_joint(last_joint_reset, speed=3.0, acceleration=2.0)


########## Other Helper Functions ##########
def to_se3_matrix(se2_pose):
    """Convert SE2 pose to SE3 matrix"""
    se2_pose = np.array(se2_pose)
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler("z", se2_pose[2], degrees=False).as_matrix()
    matrix[:2, 3] = se2_pose[:2]
    return matrix


def get_valid_push_param(
    push_param_candidates, obj_pose, obj_shape, center, bounds
):
    """Generate a valid push param or get it from the candidates"""
    # Get a push param from the candidates
    if push_param_candidates:
        push_param = push_param_candidates.pop(0)
    else:
        push_param = generate_push_params(1)[0]

    # Check if the push param will push the object out of bounds
    while True:
        se2_delta = push_physics(
            push_param[None, :],
            obj_size=obj_shape[:2],
        )[0]
        delta_pose = to_se3_matrix(se2_delta)
        new_pose = obj_pose @ delta_pose
        if object_in_bounds(new_pose[:2, 3], center, bounds):
            break

        print(
            "(Bad Param) Object may be pushed out of bounds. Try another one."
        )
        push_param_candidates.append(push_param)  # save it for later
        push_param = generate_push_params(1)[0]

    return push_param


def organize_data(obj_name, n_reps, n_data):
    """Re-organize the data"""
    obj_name = f"real_{obj_name}_{n_reps}x{n_data}"
    data_x = np.load("x_" + obj_name + ".npy")
    data_y = np.load("y_" + obj_name + ".npy")

    # Clean unnecessary data_x
    # just to confirm the data is in the correct format
    for i in range(1, n_reps):
        assert np.allclose(
            data_x[i * n_data : (i + 1) * n_data],
            data_x[:n_data],
        )
    data_x = data_x[:n_data]

    # Re-organize data_y
    assert data_y.shape[0] == n_data * n_reps
    data_y = data_y.reshape(n_reps, n_data, -1).transpose(1, 0, 2)

    # Save the re-organized data
    np.save("x_" + obj_name + ".npy", data_x)
    np.save("y_" + obj_name + ".npy", data_y)


########## Main Function ##########
def execute_push(robot: PhysicalUR10, ws_path, push_param):
    """Collect one interaction data"""
    # Adjust the camera angle so that the camera is facing the direction
    for i in range(len(ws_path)):
        r1 = R.from_quat(ws_path[i][[4, 5, 6, 3]])
        r2 = R.from_euler("z", np.pi / 2, degrees=False)
        r = r1 * r2
        ws_path[i] = np.array([*ws_path[i][:3], *r.as_quat()[[3, 0, 1, 2]]])

    # Pre-push Traj and Post-push Traj
    pre_push_pose_flat = ws_path[0].copy()
    pre_push_pose_flat[2] += 0.1
    post_push_pose_flat = ws_path[-1].copy()
    post_push_pose_flat[2] = 0.35

    # Current -> pre-push -> push -> post-push -> Camera position
    adjust_joint_limit(robot)
    robot.move_tool(
        pre_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0
    )
    robot.move_tool(ws_path[0], to_rotvec=True)
    robot.execute_ee_waypoints(ws_path, to_rotvec=True)
    robot.move_tool(
        post_push_pose_flat, to_rotvec=True, speed=1.0, acceleration=2.0
    )


def main(obj_name, n_data, n_reps=0, detection_wait=0.0):
    """Collect real data"""
    robot = PhysicalUR10()
    execution_dt = 0.008  # UR10 default dt
    tool_offset = np.array([0, 0, -0.02, 1, 0, 0, 0])

    # Object
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")
    # Robot
    robot_home = np.array([1.3, -1.571, 1.2, -1.2, -1.571, -0.271])
    robot.move_joint(robot_home)

    # Safety bounds for object reset
    center = np.array([0.0, -0.7])
    bounds = np.array([0.3, 0.2])

    # Vision system check
    rough_detect_pose = np.array([*center, 0.6, 0, np.pi, 0])
    obj_pose, _ = vision_check(robot, rough_detect_pose, height=0.15)
    input("Center of The Object?")

    # Data collection setup, always load the existing data
    # This is to prevent something wrong happends
    # in the middle of the real data collection
    if n_reps == 1:
        obj_name = f"real_{obj_name}_{n_data}"
    else:
        obj_name = f"real_{obj_name}_{n_reps}x{n_data}"

    if not os.path.exists("x_" + obj_name + ".npy"):
        data_x = []
        data_y = []
    else:
        data_x = np.load("x_" + obj_name + ".npy").tolist()[:]
        data_y = np.load("y_" + obj_name + ".npy").tolist()[:]
    count = len(data_x)
    if count >= n_data * n_reps:
        print(f"Already have {count} data, skip the rest")
        return

    # Start collecting data (regular random buffer)
    # In this case, do not set seed, otherwise the data can be repeated
    if n_reps == 1:
        candidate_push_params = []
    # If running repetitive data collection,
    # set seed for consistent pre-sampled results
    else:
        np.random.seed(42)
        candidate_push_params = generate_push_params(n_data)

    # Always save the data after each interaction
    pbar = tqdm(total=n_data * n_reps, initial=count)
    while count < n_data * n_reps:
        # Check if the object is within the safe bounds to begin with
        while not object_in_bounds(obj_pose[:2, 3], center, bounds):
            reset_object(robot, obj_pose[:2, 3], center)
            time.sleep(detection_wait)
            obj_pose, _ = get_object_pose(robot)

        # Get one push param (regular random)
        if n_reps == 1:
            push_param = get_valid_push_param(
                candidate_push_params, obj_pose, obj_shape, center, bounds
            )
        # Get one push param from the pre-sampled candidates
        else:
            push_param = candidate_push_params[count % n_data]

        # Collect one interaction data
        t_paths, ws_paths = generate_path_form_params(
            matrix_to_flat(obj_pose)[None, :],
            obj_shape,
            push_param[None, :],
            tool_offset=tool_offset,
            pre_push_offset=0.03,
            dt=execution_dt,
        )
        execute_push(robot, ws_paths[0], push_param)

        # Calculate the relative pose
        time.sleep(detection_wait)
        new_obj_pose, _ = get_object_pose(robot)
        delta_pose = np.linalg.inv(obj_pose) @ new_obj_pose
        se2_delta = project_se3_pose(matrix_to_flat(delta_pose))
        # Save result
        data_x.append(push_param)
        data_y.append(se2_delta)
        np.save("x_" + obj_name + ".npy", np.array(data_x))
        np.save("y_" + obj_name + ".npy", np.array(data_y))

        # The current pose is the new object pose
        obj_pose = new_obj_pose.copy()
        count += 1
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    args = parse_args(
        [
            ("obj_name", "cracker_box_flipped"),
            ("num_data", 1000, int),
            ("n_reps", 1, int),
            ("detection_wait", 0.0, float),
        ]
    )

    main(args.obj_name, args.num_data, args.n_reps, args.detection_wait)

    # # Run this in the end of repetitive data collection
    # organize_data(args.obj_name, args.n_reps, args.num_data)
