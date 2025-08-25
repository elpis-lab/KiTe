import time
import numpy as np
import matplotlib.pyplot as plt

from real_world.rtde import RTDE
from real_world.gripper import Gripper
from real_world.camera import Camera
from real_world.camera_6d import Camera6D

from scipy.spatial.transform import Rotation as R


class PhysicalUR10:
    def __init__(self):
        """Initialize the physical UR10 robot class"""
        self.rtde = RTDE("192.168.0.100")
        self.gripper = Gripper("192.168.0.101", "8005")
        # self.top_cam = Camera("192.168.0.101", "5001")
        # self.hand_cam = Camera("192.168.0.101", "5000")
        self.top_cam = Camera6D("192.168.0.101", "5001")
        self.hand_cam = Camera6D("192.168.0.101", "5000")

    # Joint control
    def execute_trajectory(self, trajectory, d_t: float = 0.008):
        """Execute a trajectory"""
        # Get waypoints at each time step (Only position)
        waypoints = trajectory.to_step_waypoints(d_t)
        # speed_list = []

        # Execute each waypoint
        for waypoint in waypoints:
            start_t = self.rtde.rtde_c.initPeriod()
            self.rtde.servo_joint(waypoint, time=d_t)
            self.rtde.rtde_c.waitPeriod(start_t)
            # speed_list.append(self.get_ee_speed())

        # Stop servo
        self.rtde.rtde_c.servoStop()
        time.sleep(0.2)

        # # Debug: Plot the speed
        # plt.plot(speed_list)
        # plt.show()

    def execute_ee_waypoints(
        self, waypoints: list[list[float]], d_t: float = 0.008
    ):
        """Execute a trajectory"""
        # Convert waypoints to rotation vector pose
        waypoints = [
            self.quat_pose_to_rotvec_pose(waypoint) for waypoint in waypoints
        ]
        # speed_list = []

        # Execute each waypoint
        for waypoint in waypoints:
            start_t = self.rtde.rtde_c.initPeriod()
            self.rtde.servo_tool(waypoint, time=d_t)
            self.rtde.rtde_c.waitPeriod(start_t)
            # speed_list.append(self.get_ee_speed())

        # Stop servo
        self.rtde.rtde_c.servoStop()
        time.sleep(0.2)

        # # Debug: Plot the speed
        # plt.plot(speed_list)
        # plt.show()

    def move_joint(self, joint_angles: list[float]):
        """Move the robot to a joint configuration"""
        self.rtde.move_joint(joint_angles)

    def move_tool(self, tool_pose: list[float], to_rotvec: bool = True):
        """Move the robot to a tool pose"""
        if to_rotvec:
            tool_pose = self.quat_pose_to_rotvec_pose(tool_pose)
        self.rtde.move_tool(tool_pose)

    def quat_pose_to_rotvec_pose(self, quat_pose: list[float]):
        """Convert a quaternion pose to a rotation vector pose"""
        # Quaternion in wxyz format, convert to xyzw format
        quat = np.roll(quat_pose[3:], -1)  # → [x, y, z, w]
        rotvec = R.from_quat(quat).as_rotvec()
        return list(quat_pose[:3]) + list(rotvec)

    # Gripper
    def control_gripper(self, action: str):
        """Control the gripper"""
        if action == "open":
            self.gripper.open_gripper()
        elif action == "close":
            self.gripper.close_gripper()
        else:
            raise ValueError(f"Invalid action: {action}")

    # TODO: Fix this function
    # # Cameras
    # def get_object_pose(
    #     self, x_offset: float = 0.002, y_offset: float = -0.002
    # ):
    #     """Get the pose of the object"""
    #     top_pose = self.get_object_pose_top()
    #     closing_pose = (
    #         [top_pose[0]] + [top_pose[1] + 0.1] + [0.4] + [0.0, -np.pi, 0.0]
    #     )
    #     self.move_tool(closing_pose, to_rotvec=False)
    #     time.sleep(1.2)
    #     hand_x, hand_y, hand_theta = self.get_object_pose_hand()
    #     ee_pose = self.rtde.get_tool_pose()[:2]
    #     actual_pose = np.array(
    #         [
    #             ee_pose[0] - hand_x + x_offset,
    #             ee_pose[1] + hand_y + y_offset,
    #             -hand_theta,
    #         ]
    #     )
    #     return actual_pose

    def get_object_pose_top(self):
        """Get the pose of the object from the top camera"""
        return self.top_cam.get_object_pose()

    def get_object_pose_hand(self):
        """Get the pose of the object from the hand camera"""
        return self.hand_cam.get_object_pose()

    # Getters
    def get_ee_pose(self):
        """Get the pose of the robot"""
        return self.rtde.get_tool_pose()

    def get_ee_speed(self):
        """Get the speed of the robot"""
        return self.rtde.get_tool_speed()

    def get_q_values(self):
        """Get the joint values of the robot"""
        return self.rtde.get_joint_values()
