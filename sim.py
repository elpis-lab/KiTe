import numpy as np

import mujoco
import mujoco.viewer
from mujoco import mjx
import jax

from mjx_utils import mjx_sim_reset, get_mjx_sim_step_n
from mjx_utils import mjx_get_qpos_data_into, mjx_get_qpos, render_mp4
from geometry.pose import Pose


class Sim:
    def __init__(
        self,
        xml_path,
        n_envs,
        robot_joint_dof,
        robot_ee_dof,
        dt=0.1,
        visualize=True,
        jax_cache="jax_cache",
    ):
        """Mujoco Simulation Environment

        The simulation environment should have one robot and multiple objects.
        qpos is consisting of robot joint positions [:robot_dof]
        and object poses [robot_dof:].
        """
        # Initialize Mujoco
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = None
        if visualize:
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data
            )
            self.viewer.sync()
        # Move Mujoco to MJX
        # JAX persistent compilation cache
        jax.config.update("jax_compilation_cache_dir", "/tmp/" + jax_cache)
        # Move Mujoco to GPU
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model, self.mj_data)

        # Simulation parameters
        # qpos idx
        self.robot_joint_dof = robot_joint_dof
        self.robot_ee_dof = robot_ee_dof
        self.robot_joint_idx = np.arange(robot_joint_dof)
        self.robot_ee_idx = np.arange(
            robot_joint_dof, robot_joint_dof + robot_ee_dof
        )
        obj_idx = np.arange(robot_joint_dof + robot_ee_dof, self.mj_model.nq)
        self.obj_idxs = obj_idx.reshape(-1, 7)
        # time
        self.dt = dt
        time_step = self.mj_model.opt.timestep

        # Prepare parallelization
        # get Jit functions
        self.n_envs = n_envs
        self.rng_key = jax.random.PRNGKey(0)
        self.rng_key = jax.random.split(self.rng_key, self.n_envs)
        mjx_sim_step_n = get_mjx_sim_step_n(int(self.dt // time_step))
        self.jit_reset = jax.jit(
            jax.vmap(mjx_sim_reset, in_axes=(None, 0, 0, 0, 0))
        )
        self.jit_step = jax.jit(jax.vmap(mjx_sim_step_n, in_axes=(None, 0, 0)))
        # batch the data
        self.mjx_data = jax.vmap(lambda rng: self.mjx_data.replace())(
            self.rng_key
        )

        # Store the initial state for reset
        self.init_qpos = mjx_get_qpos(self.mjx_data)

    def get_sim_info(self):
        """Return simulation infomation"""
        return (self.n_envs, self.dt)

    def run_sim(self, duration):
        """Run the simulation for a given duration"""
        n_steps = int(duration // self.dt)
        for _ in range(n_steps):
            self.step()

    def step(self, ctrl=None):
        """Step the simulation"""
        self.mjx_data = self.jit_step(self.mjx_model, self.mjx_data, ctrl)
        if self.viewer:
            self.vis_sync()

    def reset(self, wait_time=0.5):
        """Reset the simulation to the initial state"""
        # Reset the simulation
        self.mjx_data = self.jit_reset(
            self.mjx_model,
            self.mjx_data,
            self.init_qpos,
            None,
            # assume robot joint is position control
            self.init_qpos[:, : self.robot_joint_dof + self.robot_ee_dof],
        )
        # Wait for the simulation to stabilize
        self.run_sim(wait_time)

    def close(self):
        """Close the simulation"""
        if self.viewer:
            self.viewer.close()

    ########## Object-related functions ##########
    def set_obj_init_poses(self, init_pose, obj_idx=0, env_idx=None):
        """Set the initial poses of the objects"""
        init_pose, env_idx = self._preprocess_values(init_pose, env_idx)
        self.init_qpos[np.ix_(env_idx, self.obj_idxs[obj_idx])] = init_pose

    def get_obj_pose(self, obj_idx=0, env_idx=None):
        """Return object information"""
        _, env_idx = self._preprocess_values([0], env_idx)
        return np.array(
            self.mjx_data.qpos[np.ix_(env_idx, self.obj_idxs[obj_idx])]
        )

    ########## Robot-related functions ##########
    def set_robot_init_joints(self, joints, ee_joints=None, env_idx=None):
        """Set the initial joint positions"""
        joints, env_idx = self._preprocess_values(joints, env_idx)
        self.init_qpos[np.ix_(env_idx, self.robot_joint_idx)] = joints
        if ee_joints is not None:
            ee_joints, env_idx = self._preprocess_values(ee_joints, env_idx)
            self.init_qpos[np.ix_(env_idx, self.robot_ee_idx)] = ee_joints

    def get_robot_joints(self, env_idx=None):
        """Get the robot joint positions"""
        _, env_idx = self._preprocess_values([0], env_idx)
        return np.array(
            self.mjx_data.qpos[np.ix_(env_idx, self.robot_joint_idx)]
        )

    def get_robot_ee(self, env_idx=None):
        """Get the robot end-effector positions"""
        _, env_idx = self._preprocess_values([0], env_idx)
        return np.array(self.mjx_data.qpos[np.ix_(env_idx, self.robot_ee_idx)])

    def move_ee(self, ee, env_idx=None, wait_time=0.0):
        """Set the robot end-effector positions"""
        self._move_robot(self.robot_ee_idx, ee, env_idx, wait_time)

    def move_joints(self, joints, env_idx=None, wait_time=0.0):
        """Set the robot joint positions"""
        self._move_robot(self.robot_joint_idx, joints, env_idx, wait_time)

    def set_ee(self, ee, env_idx=None):
        """Set the robot end-effector positions"""
        self._set_robot(self.robot_ee_idx, ee, env_idx)

    def set_joints(self, joints, env_idx=None):
        """Set the robot joint positions"""
        self._set_robot(self.robot_joint_idx, joints, env_idx)

    def _move_robot(self, joint_idxs, values, env_idx=None, wait_time=0.0):
        """Move the robot joint positions"""
        values, env_idx = self._preprocess_values(values, env_idx)
        # Set the control
        ctrl = np.array(self.mjx_data.ctrl)
        ctrl[np.ix_(env_idx, joint_idxs)] = values
        self.mjx_data = self.mjx_data.replace(ctrl=ctrl)
        self.run_sim(wait_time)

    def _set_robot(self, joint_idxs, values, env_idx=None):
        """Set the robot joint positions"""
        values, env_idx = self._preprocess_values(values, env_idx)
        # Set the control
        self._move_robot(joint_idxs, values, env_idx, 0.0)
        # Set the qpos
        qpos = np.array(self.mjx_data.qpos)
        qpos[np.ix_(env_idx, joint_idxs)] = values
        self.mjx_data = self.mjx_data.replace(qpos=qpos)

    def execute_waypoints(
        self, waypoints, wait_time=0.0, return_intermediate=False
    ):
        """
        Run the push simulation with the given waypoints.
        Args:
            waypoints: the target joint positions at each time step
                       defined as (num_time_step, num_envs, num_joints)
        """
        n_run_steps, n_trials, n_joint = waypoints.shape
        assert n_joint == self.robot_joint_dof, "Invalid joint dimension"
        assert n_trials <= self.n_envs, (
            "required number of execution should not be larger"
            + "than the number of simulation environment"
        )
        env_idx = np.arange(n_trials)

        # Save the initial qpos first
        init_qpos = mjx_get_qpos(self.mjx_data)[:n_trials]
        if return_intermediate:
            intermediate_qpos = np.zeros(
                (n_run_steps + 1, n_trials, init_qpos.shape[1])
            )
            intermediate_qpos[0] = init_qpos

        # Start execution
        # init the joint position
        self.set_joints(waypoints[0], env_idx)

        # Run the sim with the computed trajectory
        for k in range(n_run_steps):
            self.move_joints(waypoints[k], env_idx)
            self.step()
            if return_intermediate:
                qpos = mjx_get_qpos(self.mjx_data)[:n_trials]
                intermediate_qpos[k + 1] = qpos
        # Run for extra time to stabilize the simulation
        self.run_sim(wait_time)

        # Get the last qpos
        last_qpos = mjx_get_qpos(self.mjx_data)[:n_trials]

        # Compute the relative poses
        init_obj_qpos = init_qpos[:, self.obj_idxs.flatten()]
        last_obj_qpos = last_qpos[:, self.obj_idxs.flatten()]
        relative_qpos = self._get_relative_qpos(init_obj_qpos, last_obj_qpos)

        if return_intermediate:
            return relative_qpos, intermediate_qpos
        else:
            return relative_qpos

    ########## Helper functions ##########
    def _get_relative_qpos(self, init_qpos, last_qpos):
        """Take the inital and last qposand compute the relative 6D poses"""
        relative_qpos = np.zeros_like(init_qpos)
        for i in range(relative_qpos.shape[0]):
            for j in range(int(relative_qpos.shape[1] // 7)):
                qpos1 = init_qpos[i, 7 * j : 7 * (j + 1)]
                pose1 = Pose(qpos1[:3], qpos1[3:])
                qpos2 = last_qpos[i, 7 * j : 7 * (j + 1)]
                pose2 = Pose(qpos2[:3], qpos2[3:])
                qpos = (pose1.invert @ pose2).flat
                relative_qpos[i, 7 * j : 7 * (j + 1)] = qpos
        return relative_qpos

    def _preprocess_values(self, values, env_idx):
        """Preprocess the values and env_idx to match"""
        # Preprocess env_idx first
        # if not provided, use all environments
        if env_idx is None:
            env_idx = np.arange(self.n_envs)
        # if a single environment, convert to array
        if isinstance(env_idx, int):
            env_idx = np.array([env_idx])
        env_idx = np.array(env_idx)

        # Preprocess values
        # if values is 1D, expand it to n_envs
        values = np.array(values)
        if values.ndim == 1:
            values = np.tile(values, (len(env_idx), 1))
        else:
            assert len(values) == len(
                env_idx
            ), "Values need to be 1D or have the same length as env_idx"

        return values, env_idx

    def vis_sync(self, env_idx=0):
        """Sync the simulation state to the viewer"""
        mjx_get_qpos_data_into(
            self.mj_data, self.mj_model, self.mjx_data, env_id=env_idx
        )
        self.viewer.sync()

    def render_state(self, state, filename):
        """Render the state qpos (n_frame, nq) into a mp4 video"""
        render_mp4(self.mj_model, self.mj_data, state, self.dt, filename)


def test(sim: Sim):
    # Testing
    sim.set_robot_init_joints(
        np.array([np.pi / 2, -1.7, 2, -1.87, -np.pi / 2, np.pi])
    )
    sim.set_obj_init_poses(np.array([0, -0.7, 0, 1, 0, 0, 0]), 0)
    sim.reset()

    # Test control
    ctrl = np.array([-1.5, -1.5, 1.5, -1.5, -1.5, 0])
    sim.move_joints(ctrl, wait_time=1.0)

    # Test waypoints
    waypoints = np.linspace(ctrl - 0.5, ctrl + 0.5, 30)
    waypoints = waypoints[:, None, :].repeat(sim.n_envs, axis=1)
    relative_qpos, intermediate_qpos = sim.execute_waypoints(
        waypoints, wait_time=1.0, return_intermediate=True
    )
    print(relative_qpos[0], relative_qpos.shape)
    print(intermediate_qpos[-1, 0], intermediate_qpos.shape)

    # Test getters
    print(sim.get_sim_info())
    print(sim.get_robot_joints()[0], sim.get_robot_joints().shape)
    print(sim.get_obj_pose()[0], sim.get_obj_pose().shape)

    sim.close()


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    sim = Sim(
        "mjx_sim.xml",
        n_envs=100,
        robot_joint_dof=6,
        robot_ee_dof=0,
        dt=0.01,
        visualize=True,
    )
    test(sim)
