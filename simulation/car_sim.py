import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import time
import matplotlib.pyplot as plt

import mujoco
import mujoco.viewer
from concurrent.futures import ThreadPoolExecutor, wait

from geometry.pose import euler_to_quat, quat_to_euler
from geometry.car_dynamics import CAR_WHEELBASE, CAR_TRACK_WIDTH
from geometry.car_dynamics import CAR_WHEEL_RADIUS, propagate_multi_steps
from simulation.mujoco_utils import render_mp4


class Sim:
    def __init__(
        self,
        xml,
        n_envs,
        dt=0.1,
        visualize=True,
        real_time_vis=False,
        # Car parameters
        wheelbase=CAR_WHEELBASE,
        wheel_radius=CAR_WHEEL_RADIUS,
        track_width=CAR_TRACK_WIDTH,
        rear_offset=CAR_WHEELBASE / 2,
        # Actuator names
        steering_actuator="steering_pos",
        throttle_actuator="throttle_velocity",
    ):
        """
        Mujoco Simulation Environment

        This class is implemented in a parallel simulation manner.

        The simulation environment has one car.
        The car's state is [x, y, theta, v_forward]
        The car's control is [u_vel (m/s), u_phi (rad)]
        """
        # Initialize Mujoco
        if "<mujoco" in xml:
            self.mj_model = mujoco.MjModel.from_xml_string(xml)
        else:
            self.mj_model = mujoco.MjModel.from_xml_path(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = None
        if visualize:
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data
            )
            self.viewer.sync()
        self.real_time_vis = real_time_vis

        # Simulation car constants
        self.wheelbase = wheelbase
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.rear_offset = rear_offset
        self.steering_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, steering_actuator
        )
        self.throttle_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, throttle_actuator
        )

        # Simulation step settings (push_sim.py style)
        self.dt = dt
        time_step = self.mj_model.opt.timestep
        self.n_substeps = int(self.dt // time_step)
        assert (
            self.dt % time_step == 0
        ), f"dt must be a multiple of sim time step: {time_step}"

        # Parallel env datas
        self.n_envs = int(n_envs)
        self.mj_datas = [
            mujoco.MjData(self.mj_model) for _ in range(self.n_envs)
        ]
        # save pointer to each qpos and qvel as a numpy object array
        self.mj_datas_qpos = np.empty(self.n_envs, dtype=object)
        self.mj_datas_qvel = np.empty(self.n_envs, dtype=object)
        self.mj_datas_qpos[:] = [d.qpos for d in self.mj_datas]
        self.mj_datas_qvel[:] = [d.qvel for d in self.mj_datas]
        self.executor = ThreadPoolExecutor(max_workers=self.n_envs)

        # Store initial state for reset (make a copy of the values)
        self.init_qpos = np.stack(self.mj_datas_qpos)
        self.init_qvel = np.stack(self.mj_datas_qvel)

    def get_sim_info(self):
        """Return simulation infomation"""
        return (self.n_envs, self.dt)

    ########## Parallel Simulation ##########
    def run_sim(self, duration, ctrl=None, thread_fn=None):
        """Run the simulation for a given duration with optional function"""
        if duration <= 0:
            return
        n_steps = int(duration // self.dt)
        self.step_n(n_steps, ctrl, thread_fn)

    def _step_n_thread(
        self, thread_i, n_steps, mj_model, mj_data, ctrl=None, thread_fn=None
    ):
        """Step the simulation for one thread"""
        for j in range(n_steps):
            if ctrl is not None:
                mj_data.ctrl[:] = ctrl[j]
            if thread_fn is not None:
                thread_fn(thread_i, j, mj_model, mj_data)
            for _ in range(self.n_substeps):
                mujoco.mj_step(mj_model, mj_data)

    def step_n(self, n_steps, ctrl=None, thread_fn=None):
        """Step the simulation n times with optional function"""

        def vis_thread_fn(thread_i, j, mj_model, mj_data):
            """Thread function to visualize"""
            if thread_fn is not None:
                thread_fn(thread_i, j, mj_model, mj_data)
            if thread_i == 0:
                self.vis_sync(thread_i)

        if self.viewer:
            fn = vis_thread_fn
        else:
            fn = thread_fn
        futures = [
            self.executor.submit(
                self._step_n_thread,
                i,
                n_steps,
                self.mj_model,
                self.mj_datas[i],
                ctrl,
                fn,
            )
            for i in range(self.n_envs)
        ]
        wait(futures)

    def reset(self, wait_time=0.0):
        """Reset the simulation to the initial state"""
        zero_vel = np.zeros(self.mj_model.nv)
        zero_ctrl = np.zeros(self.mj_model.nu)
        for i, mj_data in enumerate(self.mj_datas):
            mujoco.mj_resetData(self.mj_model, self.mj_datas[i])
            self.mj_datas[i].qpos[:] = self.init_qpos[i]
            self.mj_datas[i].qvel[:] = self.init_qvel[i]
            # zero control
            self.mj_datas[i].ctrl[:] = zero_ctrl
            mujoco.mj_forward(self.mj_model, self.mj_datas[i])
        # Wait for the simulation to stabilize
        self.run_sim(wait_time)

    def close(self):
        """Close the simulation"""
        if self.viewer is not None:
            self.viewer.close()
        self.executor.shutdown(wait=True)

    ########## Robot-related functions ##########
    def set_car_init_states(self, states, env_idx=None):
        """Set the initial states of the car"""
        states, env_idx = self._preprocess_values(states, env_idx)

        car_pos, car_vel = self._to_pose(states)
        self.init_qpos[env_idx, :7] = car_pos
        self.init_qvel[env_idx, :6] = car_vel

    def get_car_state(self, env_idx=None):
        """
        Get (N, 4) states with format [x_rear, y_rear, theta, v]
        at rear-axle frame.
        """
        env_idx = self._preprocess_env_idx(env_idx)

        poses = np.stack(self.mj_datas_qpos[env_idx])[:, :7]
        vels = np.stack(self.mj_datas_qvel[env_idx])[:, :7]
        return self._to_states(poses, vels)

    def _to_states(self, poses, vels):
        """
        Convert poses and vels to (N, 4) states
        with format [x_rear, y_rear, theta, v]
        """
        chassis_x, chassis_y = poses[:, 0], poses[:, 1]
        vx, vy = vels[:, 0], vels[:, 1]

        # Compute at rear axle frame
        theta = quat_to_euler(poses[:, 3:7])[:, 2]
        x_rear = chassis_x - self.rear_offset * np.cos(theta)
        y_rear = chassis_y - self.rear_offset * np.sin(theta)
        v = vx * np.cos(theta) + vy * np.sin(theta)

        # Stack
        out = np.stack([x_rear, y_rear, theta, v], axis=-1)
        return out

    def _to_pose(self, states):
        """Convert states to poses"""
        x_rear = states[:, 0]
        y_rear = states[:, 1]
        theta = states[:, 2]
        v = np.zeros_like(x_rear)
        if states.shape[1] > 3:
            v = states[:, 3]

        # Convert to chassis center
        chassis_x = x_rear + self.rear_offset * np.cos(theta)
        chassis_y = y_rear + self.rear_offset * np.sin(theta)
        angles = np.zeros((len(states), 3))
        angles[:, 2] = theta
        quat = euler_to_quat(angles)

        # Fill
        qpos = np.zeros((len(states), 7))
        qpos[:, 0] = chassis_x
        qpos[:, 1] = chassis_y
        qpos[:, 3:7] = quat
        qvel = np.zeros((len(states), 6), dtype=float)
        qvel[:, 0] = v * np.cos(theta)
        qvel[:, 1] = v * np.sin(theta)
        return qpos, qvel

    def execute_controls(
        self, controls, durations, wait_time=0.0, return_intermediate=False
    ):
        """
        Run the car simulation with the given controls and durations.
        Args:
            controls: the control commands (u_vel, u_phi) at each step
                      of n given trajectories
            durations: the duration of each control step
                      of n given trajectories
        """
        n_trajs = len(controls)
        assert len(durations) == len(controls)
        assert (
            n_trajs <= self.n_envs
        ), f"Number of trajectories should not be larger than {self.n_envs}"

        traj_ctrls = []
        traj_ctrl_steps = []
        traj_key_steps_idx = []
        t_max = 0
        # For each trajectory
        for i in range(n_trajs):
            # Handle empty control/duration: treat as single keyframe (one state only)
            if len(controls[i]) == 0 or len(durations[i]) == 0:
                ci = np.zeros((0, 2), dtype=float)
                di = np.zeros(0, dtype=float)
            else:
                ci = np.asarray(controls[i], dtype=float)  # (k_i, 2)
                di = np.asarray(durations[i], dtype=float)  # (k_i,)
                assert (
                    ci.ndim == 2 and ci.shape[1] == 2
                ), "Invalid control shape"
                assert (
                    di.ndim == 1 and di.shape[0] == ci.shape[0]
                ), "Control and duration must have same length"

            # compute the number of steps for each control
            # round to nearest
            steps_i = np.rint(di / float(self.dt)).astype(int)
            steps_i = np.maximum(
                steps_i, 1
            )  # at least 1 step (no-op when empty)
            # key step indices to record the state (state after each control)
            # boundary indices: 0, cumsum(steps); empty trajectory -> [0] only (one keyframe)
            key_idx = np.concatenate(([0], np.cumsum(steps_i)))

            traj_ctrls.append(ci)
            traj_ctrl_steps.append(steps_i)
            traj_key_steps_idx.append(key_idx)
            t_max = max(t_max, int(key_idx[-1]))

        # Pad all trajectories to same length (t_max)
        # fill each env’s timeline with repeated controls
        control_steps = np.zeros((t_max, n_trajs, 2), dtype=float)
        for env_i in range(n_trajs):
            ci = traj_ctrls[env_i]
            steps_i = traj_ctrl_steps[env_i]
            t = 0
            for k in range(ci.shape[0]):
                s = int(steps_i[k])
                control_steps[t : t + s, env_i, :] = ci[k]
                t += s
            # remaining [t:t_max] stays zeros (idle padding)

        # Execute the controls
        _, intermediate_states = self.execute_controls_steps(
            control_steps, wait_time, True
        )

        # Extract only keyframes per-trajectory
        states = []
        intermediate = []
        for env_i in range(n_trajs):
            key_idx = traj_key_steps_idx[env_i]
            traj_states = intermediate_states[key_idx, env_i, :]
            states.append(traj_states.copy())
            if return_intermediate:
                intermediate.append(
                    intermediate_states[: int(key_idx[-1]) + 1, env_i, :]
                )

        if return_intermediate:
            return states, intermediate
        return states

    def execute_controls_steps(
        self, control_steps, wait_time=0.0, return_intermediate=False
    ):
        """
        Run the car simulation with the given controls.
        Args:
            control_steps: the target car state at each time step
                           defined as (num_time_step, num_envs, num_joints)
        """
        control_steps = np.asarray(control_steps)
        n_run_steps, n_trials, control_dim = control_steps.shape
        assert control_dim == 2, "Invalid control dimension"
        assert n_trials <= self.n_envs, (
            "required number of execution should be smaller "
            + "than the number of simulation environment"
        )
        env_idx = np.arange(n_trials)
        extra_steps = int(wait_time // self.dt)

        # Save the initial state first
        init_state = self.get_car_state(env_idx)
        if return_intermediate:
            intermediate_states = np.zeros(
                (1 + n_run_steps + extra_steps, n_trials, 4)
            )

        # Define the thread_fn to be passed to parallel_step_n
        def thread_fn(env_i, step_i, mj_model, mj_data):
            """Thread function to be run in parallel"""
            u_vel, u_phi = control_steps[step_i, env_i]
            wheel_ang_vel, steering_pos = self._to_car_control(u_vel, u_phi)
            mj_data.ctrl[self.throttle_id] = wheel_ang_vel
            mj_data.ctrl[self.steering_id] = steering_pos
            if return_intermediate:
                state = self.get_car_state(env_i)[0]
                intermediate_states[step_i, env_i] = state

        # Run the sim with the controls
        self.step_n(n_run_steps, thread_fn=thread_fn)

        # Define the thread_fn to be passed to parallel_step_n
        def thread_stabilize_fn(env_i, step_i, mj_model, mj_data):
            """Thread function to be run in parallel"""
            mj_data.ctrl[self.throttle_id] = 0.0
            mj_data.ctrl[self.steering_id] = 0.0
            if return_intermediate:
                state = self.get_car_state(env_i)[0]
                intermediate_states[n_run_steps + step_i, env_i] = state

        # Run for extra time to stabilize the simulation
        self.step_n(extra_steps, thread_fn=thread_stabilize_fn)

        # Get the last state
        last_state = self.get_car_state(env_idx)
        # sim step is run after the thread_fn, store qpos after the last step
        if return_intermediate:
            intermediate_states[-1] = last_state

        if return_intermediate:
            return (init_state, last_state), intermediate_states
        return (init_state, last_state)

    def _to_car_control(self, u_vel, u_phi):
        """Convert [u_vel, u_phi] to car actuators'control command."""
        # steering position control
        steering_pos = np.asarray(u_phi)
        # convert linear vel -> wheel angular velocity for velocity control
        wheel_ang_vel = u_vel / self.wheel_radius
        # magic number to compensate
        # 1, the under-actuated P-controller
        # 2, the acceleration and deceleration process
        wheel_ang_vel = wheel_ang_vel * 1.01
        steering_pos = steering_pos * 1.01
        return wheel_ang_vel, steering_pos

    ########## Helper functions ##########
    def _preprocess_env_idx(self, env_idx):
        """Process the env_idx to match the simulation"""
        if env_idx is None:
            env_idx = np.arange(self.n_envs)
        elif np.isscalar(env_idx):
            env_idx = np.array([env_idx])
        env_idx = np.asarray(env_idx, dtype=int)
        return env_idx

    def _preprocess_values(self, values, env_idx):
        """Preprocess the values and env_idx to match"""
        values = np.asarray(values)

        # Preprocess env_idx first
        # if not provided, use environments that values need
        if env_idx is None:
            size = 1 if values.ndim == 1 else len(values)
            env_idx = np.arange(size)
        # if a single environment provided, convert to array
        if np.isscalar(env_idx):
            env_idx = np.array([env_idx])
        env_idx = np.asarray(env_idx, dtype=int)

        # Preprocess values
        # if values is 1D, expand it to n_envs
        if values.ndim == 1:
            values = np.tile(values, (len(env_idx), 1))
        else:
            assert len(values) == len(
                env_idx
            ), "Values need to be 1D or have the same length as env_idx"

        if np.any(env_idx < 0) or np.any(env_idx >= self.n_envs):
            raise ValueError("env_idx out of range")
        return values, env_idx

    ########## Visualization ##########
    def vis_sync(self, env_idx=0):
        """Sync the simulation state to the viewer"""
        if self.viewer is None:
            return
        self.mj_data.qpos[:] = self.mj_datas_qpos[env_idx]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.viewer.sync()
        if self.real_time_vis:
            time.sleep(self.dt)

    def render_state(self, state, filename):
        """Render the state qpos (n_frame, nq) into a mp4 video"""
        render_mp4(self.mj_model, self.mj_data, state, self.dt, filename)


def test(sim: Sim):
    """Test the car simulation"""
    sim.reset()
    input("Start test")

    # Random controls
    # u = np.array([[0.25, 0.0], [0.5, 0.3], [0.5, -0.3], [0.25, 0.0]])
    # u = np.array([[1.0, 0.3]])
    u = np.random.uniform([0.0, -0.3], [1.0, 0.3], size=(10, 2))
    u = u[None, :, :]
    t = np.ones(u.shape[1])[None, :] * 1.0
    # t = np.ones(len(u))[None, :] * 6.0

    # MuJoCo
    states, intermediate_states = sim.execute_controls(
        u, t, wait_time=0.5, return_intermediate=True
    )
    # print(states)

    # Analytical
    plan_states, plan_intermediate_states = propagate_multi_steps(
        u[0], t[0], return_intermediate=True
    )

    # Plot the model result and actual result
    fig, ax = plt.subplots(figsize=(10, 10))
    plan_x = [p[0] for p in plan_states]
    plan_y = [p[1] for p in plan_states]
    plan_inter_x = [p[0] for p in plan_intermediate_states]
    plan_inter_y = [p[1] for p in plan_intermediate_states]
    ax.plot(plan_inter_x, plan_inter_y, "b--", linewidth=2, label="Planned")
    ax.scatter(plan_x, plan_y, marker="o", color="blue")

    x = [p[0] for p in intermediate_states[0]]
    y = [p[1] for p in intermediate_states[0]]
    inter_x = [p[0] for p in states[0]]
    inter_y = [p[1] for p in states[0]]
    ax.plot(x, y, "r-", linewidth=2, label="Actual")
    ax.scatter(inter_x, inter_y, marker="o", color="red")
    ax.set_aspect("equal")
    plt.show()

    input("Test done")
    sim.close()


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    np.random.seed(42)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    xml = open(os.path.join(curr_dir, "car_sim.xml")).read()

    sim = Sim(xml, n_envs=10, dt=0.01, visualize=True, real_time_vis=False)
    test(sim)
