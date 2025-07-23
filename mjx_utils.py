import imageio

import mujoco
import mujoco.viewer
from mujoco import mjx
import jax
from jax import numpy as jnp
import numpy as np


##########
# Basic MJX functions, step and reset
##########
def mjx_sim_step(model, data, ctrl=None):
    """
    Basic MJX step function that takes a control input
    and perform a single step of the simulation

    in-axes for jit should be (None, 0, 0)
    """
    if ctrl is not None:
        data = data.replace(ctrl=ctrl)

    data = mjx.step(model, data)
    return data


def get_mjx_sim_step_n(n_steps):
    """Get a mjx_sim_step_n function given the number of steps
    Usage:
        from mjx_utilsimport get_mjx_sim_step_n
        mjx_sim_step_n = get_mjx_sim_step_n(n_steps)
    """

    def mjx_sim_step_n(model, data, ctrl=None):
        """
        Basic MJX step function that takes a conrol input
        and perform n steps of the simulation

        in-axes for jit should be (None, 0, 0)
        """

        def f(data, _):
            """Simple step function for jax.lax.scan"""
            return mjx.step(model, data), None

        if ctrl is not None:
            data = data.replace(ctrl=ctrl)
        data, _ = jax.lax.scan(f, data, None, length=n_steps)
        return data

    return mjx_sim_step_n


def mjx_sim_reset(model, data, qpos=None, qvel=None, ctrl=None):
    """
    Basic MJX reset function that takes qpos, qvel, ctrl
    and reset the simulation to the given state.
    If values are not provided, simply reset it to the initial state.

    in-axes for jit should be (None, 0, 0, 0, 0)
    """
    mjx_data = mjx.make_data(model)
    if qpos is None:
        qpos = model.qpos0
    if qvel is None:
        qvel = jnp.zeros(model.nv)
    if ctrl is None:
        ctrl = jnp.zeros(model.nu)
    mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

    mjx_data = mjx.forward(model, mjx_data)
    return mjx_data


##########
# MJX functions for Visualization
##########
def mjx_get_qpos(mjx_data):
    """This should return a numpy array with shape (n_envs, n_qpos)"""
    return np.array(mjx_data.qpos)


def mjx_get_qpos_data_into(mj_data, mj_model, mjx_data, env_id=0):
    """Get the qpos data from the mjx_data and put it into the mj_data"""
    single_mjx_data = jax.tree_util.tree_map(lambda x: x[env_id], mjx_data)
    mj_data.qpos[:] = single_mjx_data.qpos
    mujoco.mj_forward(mj_model, mj_data)


def get_qpos_data_into(mj_data, mj_model, qpos):
    """Put the qpos data into the mj_data"""
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)


def render_mp4(
    mj_model,
    mj_data,
    qpos_sequence,
    dt,
    file_path,
    height=720,
    width=1280,
    scene_option_flags={},
):
    """
    Render a sequence of qpos data into mp4 video

    qpos_sequence should be a list of qpos data that represents
    the qpos data at each time step
    """
    # Initialize renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    for flag, value in scene_option_flags.items():
        scene_option.flags[flag] = value

    # Render frames
    mujoco.mj_resetData(mj_model, mj_data)
    frames = []
    for qpos in qpos_sequence:
        get_qpos_data_into(mj_data, mj_model, qpos)
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

    # Save frames to mp4
    frame_rate = 1 / dt
    with imageio.get_writer(file_path, fps=frame_rate) as writer:
        for frame in frames:
            writer.append_data(frame)

    return frames
