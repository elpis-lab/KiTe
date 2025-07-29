import mujoco


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
    import imageio

    # Initialize renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    scene_option = mujoco.MjvOption()
    for flag, value in scene_option_flags.items():
        scene_option.flags[flag] = value

    # Render frames
    mujoco.mj_resetData(mj_model, mj_data)
    frames = []
    for qpos in qpos_sequence:
        mj_data.qpos[:] = qpos
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

    # Save frames to mp4
    frame_rate = 1 / dt
    with imageio.get_writer(file_path, fps=frame_rate) as writer:
        for frame in frames:
            writer.append_data(frame)

    return frames
