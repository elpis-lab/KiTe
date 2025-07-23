import time
import numpy as np
import mujoco
import mujoco.viewer

# Initialize Mujoco
mj_model = mujoco.MjModel.from_xml_path("mjx_sim.xml")
mj_data = mujoco.MjData(mj_model)
viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

print(mj_data.geom_xpos[0])

robot_init = [np.pi / 2, -1.7, 2, -1.87, -np.pi / 2, np.pi]
obj_init = [0, -0.7, 0, 1, 0, 0, 0]
mj_data.qpos[:] = robot_init + obj_init
mj_data.ctrl[:] = robot_init
for _ in range(10000):
    for _ in range(5):
        mujoco.mj_step(mj_model, mj_data)
    viewer.sync()
    time.sleep(0.01)
viewer.close()
