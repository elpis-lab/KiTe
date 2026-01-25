import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt

from simulation.car_sim import Sim
from geometry.car_dynamics import propagate_analytical, propagate_multi_steps
from geometry.car_dynamics import CAR_WHEELBASE
from lie_group.lie_se2 import to_se2_transform, inv_se2_transform, log_se2
from lie_group.propagation import se2_error, propagate_cov
from planning.planning_utils import draw_cov_ellipse, world_2d_cov


def collect_dataset(
    sim: Sim,
    N=1e5,
    durations=(0.5, 1.0),
    v_range=(0.0, 1.0),
    phi_range=(-0.3, 0.3),
    yaw_random=True,
):
    """
    Collect residuals delta = Log(T_kin^{-1} T_mj)
    and corresponding control (v, phi, T).
    """
    durations = np.asarray(durations)

    res_list = []
    control_list = []
    n_done = 0
    while n_done < N:
        batch = min(sim.n_envs, N - n_done)
        env_idx = np.arange(batch)

        # Init
        x_init = np.zeros(batch)
        y_init = np.zeros(batch)
        th_init = (
            np.random.uniform(-np.pi, np.pi, size=batch)
            if yaw_random
            else np.zeros(batch)
        )
        # v_init = np.random.uniform(v_range[0], v_range[1], size=batch)
        v_init = np.ones(batch) * (v_range[1] - v_range[0]) / 2

        init_states = np.stack([x_init, y_init, th_init, v_init], axis=-1)
        sim.set_car_init_states(init_states, env_idx)
        sim.reset(wait_time=0.0)

        # Sample controls + durations
        v_cmd = np.random.uniform(v_range[0], v_range[1], size=batch)
        phi_cmd = np.random.uniform(phi_range[0], phi_range[1], size=batch)
        t_cmd = np.random.choice(durations, size=batch)

        # Collect 1 control step for each env
        state_mj = sim.execute_controls(
            np.concatenate(
                [v_cmd[:, None, None], phi_cmd[:, None, None]], axis=-1
            ),
            t_cmd[:, None],
            wait_time=0.0,
            return_intermediate=False,
        )
        state_mj = np.array(state_mj)
        init_state = state_mj[:, 0, :3]
        state_mj = state_mj[:, 1, :3]

        # Kinematic prediction
        state_kin = np.zeros_like(init_state)
        for i in range(len(env_idx)):
            state_kin[i] = propagate_analytical(
                (v_cmd[i], phi_cmd[i]), t_cmd[i], init_state[i]
            )

        # Residual delta = Log(T_kin^{-1} T_mj)
        for i in range(len(env_idx)):
            trans_kin = to_se2_transform(state_kin[i])
            trans_mj = to_se2_transform(state_mj[i])
            trans_delta = inv_se2_transform(trans_kin) @ trans_mj
            delta = log_se2(trans_delta)

            res_list.append(delta)
            control_list.append((v_cmd[i], phi_cmd[i], t_cmd[i]))

        n_done += batch
    res = np.asarray(res_list, dtype=float)
    control = np.asarray(control_list, dtype=float)
    return res, control


def fit_params(res, control):
    """
    Fit variance coefficients:
      - qx  = X0 + X1 * |v| T
      - qy  = Y0 + Y1 * |v| T + Y2 * |v| |tan(phi)| T
      - qth = YAW0 + YAW1 * |(v / CAR_WHEELBASE) * tan(phi)| T
    using squared residuals.
    """
    v, phi, t = control[:, 0], control[:, 1], control[:, 2]
    w = (v / CAR_WHEELBASE) * np.tan(phi)

    fv = np.abs(v) * t
    fvtan = np.abs(v) * np.abs(np.tan(phi)) * t
    fw = np.abs(w) * t

    # residual squared
    dx2 = res[:, 0] ** 2
    dy2 = res[:, 1] ** 2
    dth2 = res[:, 2] ** 2

    # solve for coefficients
    ax = np.column_stack([np.ones_like(fv), fv])
    ay = np.column_stack([np.ones_like(fv), fv, fvtan])
    ath = np.column_stack([np.ones_like(fw), fw])
    x0, x1 = nnls(ax, dx2)[0]
    y0, y1, y2 = nnls(ay, dy2)[0]
    yaw0, yaw1 = nnls(ath, dth2)[0]

    # rescale
    x1 *= 2.4**2
    y2 *= 0.6**2
    yaw1 *= 0.6**2
    return {
        "X0": x0,
        "X1": x1,
        "Y0": y0,
        "Y1": y1,
        "Y2": y2,
        "YAW0": yaw0,
        "YAW1": yaw1,
    }


def validate(res, control, params):
    """Check z-scores ~ N(0,1) if Q is well calibrated."""
    v, phi, t = control[:, 0], control[:, 1], control[:, 2]

    x0, x1 = params["X0"], params["X1"]
    y0, y1, y2 = params["Y0"], params["Y1"], params["Y2"]
    yaw0, yaw1 = params["YAW0"], params["YAW1"]

    w = (v / CAR_WHEELBASE) * np.tan(phi)

    # Noise model
    qx = x0 + x1 * np.abs(v) * t
    qy = (y0 + y1 * np.abs(v) * t) + y2 * np.abs(v) * np.abs(np.tan(phi)) * t
    qth = yaw0 + yaw1 * np.abs(w) * t

    qx = np.maximum(qx, 1e-12)
    qy = np.maximum(qy, 1e-12)
    qth = np.maximum(qth, 1e-12)

    # Compute z-score
    zx = res[:, 0] / np.sqrt(qx)
    zy = res[:, 1] / np.sqrt(qy)
    zt = res[:, 2] / np.sqrt(qth)
    print("Z-score stats (should be ~ mean 0, std 1):")
    for name, z in [("x", zx), ("y", zy), ("yaw", zt)]:
        print(f"  {name:>4s}: mean={z.mean():+.3f}, std={z.std(ddof=1):.3f}")


def validate_sim(
    sim: Sim,
    params,
    k=6,
    duration=1.0,
    v_range=(0.1, 1.0),
    phi_range=(-0.3, 0.3),
    n_std=2.0,
):
    sim.reset()

    # Sample controls
    # v_range = (1.0, 1.0)
    # phi_range = (-0.3, -0.3)
    v_cmd = np.random.uniform(v_range[0], v_range[1], size=k)
    phi_cmd = np.random.uniform(phi_range[0], phi_range[1], size=k)
    u = np.stack([v_cmd, phi_cmd], axis=-1)
    t = np.ones((k,), dtype=float) * float(duration)

    # Run Simulation
    init_state = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=float)
    sim.set_car_init_states(init_state, env_idx=np.array([0]))
    sim.reset(wait_time=0.0)
    states, intermediate_states = sim.execute_controls(
        u[None, :, :], t[None, :], return_intermediate=True
    )

    # Results from Simulation
    mj_key = np.asarray(states[0], dtype=float)[:, :3]
    mj_path = np.asarray(intermediate_states[0], dtype=float)[:, :3]

    # Analytical whole Path
    kin_key, kin_path = propagate_multi_steps(
        u, t, init_state[0], return_intermediate=True, dt=sim.dt
    )

    # Covariance Propagation
    def get_process_cov(v, phi, t, params):
        X0, X1 = params["X0"], params["X1"]
        Y0, Y1, Y2 = params["Y0"], params["Y1"], params["Y2"]
        YAW0, YAW1 = params["YAW0"], params["YAW1"]

        w = (v / CAR_WHEELBASE) * np.tan(phi)
        qx = X0 + X1 * abs(v) * t
        qy = (Y0 + Y1 * abs(v) * t) + Y2 * abs(v) * abs(np.tan(phi)) * t
        qth = YAW0 + YAW1 * abs(w) * t

        qx = max(qx, 1e-12)
        qy = max(qy, 1e-12)
        qth = max(qth, 1e-12)
        return np.diag([qx, qy, qth])

    cov = np.zeros((3, 3), dtype=float)
    covs = np.zeros((k + 1, 3, 3), dtype=float)
    for i in range(k):
        proc_cov = get_process_cov(v_cmd[i], phi_cmd[i], t[i], params)
        delta = se2_error(kin_key[i], kin_key[i + 1])
        covs[i + 1] = propagate_cov(delta, covs[i], proc_cov)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # path
    ax.plot(kin_path[:, 0], kin_path[:, 1], "b--", label="Analytical Path")
    ax.plot(mj_path[:, 0], mj_path[:, 1], "r-", label="MuJoCo Path")
    # belief
    for i in range(len(kin_key)):
        x, y, yaw = kin_key[i]
        cov_xy_world = world_2d_cov(yaw, covs[i])
        draw_cov_ellipse(ax, (x, y), cov_xy_world, sigma=n_std, color="C9")
    # keypoints
    ax.scatter(
        kin_key[:, 0], kin_key[:, 1], marker="o", label="Analytical Mean"
    )
    ax.scatter(mj_key[:, 0], mj_key[:, 1], marker="o", label="MuJoCo Mean")

    ax.set_aspect("equal")
    ax.grid(True, zorder=0)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"validate_sim: ellipses = {n_std}-sigma in XY (world frame)")
    ax.legend()
    plt.show()


def main(sim):
    sim.reset()

    # Collect residual dataset
    print("Collecting dataset")
    res, control = collect_dataset(
        sim,
        N=1e4,
        durations=(1.0,),
        v_range=(0.0, 1.0),
        phi_range=(-0.3, 0.3),
    )

    # Fit params
    print("\nFitting params")
    params = fit_params(res, control)
    print("Fitted params (VARIANCE model):")
    for k, val in params.items():
        print(f"{k:5s} = {val:.6e}")

    # Validate
    print("\nValidate Results:")
    validate(res, control, params)

    # Final adapted params
    params = {
        "X0": 0.0,
        "X1": 1.45e-3,
        "Y0": 0.0,
        "Y1": 0.0,
        "Y2": 3.70e-4,
        "YAW0": 0.0,
        "YAW1": 4.10e-4,
    }
    validate(res, control, params)

    sim.close()
    return params


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    np.random.seed(0)

    # Launch simulation
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    xml = open(os.path.join(curr_dir, "car_sim.xml")).read()
    sim = Sim(xml, n_envs=20, dt=0.01, visualize=False, real_time_vis=False)

    # params = main(sim)

    params = {
        "X0": 0.0,
        "X1": 1.45e-3,
        "Y0": 0.0,
        "Y1": 0.0,
        "Y2": 3.70e-4,
        "YAW0": 0.0,
        "YAW1": 4.10e-4,
    }
    validate_sim(sim, params, k=10)
