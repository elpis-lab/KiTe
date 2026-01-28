import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import nnls, minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

from simulation.car_sim import Sim
from geometry.car_dynamics import propagate_analytical, propagate_multi_steps
from geometry.car_dynamics import CAR_WHEELBASE
from lie_group.propagation import se2_error, propagate_cov
from planning.planning_utils import draw_cov_ellipse, world_2d_cov


def collect_dataset(
    sim: Sim,
    n_data=1e4,
    n_step=10,
    durations=np.array([1.0]),
    v_range=(-0.5, 1.0),
    phi_range=(-0.3, 0.3),
):
    """
    Collect per-step residuals over multi-step trajectories.
    Residuals delta = Log(T_kin^{-1} T_mj) with
    corresponding control (v, phi, t).
    """
    n_env = sim.n_envs
    n = int(n_data // n_step // n_env)

    res_list = []
    control_list = []
    for i in tqdm(range(n)):
        sim.reset(wait_time=0.0)

        # Sample controls + durations
        v_cmd = np.random.uniform(v_range[0], v_range[1], size=(n_env, n_step))
        phi_cmd = np.random.uniform(
            phi_range[0], phi_range[1], size=(n_env, n_step)
        )
        u = np.stack([v_cmd, phi_cmd], axis=-1)
        t = np.random.choice(durations, size=(n_env, n_step))

        # Run in Mujoco
        states_mj = sim.execute_controls(u, t)
        states_mj = np.asarray(states_mj, dtype=float)[:, :, :3]

        # Compute per-step one-step residuals
        for i in range(n_env):
            for j in range(n_step):
                state_mj = states_mj[i, j + 1]
                state_kin = propagate_analytical(
                    (v_cmd[i][j], phi_cmd[i][j]), t[i][j], states_mj[i, j]
                )
                # Residual delta = Log(T_kin^{-1} T_mj)
                delta = se2_error(state_kin, state_mj)
                res_list.append(delta)
                control_list.append((v_cmd[i][j], phi_cmd[i][j], t[i][j]))

    res = np.asarray(res_list, dtype=float)
    control = np.asarray(control_list, dtype=float)
    return res, control


def fit_params(res, control):
    """
    Fit variance coefficients:
      - qx  = X0 + X1 * |v| T + X2 * |tan(phi)| T + X3 * |v| |tan(phi)| T
      - qy  = Y0 + Y1 * |v| T + Y2 * |tan(phi)| T + Y3 * |v| |tan(phi)| T
      - qth = T0 + T1 * |v| T + T2 * |tan(phi)| T + T3 * |v| |tan(phi)| T
    using squared residuals.
    """
    v, phi, t = control[:, 0], control[:, 1], control[:, 2]
    w = (v / CAR_WHEELBASE) * np.tan(phi)
    tanphi = np.tan(phi)

    f0 = np.ones_like(t)
    f1 = np.abs(v) * t
    f2 = np.abs(tanphi) * t
    f3 = np.abs(v * tanphi) * t
    col_x = np.column_stack([f3])
    col_y = np.column_stack([f3])
    col_t = np.column_stack([f3])

    # solve for coefficients for residual squared
    def fit_axis(col, res, eps=1e-12):
        """Fix params"""
        # x0 = nnls(col, res)[0]
        x0 = np.zeros(col.shape[1])
        bounds = [(0.0, None)] * col.shape[1]

        # # Negative log-likelihood
        # def nll(p):
        #     q = col @ p
        #     # keep strictly positive for log/div
        #     q = np.maximum(q, eps)
        #     return 0.5 * float(np.sum(np.log(q) + res / q))

        def z_score(p):
            q = col @ p
            q = np.maximum(q, eps)
            z = res / np.sqrt(q)
            mz = z.mean()
            sz = z.std(ddof=1)
            return float(mz * mz + (sz - 1.0) ** 2)

        out = minimize(
            z_score,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options=dict(maxiter=500),
        )
        if not out.success:
            print("[fit_axis] warning:", out.message)
        return out.x

    x3 = fit_axis(col_x, res[:, 0])
    y3 = fit_axis(col_y, res[:, 1])
    t3 = fit_axis(col_t, res[:, 2])
    x0, x1, x2 = 0.0, 0.0, 0.0
    y0, y1, y2 = 0.0, 0.0, 0.0
    t0, t1, t2 = 0.0, 0.0, 0.0
    x3 = float(x3)
    y3 = float(y3)
    t3 = float(t3)

    # rescale
    return [x0, x1, x2, x3, y0, y1, y2, y3, t0, t1, t2, t3]


def validate(res, control, params):
    """Check z-scores ~ N(0,1) if Q is well calibrated."""
    v, phi, t = control[:, 0], control[:, 1], control[:, 2]

    x0, x1, x2, x3 = params[:4]
    y0, y1, y2, y3 = params[4:8]
    t0, t1, t2, t3 = params[8:12]

    w = (v / CAR_WHEELBASE) * np.tan(phi)
    tanphi = np.tan(phi)

    # Noise model
    qx = (
        x0
        + x1 * np.abs(v) * t
        + x2 * np.abs(tanphi) * t
        + x3 * np.abs(v * tanphi) * t
    )
    qy = (
        y0
        + y1 * np.abs(v) * t
        + y2 * np.abs(tanphi) * t
        + y3 * np.abs(v * tanphi) * t
    )
    qth = (
        t0
        + t1 * np.abs(v) * t
        + t2 * np.abs(tanphi) * t
        + t3 * np.abs(v * tanphi) * t
    )

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
    v_range=(-0.5, 1.0),
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
        X0, X1, X2, X3 = params[:4]
        Y0, Y1, Y2, Y3 = params[4:8]
        T0, T1, T2, T3 = params[8:12]

        w = (v / CAR_WHEELBASE) * np.tan(phi)
        qx = (
            X0
            + X1 * abs(v) * t
            + X2 * abs(np.tan(phi)) * t
            + X3 * abs(v * np.tan(phi)) * t
        )
        qy = (
            Y0
            + Y1 * abs(v) * t
            + Y2 * abs(np.tan(phi)) * t
            + Y3 * abs(v * np.tan(phi)) * t
        )
        qth = (
            T0
            + T1 * abs(v) * t
            + T2 * abs(np.tan(phi)) * t
            + T3 * abs(v * np.tan(phi)) * t
        )

        qx = max(qx, 1e-12)
        qy = max(qy, 1e-12)
        qth = max(qth, 1e-12)
        return np.diag([qx, qy, qth])

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
    res, controls = collect_dataset(
        sim,
        n_data=1e3,
        n_step=10,
        durations=np.array([1.0]),
        v_range=(-0.5, 1.0),
        phi_range=(-0.3, 0.3),
    )
    np.save("data/car_noise_residuals.npy", res)
    np.save("data/car_noise_controls.npy", controls)
    res = np.load("data/car_noise_residuals.npy")
    controls = np.load("data/car_noise_controls.npy")

    # Fit params
    print("\nFitting params")
    params = fit_params(res, controls)
    print("Fitted params (VARIANCE model):")
    print("X =", params[:4])
    print("Y =", params[4:8])
    print("T =", params[8:12])

    # Validate
    print("\nValidate Results:")
    validate(res, controls, params)

    # Final adapted params
    # params = [None] * 12
    # params[:4] = [0.0, 5.0e-2, 0.0, 1.0e-2]
    # params[4:8] = [0.0, 0.0, 0.0, 4.5e-4]
    # params[8:12] = [0.0, 0.0, 0.0, 3.6e-3]
    # validate(res, controls, params)
    return params


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)
    np.random.seed(42)

    # Launch simulation
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    xml = open(os.path.join(curr_dir, "car_sim.xml")).read()
    sim = Sim(xml, n_envs=10, dt=0.01, visualize=True, real_time_vis=False)

    # Fit
    params = main(sim)

    # Test in Simulation
    np.random.seed(10)
    # params = [None] * 12
    # params[:4] = [0.0, 1.4e-3, 0.0, 0.0]
    # params[4:8] = [0.0, 0.0, 0.0, 3.7e-4]
    # params[8:12] = [0.0, 0.0, 0.0, 1.4e-3]
    # params[:4] = [0.0, 1.4e-3, 0.0, 0.0]
    # params[4:8] = [0.0, 0.0, 0.0, 3.7e-4]
    # params[8:12] = [0.0, 0.0, 0.0, 1.4e-3]
    validate_sim(sim, params, k=10)

    sim.close()
