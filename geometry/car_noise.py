import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import nnls

from simulation.car_sim import Sim
from geometry.car_dynamics import propagate_analytical, propagate_multi_steps
from geometry.car_dynamics import propagate_cov_linear, process_cov_body
from geometry.pose import wrap_to_pi
from planning.planning_utils import draw_cov_ellipse, world_2d_cov


def run_sim(
    sim: Sim,
    n_traj: int,
    n_step_range=(5, 10),
    v_range=(-0.3, 0.3),
    phi_range=(-0.3, 0.3),
    t_candidates=np.array([1.0]),
    random_init_yaw=True,
    return_intermediate=False,
):
    """
    Run simulation with random controls and durations
    to get n trajectories of length n_step.
    """
    n_env = sim.n_envs
    n = int(max(n_env, n_traj) // n_env)

    # Run sim n times
    states_list = []
    controls_list = []
    intermediate_list = []
    for _ in tqdm(range(n)):
        # Number of steps
        n_steps = np.random.randint(n_step_range[0], n_step_range[1])
        # Sample controls: (n_env, n_steps, 2)
        v_cmd = np.random.uniform(
            v_range[0], v_range[1], size=(n_env, n_steps)
        )
        phi_cmd = np.random.uniform(
            phi_range[0], phi_range[1], size=(n_env, n_steps)
        )
        u = np.stack([v_cmd, phi_cmd], axis=-1)
        t = np.random.choice(t_candidates, size=(n_env, n_steps))

        # Init simulation
        init_state = np.zeros((n_env, 4), dtype=float)
        if random_init_yaw:
            init_state[:, 2] = np.random.uniform(-np.pi, np.pi, size=n_env)
        sim.set_car_init_states(init_state)
        sim.reset()

        # Run simulation
        out = sim.execute_controls(
            u, t, return_intermediate=return_intermediate
        )
        if return_intermediate:
            states, intermediate = out
            states_list.extend(states)
            intermediate_list.extend(intermediate)
        else:
            states = out
            states_list.extend(states)

        # Controls as (n_env, n_steps, 3): (v, phi, t)
        controls = np.concatenate([u, t[..., None]], axis=-1)
        controls_list.extend(controls)

    if return_intermediate:
        return states_list, controls_list, intermediate_list
    return states_list, controls_list


def collect_data(sim: Sim, n_traj=1000):
    """Collect car sim data"""
    # Collect car sim data
    print("Collecting dataset")
    states, controls = run_sim(sim, n_traj=n_traj)
    states = np.asarray(states, dtype=object)
    controls = np.asarray(controls, dtype=object)
    np.save("data/car_noise_states.npy", states)
    np.save("data/car_noise_controls.npy", controls)


def process_data(states_list, controls_list):
    """
    Process states to compute per-step one-step residuals.
    """
    # Get pairs of states and corresponding controls
    states_pairs = []
    controls = []
    for i in range(len(states_list)):
        states = np.asarray(states_list[i], dtype=float)[:, :3]
        control = np.asarray(controls_list[i], dtype=float)

        x0 = states[:-1, :]
        x1 = states[1:, :]
        states_pairs.extend(np.stack([x0, x1], axis=1))
        controls.extend(control.reshape(-1, 3))
    states_pairs = np.asarray(states_pairs, dtype=float)
    controls = np.asarray(controls, dtype=float)

    # Compute residuals
    x0 = np.asarray(states_pairs[:, 0, :], dtype=float)
    x1 = np.asarray(states_pairs[:, 1, :], dtype=float)
    v, phi, t = controls[:, 0], controls[:, 1], controls[:, 2]
    res = np.zeros_like(x0)
    for i in range(states_pairs.shape[0]):
        pred = propagate_analytical((v[i], phi[i]), float(t[i]), x0[i])

        # world translation error
        e_xy_world = x1[i, :2] - pred[:2]
        th = float(pred[2])
        # body frame translation error: e_b = R(th)^T e_w
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, s], [-s, c]], dtype=float)
        e_xy_body = rot @ e_xy_world
        # yaw error
        e_th = wrap_to_pi(x1[i, 2] - pred[2])

        res[i, 0] = e_xy_body[0]
        res[i, 1] = e_xy_body[1]
        res[i, 2] = e_th

    return res, controls


def fit_params(res, controls):
    """
    Fit variance constants for linear covariance model:
        Var_x  ≈ Cx * |v*tan(phi)| * t
        Var_y  ≈ Cy * |v*tan(phi)| * t
        Var_th ≈ Cth* |v*tan(phi)| * t
    """
    v, phi, t = controls[:, 0], controls[:, 1], controls[:, 2]

    # Design feature (N,3)
    f0 = np.ones_like(t)
    # f1 = np.abs(v) * t
    f1 = t
    f2 = np.abs(np.tan(phi)) * t
    f3 = np.abs(v * np.tan(phi)) * t
    col = np.column_stack([f0, f1, f2, f3])

    # Targets are squared residuals (variance proxy)
    rx2 = res[:, 0] ** 2
    ry2 = res[:, 1] ** 2
    rth2 = res[:, 2] ** 2

    # Nonnegative least squares per axis
    x_coef, _ = nnls(col, rx2)
    y_coef, _ = nnls(col, ry2)
    th_coef, _ = nnls(col, rth2)

    return np.concatenate([x_coef, y_coef, th_coef]).astype(float)


def get_process_var(v, phi, t, params, eps=1e-12):
    """Get process variance from model parameters."""
    x0, x1, x2, x3, y0, y1, y2, y3, th0, th1, th2, th3 = params
    tanphi = np.tan(phi)

    f0 = 1.0
    f1 = np.abs(v) * t
    f2 = np.abs(tanphi) * t
    f3 = np.abs(v * tanphi) * t

    var_x = x0 + x1 * f1 + x2 * f2 + x3 * f3
    var_y = y0 + y1 * f1 + y2 * f2 + y3 * f3
    var_th = th0 + th1 * f1 + th2 * f2 + th3 * f3

    var_x = np.maximum(var_x, eps)
    var_y = np.maximum(var_y, eps)
    var_th = np.maximum(var_th, eps)

    return var_x, var_y, var_th


def validate(res, control, params):
    """Check z-scores ~ N(0,1) if Q is well calibrated."""
    v, phi, t = control[:, 0], control[:, 1], control[:, 2]
    var_x, var_y, var_th = get_process_var(v, phi, t, params)

    # Compute z-score
    zx = res[:, 0] / np.sqrt(var_x)
    zy = res[:, 1] / np.sqrt(var_y)
    zt = res[:, 2] / np.sqrt(var_th)
    print("Z-score stats (should be ~ mean 0, std 1):")
    for name, z in [("x", zx), ("y", zy), ("yaw", zt)]:
        print(f"  {name:>4s}: mean={z.mean():+.3f}, std={z.std(ddof=1):.3f}")


def fit(sim):
    sim.reset()

    states = np.load("data/car_noise_states.npy", allow_pickle=True)
    controls = np.load("data/car_noise_controls.npy", allow_pickle=True)

    # Process states
    res, controls = process_data(states.tolist(), controls.tolist())

    # Fit params
    print("\nFitting params")
    params = fit_params(res, controls)
    print("Fitted params (VARIANCE model):", params)

    # Validate
    print("\nValidate Results:")
    validate(res, controls, params)

    return params


def validate_sim_stats(params):
    """
    Validate process noise model using trajectory data
    with stats: whitened residuals, d^2, coverage.
    """
    print("\nValidating simulation stats with params:")
    print("Params:", params)

    states_list = np.load("data/car_noise_states.npy", allow_pickle=True)
    controls_list = np.load("data/car_noise_controls.npy", allow_pickle=True)
    states_list = states_list.tolist()
    controls_list = controls_list.tolist()

    all_w = []
    all_d2 = []
    for i in range(len(states_list)):
        controls = np.asarray(controls_list[i], dtype=float)
        k = controls.shape[0]
        u_i, t = controls[:, :2], controls[:, 2]

        # Mujoco states
        mj_key = np.asarray(states_list[i], dtype=float)[:, :3]
        # Analytical states
        kin_key = propagate_multi_steps(u_i, t, mj_key[0])
        kin_key = np.asarray(kin_key, dtype=float)

        # Compute covariance whitened residuals for each key step
        # along the trajectory
        cov = np.zeros((3, 3), dtype=float)
        for j in range(k):
            # cov prediction
            var_x, var_y, var_th = get_process_var(
                u_i[j, 0], u_i[j, 1], t[j], params
            )
            process_cov = np.diag([var_x, var_y, var_th])
            cov = propagate_cov_linear(
                u_i[j], t[j], kin_key[j], cov, process_cov
            )

            # residual
            r_world = mj_key[j + 1] - kin_key[j + 1]
            r_world[2] = wrap_to_pi(r_world[2])
            # cholesky whiten residual
            cov = 0.5 * (cov + cov.T) + np.eye(3) * 1e-12
            chol = np.linalg.cholesky(cov)
            w = np.linalg.solve(chol, r_world)
            d2 = float(w @ w)

            all_w.append(w)
            all_d2.append(d2)
    all_w = np.asarray(all_w)
    all_d2 = np.asarray(all_d2)

    chi2 = {"50%": 2.366, "90%": 6.251, "95%": 7.815, "99%": 11.345}
    cover = {k: np.mean(all_d2 <= v) for k, v in chi2.items()}
    print("GLOBAL whitened stats (ideal mean~0, std~1):")
    for j, name in enumerate(["x_world", "y_world", "yaw"]):
        print(
            f"  {name:>7s}: mean={all_w[:, j].mean():+.3f}, "
            + f"std={all_w[:, j].std(ddof=1):.3f}"
        )
    print(f"Mahalanobis d^2: mean={all_d2.mean():.3f} (ideal ~3.0)")
    print("Coverage:", cover)
    return all_w, all_d2, cover


def vis_sim_results(sim: Sim, params, n_std=2.0):
    sim.reset()
    states, controls, intermediate_states = run_sim(
        sim, n_traj=1, return_intermediate=True
    )
    controls = np.asarray(controls[0], dtype=float)
    k = controls.shape[0]

    # Results from Simulation (single env)
    mj_key = np.asarray(states[0], dtype=float)[:, :3]
    mj_path = np.asarray(intermediate_states[0], dtype=float)[:, :3]
    v, phi, t = controls[:, 0], controls[:, 1], controls[:, 2]

    # Analytical whole Path
    kin_key, kin_path = propagate_multi_steps(
        controls[:, :2], t, mj_key[0], return_intermediate=True
    )
    covs = np.zeros((k + 1, 3, 3), dtype=float)
    for i in range(k):
        # local process covariance
        proc_vars = get_process_var(v[i], phi[i], t[i], params)
        proc_cov = np.diag(proc_vars)
        covs[i + 1] = propagate_cov_linear(
            (v[i], phi[i]), t[i], kin_key[i], covs[i], proc_cov
        )

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # path
    ax.plot(kin_path[:, 0], kin_path[:, 1], "b--", label="Analytical Path")
    ax.plot(mj_path[:, 0], mj_path[:, 1], "r-", label="MuJoCo Path")
    # belief
    for i in range(len(kin_key)):
        x, y, yaw = kin_key[i]
        cov_xy_world = world_2d_cov(yaw, covs[i])
        # cov_xy_world = covs[i]
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


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=7)
    np.random.seed(42)

    # Launch simulation
    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml = open(os.path.join(par_dir, "simulation/car_sim.xml")).read()
    sim = Sim(xml, n_envs=10, dt=0.01, visualize=True, real_time_vis=False)

    # Collect data
    # collect_data(sim, n_traj=1000)

    # Fit
    params = fit(sim)
    params = np.array(params)

    # Validate from saved data (no sim)
    validate_sim_stats(params)

    # Scale params to fit global stats
    params[:4] *= 0.83**2  # x
    params[4:8] *= 0.83**2  # y
    params[8:] *= 0.57**2  # th
    validate_sim_stats(params)

    # Test in Simulation
    np.random.seed(42)
    vis_sim_results(sim, params)

    sim.close()
