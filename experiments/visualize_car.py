import os
import sys
import re
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

import matplotlib as mpl
from matplotlib import colors as mcolors

from planning.car import POS_RANGES, CAR_WHEELBASE, generate_car_env
from planning.planning_utils import vec_to_cov
from simulation.car_sim import Sim

import mujoco

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Match planning/planning_utils.draw_cov_ellipse (sigma=2 ~ 95% in 2D)
CHI2_LOOKUP = {1: 2.278867, 2: 5.991464547, 3: 9.210340372}


def mpl_tab10_rgb(idx: int):
    """RGB tuple for Matplotlib tab10 color C{idx}."""
    return mcolors.to_rgb(f"C{idx % 10}")


def chi2_q(sigma: int) -> float:
    if sigma not in CHI2_LOOKUP:
        raise ValueError(f"sigma must be one of {sorted(CHI2_LOOKUP)}")
    return CHI2_LOOKUP[sigma]


def cov_xy_to_ellipse_params(cov_xy: np.ndarray, sigma: int = 2):
    """
    2x2 covariance -> semi-axis lengths (major, minor) and euler-z (rad)
    for an ellipse in the xy plane (same convention as draw_cov_ellipse).
    """
    cov_xy = np.asarray(cov_xy, dtype=float)
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.maximum(vals, 0.0)
    q = chi2_q(sigma)
    radii = np.sqrt(vals * q)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    vecs = vecs[:, order]
    ang = float(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return float(radii[0]), float(radii[1]), ang


def fit_xy_spline_curve(
    states: np.ndarray,
    n_dense: int = 160,
    smoothing: float = 0.0,
):
    """
    Fit a smooth open curve through waypoint centers (x, y).

    Returns (dense_xy Nx2, waypoints_xy Mx2) where dense is sampled along the
    spline; waypoints are the original centers for covariance markers.
    """
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states[None, :]
    if len(states) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    x = states[:, 0]
    y = states[:, 1]
    m = len(states)
    k = int(min(3, m - 1))
    if k < 1:
        return np.zeros((0, 2)), np.column_stack([x, y])

    u_dense = np.linspace(0.0, 1.0, n_dense)
    try:
        tck, u = splprep([x, y], s=smoothing, k=k)
        xd, yd = splev(u_dense, tck)
    except (ValueError, TypeError):
        # Degenerate / collinear / duplicate points — chord-length linear interp
        tpar = np.zeros(m)
        for i in range(1, m):
            tpar[i] = tpar[i - 1] + np.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        if tpar[-1] < 1e-12:
            xd = np.full(n_dense, x[0])
            yd = np.full(n_dense, y[0])
        else:
            tpar = tpar / tpar[-1]
            xd = np.interp(u_dense, tpar, x)
            yd = np.interp(u_dense, tpar, y)
    dense = np.column_stack([xd, yd])
    wpts = np.column_stack([x, y])
    return dense, wpts


def workspace_bounds_geoms(
    pos_ranges,
    fill_rgba=None,
    border_rgba=None,
    border_half_thickness=0.012,
    fill_z=0.003,
    border_z=0.006,
):
    """
    Visual-only floor patch + edge rims for state-space bounds (x/y ranges).
    Operable area: tab10 C1 fill (alpha ~0.4) and C1 boundary strips.
    """
    if fill_rgba is None:
        r, g, b = mpl_tab10_rgb(5)
        fill_rgba = (r, g, b, 0.1)
    if border_rgba is None:
        r, g, b = mpl_tab10_rgb(5)
        border_rgba = (r, g, b, 1.0)

    (xmin, xmax), (ymin, ymax) = pos_ranges
    cx = 0.5 * (float(xmin) + float(xmax))
    cy = 0.5 * (float(ymin) + float(ymax))
    hx = 0.5 * (float(xmax) - float(xmin))
    hy = 0.5 * (float(ymax) - float(ymin))
    t = float(border_half_thickness)

    fr, fg, fb, fa = fill_rgba
    br, bg, bb, ba = border_rgba

    lines = []
    lines.append(
        "    <!-- Workspace bounds (operable area / state space range) -->"
    )
    lines.append(
        "    <geom "
        'name="ws_bounds_fill" type="box" '
        f'pos="{cx:.6f} {cy:.6f} {fill_z:.6f}" '
        f'size="{hx:.6f} {hy:.6f} {fill_z:.6f}" '
        f'rgba="{fr:.4f} {fg:.4f} {fb:.4f} {fa:.4f}" '
        'contype="0" conaffinity="0"/>'
    )
    edges = [
        ("ws_bounds_n", cx, ymax, hx + t, t),
        ("ws_bounds_s", cx, ymin, hx + t, t),
        ("ws_bounds_e", xmax, cy, t, hy + t),
        ("ws_bounds_w", xmin, cy, t, hy + t),
    ]
    for name, ex, ey, sx, sy in edges:
        lines.append(
            "    <geom "
            f'name="{name}" type="box" '
            f'pos="{ex:.6f} {ey:.6f} {border_z:.6f}" '
            f'size="{sx:.6f} {sy:.6f} {border_z:.6f}" '
            f'rgba="{br:.4f} {bg:.4f} {bb:.4f} {ba:.4f}" '
            'contype="0" conaffinity="0"/>'
        )
    return lines


def goal_c2_geoms(goals, goal_height, goal_size):
    """Translucent C2 cylinder; radius = env goal_size, height = goal_height."""
    goals = np.asarray(goals, dtype=float)
    r, g, b = mpl_tab10_rgb(2)
    a = 0.2
    rad = float(goal_size)
    hz = 0.5 * float(goal_height)
    lines = []
    for gi, goal in enumerate(goals):
        gx, gy = float(goal[0]), float(goal[1])
        lines.append(
            "    <geom "
            f'name="goal_{gi}" type="cylinder" '
            f'pos="{gx:.6f} {gy:.6f} {hz:.6f}" '
            f'size="{rad:.6f} {hz:.6f}" '
            f'rgba="{r:.4f} {g:.4f} {b:.4f} {a:.4f}" '
            'contype="0" conaffinity="0"/>'
        )
    return lines


def capsule_between(p0, p1, radius, rgba, name, z_mid=0.012):
    """MuJoCo capsule along segment (x,y,z)."""
    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    z0 = z_mid if len(p0) < 3 else float(p0[2])
    z1 = z_mid if len(p1) < 3 else float(p1[2])
    r, g, b, a = rgba
    return (
        "    <geom "
        f'name="{name}" type="capsule" '
        f'fromto="{x0:.6f} {y0:.6f} {z0:.6f} {x1:.6f} {y1:.6f} {z1:.6f}" '
        f'size="{radius:.6f}" '
        f'rgba="{r:.4f} {g:.4f} {b:.4f} {a:.4f}" '
        'contype="0" conaffinity="0"/>'
    )


def trajectory_curve_geoms(
    states: np.ndarray,
    color_idx: int,
    prefix: str,
    line_radius: float = 0.006,
    z_curve: float = 0.012,
    n_dense: int = 160,
):
    """Spline through waypoints + capsule chain in tab10 color."""
    dense, _ = fit_xy_spline_curve(states, n_dense=n_dense)
    if len(dense) < 2:
        return []
    r, g, b = mpl_tab10_rgb(color_idx)
    rgba_line = (r, g, b, 1.0)
    lines = []
    for i in range(len(dense) - 1):
        p0 = np.array([dense[i, 0], dense[i, 1], z_curve])
        p1 = np.array([dense[i + 1, 0], dense[i + 1, 1], z_curve])
        lines.append(
            capsule_between(
                p0,
                p1,
                line_radius,
                rgba_line,
                f"{prefix}_seg_{i}",
                z_mid=z_curve,
            )
        )
    return lines


def trajectory_waypoint_spheres(
    states: np.ndarray,
    prefix: str,
    z_wp: float = 0.015,
    radius: float = 0.009,
):
    """Black spheres at each key waypoint (x, y) along a trajectory."""
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states[None, :]
    lines = []
    for wi, st in enumerate(states):
        x, y = float(st[0]), float(st[1])
        lines.append(
            "    <geom "
            f'name="{prefix}_wp_{wi}" type="sphere" '
            f'pos="{x:.6f} {y:.6f} {z_wp:.6f}" '
            f'size="{radius:.6f}" '
            'rgba="0.0 0.0 0.0 1.0" '
            'contype="0" conaffinity="0"/>'
        )
    return lines


def covariance_geoms_at_waypoints(
    states: np.ndarray,
    color_idx: int,
    prefix: str,
    sigma: int = 2,
    fill_alpha: float = 0.4,
    edge_alpha: float = 1.0,
    z_fill: float = 0.014,
    z_edge: float = 0.016,
    n_edge_seg: int = 36,
    edge_radius: float = 0.004,
):
    """
    Covariance ellipses on key waypoints (state dim >= 9: x,y,yaw + 6 cov).
    Marginal xy uses cov[:2,:2] from vec_to_cov(state[3:]).
    """
    states = np.asarray(states, dtype=float)
    if states.ndim == 1:
        states = states[None, :]
    r, g, b = mpl_tab10_rgb(color_idx)
    rgba_fill = (r, g, b, fill_alpha)
    rgba_edge = (r, g, b, edge_alpha)
    lines = []
    for wi, st in enumerate(states):
        if st.shape[0] < 9:
            continue
        x, y = float(st[0]), float(st[1])
        cov = vec_to_cov(st[3:9])
        cov_xy = cov[:2, :2]
        a_ax, b_ax, ang = cov_xy_to_ellipse_params(cov_xy, sigma=sigma)
        # Flat ellipsoid (fill)
        lines.append(
            "    <geom "
            f'name="{prefix}_covf_{wi}" type="ellipsoid" '
            f'pos="{x:.6f} {y:.6f} {z_fill:.6f}" '
            f'size="{a_ax:.6f} {b_ax:.6f} {0.5 * (z_fill + 0.001):.6f}" '
            f'euler="0 0 {ang:.6f}" '
            f'rgba="{rgba_fill[0]:.4f} {rgba_fill[1]:.4f} {rgba_fill[2]:.4f} {rgba_fill[3]:.4f}" '
            'contype="0" conaffinity="0"/>'
        )
        # Edge: short capsules along analytic ellipse
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c, -s], [s, c]])
        tvals = np.linspace(0.0, 2.0 * np.pi, n_edge_seg + 1, endpoint=True)
        for k in range(n_edge_seg):
            u0 = np.array([a_ax * np.cos(tvals[k]), b_ax * np.sin(tvals[k])])
            u1 = np.array(
                [a_ax * np.cos(tvals[k + 1]), b_ax * np.sin(tvals[k + 1])]
            )
            q0 = R @ u0 + np.array([x, y])
            q1 = R @ u1 + np.array([x, y])
            lines.append(
                capsule_between(
                    np.array([q0[0], q0[1], z_edge]),
                    np.array([q1[0], q1[1], z_edge]),
                    edge_radius,
                    rgba_edge,
                    f"{prefix}_cove_{wi}_{k}",
                    z_mid=z_edge,
                )
            )
    return lines


def workspace_worldbody_overlay(
    env,
    pos_ranges=None,
    obstacle_height=0.25,
    goal_height=0.01,
    trajectory_c4=None,
    trajectory_c0=None,
):
    """Create MuJoCo worldbody geoms for workspace, obstacles, goals, paths."""
    if pos_ranges is None:
        pos_ranges = POS_RANGES

    obstacles = np.asarray(env.get("obstacles", []), dtype=float)
    goals = np.asarray(env.get("goals", []), dtype=float)
    goal_size = float(env.get("goal_size", 0.075))

    lines = []
    lines.append("  <worldbody>")
    lines.append("    <!-- Added by experiments/visualize_car.py -->")
    lines.extend(workspace_bounds_geoms(pos_ranges))

    # Optional trajectories (spline + covariance on key waypoints)
    if trajectory_c4 is not None and np.asarray(trajectory_c4).size > 0:
        tc4 = np.asarray(trajectory_c4, dtype=float)
        lines.extend(
            trajectory_curve_geoms(tc4, color_idx=4, prefix="traj_c4")
        )
        lines.extend(trajectory_waypoint_spheres(tc4, prefix="traj_c4"))
        lines.extend(
            covariance_geoms_at_waypoints(tc4, color_idx=4, prefix="cov_c4")
        )
    if trajectory_c0 is not None and np.asarray(trajectory_c0).size > 0:
        tc0 = np.asarray(trajectory_c0, dtype=float)
        lines.extend(
            trajectory_curve_geoms(tc0, color_idx=0, prefix="traj_c0")
        )
        lines.extend(trajectory_waypoint_spheres(tc0, prefix="traj_c0"))
        lines.extend(
            covariance_geoms_at_waypoints(tc0, color_idx=0, prefix="cov_c0")
        )

    obstacles = obstacles[:2]
    for i, obstacle in enumerate(obstacles):
        x, y, w, h = obstacle[:4]
        lines.append(
            "    <geom "
            + f'name="obs_{i}" type="box" '
            + f'pos="{x:.6f} {y:.6f} {0.5 * obstacle_height:.6f}" '
            + f'size="{0.5 * w:.6f} {0.5 * h:.6f} {0.5 * obstacle_height:.6f}" '
            + 'rgba="0.0 0.0 0.0 1.0" '
            + 'friction=".8 .01 .0001" condim="3"/>'
        )

    lines.extend(goal_c2_geoms(goals, goal_height, goal_size))

    lines.append("  </worldbody>")
    return "\n".join(lines) + "\n"


def clean_floor_grid(xml_text):
    """Lighten floor checker texture for a clean background."""
    xml_text = re.sub(
        r'rgb1="[^"]+"\s+rgb2="[^"]+"',
        'rgb1="1.0 1.0 1.0" rgb2="0.975 0.975 0.975"',
        xml_text,
    )
    xml_text = re.sub(
        r'material name="grid"[^>]*/>',
        (
            'material name="grid" texture="grid" texrepeat="10 10" '
            'texuniform="true" reflectance=".05"/>'
        ),
        xml_text,
    )
    return xml_text


def build_car_visualization_xml(
    env,
    base_xml_path=None,
    pos_ranges=None,
    obstacle_height=0.05,
    goal_height=0.03,
    trajectory_c4=None,
    trajectory_c0=None,
):
    """
    Build an XML string for visualizing a planning/car.py environment in MuJoCo.
    """
    if base_xml_path is None:
        base_xml_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "simulation",
            "car_sim.xml",
        )

    with open(base_xml_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    xml_text = clean_floor_grid(xml_text)
    overlay = workspace_worldbody_overlay(
        env,
        pos_ranges=pos_ranges,
        obstacle_height=obstacle_height,
        goal_height=goal_height,
        trajectory_c4=trajectory_c4,
        trajectory_c0=trajectory_c0,
    )
    return xml_text.replace("</mujoco>", f"{overlay}</mujoco>")


def set_visualization_camera(viewer, pos_ranges=None):
    """
    Aim the free camera ~45° down at the workspace center (good for screenshots).
    Call after Sim is created with visualize=True.
    """
    if viewer is None or pos_ranges is None:
        return
    (xmin, xmax), (ymin, ymax) = pos_ranges
    cx = 0.5 * (float(xmin) + float(xmax))
    cy = 0.5 * (float(ymin) + float(ymax))
    hx = 0.5 * (float(xmax) - float(xmin))
    hy = 0.5 * (float(ymax) - float(ymin))
    span = float(max(hx, hy, 0.2))

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [cx, cy, 0.12]
    viewer.cam.distance = 2.2 * span
    viewer.cam.azimuth = 180.0
    # viewer.cam.elevation = -45.0
    viewer.cam.elevation = -90.0
    if hasattr(viewer, "sync"):
        viewer.sync()


def visualize_car_env_mujoco(
    env=None,
    pos_ranges=None,
    obstacle_height=0.05,
    goal_height=0.03,
    dt=0.01,
    trajectory_1=None,
    trajectory_2=None,
):
    """
    Launch a MuJoCo viewer for a car environment.

    Workspace uses C1 fill + boundary; obstacles black; goals C2 (alpha 0.7).
    Optional trajectories: C4 and C0 splines, black spheres at key waypoints,
    and covariance on waypoints (9-D states: x,y,theta + 6 cov).
    """
    if env is None:
        env = generate_car_env()
    if pos_ranges is None:
        pos_ranges = POS_RANGES

    xml = build_car_visualization_xml(
        env,
        pos_ranges=pos_ranges,
        obstacle_height=obstacle_height,
        goal_height=goal_height,
        trajectory_c4=trajectory_1,
        trajectory_c0=trajectory_2,
    )
    sim = Sim(xml, n_envs=1, dt=dt, visualize=True, real_time_vis=False)

    set_visualization_camera(sim.viewer, pos_ranges=pos_ranges)

    start = np.asarray(env["start"], dtype=float)
    if start.shape[0] < 4:
        start = np.concatenate([start[:3], [0.0]])
    sim.set_car_init_states(start[:4], env_idx=0)
    sim.reset(wait_time=0.1)

    try:
        if sim.viewer is not None:
            while sim.viewer.is_running():
                sim.vis_sync(0)
                time.sleep(0.01)
    finally:
        sim.close()


if __name__ == "__main__":
    # Load environment
    env_i = 9
    envs = np.load("data/planning_car_envs.npy", allow_pickle=True)
    env = envs[env_i]
    # visualize_car_env(env, [env["start"]], draw_car_shape=True)
    # plt.show()

    # Load path
    kite_path = np.load(
        f"results/planning_car/car_aorrt_w2_50.0_plan_states.npy",
        allow_pickle=True,
    )
    gbt_path = np.load(
        f"results/planning_car/car_aorrt_w2_0.0_plan_states.npy",
        allow_pickle=True,
    )
    kite_path = kite_path[0, env_i, -1]
    # clean a bit for visualization
    kite_path = kite_path[:5] + kite_path[7:10] + kite_path[11:]
    gbt_path = gbt_path[2, env_i, -1]

    # adjusted the path for more intuitive visualization (center of mass frame)
    adjusted_kite_path = kite_path.copy()
    adjusted_kite_path[:, 0] += CAR_WHEELBASE / 2 * np.cos(kite_path[:, 2])
    adjusted_kite_path[:, 1] += CAR_WHEELBASE / 2 * np.sin(kite_path[:, 2])
    adjusted_gbt_path = gbt_path.copy()
    adjusted_gbt_path[:, 0] += CAR_WHEELBASE / 2 * np.cos(gbt_path[:, 2])
    adjusted_gbt_path[:, 1] += CAR_WHEELBASE / 2 * np.sin(gbt_path[:, 2])
    # adjusted the goal simply for more intuitive visualization
    adjusted_env = env.copy()
    adjusted_env["goals"] = env["goals"] + np.array([0, CAR_WHEELBASE / 2, 0])
    visualize_car_env_mujoco(
        env=adjusted_env,
        trajectory_1=adjusted_kite_path,
        trajectory_2=adjusted_gbt_path,
    )
