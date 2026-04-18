import os
import re
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planning.car import POS_RANGES, CAR_SIZE, generate_car_env
from simulation.car_sim import Sim


def _workspace_bounds_geoms(
    pos_ranges,
    fill_rgba=(0.25, 0.45, 0.85, 0.14),
    border_rgba=(0.1412, 0.4431, 0.6392, 0.5),
    border_half_thickness=0.012,
    fill_z=0.003,
    border_z=0.006,
):
    """
    Visual-only floor patch + edge rims for state-space bounds (x/y ranges).
    Matches SE2 position bounds in planning/car.py (POS_RANGES).
    """
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
        "    <!-- Workspace bounds (position / state space range) -->"
    )
    lines.append(
        "    <geom "
        'name="ws_bounds_fill" type="box" '
        f'pos="{cx:.6f} {cy:.6f} {fill_z:.6f}" '
        f'size="{hx:.6f} {hy:.6f} {fill_z:.6f}" '
        f'rgba="{fr:.3f} {fg:.3f} {fb:.3f} {fa:.3f}" '
        'contype="0" conaffinity="0"/>'
    )
    # Four edge strips (flat boxes on the floor, collision-free).
    edges = [
        ("ws_bounds_n", cx, ymax, hx + t, t, "north"),
        ("ws_bounds_s", cx, ymin, hx + t, t, "south"),
        ("ws_bounds_e", xmax, cy, t, hy + t, "east"),
        ("ws_bounds_w", xmin, cy, t, hy + t, "west"),
    ]
    for name, ex, ey, sx, sy, _label in edges:
        lines.append(
            "    <geom "
            f'name="{name}" type="box" '
            f'pos="{ex:.6f} {ey:.6f} {border_z:.6f}" '
            f'size="{sx:.6f} {sy:.6f} {border_z:.6f}" '
            f'rgba="{br:.3f} {bg:.3f} {bb:.3f} {ba:.3f}" '
            'contype="0" conaffinity="0"/>'
        )
    return lines


def _build_worldbody_overlay(
    env,
    pos_ranges=None,
    obstacle_height=0.25,
    goal_height=0.01,
):
    """Create MuJoCo worldbody geoms for workspace, obstacles, and goals."""
    if pos_ranges is None:
        pos_ranges = POS_RANGES

    obstacles = np.asarray(env.get("obstacles", []), dtype=float)
    goals = np.asarray(env.get("goals", []), dtype=float)
    goal_size = float(env.get("goal_size", 0.05))

    lines = []
    lines.append("  <worldbody>")
    lines.append("    <!-- Added by experiments/visualize_car.py -->")
    lines.extend(_workspace_bounds_geoms(pos_ranges))

    # Obstacles are axis-aligned cuboids from planning/car.py: [x, y, w, h].
    # planning/car.py returns obstacles shaped (n_obstacles, 4): [x, y, w, h]

    obstacles = obstacles[:2]
    for i, obstacle in enumerate(obstacles):
        x, y, w, h = obstacle[:4]
        lines.append(
            "    <geom "
            + f'name="obs_{i}" type="box" '
            + f'pos="{x:.6f} {y:.6f} {0.5 * obstacle_height:.6f}" '
            + f'size="{0.5 * w:.6f} {0.5 * h:.6f} {0.5 * obstacle_height:.6f}" '
            + 'rgba="0.20 0.20 0.20 1.0" '
            + 'friction=".8 .01 .0001" condim="3"/>'
        )

    # Goal markers as thin translucent green rectangles on the floor.
    for i, goal in enumerate(goals):
        gx, gy, gyaw = goal[:3]
        # MuJoCo box `size` is half-extents. Use goal_size as a 2D "radius",
        # so we draw a square of half-side = goal_size.
        half_x = 1.5 * CAR_SIZE[0] / 2
        half_y = 1.5 * CAR_SIZE[1] / 2
        rz = float(gyaw)
        lines.append(
            "    <geom "
            + f'name="goal_{i}" type="box" '
            + f'pos="{gx:.6f} {gy:.6f} {0.5 * goal_height:.6f}" '
            + f'size="{half_x:.6f} {half_y:.6f} {0.5 * goal_height:.6f}" '
            + f'euler="0 0 {rz:.6f}" '
            + 'rgba="0.1333 0.6000 0.3294 0.5" '
            + 'contype="0" conaffinity="0"/>'
        )

    lines.append("  </worldbody>")
    return "\n".join(lines) + "\n"


def _lighten_floor_grid(xml_text):
    """Switch floor checker texture to a light-gray grid."""
    xml_text = re.sub(
        r'rgb1="[^"]+"\s+rgb2="[^"]+"',
        'rgb1="0.90 0.90 0.90" rgb2="0.80 0.80 0.80"',
        xml_text,
    )
    xml_text = re.sub(
        r'material name="grid"[^>]*/>',
        (
            'material name="grid" texture="grid" texrepeat="14 14" '
            'texuniform="true" reflectance=".05"/>'
        ),
        xml_text,
    )
    return xml_text


def _clean_floor_grid(xml_text):
    """Switch floor checker texture to a light-gray grid."""
    xml_text = re.sub(
        r'rgb1="[^"]+"\s+rgb2="[^"]+"',
        'rgb1="1.0 1.0 1.0" rgb2="0.97 0.97 0.97"',
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
    obstacle_height=0.25,
    goal_height=0.01,
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

    # xml_text = _lighten_floor_grid(xml_text)
    xml_text = _clean_floor_grid(xml_text)
    overlay = _build_worldbody_overlay(
        env,
        pos_ranges=pos_ranges,
        obstacle_height=obstacle_height,
        goal_height=goal_height,
    )
    return xml_text.replace("</mujoco>", f"{overlay}</mujoco>")


def visualize_car_env_mujoco(
    env=None,
    pos_ranges=None,
    obstacle_height=0.05,
    goal_height=0.03,
    dt=0.01,
    hold_time=60.0,
):
    """
    Launch a MuJoCo viewer for a car environment with cuboid obstacles.

    - Workspace (state-space x/y bounds) is shown as a tinted floor patch
      and blue edge strips matching POS_RANGES in planning/car.py.
    - Obstacles are added as dark cuboids.
    - Goal regions are shown as translucent green rectangles.
    - Floor texture is changed to light gray checker grid.
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
    )
    sim = Sim(xml, n_envs=1, dt=dt, visualize=True, real_time_vis=True)

    # Set the car to the environment's start state.
    start = np.asarray(env["start"], dtype=float)
    if start.shape[0] < 4:
        start = np.concatenate([start[:3], [0.0]])
    sim.set_car_init_states(start[:4], env_idx=0)
    sim.reset(wait_time=0.1)

    # Keep the viewer open by running no-op simulation.
    if hold_time is None:
        hold_time = 60.0
    sim.run_sim(float(hold_time))
    sim.close()


if __name__ == "__main__":
    np.random.seed(40)
    visualize_car_env_mujoco()
