import numpy as np
from geometry.pose import wrap_to_pi

# Constants
CAR_SIZE = (0.11, 0.25)
CAR_WHEELBASE = 0.30
CAR_TRACK_WIDTH = 0.17
CAR_WHEEL_RADIUS = 0.036


def propagate_dt(u, t, init_state=(0, 0, 0, 0), dt=0.01):
    """
    Physics of car dynamics for t / dt steps with control u.
    Deprecated since for loop is slow.
    Kept just for analytical solution reference.
    """
    v, phi = u
    x, y, theta = init_state[:3]
    n_steps = max(1, round(t / dt))
    h = t / n_steps

    w = (v / CAR_WHEELBASE) * np.tan(phi)
    for _ in range(n_steps):
        x += v * np.cos(theta) * h
        y += v * np.sin(theta) * h
        theta += w * h
    return x, y, theta


def propagate_analytical(u, t, init_state=(0, 0, 0, 0)):
    """
    Analytical solution of car dynamics for t time with control u.
    Exact ODE solution of car kinematics dynamics
    """
    v, phi = u
    x0, y0, yaw0 = init_state[:3]

    # Yaw rate
    w = (v / CAR_WHEELBASE) * np.tan(phi)
    # Straight / near-straight motion (avoid division by tiny w)
    if abs(w) < 1e-9:
        x1 = x0 + v * np.cos(yaw0) * t
        y1 = y0 + v * np.sin(yaw0) * t
        yaw1 = yaw0
    # Solution for non-zero yaw rate
    else:
        # yaw rate integration
        yaw1 = yaw0 + w * t
        # turning radius (signed)
        r = v / w
        # exact integration
        x1 = x0 + r * (np.sin(yaw1) - np.sin(yaw0))
        y1 = y0 - r * (np.cos(yaw1) - np.cos(yaw0))
    yaw1 = wrap_to_pi(yaw1)

    return x1, y1, yaw1


def propagate_multi_steps(
    u, t, init_state=(0, 0, 0, 0), return_intermediate=False, dt=0.01
):
    """
    Physics of car dynamics for a sequence of controls u and time t.
    Vectorization implementation.
    """
    u = np.asarray(u)
    t = np.asarray(t)
    assert u.ndim == 2 and t.ndim == 1
    assert u.shape[0] == t.shape[0]
    curr_state = init_state[:3]
    steps = u.shape[0]

    # Keyframe states
    states = np.zeros((steps + 1, 3), dtype=float)
    states[0, :] = init_state[:3]
    for i in range(steps):
        curr_state = propagate_analytical(u[i], t[i], curr_state)
        states[i + 1, :] = curr_state
    if not return_intermediate:
        return states

    # Intermediate states
    curr_state = init_state[:3]
    intermediate = [curr_state]
    for i in range(steps):
        # number of internal steps for this segment (>=1)
        substeps = int(max(1, int(np.round(t[i] / dt))))
        h = t[i] / substeps  # so we land exactly on the segment end
        for _ in range(substeps):
            curr_state = propagate_analytical(u[i], h, curr_state)
            intermediate.append(curr_state)

    intermediate = np.asarray(intermediate, dtype=float)
    return states, intermediate
