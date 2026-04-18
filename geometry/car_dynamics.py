import numpy as np
from geometry.pose import wrap_to_pi

# Car Constants
CAR_SIZE = (0.50, 0.22)
CAR_WHEELBASE = 0.30
CAR_TRACK_WIDTH = 0.17
CAR_WHEEL_RADIUS = 0.036

# Covariance constants (Fitted from car_noise.py)
COV_X = (1.0e-5, 56.0e-5)
COV_Y = (1.0e-5, 14.0e-5)
COV_YAW = (1.0e-5, 127.0e-5)


def propagate_dt(u, t, init_state=(0, 0, 0, 0), dt=0.1):
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

    return float(x1), float(y1), float(yaw1)


def propagate_multi_steps(
    u,
    t,
    init_state=(0, 0, 0, 0),
    return_intermediate=False,
    dt=0.01,
    propagate_fn=propagate_analytical,
):
    """Physics of car dynamics for a sequence of controls u and time t."""
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
        curr_state = propagate_fn(u[i], t[i], curr_state)
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
            curr_state = propagate_fn(u[i], h, curr_state)
            intermediate.append(curr_state)

    intermediate = np.asarray(intermediate, dtype=float)
    return states, intermediate


def process_cov_body(u, t, eps=1e-12):
    """Get the process covariance in the body frame."""
    v, phi = u
    tanphi = np.tan(phi)

    # Base
    # f0 = np.ones_like(t)
    f1 = np.abs(v) * t
    # f2 = np.abs(tanphi) * t
    f3 = np.abs(v * tanphi) * t
    # Linear model of process covariance
    var_x = COV_X[0] * f1 + COV_X[1] * f3
    var_y = COV_Y[0] * f1 + COV_Y[1] * f3
    var_th = COV_YAW[0] * f1 + COV_YAW[1] * f3

    var_x = np.maximum(var_x, eps)
    var_y = np.maximum(var_y, eps)
    var_th = np.maximum(var_th, eps)
    return np.diag([var_x, var_y, var_th])


def propagate_cov_linear(u, t, init_state, cov, process_cov_body, dt=0.1):
    """Propagate the covariance of the car dynamics using a linear model."""
    v, phi = u
    th = float(init_state[2])
    # yaw rate
    w = (v / CAR_WHEELBASE) * np.tan(phi)

    # Convert process covariance to world frame
    c, s = np.cos(th), np.sin(th)
    g = np.eye(3, dtype=float)
    g[:2, :2] = np.array([[c, -s], [s, c]], dtype=float)
    process_cov = g @ process_cov_body @ g.T

    # Covariance propagation
    n_steps = max(1, int(np.ceil(float(t) / float(dt))))
    h = float(t) / n_steps
    process_cov_step = process_cov / n_steps
    for _ in range(n_steps):
        state_jac = np.eye(3, dtype=float)

        # Straight: x += v cos(th) h, y += v sin(th) h, th unchanged
        if abs(w) < 1e-9:
            state_jac[0, 2] = -v * np.sin(th) * h
            state_jac[1, 2] = v * np.cos(th) * h
        # Turning: use exact closed-form sensitivity wrt th for duration h
        else:
            th1 = th + w * h
            r = v / w
            # dx/dth0, dy/dth0 over the substep
            state_jac[0, 2] = r * (np.cos(th1) - np.cos(th))
            state_jac[1, 2] = r * (np.sin(th1) - np.sin(th))
            # update nominal yaw
            th = th1

        cov = state_jac @ cov @ state_jac.T + process_cov_step
    return cov
