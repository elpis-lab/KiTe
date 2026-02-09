import math
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse

from geometry.pose import angle_diff
import ompl.util as ou


# OMPL
def set_ompl_seed(seed):
    """Set the seed for the OMPL random number generator"""
    ou.RNG.setSeed(seed)


def vec_to_cov(vector):
    """Convert vector to covariance matrix"""
    return np.array(
        [
            [vector[0], vector[1], vector[2]],
            [vector[1], vector[3], vector[4]],
            [vector[2], vector[4], vector[5]],
        ]
    )


def cov_to_vec(cov):
    """Convert covariance matrix to vector"""
    return np.array(
        [cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2]]
    )


# Validity Check
# point and rectangles
def point_in_rects(x, y, rects):
    """
    Check if (x, y) lies inside any axis-aligned rectangle in rects.
    Each rectangle is (cx, cy, w, h).
    """
    rects = np.asarray(rects)
    cx, cy, w, h = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    inside = (np.abs(x - cx) <= 0.5 * w) & (np.abs(y - cy) <= 0.5 * h)
    return bool(np.any(inside))


def point_dist_to_rects(x, y, rects):
    """
    Compute the minimum distance from (x, y) to any axis-aligned rectangles.
    Rects are n by 4 numpy array with columns (cx, cy, w, h).
    """
    rects = np.asarray(rects)
    cx, cy, w, h = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3]
    dx = np.maximum(np.abs(x - cx) - 0.5 * w, 0.0)
    dy = np.maximum(np.abs(y - cy) - 0.5 * h, 0.0)
    return np.min(np.hypot(dx, dy))


def points_out_of_bound(points, bound):
    """
    Check if (x, y) lies inside any axis-aligned rectangle in rects.
    Each rectangle is (cx, cy, w, h).
    """
    points = np.asarray(points)
    single = points.ndim == 1
    if single:
        points = points[None, :]

    bound = np.asarray(bound)
    lower = bound[:, 0]
    upper = bound[:, 1]
    outs = np.any((points[:, :2] < lower) | (points[:, :2] > upper), axis=1)

    if single:
        return outs[0]
    return outs


def se2_points_in_region(points, center, size, rot_weight=0.2):
    """Check if SE2 poses are in the region"""
    points = np.asarray(points)
    single = points.ndim == 1
    if single:
        points = points[None, :]
    center = np.asarray(center)

    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    dyaw = angle_diff(points[:, 2], center[2])
    d = (dx**2 + dy**2 + (rot_weight * dyaw) ** 2) ** 0.5

    if single:
        return d[0] <= size
    return d <= size


# rectangles and circles
def rects_circles_in_collision(pose, shape, circles):
    """
    Check if a rectangle object is in collision with a list of circles.
    The pose is a SE2 pose (x, y, yaw) and shape is a tuple (w, h).
    The circles is a list of circles with columns (x, y, r).
    """
    if len(circles) == 0:
        return False
    circles = np.asarray(circles)
    circle_poses = circles[:, :2]
    circle_radius = circles[:, 2]

    pose = np.array(pose)
    single = pose.ndim == 1
    if single:
        pose = pose[None, :]

    x, y, yaw = pose[:, 0], pose[:, 1], pose[:, 2]
    w, h = shape[:2]
    hx, hy = 0.5 * w, 0.5 * h

    # Translate circle centers into object frame
    dx = circle_poses[:, 0][None, :] - x[:, None]
    dy = circle_poses[:, 1][None, :] - y[:, None]
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    lx = c * dx + s * dy
    ly = -s * dx + c * dy

    # Find closest point on the object to the circle center
    # Clamp each circle center to object extents
    qx = np.clip(lx, -hx, hx)
    qy = np.clip(ly, -hy, hy)

    # Distance squared from circle center to nearest point on object
    ddx = lx - qx
    ddy = ly - qy
    dist2 = ddx * ddx + ddy * ddy

    # Collision check
    circle_radius.reshape((-1, circle_radius.shape[0]))
    hits = np.any(dist2 <= circle_radius**2, axis=1)

    # Return result
    if single:
        return hits[0]
    return hits


def approx_rect_to_circles(rect, margin=0.0, overlap_frac=0.1):
    """Approximate a rectangle to a list of circles"""
    x, y, w, h = rect
    # Decide dominant axis
    if w >= h:
        axis = "x"
        long_len, short_len = w, h
    else:
        axis = "y"
        long_len, short_len = h, w
    long_len += 2.0 * margin

    # Radius = half of short side
    r = 0.5 * short_len
    # if the long dimension fits within one circle's diameter
    if long_len <= 2.0 * r:
        return np.array([[x, y, r]], dtype=float)

    # Step of placing circles
    step = 2.0 * r * (1.0 - overlap_frac)
    start = -0.5 * long_len
    end = 0.5 * long_len
    n_intervals = max(1, int(np.ceil((end - start) / step)))

    # Generate coords with spacing <= step, including end
    coords = np.linspace(start, end, n_intervals + 1)
    circles = np.zeros((coords.shape[0], 3), dtype=float)
    if axis == "x":
        circles[:, 0] = x + coords
        circles[:, 1] = y
    else:
        circles[:, 0] = x
        circles[:, 1] = y + coords
    circles[:, 2] = r

    return circles


# rectangles and rectangles
def rects_rects_in_collision(pose, shape, rects):
    """
    Check if a rectangle object is in collision with a list of rectangles.
    The pose is a SE2 pose (x, y, yaw) and shape is a tuple (w, h).
    The rects is a list of rectangles with columns (cx, cy, w, h).
    """
    if len(rects) == 0:
        return False

    rects = np.asarray(rects, dtype=float)
    r_cx = rects[:, 0]
    r_cy = rects[:, 1]
    r_bx = 0.5 * rects[:, 2]  # half width
    r_by = 0.5 * rects[:, 3]  # half height

    pose = np.asarray(pose, dtype=float)
    single = pose.ndim == 1
    if single:
        pose = pose[None, :]

    x = pose[:, 0]
    y = pose[:, 1]
    yaw = pose[:, 2]

    w, h = shape[:2]
    a0 = 0.5 * w  # hx
    a1 = 0.5 * h  # hy

    # Relative translation from object center to each rect center in world frame
    # (N, M)
    dx = r_cx[None, :] - x[:, None]
    dy = r_cy[None, :] - y[:, None]
    # Object local axes in world:
    # u = (cos, sin), v = (-sin, cos)
    c = np.cos(yaw)[:, None]  # (N,1)
    s = np.sin(yaw)[:, None]  # (N,1)

    # Rotation terms between object axes (u,v) and world axes (ex, ey)
    # R = [[u·ex, u·ey],
    #      [v·ex, v·ey]] = [[c, s],
    #                      [-s, c]]
    eps = 1e-12  # Add tiny epsilon to avoid issues in near-parallel cases
    abs_tf00 = np.abs(c) + eps
    abs_tf01 = np.abs(s) + eps
    abs_tf10 = np.abs(-s) + eps
    abs_tf11 = np.abs(c) + eps

    # Projections of T onto object axes T·u and T·v
    # (N, M)
    tf_u = c * dx + s * dy
    tf_v = -s * dx + c * dy

    # SAT tests (OBB vs AABB), 4 separating axes: u, v, ex, ey
    # 1) axis u
    ra = a0
    rb = r_bx[None, :] * abs_tf00 + r_by[None, :] * abs_tf01
    sep_u = np.abs(tf_u) > (ra + rb)
    # 2) axis v
    ra = a1
    rb = r_bx[None, :] * abs_tf10 + r_by[None, :] * abs_tf11
    sep_v = np.abs(tf_v) > (ra + rb)
    # 3) axis ex (world x)
    ra = a0 * abs_tf00 + a1 * abs_tf10
    rb = r_bx[None, :]
    sep_ex = np.abs(dx) > (ra + rb)
    # 4) axis ey (world y)
    ra = a0 * abs_tf01 + a1 * abs_tf11
    rb = r_by[None, :]
    sep_ey = np.abs(dy) > (ra + rb)

    # Collision if NOT separated on any axis
    separated = sep_u | sep_v | sep_ex | sep_ey
    collide_nm = ~separated
    hits = np.any(collide_nm, axis=1)

    return hits[0] if single else hits


# probability of collision (circles version)
def circles_in_collision(center_pose, body_circles, obstacles):
    """Check if the circles are in collision"""
    if obstacles is None or len(obstacles) == 0:
        return 0.0
    obs_p = obstacles[:, :2]
    obs_r = obstacles[:, 2]

    # Main body represented as K circles
    body_circles = np.asarray(body_circles)
    body_p = body_circles[:, :2]
    body_r = body_circles[:, 2]
    # compute centers & covs in world frame for each circleq
    w_body_p, _ = circles_world_centers_and_covs(
        center_pose, np.zeros((3, 3)), body_p
    )

    # Pairwise distance check: ||pi - pj|| <= ri + rj
    diff = w_body_p[:, None, :] - obs_p[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    # (K, M) threshold squared
    threshold = body_r[:, None] + obs_r[None, :]
    in_collision = np.any(d2 <= threshold**2)
    return bool(in_collision)


def circles_collision_risk(center_pose, center_cov, body_circles, obstacles):
    if obstacles is None or len(obstacles) == 0:
        return 0.0
    obs_p = obstacles[:, :2]
    obs_r = obstacles[:, 2]

    # Main body represented as K circles
    body_circles = np.asarray(body_circles)
    body_p = body_circles[:, :2]
    body_r = body_circles[:, 2]
    # compute centers & covs in world frame for each circle
    w_body_p, w_body_cov = circles_world_centers_and_covs(
        center_pose, center_cov, body_p
    )

    # Broadcast distance matrix over (K, M)
    d = w_body_p[:, None, :] - obs_p[None, :, :]
    s = np.linalg.norm(d, axis=2)

    # Unit directions u
    u = d / (s[..., None] + 1e-12)  # (K, M, 2)
    # Directional variance
    # sigma^2 = u^T Sy u, with Sy only depends on k
    # einsum: u(k,m,i) * Sy(k,i,j) * u(k,m,j)
    var = np.einsum("kmi,kij,kmj->km", u, w_body_cov, u)
    var = np.maximum(var, 1e-12)
    sigma = np.sqrt(var)

    # Collision radius rho(k, m) = body_r[k] + obs_r[m]
    rho = body_r[:, None] + obs_r[None, :]
    margin = s - rho  # >0 safe

    # If mean already in collision, probability ~ 1
    # Otherwise, probability = Phi(-(margin) / sigma)
    z = -(margin / sigma)
    # normal cumulative distribution function
    p = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    p = np.clip(p, 0.0, 1.0)
    risk = np.max(p)
    # p_obs = 1.0 - np.prod(1.0 - p, axis=0)
    # risk = 1.0 - np.prod(1.0 - p_obs)
    return float(risk)


def circles_world_centers_and_covs(pose, cov, body_p):
    """Convert body frame circle centers & covs to world frame"""
    x, y, yaw = pose
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    # Convert to world positions
    w_body_p = body_p @ rot.T + np.array([x, y])

    # Convert to world covs
    # build Jacobian
    px, py = body_p[:, 0], body_p[:, 1]
    dth_x = -s * px - c * py
    dth_y = +c * px - s * py
    jac = np.zeros((body_p.shape[0], 2, 3), dtype=float)
    jac[:, 0, 0] = 1.0
    jac[:, 1, 1] = 1.0
    jac[:, 0, 2] = dth_x
    jac[:, 1, 2] = dth_y
    # to world covs
    w_body_cov = jac @ cov @ np.transpose(jac, (0, 2, 1))
    w_body_cov = (
        0.5 * (w_body_cov + np.transpose(w_body_cov, (0, 2, 1)))
        + 1e-12 * np.eye(2)[None, :, :]
    )
    return w_body_p, w_body_cov


# Plotting
def draw_circle(ax, x, y, r, color="r", alpha=0.5, linewidth=0, label=None):
    """Draw a circle on a given axis, with proper legend support."""
    circ = plt.Circle(
        (x, y), r, color=color, alpha=alpha, linewidth=linewidth, label=label
    )
    ax.add_patch(circ)
    return circ


def draw_gradient_circle(
    ax, center, radius, n_rings=30, cmap="Greens", alpha=0.8, label=None
):
    """
    Approximate a radial gradient by drawing multiple concentric
    semi-transparent circles.
    """
    cm = plt.get_cmap(cmap)
    cx, cy = center

    for i in range(n_rings, 0, -1):
        r = radius * i / n_rings
        color = cm(0.25 + 0.5 * (1 - i / n_rings))
        lab = label if i == n_rings // 2 else None
        # first draw a white circle on top to cover the previous circle
        if i != n_rings:
            draw_circle(ax, cx, cy, r, color="w", alpha=alpha)
        draw_circle(ax, cx, cy, r, color, alpha=alpha, label=lab)


def draw_rect(
    ax,
    x,
    y,
    width,
    height,
    yaw=0,
    color="b",
    edgecolor=None,
    alpha=0.5,
    linewidth=0,
    label=None,
):
    """Draw a rectangle at the given position with the given orientation."""
    # Rectangle corners (centered)
    corners = np.array(
        [
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
            [width / 2, height / 2],
            [-width / 2, height / 2],
        ]
    )

    # Rotation
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = corners @ R.T + np.array([x, y])

    edgecolor = color if edgecolor is None else edgecolor
    poly = Polygon(
        corners,
        closed=True,
        facecolor=color,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
    )
    ax.add_patch(poly)
    return poly


def draw_gradient_rect(
    ax,
    x,
    y,
    width,
    height,
    yaw=0,
    n_strips=30,
    cmap="Greens",
    alpha=0.8,
    label=None,
):
    """
    Approximate a gradient-filled rectangle by tiling small rectangles.
    Only one legend entry is created.
    """
    cm = plt.get_cmap(cmap)
    cx, cy = x, y

    x0 = cx - width / 2.0
    y0 = cy - height / 2.0
    dx = width / n_strips
    dy = height / n_strips

    for iy in range(n_strips):
        for ix in range(n_strips):
            x = x0 + (ix + 0.5) * dx
            y = y0 + (iy + 0.5) * dy

            # normalized radial distance (square-style)
            d = math.hypot(x - cx, y - cy) / (width / 2.0)
            d = min(1.0, d)
            color = cm(0.25 + 0.5 * (1 - d))

            lab = None
            if ix == (n_strips - 1) // 2 and iy == (n_strips - 1) // 2:
                lab = label
            draw_rect(ax, x, y, dx, dy, yaw, color, color, alpha, label=lab)


def draw_cov_ellipse(
    ax, mean, cov, sigma=2, color="b", alpha=0.5, linewidth=0, label=None
):
    """Draw a confidence ellipse for a 2x2 covariance at mean."""
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)

    def chi2_quantile_2d(sigma):
        """
        Chi-square quantiles for df=2,
        restricted to common 1/2/3-sigma levels in 2D.
        - 0.68, 0.95, 0.99
        """
        lookup = {
            1: 2.278867,  # ~ 68%
            2: 5.991464547,  # 95%
            3: 9.210340372,  # 99.7%
        }
        if sigma not in lookup:
            raise ValueError(f"sigma must be one of {sorted(lookup.keys())}")
        return lookup[sigma]

    # Scale by chi-square quantile for df=2
    q = chi2_quantile_2d(sigma)

    # Radii = sqrt(eigenvalues * q) ; sort so radii[0] is major axis
    radii = np.sqrt(vals * q)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    vecs = vecs[:, order]
    # Angle of major axis (vecs[:,0])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    e = Ellipse(
        xy=mean,
        width=2 * radii[0],
        height=2 * radii[1],
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
    )
    ax.add_patch(e)
    return e


def world_2d_cov(yaw, cov):
    """Local 2D covariance to world covariance given rotation yaw"""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    cov_body = np.asarray(cov)[:2, :2]
    return rot @ cov_body @ rot.T
