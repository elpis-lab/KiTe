import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse

from geometry.pose import angle_diff
import ompl.util as ou


# OMPL
def set_ompl_seed(seed):
    """Set the seed for the OMPL random number generator"""
    ou.RNG.setSeed(seed)


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
        lab = label if i == 1 else None
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
