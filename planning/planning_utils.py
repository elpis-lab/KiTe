import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import ompl.util as ou


# OMPL
def set_ompl_seed(seed):
    """Set the seed for the OMPL random number generator"""
    ou.RNG.setSeed(seed)


# State Utils
def get_random_se2_states(
    n_data, pos_range=((-0.4, 0.4), (-1.0, -0.4)), euler_range=(-np.pi, np.pi)
):
    """Generate random initial states"""
    pos_x = np.random.uniform(pos_range[0][0], pos_range[0][1], (n_data, 1))
    pos_y = np.random.uniform(pos_range[1][0], pos_range[1][1], (n_data, 1))
    euler = np.random.uniform(euler_range[0], euler_range[1], (n_data, 1))
    states = np.concatenate([pos_x, pos_y, euler], axis=-1)
    return states


# Validity Check
def out_of_bounds(obj_pos, pos_range=((-0.76, 0.76), (-1.1, -0.3))):
    """Check if the object is out of bounds"""
    obj_pos = np.array(obj_pos)
    single = obj_pos.ndim == 1
    if single:
        obj_pos = obj_pos[None, :]

    # Check bounds
    lower = np.array([r[0] for r in pos_range])
    upper = np.array([r[1] for r in pos_range])
    outs = np.any((obj_pos < lower) | (obj_pos > upper), axis=1)

    if single:
        return outs[0]
    return outs


def in_collision_with_circles(
    obj_pose, obj_shape, circle_poses, circle_radius
):
    """Check if the object is in collision with a list of circles"""
    if len(circle_poses) == 0 or len(circle_radius) == 0:
        return False
    circle_poses = np.asarray(circle_poses)
    circle_radius = np.asarray(circle_radius)

    obj_pose = np.array(obj_pose)
    single = obj_pose.ndim == 1
    if single:
        obj_pose = obj_pose[None, :]

    x, y, yaw = obj_pose[:, 0], obj_pose[:, 1], obj_pose[:, 2]
    w, h = obj_shape[:2]
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


def get_box_corners(poses, obj_shape):
    """Get the corners of the box of given poses"""
    poses = np.atleast_2d(poses)
    n_data = poses.shape[0]
    w, h = obj_shape[:2]

    # Local box corners
    local_corners = np.array(
        [
            [-w / 2, -h / 2, 1.0],
            [-w / 2, h / 2, 1.0],
            [w / 2, -h / 2, 1.0],
            [w / 2, h / 2, 1.0],
        ]
    ).T

    # Transforms
    sin_yaw = np.sin(poses[:, 2])
    cos_yaw = np.cos(poses[:, 2])
    h_matrices = np.tile(np.eye(3)[None], (n_data, 1, 1))
    h_matrices[:, :2, :2] = np.stack(
        (cos_yaw, -sin_yaw, sin_yaw, cos_yaw), axis=1
    ).reshape(-1, 2, 2)
    h_matrices[:, :2, 2] = poses[:, :2]

    # Transform corners
    corners = h_matrices @ local_corners
    corners = corners[:, :2, :].transpose(0, 2, 1)
    return corners


def is_edge_success(poses, obj_shape, edge=0.76, threshold=0.025):
    """Check if the push-to-edge plan is successful"""
    poses = np.atleast_2d(poses)
    xs = poses[:, 0]
    corners = get_box_corners(poses, obj_shape)  # (N, 4, 2)
    max_xs = np.max(corners[:, :, 0], axis=1)
    # check if the object has enough space to be grasped
    success = (max_xs - edge >= threshold) & (xs < edge)
    return success


def is_state_success(poses, goal, goal_region):
    """Check if the state is successful"""
    poses = np.atleast_2d(poses)
    goal = np.asarray(goal)
    goal_region = np.asarray(goal_region)

    # Check if it is in the goal region
    d = poses - goal
    d[:, 2] = (d[:, 2] + np.pi) % (2.0 * np.pi) - np.pi  # wrap to pi
    in_x = (goal_region[0, 0] <= d[:, 0]) & (d[:, 0] <= goal_region[0, 1])
    in_y = (goal_region[1, 0] <= d[:, 1]) & (d[:, 1] <= goal_region[1, 1])
    in_yaw = (goal_region[2, 0] <= d[:, 2]) & (d[:, 2] <= goal_region[2, 1])
    return in_x & in_y & in_yaw


# Plotting
def draw_gradient_circle(ax, center, radius, n_rings=80, cmap="Greens"):
    """
    Approximate a radial gradient by drawing multiple concentric
    semi-transparent circles.
    """
    cm = plt.get_cmap(cmap)
    cx, cy = center
    for i in range(n_rings, 0, -1):
        r = radius * i / n_rings
        color = cm(0.2 + 0.5 * (1 - i / n_rings))
        circ = plt.Circle((cx, cy), r, color=color, alpha=0.35, linewidth=0)
        ax.add_patch(circ)


def plot_states(
    states,
    obstacles=np.array([]),
    planned_states=None,
    obj_shape=None,
    goal=None,
    goal_size=None,
):
    """Plot the states of the object."""
    states = np.array(states)
    if planned_states is not None:
        planned_states = np.array(planned_states)

    plt.figure(figsize=(8, 6))
    # Plot the table as the background
    draw_rectangle(0, -0.505, 1.524, 1.524, 0, "k", alpha=0.1)
    # Plot a robot
    draw_rectangle(0, 0, 0.2, 0.2, 0, "gray", alpha=0.5, label="Robot")
    # plot the goal
    if goal is not None and goal_size is not None:
        draw_rectangle(
            goal[0], goal[1], goal_size[0], goal_size[1], 0, "g", 0.5, "Goal"
        )
    # Plot the obstacles
    if len(obstacles) > 0:
        circle_poses = obstacles[:, :2]
        circle_rads = obstacles[:, 2]
        for circle_pose, circle_rad in zip(circle_poses, circle_rads):
            draw_circle(
                circle_pose[0],
                circle_pose[1],
                circle_rad,
                "r",
                alpha=0.3,
                label="Obstacle",
            )

    # Plot the states path
    if planned_states is not None:
        plt.plot(
            planned_states[:, 0],
            planned_states[:, 1],
            "o-",
            color="b",
            label="Planned Path",
        )
    plt.plot(
        states[:, 0],
        states[:, 1],
        "o-",
        color="g",
        label="Actual Path",
    )
    # If object shape is provided, draw rectangles for start and goal
    if obj_shape is not None:
        w, l = obj_shape[0], obj_shape[1]

        # Draw rectangles
        if planned_states is not None:
            for state in planned_states:
                state_x, state_y, state_theta = state
                draw_rectangle(
                    state_x, state_y, w, l, state_theta, "b", alpha=0.3
                )
        for state in states:
            state_x, state_y, state_theta = state
            draw_rectangle(state_x, state_y, w, l, state_theta, "g", alpha=0.3)

    # Plot start positions
    plt.plot(states[0, 0], states[0, 1], "ro", label="Start")

    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Push Path")
    plt.legend()
    plt.show()


def plot_belief_states(
    beliefs,
    obstacles=np.array([]),
    obj_shape=None,
    goal=None,
    goal_size=None,
    conf=0.95,
):
    """
    Plot belief states for an SE(2) process:
      beliefs: list/array of (T, Sigma) where:
        - T is 3x3 SE(2) homogeneous transform (world pose)
        - Sigma is 3x3 covariance in se(2) tangent [dx, dy, dtheta] at the pose
      goal: (x, y) of goal center
      goal_size: (width, length) of a rectangular goal region
      conf: confidence level for covariance ellipse (2D chi-square region)
      *_kwargs: style overrides for points and ellipses
    """

    # --- Helpers ---
    def to_se2_vec(transform):
        """Convert a SE(2) transform to a 3x1 vector [x, y, theta]."""
        return np.array(
            [
                transform[0, 2],
                transform[1, 2],
                np.arctan2(transform[1, 0], transform[0, 0]),
            ]
        )

    def world_cov_xy(T, Sigma):
        """
        Map the se(2) tangent covariance to world XY for plotting.
        We take the 2x2 translational block and rotate it by R (pose orientation).
        Cross-terms with yaw are ignored for the ellipse.
        """
        R = T[:2, :2]
        Cxy_body = Sigma[:2, :2]
        return R @ Cxy_body @ R.T

    # Defaults
    point_kwargs = dict(marker="o", ms=4, color="g", linestyle="-")
    beliefs = list(beliefs)

    # --- Begin plot ---
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    # Background table
    draw_rectangle(0, -0.505, 1.524, 1.524, 0, "k", alpha=0.1)
    # A robot marker (same as your original)
    draw_rectangle(0, 0, 0.2, 0.2, 0, "gray", alpha=0.5, label="Robot")
    # Goal region
    if goal is not None and goal_size is not None:
        draw_rectangle(
            goal[0], goal[1], goal_size[0], goal_size[1], 0, "g", 0.5, "Goal"
        )
    # Obstacles as circles
    if len(obstacles) > 0:
        circle_poses = obstacles[:, :2]
        circle_rads = obstacles[:, 2]
        for circle_pose, circle_rad in zip(circle_poses, circle_rads):
            draw_circle(
                circle_pose[0],
                circle_pose[1],
                circle_rad,
                "r",
                alpha=0.3,
                label="Obstacle",
            )

    # Extract means for polylines (for visual continuity)
    means = np.array([to_se2_vec(T)[0:2] for T, _ in beliefs])
    ax.plot(
        means[:, 0],
        means[:, 1],
        "o-",
        color=point_kwargs.get("color", "g"),
        label="Actual Path",
    )
    # If object shape is provided, draw rectangles for start and goal
    if obj_shape is not None:
        w, l = obj_shape[0], obj_shape[1]
        means = np.array([to_se2_vec(T) for T, _ in beliefs])
        for state in means:
            state_x, state_y, state_theta = state
            draw_rectangle(
                state_x, state_y, w, l, state_theta, "b", alpha=0.05
            )

    # Actual beliefs
    for T, Sigma in beliefs:
        x, y, _ = to_se2_vec(T)
        ax.plot(
            [x],
            [y],
            marker=point_kwargs.get("marker", "o"),
            ms=point_kwargs.get("ms", 4),
            color=point_kwargs.get("color", "g"),
            linestyle="None",
        )
        Cxy = world_cov_xy(T, Sigma)
        draw_cov_ellipse((x, y), Cxy, conf, "b", alpha=0.2)

    # Start marker
    ax.plot(means[0, 0], means[0, 1], "ro", label="Start")
    ax.grid(True)
    ax.axis("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Belief Path (SE(2) means with XY covariance)")
    ax.legend()
    plt.show()


def draw_circle(x, y, r, color="r", alpha=0.3, label=None):
    """Draw a circle at the given position with the given radius."""
    circle = plt.Circle((x, y), r, color=color, alpha=alpha)
    plt.gca().add_patch(circle)
    if label is not None:
        plt.text(x, y, label, ha="center", va="center", color="black")


def draw_rectangle(
    x, y, width, length, theta, color="b", alpha=0.5, label=None
):
    """Draw a rectangle at the given position with the given orientation."""
    # Calculate the four corners of the rectangle
    corners = np.array(
        [
            [-width / 2, -length / 2],
            [width / 2, -length / 2],
            [width / 2, length / 2],
            [-width / 2, length / 2],
            [-width / 2, -length / 2],  # Close the rectangle
        ]
    )

    # Rotate the corners
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotated_corners = np.dot(corners, rot_matrix.T)
    # Translate the corners
    translated_corners = rotated_corners + np.array([x, y])

    # Plot the rectangle
    plt.plot(
        translated_corners[:, 0],
        translated_corners[:, 1],
        color=color,
        alpha=alpha,
    )
    plt.fill(
        translated_corners[:, 0],
        translated_corners[:, 1],
        color=color,
        alpha=alpha,
    )

    # Add text label
    if label is not None:
        plt.text(x, y, label, ha="center", va="center", color="black")


def draw_cov_ellipse(
    mean, cov, conf_level=0.95, color="b", alpha=0.2, label=None
):
    """
    Draw a confidence ellipse for a 2x2 covariance at mean.
    """
    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0.0)  # numerical safety

    def chi2_quantile_2d(conf_level):
        """
        Chi-square quantile for df=2.
        Common exacts: 0.50→1.3863, 0.6827→2.279, 0.90→4.6052, 0.95→5.9915, 0.99→9.2103
        """
        lookup = {
            0.50: 1.38629436112,
            0.6827: 2.278867,  # ≈1-sigma in 2D
            0.90: 4.605170186,
            0.95: 5.991464547,
            0.99: 9.210340372,
        }
        if conf_level in lookup:
            return lookup[conf_level]
        # Simple monotone fallback around 95% if a custom conf is passed
        return 5.991464547 * (conf_level / 0.95)

    # Scale by chi-square quantile for the chosen confidence
    q = chi2_quantile_2d(conf_level)
    # Ellipse radii are sqrt(eigenvalues * q)
    radii = np.sqrt(vals * q)
    # Angle of the major axis
    angle = np.degrees(
        np.arctan2(vecs[1, 1], vecs[0, 1])
    )  # eigenvector of larger eigenvalue in column 1 after sorting

    # Ensure the first radius is the larger one (matplotlib expects width=2*a, height=2*b)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    e = Ellipse(
        xy=mean,
        width=2 * radii[0],
        height=2 * radii[1],
        angle=angle,
        facecolor=color,
        alpha=alpha,
    )
    plt.gca().add_patch(e)


if __name__ == "__main__":
    plot_states(
        [[0, -0.7, 0]],
        obstacles=np.array(
            [
                [0.0, -0.7, 0.15],
                [-0.4, -1.0, 0.15],
                #
                [0, -0.5, 0.05],
                [-0.25, -0.8, 0.05],
                [0.25, -0.8, 0.05],
            ]
        ),
        planned_states=None,
        # obj_shape=[0.18, 0.22],
        obj_shape=[0.8, 0.6],
    )
