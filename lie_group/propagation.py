import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import multivariate_normal
from itertools import product

from lie_group.lie_se2 import exp_se2, log_se2, right_jacobian_se2, adjoint_se2
from lie_group.lie_se2 import to_se2_vec, to_se2_transform, inv_se2_transform
from lie_group.plot_utils import plot_results_3d


def se2_error(s1, s2, weight=None):
    """Get the delta error for SE2 Pose."""
    t1 = to_se2_transform(s1)
    t2 = to_se2_transform(s2)
    err_mat = inv_se2_transform(t1) @ t2
    delta = log_se2(err_mat)  # (N, 3)

    if weight is None:
        return delta
    elif isinstance(weight, np.ndarray):
        if weight.shape == (3, 3):
            return weight @ delta
        elif len(weight) == 3:
            return np.diag(weight) @ delta
    elif isinstance(weight, float):
        return np.diag([1, 1, weight]) @ delta
    elif isinstance(weight, list):
        return np.diag(weight) @ delta

    else:
        raise ValueError(f"Invalid weight type: {type(weight)}")


def propagate_cov(delta, cov, delta_cov):
    """Propagate the covariance of the SE(2) state."""
    # action covariance (right Jacobian(delta))
    jac_r = right_jacobian_se2(delta)
    # state covariance (inverse of Adjoint(delta))
    adj = adjoint_se2(exp_se2(-delta))
    # Propagate error covariance
    cov = adj @ cov @ adj.T + jac_r @ delta_cov @ jac_r.T
    return cov


class Propagation_SE2:
    """
    A class that carries out only the 'prediction' step in regular IEKF,
    Used to propagate a prediction and uncertainty of a SE(2) state.

    Attributes:
        mean: 3x3 matrix as SE(2) state
        cov:  3x3 covariance in the tangent space
    """

    def __init__(self, init_mean=np.eye(3), init_cov=np.zeros((3, 3))):
        """Initialize the SE(2) state and covariance."""
        self.mean = init_mean  # SE2 state transform
        self.cov = init_cov  # SE2 tangent space covariance

    def propagate(self, deltas, delta_covs):
        """
        Given a list of tangent space motions and their covariances,
        update the mean and covariance of the SE(2) state.
        """
        for delta, delta_cov in zip(deltas, delta_covs):
            self.propagate_step(delta, delta_cov)

    def propagate_step(self, delta, delta_cov):
        """
        Given the tangent space motion delta and tangent space noise Q,
        update the mean and covariance of the SE(2) state.
        """
        # Propagate mean
        # apply the tangent space motion to the current state
        t_delta = exp_se2(delta)
        self.mean = self.mean @ t_delta

        # Propagate covariance
        self.cov = propagate_cov(delta, self.cov, delta_cov)

    def get_end_state(self):
        """Get the (x, y, theta) from the SE(2) state with se(2) covariance."""
        return to_se2_vec(self.mean), self.cov


def mvn_box_cdf(lower, upper, mean, cov):
    """
    Computes P(lower <= X <= upper) for multivariate normal X ~ N(mean, cov)
    using inclusion-exclusion over all corners.
    """
    dim = len(mean)
    mvn = multivariate_normal(mean=mean, cov=cov)

    total = 0.0
    for signs in product([0, 1], repeat=dim):
        point = np.where(np.array(signs) == 1, upper, lower)
        sign = (-1) ** sum(1 - np.array(signs))  # inclusion-exclusion
        total += sign * mvn.cdf(point)

    return total


def to_tangent_ranges(ranges):
    """Convert SE2 ranges to tangent space ranges
    The tangent space ranges uses conservative inner bound
    """
    # Build grid
    x_l, y_l, yaw_l = ranges[:, 0]
    x_h, y_h, yaw_h = ranges[:, 1]
    yaw_l, yaw_h = wrap_to_pi(yaw_l), wrap_to_pi(yaw_h)
    if (yaw_h - yaw_l) > np.pi:
        yaw_l, yaw_h = yaw_h - 2 * np.pi, yaw_l
    grid = np.array([[x_l, y_l], [x_l, y_h], [x_h, y_l], [x_h, y_h]])

    # Build Axis-aligned tangent boxes - Sample yaw angles
    yaw_samples = np.linspace(yaw_l, yaw_h, 21)
    x_list, y_list = [], []
    for yaw in yaw_samples:
        jac_inv = jac_inv_se2(yaw)
        pts = (jac_inv @ grid.T).T
        x_list.append(np.max(np.abs(pts[:, 0])))
        y_list.append(np.max(np.abs(pts[:, 1])))
    # Inner = intersection across yaw -> min half-extent
    x_min = np.min(x_list)
    y_min = np.min(y_list)

    # Tangent space ranges
    lower = np.array([-x_min, -y_min, yaw_l])
    upper = np.array([x_min, y_min, yaw_h])
    return np.array([lower, upper]).T


def jac_inv_se2(w):
    """Convert yaw to Jacobian inverse"""
    if abs(w) < 1e-6:
        return np.eye(2)
    alpha = w / (2.0 * np.sin(w / 2.0))
    c = np.cos(w / 2.0)
    s = np.sin(w / 2.0)
    return alpha * np.array([[c, s], [-s, c]])


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    ########## Test Propagation_SE2 ##########
    # initial state
    t0 = to_se2_transform((1.0, 1.0, np.deg2rad(60)))
    q0 = 1e-8 * np.eye(3)
    # define motions - n same steps
    n_steps = 5
    delta = np.array([0.1, 0.1, np.deg2rad(45)])
    delta_cov = np.diag([0.01, 0.01, np.deg2rad(10) ** 2])
    deltas = np.repeat(delta[np.newaxis, :], n_steps, axis=0)
    delta_covs = np.repeat(delta_cov[np.newaxis, :, :], n_steps, axis=0)

    # Use propagation theory
    prop = Propagation_SE2(t0, q0)
    prop.propagate(deltas, delta_covs)
    mean, cov = prop.get_end_state()
    mean = prop.mean  # transform

    # Use Monte Carlo
    n_samples = 10000
    motions = np.random.multivariate_normal(
        delta, delta_cov, size=(n_samples, n_steps)
    )
    transform_samples = [t0] * n_samples
    for n in range(n_steps):
        transform_samples = [
            transform_samples[i] @ exp_se2(motions[i, n])
            for i in range(n_samples)
        ]
    # get error in tangent space w.r.t. to predicted mean
    mean_inv = inv_se2_transform(mean)
    sample_errors = np.array(
        [log_se2(mean_inv @ T) for T in transform_samples]
    )

    # Plot to see if the error matches the predicted cov
    plot_results_3d(sample_errors, np.zeros(3), cov)

    ########## Test Probability ##########
    # define region
    region = to_se2_transform((1, 1, -1.0))

    se2_lower = np.array([-0.4, -0.4, -0.4])
    se2_upper = np.array([0.4, 0.4, 0.4])
    t_ranges = to_tangent_ranges(np.array([se2_lower, se2_upper]).T)
    lower, upper = t_ranges[:, 0], t_ranges[:, 1]

    # Use probabilistic theory
    # express error in region's frame
    rel_transform = inv_se2_transform(region) @ mean
    rel_ad = adjoint_se2(rel_transform)
    rel_mu = log_se2(rel_transform)
    rel_cov = rel_ad @ cov @ rel_ad.T

    # Box CDF probability
    prob = mvn_box_cdf(lower, upper, rel_mu, rel_cov)

    # Use Monte Carlo
    region_inv = inv_se2_transform(region)
    rel_transforms = [region_inv @ tr for tr in transform_samples]
    rel_ts = np.array([log_se2(tr) for tr in rel_transforms])

    t_count = 0
    for t in rel_ts:
        if (
            lower[0] < t[0] < upper[0]
            and lower[1] < t[1] < upper[1]
            and lower[2] < t[2] < upper[2]
        ):
            t_count += 1

    s_count = 0
    rel_vecs = np.array([to_se2_vec(tr) for tr in rel_transforms])
    rel_vecs[:, 2] = wrap_to_pi(rel_vecs[:, 2])
    for rel_vec in rel_vecs:
        if (
            se2_lower[0] < rel_vec[0] < se2_upper[0]
            and se2_lower[1] < rel_vec[1] < se2_upper[1]
            and se2_lower[2] < rel_vec[2] < se2_upper[2]
        ):
            s_count += 1

    # Check results
    print(
        f"Theory: {prob:.4f}"
        + f"\nMC(Tangent): {t_count / n_samples:.4f}"
        + f"\nMC(SE2): {s_count / n_samples:.4f}"
    )
    plot_results_3d(rel_ts, rel_mu, rel_cov)
