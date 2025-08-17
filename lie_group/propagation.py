import numpy as np
from scipy.stats import multivariate_normal
from itertools import product

from lie_group.lie_se2 import exp_se2, log_se2, right_jacobian_se2, adjoint_se2
from lie_group.lie_se2 import to_se2_vec, to_se2_transform, inv_se2_transform
from lie_group.plot_utils import plot_results_3d


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
        # action covariance (right Jacobian(delta))
        jac_r = right_jacobian_se2(delta)
        # state covariance (inverse of Adjoint(delta))
        adj = adjoint_se2(exp_se2(-delta))
        # Propagate error covariance
        self.cov = adj @ self.cov @ adj.T + jac_r @ delta_cov @ jac_r.T

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
    n_samples = 1000
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
    region_range = np.array([[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]])

    # Use probabilistic theory
    # express error in region's frame
    rel_transofrm = inv_se2_transform(region) @ mean
    rel_ad = adjoint_se2(rel_transofrm)
    rel_mu = log_se2(rel_transofrm)
    rel_cov = rel_ad @ cov @ rel_ad.T
    lower = region_range[:, 0]
    upper = region_range[:, 1]
    prob = mvn_box_cdf(lower, upper, rel_mu, rel_cov)

    # Use Monte Carlo
    region_inv = inv_se2_transform(region)
    rel_ts = np.array(
        [log_se2(region_inv @ trans) for trans in transform_samples]
    )
    count = 0
    for t in rel_ts:
        if (
            lower[0] < t[0] < upper[0]
            and lower[1] < t[1] < upper[1]
            and lower[2] < t[2] < upper[2]
        ):
            count += 1

    # Check results
    print("Theory: ", prob, "Monte Carlo: ", count / n_samples)
    plot_results_3d(rel_ts, rel_mu, rel_cov)
