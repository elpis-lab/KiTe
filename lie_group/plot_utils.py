import numpy as np
import matplotlib.pyplot as plt


def plot_results_3d(samples, mean, cov):
    """
    Plot the se(2) tangent space (delta_x, delta_y, omega) distribution.
    Shows both the IEKF tangent covariance (from the filter) and the Monte Carlo
    tangent covariance (obtained by mapping the SE(2) samples via log_se2).
    """
    sample_mean = np.mean(samples, axis=0)
    sample_cov = np.cov(samples.T)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Tangent Space (se2) Samples and Covariances")
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        samples[:, 2],
        s=8,
        alpha=0.2,
        label="MC samples",
    )

    ax.scatter(
        [sample_mean[0]],
        [sample_mean[1]],
        [sample_mean[2]],
        c="g",
        marker="x",
        s=100,
        label="MC mean",
    )
    ax.scatter(
        [mean[0]],
        [mean[1]],
        [mean[2]],
        c="r",
        marker="*",
        s=100,
        label="Predicted mean",
    )

    # IEKF tangent covariance ellipsoid (using iekf.cov directly)
    plot_3d_cov_ellipsoid(ax, mean, cov, n_std=2, color="r", alpha=0.3)
    # Monte Carlo tangent covariance ellipsoid
    plot_3d_cov_ellipsoid(
        ax, sample_mean, sample_cov, n_std=2, color="g", alpha=0.3
    )
    ax.set_xlabel("delta_x")
    ax.set_ylabel("delta_y")
    ax.set_zlabel("omega")
    ax.legend()
    plt.show()


def plot_3d_cov_ellipsoid(
    ax, mu, sigma, n_std=1, resolution=20, color="r", alpha=0.2
):
    """
    Plot a 3D covariance ellipsoid representing the Gaussian N(mu, sigma)
    on the provided 3D axis 'ax'.
    """
    # Create a grid of points for a unit sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere_points = np.stack(
        (x.flatten(), y.flatten(), z.flatten()), axis=0
    )  # shape (3, N)

    # Eigen-decomposition of covariance
    eigvals, eigvecs = np.linalg.eigh(sigma)
    scaling = n_std * np.sqrt(eigvals)  # scale factors along principal axes

    # Transformation matrix: rotate and scale unit sphere points
    transform = eigvecs @ np.diag(scaling)
    ellipsoid_points = transform @ sphere_points
    ellipsoid_points = ellipsoid_points + mu.reshape(3, 1)

    # Reshape back to 2D arrays for plotting
    xx = ellipsoid_points[0, :].reshape((resolution, resolution))
    yy = ellipsoid_points[1, :].reshape((resolution, resolution))
    zz = ellipsoid_points[2, :].reshape((resolution, resolution))
    ax.plot_wireframe(xx, yy, zz, color=color, alpha=alpha)
