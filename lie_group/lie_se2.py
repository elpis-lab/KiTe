import numpy as np


########## Lie Operations ##########
def log_se2(transform):
    """
    Logarithm map from SE(2) -> se(2).
    'transform' is a 3x3 homogeneous transformation matrix.
    Returns a 3x1 vector [delta_x, delta_y, omage] in tangent space.

    omega = arctan2(rot[1, 0], rot[0, 0])
    [dx, dy] = V(w)^-1 * t
    """
    rot = transform[0:2, 0:2]
    t = transform[0:2, 2]

    # Extract rotation angle directly
    # The trace of R = 2 cos(omega), so omega = atan2(R[1,0], R[0,0])
    omega = np.arctan2(rot[1, 0], rot[0, 0])

    # If rotation is very small, approximate to avoid numerical issues
    eps = 1e-3
    if abs(omega) < eps:
        # Both R and V are close to identity, so V^-1[delta_x, delta_y] = t
        dx, dy = t

    else:
        # For SE(2), we can use the closed-form formula for V(w)^-1
        # V_inv = omega / (2 * (1 - cos(w))) *
        #         [[    sin(w)   , 1 - cos(w)]
        #          [-(1 - cos(w)),   sin(w)  ]]
        c = np.cos(omega)
        s = np.sin(omega)
        V_inv = (omega / (2 * (1 - c))) * np.array([[s, 1 - c], [c - 1, s]])
        dx, dy = V_inv @ t

    return np.array([dx, dy, omega])


def exp_se2(xi):
    """
    Exponential map from se(2) -> SE(2).
    'xi' is a 3x1 vector [delta_x, delta_y, omega] in the tangent space.
    Returns a 3x3 homogeneous transformation matrix.

    exp([v; w]) = [ R(w) , J(w)*v ]
                  [   0  ,   1    ]
    """
    dx, dy, omega = xi

    # If rotation is close to zero, use approximation
    eps = 1e-3
    if abs(omega) < eps:
        rot = np.eye(2)
        t = np.array([dx, dy])

    else:
        c = np.cos(omega)
        s = np.sin(omega)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        # R(omega) = (sin(omega)/omega) * I + (1-cos(omega))/omega^2 * [omega]
        #          = (sin(omega)/omega) * [1, 0; 0, 1]
        #          + (1-cos(omega))/omega^2 * [0, -omega; omega, 0]
        rot_omega = np.array(
            [[s / omega, (c - 1) / omega], [(1 - c) / omega, s / omega]]
        )
        t = rot_omega @ np.array([dx, dy])

    transform = np.eye(3)
    transform[0:2, 0:2] = rot
    transform[0:2, 2] = t
    return transform


def adjoint_se2(transform):
    """
    Returns the 3x3 adjoint matrix Ad_T for T in SE(2).
    If T = [R  t]
           [0  1],
    then Ad_T = [[ R,   [t]R ],
                 [ 0,    1   ]]
    """
    rot = transform[0:2, 0:2]
    t = transform[0:2, 2]

    # [t]R in SE2 is simply -[1]t = [t1; -t0]
    t_cross_rot = np.array([t[1], -t[0]]).reshape((2, 1))
    adjoint = np.block([[rot, t_cross_rot], [np.zeros((1, 2)), 1]])
    return adjoint


def right_jacobian_se2(xi):
    """
    Returns the 3x3 right Jacobian matrix J for T in SE(2).
    J(xi) = [[ J(omega), A(xi) ],
             [    0    ,  1    ]]
    """
    dx, dy, omega = xi

    # If rotation is close to zero, use approximation
    eps = 1e-3
    if abs(omega) < eps:
        # Use 1st-order Taylor expansion for small angles
        jac_omega = np.eye(2)
        a_xi = np.array([-dy / 2.0, dx / 2.0])

    else:
        c = np.cos(omega)
        s = np.sin(omega)
        omega_sq = omega * omega

        # J(omega) = [[sin(omega) / omega, (1 - cos(omega)) / omega],
        #             [(cos(omega) - 1) / omega, sin(omega) / omega]]
        jac_omega = np.array(
            [[s / omega, (1 - c) / omega], [(c - 1) / omega, s / omega]]
        )
        # A(xi) = 1 / omega^2 * [
        #     dx(omega - sin(omega)) + dy(cos(omega) - 1),
        #     dy(omega - sin(omega)) - dx(1 - cos(omega))
        # ]
        a1 = (dx * (omega - s) + dy * (c - 1)) / omega_sq
        a2 = (dy * (omega - s) + dx * (1 - c)) / omega_sq
        a_xi = np.array([a1, a2])

    jac = np.eye(3)
    jac[0:2, 0:2] = jac_omega
    jac[0:2, 2] = a_xi
    return jac


########## Transformations ##########
def inv_se2_transform(transform):
    """Inverse of the SE(2) transform."""
    rot = transform[:2, :2]
    t = transform[:2, 2]

    rot_inv = rot.T
    t_inv = -rot_inv @ t

    inv_transform = np.eye(3)
    inv_transform[:2, :2] = rot_inv
    inv_transform[:2, 2] = t_inv
    return inv_transform


def to_se2_vec(transform):
    """
    Extract the (x, y, theta) from the SE(2) state.
    """
    x = transform[0, 2]
    y = transform[1, 2]
    theta = np.arctan2(transform[1, 0], transform[0, 0])
    return np.array([x, y, theta])


def to_se2_transform(params):
    """
    Get the 3x3 homogeneous transformation matrix from params (x, y, theta).
    """
    c = np.cos(params[2])
    s = np.sin(params[2])
    return np.array([[c, -s, params[0]], [s, c, params[1]], [0, 0, 1]])


########## Statistics ##########
def se2_stats(se2_poses, alpha=0.5, tol=1e-3, max_iters=100):
    """Compute Karcher mean and variances of a list of SE2 poses."""
    # Initialize
    transforms = np.stack([to_se2_transform(p) for p in se2_poses], axis=0)

    # Compute Karcher mean
    t_mean = transforms[0]
    deltas = np.zeros((transforms.shape[0], 3))
    for i in range(max_iters):

        # get tangent error given the current mean
        inv_mean = inv_se2_transform(t_mean)[None, :, :]
        t_rel = inv_mean @ transforms  # T_mean⁻¹ T_i
        deltas = np.stack([log_se2(t) for t in t_rel], axis=0)  # in R³

        # get mean tangent vector
        v = deltas.mean(axis=0)
        if np.linalg.norm(v) < tol:
            break

        # update mean on the manifold
        t_mean = t_mean @ exp_se2(alpha * v)

    # Get mean and vairances
    mean = to_se2_vec(t_mean)
    std = deltas.std(axis=0, ddof=1)
    return mean, std
