import numpy as np

import ompl.base as ob
import ompl.control as oc
import ompl.util as ou

from active_learning.kernel import get_posteriors
from lie_group.lie_se2 import adjoint_se2, log_se2
from lie_group.lie_se2 import to_se2_transform, inv_se2_transform
from lie_group.propagation import mvn_box_cdf
from planning.planning_utils import in_collision_with_circles


class SE2ControlPlanner:
    """SE2 Belief control planner using SST"""

    def __init__(
        self,
        bounds,
        control_bounds,
        obj_shape,
        obstacles,
        model,
        x_train,
        belief,
        active_sampling,
        active_selection,
        controls=None,
    ):
        """Initialize the OMPL SE2 planner"""
        self.belief = belief
        self.c_dim = len(control_bounds)

        # For collision checking
        self.obj_shape = obj_shape
        self.obstacles = np.asarray(obstacles)
        if len(self.obstacles) == 0:
            self.obstacle_poses = []
            self.obstacle_rads = []
        else:
            self.obstacle_poses = self.obstacles[:, :2]
            self.obstacle_rads = self.obstacles[:, 2]

        # Initialize the spaces and set up the planner
        self.space = self.init_state_space(bounds)
        self.control_space = self.init_control_space(control_bounds)
        self.si = ControlSpaceInformation(self.space, self.control_space)
        self.ss = oc.SimpleSetup(self.si)
        self.pdef = self.ss.getProblemDefinition()
        self.set_up_planner(
            model, x_train, belief, active_sampling, active_selection, controls
        )

    def init_state_space(self, bounds):
        """Initialize the state space"""
        # Define space
        space = ob.SE2BeliefStateSpace()

        # Set position bounds
        assert len(bounds) == 2, "Bounds should only include position"
        state_bounds = ob.RealVectorBounds(2)
        state_bounds.setLow(0, bounds[0][0])
        state_bounds.setHigh(0, bounds[0][1])
        state_bounds.setLow(1, bounds[1][0])
        state_bounds.setHigh(1, bounds[1][1])
        space.setBounds(state_bounds)

        return space

    def init_control_space(self, cbounds):
        """Initialize the control space"""
        # Control space will hold both the control and the control effect.
        # Control effect / Delta state is the state change from the control,
        # and has the same dimension as the state.
        c_dim = len(cbounds)
        s_dim = self.space.getDimension()
        control_space = oc.RealVectorControlSpace(self.space, c_dim + s_dim)

        # But the control bounds are for the actual control values
        control_bounds = ob.RealVectorBounds(c_dim + s_dim)
        for i in range(c_dim):
            control_bounds.setLow(i, cbounds[i][0])
            control_bounds.setHigh(i, cbounds[i][1])
        for i in range(c_dim, c_dim + s_dim):
            control_bounds.setLow(i, 0)
            control_bounds.setHigh(i, 0)
        control_space.setBounds(control_bounds)

        return control_space

    def set_up_planner(
        self,
        model,
        x_train,
        belief,
        active_sampling,
        active_selection,
        controls,
    ):
        """Initialize the planner"""
        # State validity checker
        self.set_state_validity_checker_fn(self.is_state_valid)

        # State propagator
        propagator = SE2BeliefPropagator(self.si, model)
        self.set_state_propagator_fn(propagator.propagate)

        # Control sampler
        if active_sampling:
            control_sampler = lambda c_space: ActiveControlBatchSampler(
                self.space, c_space, model, x_train, controls
            )
        else:
            control_sampler = lambda c_space: ControlBatchSampler(
                self.space, c_space, model, x_train, controls
            )
        self.set_control_sampler(control_sampler)

        # Optimization objective
        objective = ob.PathLengthOptimizationObjective(self.si)
        self.set_optimization_objective(objective)
        # Clearance is not helpful somehow, deprecated
        # if len(self.obstacle_poses) == 0:
        #     objective = ob.PathLengthOptimizationObjective(self.si)
        #     self.set_optimization_objective(objective)
        # else:
        #     combo = ob.MultiOptimizationObjective(self.si)
        #     # Path length, with obstacle avoidance
        #     length_obj = ob.PathLengthOptimizationObjective(self.si)
        #     combo.addObjective(length_obj, 1.0)
        #     # Maximize min clearance
        #     clearance_obj = ClearanceObjective(self.si, self.clearance)
        #     combo.addObjective(clearance_obj, 0.05)
        #     self.set_optimization_objective(combo)

        # Planner algorithm
        if active_selection:
            algo = oc.BeliefSST
        else:
            algo = oc.SST
        self.set_planner(algo)
        self.si.setPropagationStepSize(0.5)  # Validity checking interpolation
        self.si.setMinMaxControlDuration(1, 1)

    def get_state(self, state_values):
        """Get a scoped state from the state space"""
        # Get a state from the space
        state = ob.State(self.space)
        # Set the state values
        state().setX(float(state_values[0]))
        state().setY(float(state_values[1]))
        state().setYaw(float(state_values[2]))
        for i in range(6):
            state().setCovariance(i, float(state_values[i + 3]))
        return state

    def plan(
        self,
        start,
        goal,
        goal_ranges,
        planning_time=5,
        accept_approximate=False,
        verbose=True,
    ):
        """Plan to goal"""
        # Set start
        start_state = self.get_state(
            [start[0], start[1], start[2], 1e-6, 0, 0, 1e-6, 0, 1e-6]
        )
        self.ss.setStartState(start_state)

        # Set goal
        goal_state = self.get_state(
            [goal[0], goal[1], goal[2], 1e-6, 0, 0, 1e-6, 0, 1e-6]
        )
        goal_bounds = ob.RealVectorBounds(3)
        for i in range(3):
            goal_bounds.setLow(i, goal_ranges[i][0])
            goal_bounds.setHigh(i, goal_ranges[i][1])
        self.ss.setGoal(ob.SE2GoalState(self.si, goal_state, goal_bounds))

        # Set optimization objective
        # This is equivalent to path length objective if not in belief space
        objective = SE2BeliefOptimizationObjective(
            self.si, goal, goal_ranges, self.belief
        )
        self.set_optimization_objective(objective)

        # Solve
        self.ss.setup()
        solved = self.ss.solve(planning_time)
        # if use exact solution only
        if not accept_approximate:
            while solved.asString() == "Approximate solution":
                if verbose:
                    print(f"Approximate solution, re-planning!")
                self.ss.clear()
                solved = self.ss.solve(planning_time)

        # Extract the path
        if solved:
            if verbose:
                print(f"Planning succeed!")
            path = self.ss.getSolutionPath()
            # Extract the states
            states = []
            for i in range(path.getStateCount()):
                state = path.getState(i)
                s = [state.getX(), state.getY(), state.getYaw()]
                bs = [state.getCovariance(i) for i in range(6)]
                states.append(s + bs)
            # Extract the controls
            controls = []
            for i in range(path.getControlCount()):
                control = path.getControl(i)
                controls.append([control[j] for j in range(self.c_dim)])
            self.ss.clear()
            return states, controls

        else:
            if verbose:
                print(f"Planning failed!")
            self.ss.clear()
            return [], []

    # Setters
    def set_state_validity_checker_fn(self, state_validity_checker_fn):
        """Set the state validity checker"""
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(state_validity_checker_fn)
        )

    def set_planner(self, planner):
        """Set the planner"""
        self.ss.setPlanner(planner(self.si))

    def set_state_propagator_fn(self, state_propagator_fn):
        """Set the state propagator"""
        self.ss.setStatePropagator(oc.StatePropagatorFn(state_propagator_fn))

    def set_control_sampler(self, control_sampler):
        """Set the control sampler"""
        self.control_space.setControlSamplerAllocator(
            oc.ControlSamplerAllocator(control_sampler)
        )

    def set_optimization_objective(self, objective):
        """Set the optimization objective"""
        self.pdef.setOptimizationObjective(objective)

    def is_state_valid(self, state):
        """Check if the state is in the bounds"""
        # In bounds
        in_bounds = self.si.satisfiesBounds(state)
        if not in_bounds:
            return False

        # In collision
        if len(self.obstacle_poses) > 0:
            pose = np.array([state.getX(), state.getY(), state.getYaw()])
            in_collision = in_collision_with_circles(
                pose, self.obj_shape, self.obstacle_poses, self.obstacle_rads
            )
            if in_collision:
                return False

        return True

    def clearance(self, state):
        """Check the clearance to the nearest obstacle"""
        if len(self.obstacle_poses) == 0:
            return 0

        dists = np.linalg.norm(
            np.array([state.getX(), state.getY()]) - self.obstacle_poses,
            axis=1,
        )
        return np.min(dists)


class ControlSpaceInformation(oc.SpaceInformation):
    """
    A control space information that runs collision checking differently.

    Unlike geometric collision checking which uses interpolation,
    regular control collision checking checks intermediate state validity.

    However, the "control" defined here is single-step push action, so
    there is no intermediate states. This class runs interpolation
    to simulate the intermediate state.
    """

    def __init__(self, space, control_space):
        super().__init__(space, control_space)

    def propagateWhileValid(self, state, control, steps, result):
        """
        Override the default validity checking to
        have intermediate collision checking
        """
        # Steps does not matter as long as it is non-zero
        if steps == 0:
            if result != state:
                self.copyState(result, state)
            return 0

        # Step size is considered to be the interpolation resolution
        step_size = abs(self.getPropagationStepSize())
        if step_size > 1.0:
            step_size = 1.0

        # Propagate the state
        self.getStatePropagator().propagate(state, control, 1.0, result)

        # Start to check validity (from step_size * result to 1 * result)
        valid = True
        temp = self.allocState()
        space = self.getStateSpace()
        t = 0
        while t < 1.0:
            t = min(t + step_size, 1.0)
            space.interpolate(state, result, t, temp)
            if not self.isValid(temp):
                valid = False
                break
        self.freeState(temp)

        # Valid
        if valid:
            return steps
        # Invalid
        if result != state:
            self.copyState(result, state)
        return 0


class ClearanceObjective(ob.StateCostIntegralObjective):
    """
    Objective function that maximizes the clearance to the nearest obstacle.
    """

    def __init__(self, si, clearance_fn):
        super().__init__(si, True)
        self.clearance_fn = clearance_fn

    def stateCost(self, state):
        return ob.Cost(1 / self.clearance_fn(state))


class SE2BeliefPropagator(oc.SE2BeliefPropagator):
    """
    See ControlBatchSampler for more details.
    Propagator is now only responsible to propagate given the
    control effect (delta state), but not based on the individual control values.

    The control contains both the control values and control effect (delta state).
    """

    def __init__(self, si, model):
        """Initialize the SE2 belief propagator"""
        super().__init__(si)
        self.space = si.getStateSpace()
        self.model = model
        dim = si.getControlSpace().getDimension()
        s_dim = self.space.getDimension()
        self.c_dim = dim - s_dim

    def propagate(self, state, control, duration, result):
        """Extract the delta state from the control values and propagate"""
        # Create delta state
        delta_state = ob.State(self.space)
        delta_state().setX(float(control[self.c_dim + 0]))
        delta_state().setY(float(control[self.c_dim + 1]))
        delta_state().setYaw(float(control[self.c_dim + 2]))
        for i in range(6):
            delta_state().setCovariance(i, float(control[self.c_dim + 3 + i]))
        self.propagateSE2Belief(state, delta_state, result)


class ControlBatchSampler(oc.ControlSampler):
    """
    Since control propagator uses NN to predict the control effect
    and is much faster to conduct in batch, the control sampler is modified to
    - Sample controls in batch, and
    - Predict control effect in batch (this is previously done in Propagator)
    Propagator is now only responsible to propagate given the
    control effect (delta state), but not based on the individual control values.

    Also, as the control sampler now should pass down delta state,
    the control dimension should be the sum of control and state dimensions.
    But the control bounds are for the actual control,
    so the valid control bounds dimension is different from the total control dimension.
    """

    def __init__(
        self,
        space,
        c_space,
        model,
        x_train=None,
        control_list=None,
        cache_size=10000,
    ):
        """Initialize the control batch sampler"""
        super().__init__(c_space)
        self.dim = c_space.getDimension()
        self.s_dim = space.getDimension()
        self.c_dim = self.dim - self.s_dim

        self.c_space = c_space
        self.model = model
        self.x_train = x_train
        self.c_lows = np.array(self.c_space.getBounds().low)[: self.c_dim]
        self.c_highs = np.array(self.c_space.getBounds().high)[: self.c_dim]

        # if control list provided
        if control_list is not None:
            control_list = np.asarray(control_list)
        self.control_list = control_list

        # store a list of cache to be sampled
        self.cache_size = cache_size
        self.cache_idx = 0
        self.controls = []
        self.d_states = []
        self.total_var = []
        self.sample_cache()

    def sample_cache(self):
        """Sample the cache to get a batch of controls and delta states"""
        self.cache_idx = 0

        # Pick a control from a predefined list
        if self.control_list is not None:
            idx = np.random.randint(0, len(self.control_list), self.cache_size)
            self.controls = self.control_list[idx]

        # Regular sampling
        else:
            self.controls = np.random.uniform(
                self.c_lows,
                self.c_highs,
                size=(self.cache_size, len(self.c_lows)),
            )
            # normalize the rotation value of control
            self.controls[:, 0] = self.controls[:, 0].astype(int) / 4

        # Predict the control effect (in mini-batch)
        batch_size = 2000
        self.d_states = np.zeros((self.cache_size, self.s_dim))
        self.total_var = np.zeros(self.cache_size)
        for i in range(0, self.cache_size, batch_size):
            i_l, i_h = i, i + batch_size

            pred = self.model(self.controls[i_l:i_h])

            # Mean
            self.d_states[i_l:i_h, :3] = pred[:, :3]

            # Variance
            if self.d_states.shape[1] > 3:
                self.d_states[i_l:i_h, [3, 6, 8]] = 1e-3
            # Kernel method (epistemic) if available
            if self.s_dim == 3 or pred.shape[1] == 3:
                if self.x_train is not None:
                    self.total_var[i_l:i_h] = get_posteriors(
                        self.model.model,
                        self.x_train,
                        self.controls[i_l:i_h],
                        sigma=5e-3,
                    )
            # NLL variance prediction
            elif pred.shape[1] == 2 * 3:
                variances = np.exp(pred[:, 3:])
                if self.d_states.shape[1] > 3:
                    self.d_states[i_l:i_h, [3, 6, 8]] = variances
                self.total_var[i_l:i_h] = np.sum(variances, axis=1)
            # Evidential regression prediction
            elif pred.shape[1] == 4 * 3:
                nu, alpha, beta = pred[:, 3:6], pred[:, 6:9], pred[:, 9:]
                # Original std
                # variances = beta / (nu * (alpha - 1.0))
                # Better aleatoric Proxy
                aleatoric = beta * (1 + nu) / (alpha * nu)
                epistemic = 1 / nu
                total_var = aleatoric + epistemic
                if self.d_states.shape[1] > 3:
                    self.d_states[i_l:i_h, [3, 6, 8]] = aleatoric
                self.total_var[i_l:i_h] = np.sum(aleatoric, axis=1)
            else:
                raise ValueError(
                    f"Invalid model prediction dimension: {pred.shape[1]}"
                )

    def sample(self, control):
        """Take a control from the cache"""
        # Pick a control randomly from the predefined list if given
        c = self.controls[self.cache_idx]
        s = self.d_states[self.cache_idx]
        for i in range(self.c_dim):
            control[i] = float(c[i])
        for i in range(self.s_dim):
            control[i + self.c_dim] = float(s[i])

        # Update the cache index
        self.cache_idx += 1
        if self.cache_idx >= self.cache_size:
            self.sample_cache()


class ActiveControlBatchSampler(ControlBatchSampler):
    """
    Since control propagator uses NN to predict the control effect
    and is much faster to conduct in batch, the control sampler is modified to
    - Sample controls in batch, and
    - Predict control effect in batch (this is previously done in Propagator)
    Propagator is now only responsible to propagate given the
    control effect (delta state), but not based on the individual control values.

    Also, as the control sampler now should pass down delta state,
    the control dimension should be the sum of control and state dimensions.
    But the control bounds are for the actual control,
    so the valid control bounds dimension is different from the total control dimension.
    """

    def __init__(
        self,
        space,
        c_space,
        model,
        x_train,
        control_list=None,
        pool_size=5,
        cache_size=10000,
    ):
        """Initialize the active batch control sampler"""
        self.n_pool = cache_size
        self.pool_size = pool_size
        super().__init__(
            space,
            c_space,
            model,
            x_train,
            control_list,
            pool_size * cache_size,
        )

    def sample_cache(self):
        """
        Sample the cache to get a batch of controls and delta states

        Different from the parent class, the cache is reshaped to make
        controls and d_states a list of control pools instead single control.
        Then we bias the control selection with the variance.
        """
        super().sample_cache()

        # Reshape to make controls and d_states
        # a list of control pools instead single control
        self.controls = self.controls.reshape(self.n_pool, self.pool_size, -1)
        self.d_states = self.d_states.reshape(self.n_pool, self.pool_size, -1)
        self.total_var = self.total_var.reshape(self.n_pool, self.pool_size)

        # Option 1 - Given the variances, assign weights to each control
        # then sample from the pool with the weights
        # weights = 1.0 / (self.total_var + 1e-6)
        # weights = weights / weights.sum(axis=1, keepdims=True)
        # u = np.random.uniform(size=self.n_pool)
        # cdf = np.cumsum(weights, axis=1)
        # indices = (u[:, None] <= cdf).argmax(axis=1)
        # Option 2 - Simply choose the control with the lowest variance
        indices = np.argmin(self.total_var, axis=1)

        # Select the best control and corresponding delta state
        rows = np.arange(self.n_pool)
        self.controls = self.controls[rows, indices]
        self.d_states = self.d_states[rows, indices]

    def sample(self, control):
        """Take a control from the cache"""
        super().sample(control)
        if self.cache_idx >= self.n_pool:
            self.sample_cache()


class SE2BeliefOptimizationObjective(ob.PathLengthOptimizationObjective):
    """SE2 optimization objective in belief space

    The accumulated motion cost is computed with Wasserstein2 distance.
    The state cost is computed with negative log likelihood of the state
    being in the goal region.
    """

    def __init__(self, si, goal, goal_ranges, wasserstein=True, max_nll=10.0):
        """Initialize the optimization objective"""
        super().__init__(si)
        self.si = si
        self.goal = to_se2_transform(goal)
        self.goal_ranges = np.asarray(goal_ranges)
        self.wasserstein = wasserstein

        # a value to prevent nll from being too large
        self.min_prob = np.exp(-max_nll)
        # SE2 ranges, to convert to tangent space ranges
        self.l_ranges, self.h_ranges = self.to_tangent_ranges(self.goal_ranges)

    def to_tangent_ranges(self, goal_ranges):
        """Convert SE2 ranges to tangent space ranges
        The tangent space ranges uses conservative inner bound
        """
        # Build grid
        x_l, y_l, yaw_l = goal_ranges[:, 0]
        x_h, y_h, yaw_h = goal_ranges[:, 1]
        grid = np.array([[x_l, y_l], [x_l, y_h], [x_h, y_l], [x_h, y_h]])
        # Build Axis-aligned tangent boxes - Sample yaw angles
        yaw_samples = np.linspace(yaw_l, yaw_h, 9)
        x_list, y_list = [], []
        for yaw in yaw_samples:
            jac_inv = self.jac_inv_se2(yaw)
            pts = (jac_inv @ grid.T).T
            x_list.append(np.max(np.abs(pts[:, 0])))
            y_list.append(np.max(np.abs(pts[:, 1])))
        # Inner = intersection across yaw -> min half-extent
        x_min = np.min(x_list)
        y_min = np.min(y_list)

        # Tangent space ranges
        lower = np.array([-x_min, -y_min, yaw_l])
        upper = np.array([x_min, y_min, yaw_h])
        return lower, upper

    def motionCost(self, s1, s2):
        """Compute the cost of the motion from s1 to s2"""
        # Euclidean distance
        s1_x, s1_y, s1_yaw = s1.getX(), s1.getY(), s1.getYaw()
        s2_x, s2_y, s2_yaw = s2.getX(), s2.getY(), s2.getYaw()
        t1 = to_se2_transform([s1_x, s1_y, s1_yaw])
        t2 = to_se2_transform([s2_x, s2_y, s2_yaw])
        t_delta = inv_se2_transform(t1) @ t2
        dist = np.linalg.norm(log_se2(t_delta))
        # OMPL distance
        # dist = ((s1_x - s2_x) ** 2 + (s1_y - s2_y) ** 2) ** 0.5
        # dist += 0.5 * abs(angle_diff(s1_yaw, s2_yaw))

        # Belief distance
        if self.wasserstein:
            cov1 = self.vec_to_cov([s1.getCovariance(i) for i in range(6)])
            cov2 = self.vec_to_cov([s2.getCovariance(i) for i in range(6)])
            dist += self.wasserstein_distance(cov1, cov2)
        return ob.Cost(dist)

    def stateCost(self, state):
        """
        Compute the negative log likelihood of the state
        being in the goal region
        """
        # State
        x, y, yaw = state.getX(), state.getY(), state.getYaw()
        mean = to_se2_transform([x, y, yaw])
        var_state = [state.getCovariance(i) for i in range(6)]
        cov = self.vec_to_cov(var_state)

        # Compute mu and cov in the goal frame in tangent space
        rel_transofrm = inv_se2_transform(self.goal) @ mean
        rel_ad = adjoint_se2(rel_transofrm)
        rel_mu = log_se2(rel_transofrm)
        rel_cov = rel_ad @ cov @ rel_ad.T

        # Theoretical probability - Success rate
        prob = mvn_box_cdf(self.l_ranges, self.h_ranges, rel_mu, rel_cov)
        nll = -np.log(max(prob, self.min_prob))

        return ob.Cost(nll)

    @staticmethod
    def wasserstein_distance(cov1, cov2):
        """
        Compute the Wasserstein2 distance between two covariance matrices
        dist = trace(cov1 + cov2 - 2 * sqrt(sqrt(cov1) @ cov2 @ sqrt(cov1)))
        optimized version:
        dist = trace(cov1) + trace(cov2)
             - 2 * Sum(Sqrt(Eigenvalues( sqrt(cov1) @ cov2 @ sqrt(cov1) )))
        """
        sqrt_cov1 = SE2BeliefOptimizationObjective.sqrtm_spd(cov1)
        eigenvalues = np.linalg.eigvals(sqrt_cov1 @ cov2 @ sqrt_cov1)
        sqrt_tr = np.sum(np.sqrt(eigenvalues))
        return np.trace(cov1) + np.trace(cov2) - 2 * sqrt_tr

    @staticmethod
    def sqrtm_spd(matrix):
        """Fast matrix square root for SPD matrices using eigendecomposition"""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    @staticmethod
    def vec_to_cov(vector):
        """Convert vector to covariance matrix"""
        return np.array(
            [
                [vector[0], vector[1], vector[2]],
                [vector[1], vector[3], vector[4]],
                [vector[2], vector[4], vector[5]],
            ]
        )

    @staticmethod
    def jac_inv_se2(w):
        """Convert yaw to Jacobian inverse"""
        if abs(w) < 1e-12:
            return np.eye(2)
        alpha = w / (2.0 * np.sin(w / 2.0))
        c = np.cos(w / 2.0)
        s = np.sin(w / 2.0)
        return alpha * np.array([[c, s], [-s, c]])


def set_ompl_seed(seed):
    """Set the seed for the OMPL random number generator"""
    ou.RNG.setSeed(seed)
