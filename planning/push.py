import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

import ompl.base as ob
import ompl.control as oc
import ompl.util as ou

from geometry.pose import angle_diff
from active_learning.kernel import get_posteriors
from lie_group.lie_se2 import adjoint_se2, log_se2
from lie_group.lie_se2 import to_se2_transform, inv_se2_transform

from planning.planning_utils import rects_circles_in_collision
from planning.planning_utils import draw_rect, draw_circle, draw_cov_ellipse
from planning.planning_utils import draw_gradient_circle

########## Pushing Environment ##########
POS_RANGES = ((-0.4, 0.4), (-1.0, -0.4))
OBSTACLES = [(0, -0.5, 0.05), (-0.25, -0.8, 0.05), (0.25, -0.8, 0.05)]


def get_random_se2_states(
    n_data, pos_range=POS_RANGES, euler_range=(-np.pi, np.pi)
):
    """Generate random initial states"""
    pos_x = np.random.uniform(pos_range[0][0], pos_range[0][1], (n_data, 1))
    pos_y = np.random.uniform(pos_range[1][0], pos_range[1][1], (n_data, 1))
    euler = np.random.uniform(euler_range[0], euler_range[1], (n_data, 1))
    states = np.concatenate([pos_x, pos_y, euler], axis=-1)
    return states


def generate_push_env(obstacles=OBSTACLES, safe_start_range=0.15):
    """Generate initial states"""
    # Obstacles are circles
    obstacles = np.asarray(obstacles)
    # Find a valid random start
    while True:
        # generate a random state
        start = get_random_se2_states(1)[0]
        # check clearance
        dists = np.linalg.norm(start[:2] - obstacles[:, :2], axis=-1)
        clearance = dists - (obstacles[:, 2] + safe_start_range)
        if clearance.min() > 0.0:
            break

    # Fixed Goal (center of the table)
    goal_center = np.array([0, -0.7, 0])
    goal_size = 0.05

    return {
        "start": start,
        "goal": goal_center,
        "goal_size": goal_size,
        "obstacles": obstacles,
    }


########## Visualization ##########
def visualize_push_env(
    env,
    path=None,
    exec_path=None,
    obj_shape=None,
    title="Object Push Environment",
):
    """Visualize Object Push Environment"""
    (xmin, xmax), (ymin, ymax) = POS_RANGES
    tx, ty = (xmin + xmax) / 2, (ymin + ymax) / 2
    tw, th = (xmax - xmin), (ymax - ymin)

    start, obstacles = env["start"], env["obstacles"]
    goal, goal_size = env["goal"], env["goal_size"]
    if path is not None:
        path = np.asarray(path, dtype=float)
    if exec_path is not None:
        exec_path = np.asarray(exec_path)
    if obj_shape is not None:
        w, h = obj_shape[0], obj_shape[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    # Background table
    # draw_rect(ax, 0, -0.505, 1.524, 1.524, 0, "k", alpha=0.1)
    draw_rect(ax, tx, ty, tw, th, 0, "k", alpha=0.1)
    # # Robot
    # draw_rect(ax, 0, 0, 0.2, 0.2, 0, "gray", alpha=0.5, label="Robot")

    # Goal
    draw_gradient_circle(ax, goal[:2], goal_size, label="Goal Region")
    # Obstacles
    if len(obstacles) > 0:
        obstacles = np.asarray(obstacles)
        circle_poses = obstacles[:, :2]
        circle_rads = obstacles[:, 2]
        first = True
        for (ox, oy), r in zip(circle_poses, circle_rads):
            label = "Obstacle" if first else None
            draw_circle(ax, ox, oy, r, "k", 1.0, label=label)
            first = False

    # Planned path (x, y)
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], "o-", color="b", label="Planned Path")
        if obj_shape is not None:
            for state in path:
                x, y, yaw = state[:3]
                draw_rect(ax, x, y, w, h, yaw, "b", alpha=0.3)

        # if covariance provided, plot belief path
        def world_cov_xy(yaw, cov):
            """Local covariance to world covariance"""
            c, s = np.cos(yaw), np.sin(yaw)
            rot = np.array([[c, -s], [s, c]])
            cov_body = np.asarray(cov)[:2, :2]
            return rot @ cov_body @ rot.T

        if path.shape[1] > 3:
            for i, state in enumerate(path):
                x, y, yaw = state[:3]
                cov_local = world_cov_xy(
                    yaw, SE2PushOptimizationObjective.vec_to_cov(state[3:])
                )
                label = "Belief" if i == 0 else None
                draw_cov_ellipse(
                    ax, (x, y), cov_local, 1, "b", 0.2, label=label
                )

    # Executed path (x, y)
    if exec_path is not None:
        ax.plot(
            exec_path[:, 0],
            exec_path[:, 1],
            "o-",
            color="g",
            label="Actual Path",
        )
        if obj_shape is not None:
            for state in exec_path:
                x, y, yaw = state[:3]
                draw_rect(ax, x, y, w, h, yaw, "g", alpha=0.3)

    # Start
    draw_circle(ax, start[0], start[1], 0.01, "C9", label="Start")

    ax.set_axisbelow(True)
    ax.grid(True, zorder=0)
    ax.set_aspect("equal", adjustable="box")
    # limit to pos ranges left and right
    plot_size = max(0.5 * tw + 0.1, 0.5 * th + 0.1)
    ax.set_xlim(tx - plot_size, tx + plot_size)
    ax.set_ylim(ty - plot_size, ty + plot_size)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.legend()
    return fig, ax


########## OMPL Main ##########
class SE2PushPlanner:
    """Planner push planner class"""

    def __init__(
        self,
        obstacles,
        obj_shape,
        model,  # dynamics model
        x_train=None,  # use active sampling
        belief=True,  # use Wasserstein distance
        algo="aorrt",  # planner algorithm
        terminal_weight=2.0,  # terminal weight (only for aorrt)
        controls=None,  # control set for sampling
    ):
        """Initialize the OMPL SE2 planner"""
        bounds = POS_RANGES
        control_bounds = ((0, 4), (-0.4, 0.4), (0.0, 0.3))
        self.c_dim = len(control_bounds)
        self.belief = belief
        self.terminal_weight = terminal_weight

        # For collision checking
        self.obj_shape = obj_shape
        self.obstacles = np.asarray(obstacles)

        # Initialize the spaces and set up the planner
        self.space = self.init_state_space(bounds)
        self.control_space = self.init_control_space(control_bounds)
        self.si = ControlSpaceInformation(self.space, self.control_space, 0.5)
        self.ss = oc.SimpleSetup(self.si)
        self.pdef = self.ss.getProblemDefinition()
        self.set_up_planner(model, x_train, belief, algo, controls)

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
        # Control effect (Delta state) is the state change from the control,
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
        algo,
        controls,
    ):
        """Initialize the planner"""
        # State validity checker
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(self.is_state_valid)
        )

        # State propagator
        propagator = SE2BeliefPropagator(self.si, model)
        self.ss.setStatePropagator(oc.StatePropagatorFn(propagator.propagate))

        # Control sampler
        # Active sampling with epistemic uncertainty if given training data
        if x_train is not None:
            control_sampler = lambda c_space: ActiveControlBatchSampler(
                self.space, c_space, model, belief, x_train, controls
            )
        else:
            control_sampler = lambda c_space: ControlBatchSampler(
                self.space, c_space, model, controls
            )
        self.control_space.setControlSamplerAllocator(
            oc.ControlSamplerAllocator(control_sampler)
        )

        # Optimization objective (set later with goal)
        # obj = SE2PushOptimizationObjective(
        #     self.si, goal, self.belief, self.terminal_weight
        # )
        # self.pdef.setOptimizationObjective(obj)

        # Planner algorithm
        if algo == "aorrt":
            planner = oc.AORRT(self.si)
        else:
            planner = oc.SST(self.si)
            # planner.setPruningRadius(0.03)
        self.ss.setPlanner(planner)
        self.si.setPropagationStepSize(1.0)
        self.si.setMinMaxControlDuration(1, 1)

    def get_belief_state(self, state, cov=(1e-6, 0, 0, 1e-6, 0, 1e-6)):
        """Get a scoped state from the state space"""
        # Get a state from the space
        s = ob.State(self.space)
        # Set the state values
        s().setX(float(state[0]))
        s().setY(float(state[1]))
        s().setYaw(float(state[2]))
        for i in range(6):
            s().setCovariance(i, float(cov[i]))
        return s

    # Validity checkers
    def is_state_valid(self, state):
        """Check if the state is in the bounds"""
        # In bounds
        in_bounds = self.si.satisfiesBounds(state)
        if not in_bounds:
            return False

        # In collision
        if len(self.obstacles) > 0:
            pose = np.array([state.getX(), state.getY(), state.getYaw()])
            in_collision = rects_circles_in_collision(
                pose, self.obj_shape, self.obstacles
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

    # Plan
    def plan(
        self,
        start,
        goal,
        goal_size,
        planning_times=(1.0, 5.0),
        verbose=True,
    ):
        """Plan to goal"""
        # Set start
        start_state = self.get_belief_state([start[0], start[1], start[2]])
        self.ss.setStartState(start_state)

        # Set goal
        self.ss.setGoal(SE2PushGoal(self.si, goal, goal_size))
        # Set optimization objective
        self.obj = SE2PushOptimizationObjective(
            self.si, goal, self.belief, self.terminal_weight
        )
        self.pdef.setOptimizationObjective(self.obj)

        # Set up the planner
        self.ss.setup()

        # Incremental solve over budgets
        n_times = len(planning_times)
        states_over_time = [None] * n_times  # (n_times, n_steps, s_dim)
        controls_over_time = [None] * n_times  # (n_times, n_steps, c_dim)
        costs_over_time = [None] * n_times  # (n_times, 2)

        # Solve over time budgets
        prev_t = 0.0
        for t_i, t in enumerate(planning_times):
            # solve for dt
            dt = t - prev_t
            prev_t = t
            status = self.ss.solve(dt)

            # no new solution found
            if (
                status.asString() != "Exact solution"
                or not self.ss.haveSolutionPath()
            ):
                if verbose:
                    print(f"No new solution found for timestep {t_i}: {t}")
                if t_i > 0 and len(controls_over_time[t_i - 1]) > 0:
                    states_over_time[t_i] = states_over_time[t_i - 1]
                    controls_over_time[t_i] = controls_over_time[t_i - 1]
                    costs_over_time[t_i] = costs_over_time[t_i - 1]
                else:
                    states_over_time[t_i] = [start]
                    controls_over_time[t_i] = []
                    costs_over_time[t_i] = [-1.0, -1.0]

            # found new solution
            else:
                if verbose:
                    print(f"New solution found for timestep {t_i}: {t}")
                states, controls, costs = self.extract_current_solution()
                states_over_time[t_i] = states
                controls_over_time[t_i] = controls
                costs_over_time[t_i] = costs

            # OMPL does not store solution based on insertion order
            # need to clear all stored solution paths here so that
            # future solution paths won't be hidden
            self.pdef.clearSolutionPaths()

        self.ss.clear()
        return states_over_time, controls_over_time, costs_over_time

    def extract_current_solution(self):
        """Extract the current solution if there is one"""
        path = self.ss.getSolutionPath()

        # States
        ompl_states = []
        states = []
        for i in range(path.getStateCount()):
            s = path.getState(i)
            mean = [s.getX(), s.getY(), s.getYaw()]
            cov = [s.getCovariance(i) for i in range(6)]
            ompl_states.append(s)
            states.append(mean + cov)

        # Controls
        # Note that the control itself also contains delta state
        # and we only need the actual control
        controls = []
        for i in range(path.getControlCount()):
            control = path.getControl(i)
            controls.append([control[j] for j in range(self.c_dim)])

        # Costs
        running_cost = 0.0
        for i in range(len(ompl_states) - 1):
            running_cost += self.obj.motionCost(
                ompl_states[i], ompl_states[i + 1]
            ).value()
        terminal_cost = self.obj.terminalCost(ompl_states[-1]).value()

        return states, controls, [running_cost, terminal_cost]


########## OMPL Components ##########
class ControlSpaceInformation(oc.SpaceInformation):
    """
    A control space information that runs collision checking differently.

    Unlike geometric collision checking which uses interpolation,
    regular control collision checking checks intermediate state validity.

    However, the "control" defined here is single-step push action, so
    there is no intermediate states. This class runs interpolation
    to simulate the intermediate state.
    """

    def __init__(self, space, control_space, step_size=0.5):
        super().__init__(space, control_space)
        # step size is considered to be the interpolation resolution
        self.step_size = min(step_size, 1.0)

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

        # Propagate the state
        self.getStatePropagator().propagate(state, control, 1.0, result)

        # Start to check validity (from step_size * result to 1 * result)
        valid = True
        temp = self.allocState()
        space = self.getStateSpace()
        t = 0
        while t < 1.0:
            t = min(t + self.step_size, 1.0)
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
        self.preds = None
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
        preds_list = []
        for i in range(0, self.cache_size, batch_size):
            i_l, i_h = i, i + batch_size

            pred = self.model(self.controls[i_l:i_h])
            preds_list.append(pred)
            # Mean
            self.d_states[i_l:i_h, :3] = pred[:, :3]
            # Covariance
            if self.s_dim > 3 and pred.shape[1] > 3:
                self.d_states[i_l:i_h, [3, 6, 8]] = np.exp(pred[:, 3:])

        self.preds = np.concatenate(preds_list, axis=0)

    def sample(self, control):
        """Take a control from the cache"""
        # Pick a control randomly from the predefined list if given
        c = self.controls[self.cache_idx]
        s = self.d_states[self.cache_idx]
        for i in range(self.c_dim):
            control[i] = float(c[i])
        for i in range(self.s_dim):
            control[self.c_dim + i] = float(s[i])

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
        belief,
        x_train,
        control_list=None,
        pool_size=10,
        p_random=0.1,
        cache_size=10000,
    ):
        """Initialize the active batch control sampler"""
        self.belief = belief
        self.x_train = x_train
        self.pool_size = pool_size
        self.p_random = p_random
        self.n_pool = cache_size
        super().__init__(
            space, c_space, model, control_list, pool_size * cache_size
        )

    def sample_cache(self):
        """
        Sample the cache to get a batch of controls and delta states

        Different from the parent class, the cache is reshaped to make
        controls and d_states a list of control pools instead single control.
        Then we bias the control selection with the variance.
        """
        super().sample_cache()

        # Compute variance for action selection
        batch_size = 2000
        select_var = np.zeros(self.cache_size)
        for i in range(0, self.cache_size, batch_size):
            i_l, i_h = i, i + batch_size

            # NLL variance (belief, aleatoric)
            if self.belief:
                select_var[i_l:i_h] = np.sum(
                    self.d_states[i_l:i_h, [3, 6, 8]], axis=1
                )
            # Kernel method (epistemic)
            elif self.x_train is not None:
                select_var[i_l:i_h] = get_posteriors(
                    self.model.model,
                    self.x_train,
                    self.controls[i_l:i_h],
                    sigma=5e-3,
                )
            else:
                raise ValueError(
                    f"Invalid model prediction dimension: {self.preds.shape[1]}"
                )

        # Reshape to make controls and d_states
        # a list of control pools instead single control
        self.controls = self.controls.reshape(self.n_pool, self.pool_size, -1)
        self.d_states = self.d_states.reshape(self.n_pool, self.pool_size, -1)
        select_var = select_var.reshape(self.n_pool, self.pool_size)

        # Option 1 - Given the variances, assign weights to each control
        # then sample from the pool with the weights
        # weights = 1.0 / (select_var + 1e-6)
        # weights = weights / weights.sum(axis=1, keepdims=True)
        # u = np.random.uniform(size=self.n_pool)
        # cdf = np.cumsum(weights, axis=1)
        # indices = (u[:, None] <= cdf).argmax(axis=1)

        # Option 2 - Simply choose the control with the lowest variance
        indices = np.argmin(select_var, axis=1)
        # To ensure probabilistic complete, include random sampling
        n_random = max(1, int(self.p_random * self.n_pool))
        random_pools = np.random.choice(
            self.n_pool, size=n_random, replace=False
        )
        random_indices = np.random.randint(0, self.pool_size, size=n_random)
        indices[random_pools] = random_indices

        # Select the best control and corresponding delta state
        rows = np.arange(self.n_pool)
        self.controls = self.controls[rows, indices]
        self.d_states = self.d_states[rows, indices]

    def sample(self, control):
        """Take a control from the cache"""
        super().sample(control)
        if self.cache_idx >= self.n_pool:
            self.sample_cache()


class SE2PushGoal(ob.GoalState):
    """Goal (represented as a ellipsoid) region for push task"""

    def __init__(self, si, goal, goal_size, rot_weight=0.2):
        """Initialize the goal for push task"""
        super().__init__(si)
        self.goal = goal  # (x, y, yaw)
        self.goal_size = goal_size
        self.rot_w = rot_weight

        # for GoalState
        goal_state = ob.State(si.getStateSpace())
        goal_state().setX(float(goal[0]))
        goal_state().setY(float(goal[1]))
        goal_state().setYaw(float(goal[2]))
        self.setState(goal_state)
        self.setThreshold(goal_size)

    def distanceGoal(self, state):
        """Compute the distance to an ellipsoidal goal"""
        dx = state.getX() - self.goal[0]
        dy = state.getY() - self.goal[1]
        dyaw = angle_diff(state.getYaw(), self.goal[2])
        d = (dx**2 + dy**2 + (self.rot_w * dyaw) ** 2) ** 0.5
        return d


class SE2PushOptimizationObjective(ob.PathLengthOptimizationObjective):
    """SE2 Push Optimization Objective

    When belief is True, this uses Wasserstein distance in belief space.
    Otherwise, this simply uses SE2 distance.
    When AO-RRT is used, terminal cost will be considered.
    """

    def __init__(
        self,
        si,
        goal,
        belief=True,
        terminal_weight=2.0,
        rot_weight=0.2,
    ):
        """Initialize the optimization objective"""
        super().__init__(si)
        self.si = si
        self.goal = goal
        self.belief = belief
        self.terminal_weight = terminal_weight
        self.weight = np.diag([1.0, 1.0, rot_weight])

    def se2_distance(self, s1, s2, cov1=None, cov2=None):
        """
        Compute the Lie SE2 distance between two states
        When no covariance is provided, this is regular SE2 distance.
        When both covariances are provided, this is Wasserstein distance.
        When only one covariance is provided, this is Wasserstein distance
        from a distribution to a dirac measure.
        """
        # Log-based SE2 distance
        t1 = to_se2_transform(s1)
        t2 = to_se2_transform(s2)
        t_delta = inv_se2_transform(t1) @ t2
        # apply weights
        dist_vec = self.weight @ log_se2(t_delta)
        # distance squared
        dist2 = dist_vec.T @ dist_vec

        # No belief state
        if cov1 is None:
            return np.sqrt(dist2)
        # apply weights
        cov1 = self.weight @ cov1 @ self.weight.T

        # Belief Wasserstein distance to a dirac measure at s2
        if cov2 is None:
            return np.sqrt(dist2 + np.trace(cov1))

        # Wasserstein distance from one distribution to another
        # express cov2 in the frame of cov1
        adj = adjoint_se2(t_delta)
        cov2 = adj @ cov2 @ adj.T
        # apply weights
        cov2 = self.weight @ cov2 @ self.weight.T
        return np.sqrt(dist2 + self.bures(cov1, cov2))

    def motionCost(self, s1, s2):
        """Compute the cost of the motion from s1 to s2"""
        # Regular SE2 distance
        s1_x, s1_y, s1_yaw = s1.getX(), s1.getY(), s1.getYaw()
        s2_x, s2_y, s2_yaw = s2.getX(), s2.getY(), s2.getYaw()
        if self.belief:
            cov1 = self.vec_to_cov([s1.getCovariance(i) for i in range(6)])
            cov2 = self.vec_to_cov([s2.getCovariance(i) for i in range(6)])
        else:
            cov1 = None
            cov2 = None

        dist = self.se2_distance(
            [s1_x, s1_y, s1_yaw], [s2_x, s2_y, s2_yaw], cov1, cov2
        )
        return ob.Cost(dist)

    def costToGo(self, state, goal):
        """
        Compute the cost to goal from the current state to the goal region
        This needs to be admissible (under-estimate the true cost)
        """
        threshold = goal.getThreshold()
        goal_state = goal.getState()
        # no covariance in goal state, simply assume goal cov
        # is the same as state cov
        # then we can just skip covariance in the distance computation
        dist_to_goal = self.se2_distance(
            [state.getX(), state.getY(), state.getYaw()],
            [goal_state.getX(), goal_state.getY(), goal_state.getYaw()],
        )
        return ob.Cost(max(dist_to_goal - threshold, 0))

    def terminalCost(self, state):
        """
        Compute the cost of the state (should be inside goal region)
        Distance to the goal center.
        If in belief space, this is Wasserstein distance
        to Dirac measure of the goal.
        """
        if self.terminal_weight == 0.0:
            return ob.Cost(0.0)

        # state
        x, y, yaw = state.getX(), state.getY(), state.getYaw()
        # if in belief space, include covariance for Wasserstein distance
        if self.belief:
            cov = self.vec_to_cov([state.getCovariance(i) for i in range(6)])
        else:
            cov = None

        # regular SE2 distance
        dist = self.se2_distance([x, y, yaw], self.goal, cov)
        return ob.Cost(self.terminal_weight * dist)

    @staticmethod
    def bures(cov1, cov2):
        """
        Compute the 2-Wasserstein distance between two covariance matrices
        This is equivalent to Bures metric since means are not considered.

        dist = trace(cov1 + cov2 - 2 * sqrt(sqrt(cov1) @ cov2 @ sqrt(cov1)))
        optimized version:
        dist = trace(cov1) + trace(cov2)
             - 2 * Sum(Sqrt(Eigenvalues( sqrt(cov1) @ cov2 @ sqrt(cov1) )))
        """

        def sqrtm_spd(matrix):
            """Fast matrix square root for SPD matrices"""
            eigvals, eigvecs = np.linalg.eigh(matrix)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        sqrt_cov1 = sqrtm_spd(cov1)
        eigenvalues = np.linalg.eigvalsh(sqrt_cov1 @ cov2 @ sqrt_cov1)
        sqrt_tr = np.sum(np.sqrt(eigenvalues))
        return np.trace(cov1) + np.trace(cov2) - 2 * sqrt_tr

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


########## Test ##########
if __name__ == "__main__":
    from geometry.object_model import get_obj_shape
    from experiments.train_push_model import load_model
    from experiments.run_push_plans_pool import run_plans_pool
    from experiments.utils import DataLoader, set_seed, get_names

    set_seed(10)
    env = generate_push_env()
    # envs = np.load("data/planning_push_envs.npy")
    # env = envs[0]
    visualize_push_env(env)
    plt.show()

    # Dynamics model
    model_type = "mlp"
    use_var = 1
    n_data = 1000
    m_id = 0
    # Load object
    obj_name = "cracker_box_flipped"
    model_name, data_name, rep_data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    # Load trained model
    model = load_model(model_type, obj_shape, use_var)
    name = f"{obj_name}_{model_type}_{use_var}_{n_data}_{m_id}"
    model.load(f"results/models/{name}.pth")
    # Load data that is used for training, for active planning
    data_loader = DataLoader(data_name)
    dataset = data_loader.load_data()
    used_indices = np.load(f"results/learning/idx_used_{name}.npy")
    x_train = dataset["x_pool"][used_indices]
    # Load data as control set for sampling (not used for training)
    mask = np.ones(len(dataset["x_pool"]), dtype=bool)
    mask[used_indices] = False
    control_list = dataset["x_pool"][mask].astype(np.float64)

    # Plan
    algo = "aorrt"
    belief = True
    planner = SE2PushPlanner(
        env["obstacles"],
        obj_shape,
        model,
        x_train=None,
        belief=belief,
        algo=algo,
        terminal_weight=2.0,
        controls=control_list,
    )
    times = list(np.linspace(1.0, 30.0, 30))
    states, controls, costs = planner.plan(
        env["start"], env["goal"], env["goal_size"], times
    )
    for i in range(len(times)):
        print(f"{times[i]:.2f}: {costs[i][0]:.2f}, {costs[i][1]:.2f}")

    # Execution
    exec_path, _, _ = run_plans_pool(
        obj_name,
        [states[-1]],
        [controls[-1]],
        obj_shape,
        env["obstacles"],
        dataset,
    )

    # Visualization
    visualize_push_env(env, states[-1], exec_path[-1], obj_shape=None)
    plt.show()
