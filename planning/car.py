import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from ompl import base as ob
from ompl import control as oc
from ompl import util as ou

from geometry.pose import angle_diff, wrap_to_pi
from geometry.car_dynamics import CAR_WHEELBASE, CAR_SIZE, propagate_analytical
from lie_group.lie_se2 import adjoint_se2, exp_se2, right_jacobian_se2
from lie_group.lie_se2 import to_se2_transform, inv_se2_transform, log_se2
from lie_group.propagation import propagate_cov, se2_error

from planning.planning_utils import vec_to_cov, cov_to_vec, world_2d_cov
from planning.planning_utils import rects_rects_in_collision
from planning.planning_utils import draw_rect, draw_circle, draw_cov_ellipse
from planning.planning_utils import draw_gradient_circle

########## Pushing Environment ##########
# four obstacles to represent two cars
OBSTACLES = np.zeros((3, 4))
OBSTACLES[:, 0] = np.array([3.75, 3.75, 5.25]) * CAR_SIZE[0]
OBSTACLES[:, 1] = np.array([0.75, 3.75, 3.75]) * CAR_SIZE[1]
OBSTACLES[:, 2] = 1.5 * CAR_SIZE[0]
OBSTACLES[:, 3] = np.array([1.5, 1.5, 7.5]) * CAR_SIZE[1]
POS_RANGES = [[0.0, 4.5 * CAR_SIZE[0]], [0.0, 7.5 * CAR_SIZE[1]]]
SAFE_RANGE = np.linalg.norm(CAR_SIZE)
# car kinematic model process noise model parameters
# from simulation/car_noise_fitting.py
COV_X = 1.45e-3
COV_Y = 3.70e-4
COV_YAW = 4.10e-4


def get_random_se2_states(
    n_data, pos_range=POS_RANGES, euler_range=(-np.pi, np.pi)
):
    """Generate random initial states"""
    pos_x = np.random.uniform(pos_range[0][0], pos_range[0][1], (n_data, 1))
    pos_y = np.random.uniform(pos_range[1][0], pos_range[1][1], (n_data, 1))
    euler = np.random.uniform(euler_range[0], euler_range[1], (n_data, 1))
    states = np.concatenate([pos_x, pos_y, euler], axis=-1)
    return states


def generate_car_env(obstacles=OBSTACLES, safe_start_range=SAFE_RANGE):
    """Generate initial states"""
    # Obstacles are circles
    obstacles = np.asarray(obstacles)
    # Find a valid random start
    while True:
        # generate a random state
        free = [r.copy() for r in POS_RANGES]
        free[0][1] = 2.0 / 3.0 * free[0][1]
        free[1][1] = 1.5 / 2.0 * free[1][1]
        start = get_random_se2_states(1, free, (0, np.pi))[0]
        # check clearance
        dists = np.linalg.norm(start[:2] - obstacles[:, :2], axis=-1)
        clearance = dists - safe_start_range
        if clearance.min() > 0.0:
            break

    # Fixed Two Goal Regions (Two parking spots)
    goals_center = np.zeros((2, 3))
    goals_center[:, 0] = 3.75 * CAR_SIZE[0]
    goals_center[:, 1] = np.array([5.25, 2.25]) * CAR_SIZE[1]
    goals_center[:, 2] = 1.5708
    goal_size = 0.05
    return {
        "start": start,
        "goals": goals_center,
        "goal_size": goal_size,
        "obstacles": obstacles,
    }


########## Visualization ##########
def visualize_car_env(
    env,
    path=None,
    exec_path=None,
    car_shape=CAR_SIZE,
    draw_car_shape=False,
    title="Object Push Environment",
):
    """Visualize Object Push Environment"""
    (xmin, xmax), (ymin, ymax) = POS_RANGES
    tx, ty = (xmin + xmax) / 2, (ymin + ymax) / 2
    tw, th = (xmax - xmin), (ymax - ymin)

    start, obstacles = env["start"], env["obstacles"]
    goals, goal_size = env["goals"], env["goal_size"]
    if path is not None:
        path = np.asarray(path, dtype=float)
    if exec_path is not None:
        exec_path = np.asarray(exec_path)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Background table
    # draw_rect(ax, 0, -0.505, 1.524, 1.524, 0, "k", alpha=0.1)
    draw_rect(ax, tx, ty, tw * 1.7, th * 1.2, 0, "k", alpha=0.1)

    # Goal
    for i, goal in enumerate(goals):
        al = 1.0 - 0.5 * (i / max(len(goals) - 1, 1))
        label = "Goal Region " + "I" * (i + 1)
        draw_gradient_circle(ax, goal[:2], goal_size, alpha=al, label=label)
    # Obstacles
    if len(obstacles) > 0:
        first = True
        for x, y, w, h in obstacles:
            label = "Obstacle" if first else None
            draw_rect(ax, x, y, w, h, 0, "k", alpha=1.0, label=label)
            first = False

    # Planned path (x, y)
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], "o-", color="C0", label="Planned Path")

        # if covariance provided, plot
        if path.shape[1] > 3:
            for i, state in enumerate(path):
                x, y, yaw = state[:3]
                cov_world = world_2d_cov(yaw, vec_to_cov(state[3:]))
                label = "Belief" if i == 0 else None
                draw_cov_ellipse(
                    ax, (x, y), cov_world, 2.0, "C9", 0.5, label=label
                )

    # Executed path (x, y)
    if exec_path is not None:
        ax.plot(
            exec_path[:, 0],
            exec_path[:, 1],
            "o-",
            color="C5",
            label="Actual Path",
        )

    # Car
    if draw_car_shape:
        w, h = car_shape[0], car_shape[1]
        if path is not None:
            for state in path:
                x, y, yaw = state[:3]
                draw_rect(ax, x, y, w, h, yaw, "C0", alpha=0.1)

        if exec_path is not None:
            for state in exec_path:
                x, y, yaw = state[:3]
                draw_rect(ax, x, y, w, h, yaw, "C5", alpha=0.1)

    # Start
    draw_circle(ax, start[0], start[1], 0.01, "C8", label="Start")

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
class SE2CarPlanner:
    """Car planner class"""

    def __init__(
        self,
        obstacles,
        car_shape=CAR_SIZE,
        belief=True,  # use Wasserstein distance
        algo="aorrt",  # planner algorithm
        terminal_weight=2.0,  # terminal weight (only for aorrt)
        controls=None,  # control set for sampling
    ):
        """Initialize the OMPL SE2 planner"""
        bounds = POS_RANGES
        control_bounds = ((-0.5, 1.0), (-0.3, 0.3))
        self.c_dim = len(control_bounds)
        self.belief = belief
        self.terminal_weight = terminal_weight

        # For collision checking
        self.car_shape = car_shape
        self.obstacles = np.asarray(obstacles)

        # Initialize the spaces and set up the planner
        self.space = self.init_state_space(bounds)
        self.control_space = self.init_control_space(control_bounds)
        self.si = ControlSpaceInformation(self.space, self.control_space, 0.5)
        self.ss = oc.SimpleSetup(self.si)
        self.pdef = self.ss.getProblemDefinition()
        self.set_up_planner(belief, algo, controls)

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

    def set_up_planner(self, belief, algo, controls):
        """Initialize the planner"""
        # State validity checker
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(car_propagate)
        )

        # State propagator
        propagator = CarPropagator(belief=True)  # always propagate belief
        self.ss.setStatePropagator(oc.StatePropagatorFn(propagator.propagate))

        # # Control sampler
        # control_sampler = lambda c_space: ControlSampler(c_space, controls)
        # self.control_space.setControlSamplerAllocator(
        #     oc.ControlSamplerAllocator(control_sampler)
        # )

        # Optimization objective (set later with goal)
        # obj = SE2PushOptimizationObjective(
        #     self.si, goal, belief, self.terminal_weight
        # )
        # self.pdef.setOptimizationObjective(obj)

        # Planner algorithm
        if algo == "aorrt":
            planner = oc.AORRT(self.si)
        else:
            planner = oc.SST(self.si)
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
            in_collision = rects_rects_in_collision(
                pose, self.car_shape, self.obstacles
            )
            if in_collision:
                return False

        return True

    def clearance(self, state):
        """Check the clearance to the nearest obstacle"""
        if len(self.obstacles) == 0:
            return 0

        dists = np.linalg.norm(
            np.array([state.getX(), state.getY()]) - self.obstacles[:, :2],
            axis=1,
        )
        return np.min(dists)

    # Plan
    def plan(
        self,
        start,
        goals,
        goal_size,
        desired_goal_idx=0,
        planning_times=(1.0, 5.0),
        verbose=True,
    ):
        """Plan to goal"""
        # Set start
        start_state = self.get_belief_state([start[0], start[1], start[2]])
        self.ss.setStartState(start_state)

        # Set goal
        self.ss.setGoal(SE2CarGoal(self.si, goals, goal_size))
        # Set optimization objective
        self.obj = SE2CarOptimizationObjective(
            self.si, goals[desired_goal_idx], self.belief, self.terminal_weight
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


class CarPropagator(oc.StatePropagator):
    """Car propagator with/without belief"""

    def __init__(self, belief=True):
        """Initialize the car propagator"""
        super().__init__()
        self.belief = belief

    def propagate(self, start, control, duration, state):
        """Propagate the car state"""
        v, phi = control[0], control[1]
        init_state = (start.getX(), start.getY(), start.getYaw())

        # Propagate the state
        next_state = propagate_analytical((v, phi), duration, init_state)
        state.setX(next_state[0])
        state.setY(next_state[1])
        state.setYaw(next_state[2])

        if not self.belief:
            return

        # We consider uncertainty
        cov = vec_to_cov([state.getCovariance(i) for i in range(6)])
        # A simple linear model of the process noise
        delta_cov = np.diag(
            [
                COV_X * abs(v) * duration,
                COV_Y * abs(v) * abs(np.tan(phi)) * duration,
                COV_YAW * abs((v / CAR_WHEELBASE) * np.tan(phi)) * duration,
            ]
        )

        # Propagate covariance
        # log-based SE2 change
        delta = se2_error(init_state, next_state)
        # cov
        cov = propagate_cov(delta, cov, delta_cov)
        cov_vec = cov_to_vec(cov)
        for i in range(6):
            state.setCovariance(i, float(cov_vec[i]))


class SE2CarGoal(ob.GoalState):
    """Goal (represented as a ellipsoid) region for car task"""

    def __init__(self, si, goals, goal_size, rot_weight=0.2):
        """Initialize the goal for car task"""
        super().__init__(si)
        self.goals = goals  # a list of(x, y, yaw)
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


class SE2CarOptimizationObjective(ob.PathLengthOptimizationObjective):
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
            cov1 = vec_to_cov([s1.getCovariance(i) for i in range(6)])
            cov2 = vec_to_cov([s2.getCovariance(i) for i in range(6)])
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
            cov = vec_to_cov([state.getCovariance(i) for i in range(6)])
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


class PathOptimizationObjective(ob.PathLengthOptimizationObjective):
    def __init__(
        self,
        si,
        pdef,
        goal,
        goal_radius,
        obstacle_poses,
        obstacle_rads,
        terminal_weight=20.0,
        clearance_weight=1.0,
    ):
        """Initialize the optimization objective"""
        super().__init__(si)
        self.si = si
        self.pdef = pdef
        self.goal = goal
        self.goal_radius = goal_radius
        self.obstacle_poses = obstacle_poses
        self.obstacle_rads = obstacle_rads
        self.terminal_weight = terminal_weight
        self.clearance_weight = clearance_weight

    def motionCost(self, s1, s2):
        """Compute the cost of the motion from s1 to s2"""
        cost = 0

        # Path length cost
        p_cost = self.si.getStateSpace().distance(s1, s2)
        cost += p_cost

        # # Clearance cost
        # c_cost = self.clearance_weight * (
        #     0.5 * (self.clearanceCost(s1) + self.clearanceCost(s2)) * p_cost
        # )
        # cost += c_cost

        return ob.Cost(cost)

    def costToGo(self, state, goal):
        """
        Compute the cost to goal from the current state to the goal region
        This needs to be admissible (under-estimate the true cost)
        """
        dist = goal.distanceGoal(state)
        threshold = goal.getThreshold()
        return ob.Cost(max(dist - threshold, 0))

    def terminalCost(self, state):
        """Compute the cost of the state (should be inside goal region)"""
        x, y, yaw = state.getX(), state.getY(), state.getYaw()
        cost = ((x - self.goal[0]) ** 2 + (y - self.goal[1]) ** 2) ** 0.5
        cost += 0.5 * abs(angle_diff(yaw, self.goal[2]))
        return self.terminal_weight * cost

    def clearanceCost(self, state):
        """Compute the minimum distance to the nearest obstacle"""
        if len(self.obstacle_poses) == 0:
            return 0

        dists = np.linalg.norm(
            np.array([state.getX(), state.getY()]) - self.obstacle_poses,
            axis=1,
        )
        return 1.0 / (np.min(dists) + 1e-6)


def isStateValid(
    spaceInformation, state, system=None, config=None, obstacle_config=None
):
    # First check bounds
    if not spaceInformation.satisfiesBounds(state):
        return False

    # If no obstacle checking is requested, return True
    if obstacle_config is None or not obstacle_config.get("enabled", False):
        return True

    # Extract state position based on system type
    if system in ["simple_car", "pushing"]:
        # SE2 state: x, y, yaw
        try:
            x = state.getX()
            y = state.getY()
            pos = np.array([x, y])
        except AttributeError:
            # Fallback: state might be a compound state wrapper
            try:
                if callable(state):
                    state_obj = state()
                else:
                    state_obj = state
                x = state_obj.getX()
                y = state_obj.getY()
                pos = np.array([x, y])
            except (AttributeError, TypeError):
                # If we can't extract position, skip obstacle check
                return True

    elif system == "dublin_airplane":
        # SE3 state: x, y, z (we check obstacles in 2D, ignoring z)
        try:
            if callable(state):
                compound_state = state()
            else:
                compound_state = state
            x = compound_state[0][0]  # x position
            y = compound_state[0][1]  # y position
            pos = np.array([x, y])
        except (AttributeError, TypeError, IndexError):
            # If we can't extract position, skip obstacle check
            return True
    else:
        # Unknown system, skip obstacle check
        return True

    # Check obstacles
    safety_radius = obstacle_config.get("safety_radius", 0.10)

    # Check circular obstacles
    circles = obstacle_config.get("circles", [])
    for cx, cy, r in circles:
        dist = np.hypot(pos[0] - cx, pos[1] - cy)
        if dist < (r + safety_radius):
            return False

    # Check axis-aligned bounding boxes (AABBs)
    aabbs = obstacle_config.get("aabbs", [])
    for xmin, ymin, xmax, ymax in aabbs:
        # Inflate AABB by safety_radius
        xmin_inflated = xmin - safety_radius
        ymin_inflated = ymin - safety_radius
        xmax_inflated = xmax + safety_radius
        ymax_inflated = ymax + safety_radius
        if (
            xmin_inflated <= pos[0] <= xmax_inflated
            and ymin_inflated <= pos[1] <= ymax_inflated
        ):
            return False

    # Check oriented boxes
    boxes = obstacle_config.get("boxes", [])
    for cx, cy, hx, hy, yaw in boxes:
        # Transform point to box-local coordinates
        c, s = np.cos(-yaw), np.sin(-yaw)
        px = pos[0] - cx
        py = pos[1] - cy
        plx = c * px - s * py
        ply = s * px + c * py
        # Check if point is inside box (with safety_radius inflation)
        hx_inflated = hx + safety_radius
        hy_inflated = hy + safety_radius
        if abs(plx) < hx_inflated and abs(ply) < hy_inflated:
            return False

    return True


def plan(
    system: str,
    startState: np.ndarray,
    goalState: np.ndarray,
    goalThreshold: float,
    propagator: oc.StatePropagatorFn,
    minControlDuration: int,
    maxControlDuration: int,
    propagationStepSize: float,
    planningTime: float = 20.0,
    plannerName: str = "fusion",
    pruningRadius: float = 0.1,
    config: dict = None,
    visualize: bool = False,
):
    # print(f"[INFO] Plan function started:")
    # print(f"     - system: {system}")
    # print(f"     - startState: {startState}")
    # print(f"     - goalState: {goalState}")
    # print(f"     - planningTime: {planningTime}")
    # print(f"     - plannerName: {plannerName}")
    # print(f"     - minControlDuration: {minControlDuration}")
    # print(f"     - maxControlDuration: {maxControlDuration}")
    # print(f"     - propagationStepSize: {propagationStepSize}")

    space, cspace = configurationSpace(system)

    # Define a simple setup class
    ss = oc.SimpleSetup(cspace)

    # Extract obstacle configuration from config if available
    obstacle_config = None
    if config is not None:
        obstacle_config = config.get("obstacles", None)
        # If obstacles key exists but is a dict, use it directly
        if isinstance(obstacle_config, dict):
            # Ensure enabled flag is set if obstacles are defined
            if (
                obstacle_config.get("circles")
                or obstacle_config.get("aabbs")
                or obstacle_config.get("boxes")
            ):
                obstacle_config["enabled"] = obstacle_config.get(
                    "enabled", True
                )

    # Create state validity checker with system, config, and obstacle_config
    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(
            partial(
                isStateValid,
                ss.getSpaceInformation(),
                system=system,
                config=config,
                obstacle_config=obstacle_config,
            )
        )
    )

    # Set the propagator
    ss.setStatePropagator(oc.StatePropagatorFn(propagator))
    ss.getSpaceInformation().setMinMaxControlDuration(
        minControlDuration, maxControlDuration
    )
    ss.getSpaceInformation().setPropagationStepSize(propagationStepSize)

    # Create a start state
    start = pickStartState(system, space, startState)
    ss.setStartState(start)

    # Create a goal state
    goal = pickGoalState(system, goalState, ss, threshold=goalThreshold)
    goal.setThreshold(goalThreshold)
    ss.setGoal(goal)

    # Choose planner based on parameter
    planner = pickPlanner(plannerName, ss, pruningRadius=pruningRadius)
    ss.setPlanner(planner)

    # Set the optimization objective to control count (number of controls in path)
    ss.setOptimizationObjective(
        ControlCountObjective(
            ss.getSpaceInformation(), cost_per_control=propagationStepSize
        )
    )

    # Set the control sampler
    if system == "pushing":
        objectShape = pickObjectShape(config.get("objectName"))
        controlSampler = pickControlSampler(system, objectShape)
        cspace.setControlSamplerAllocator(
            oc.ControlSamplerAllocator(controlSampler)
        )

    if planningTime < 0:
        print(f"[INFO] Using exact solution termination condition")
        ptc = ob.exactSolnPlannerTerminationCondition(
            ss.getProblemDefinition()
        )
    else:
        ptc = ob.timedPlannerTerminationCondition(planningTime)

    solved = ss.solve(ptc)

    # Retry logic: only retry for timed planning (planningTime > 0)
    # For exact solution termination (planningTime < 0), accept the first solution found
    if planningTime < 0:
        # For exact solution termination, accept the first solution regardless of control count
        if ss.haveExactSolutionPath():
            control_count = ss.getSolutionPath().getControlCount()
            print(
                f"Solution found with {plannerName} "
                f"({control_count} controls) - accepting first exact solution"
            )
        else:
            log(
                "[WARNING] No exact solution found with exact solution termination condition",
                "warning",
            )
    else:
        # For timed planning, retry until we get a solution with reasonable number of controls
        maxRetries = 1000 if system == "simple_car" else 100
        max_attempts = 10  # Prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            # Check if we have an exact solution
            if ss.haveExactSolutionPath():
                control_count = ss.getSolutionPath().getControlCount()
                if control_count <= maxRetries:
                    print(
                        f"Solution found in {planningTime:.1f}s with {plannerName} "
                        f"({control_count} controls)"
                    )
                    break
                else:
                    print(
                        f"[WARNING] Solution found but has {control_count} "
                        f"controls (> {maxRetries})... retrying... (attempt {attempts}/{max_attempts})"
                    )
            else:
                print(
                    f"[WARNING] No exact solution found in {planningTime:.1f}s... retrying... (attempt {attempts}/{max_attempts})"
                )

            # Clear and retry
            ss.getPlanner().clear()
            solved = ss.solve(planningTime)

        if attempts >= max_attempts:
            log(
                f"[WARNING] Maximum retry attempts ({max_attempts}) reached, using current solution",
                "warning",
            )

    return getSolutionsInfo(ss), ss


def plan():
    # construct the state space we are planning in
    space = ob.SE2StateSpace()

    # set the bounds for the R^2 part of SE(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)

    # create a control space
    cspace = oc.RealVectorControlSpace(space, 2)

    # set the bounds for the control space
    cbounds = ob.RealVectorBounds(2)
    cbounds.setLow(-0.3)
    cbounds.setHigh(0.3)
    cspace.setBounds(cbounds)

    # obstalces
    # obstacle_poses = [(-0.25, 0.25)]
    # obstacle_rads = [0.1]
    obstacles = [(-0.25, 0.25, 0.1)]

    # goal
    goal = [0.0, 0.5, 0.0]
    goal_radius = 0.03

    # define a simple setup class
    ss = oc.SimpleSetup(cspace)
    ss.setStateValidityChecker(
        ob.StateValidityCheckerFn(
            partial(isStateValid, ss.getSpaceInformation(), obstacles)
        )
    )
    ss.setStatePropagator(oc.StatePropagatorFn(propagate))
    p_objective = PathOptimizationObjective(
        ss.getSpaceInformation(),
        ss.getProblemDefinition(),
        goal,
        goal_radius,
        obstacle_poses,
        obstacle_rads,
        terminal_weight=10.0,
        clearance_weight=0.1,
    )
    # c_objective = ClearanceObjective(
    #     ss.getSpaceInformation(),
    #     obstacle_poses,
    #     obstacle_rads,
    #     clearance_weight=0.1,
    # )
    # objective = ob.MultiOptimizationObjective(ss.getSpaceInformation())
    # objective.addObjective(p_objective, 1.0)
    # objective.addObjective(c_objective, 1.0)
    ss.setOptimizationObjective(p_objective)

    # create a start state
    start_state = ob.State(space)
    start_state().setX(-0.5)
    start_state().setY(0.0)
    start_state().setYaw(0.0)

    # create a goal state
    goal_state = ob.State(space)
    goal_state().setX(goal[0])
    goal_state().setY(goal[1])
    goal_state().setYaw(goal[2])

    # set the start and goal states
    ss.setStartAndGoalStates(start_state, goal_state, goal_radius)

    # (optionally) set planner
    si = ss.getSpaceInformation()
    planner = oc.AORRT(si)
    # planner = oc.RRT(si)
    # planner = oc.SST(si)
    # planner = oc.SSTstar(si)
    # planner = oc.BeliefSST(si)
    # planner.setSelectionRadius(0.02)
    # planner.setPruningRadius(0.01)
    # planner.setShrinkFactor(0.95)

    # planner = oc.EST(si)
    # planner = oc.KPIECE1(si) # this is the default
    # SyclopEST and SyclopRRT require a decomposition to guide the search
    # decomp = MyDecomposition(32, bounds)
    # planner = oc.SyclopEST(si, decomp)
    # planner = oc.SyclopRRT(si, decomp)

    ss.setPlanner(planner)
    # (optionally) set propagation step size
    si.setPropagationStepSize(0.1)

    # attempt to solve the problem
    solved = ss.solve(20.0)
    if not solved:
        print("No solution found")
        return

    # # Visualize solving process
    # c = 0
    # ss.setup()
    # while True:
    #     ptc = ob.timedPlannerTerminationCondition(10)
    #     solved = planner.solveOnce(ptc)
    #     if solved.asString() != "Exact solution":
    #         break
    #     planner.setBestCost(planner.getPreviousSolutionCost())

    #     pd = ob.PlannerData(si)
    #     planner.getPlannerData(pd)
    #     plot_graph(pd, f"graph_{c}.png")
    #     c += 1

    #     planner.clear()

    # print the path to screen
    # print("Found solution:\n%s" % ss.getSolutionPath().printAsMatrix())
    states = []
    for i in range(ss.getSolutionPath().getStateCount()):
        state = ss.getSolutionPath().getState(i)
        states.append(state)

    p_cost = 0
    for i in range(len(states) - 1):
        p_cost += si.getStateSpace().distance(states[i], states[i + 1])

    t_cost = p_objective.terminalCost(
        ss.getSolutionPath().getState(len(states) - 1)
    )

    c_cost = 0
    for i in range(len(states) - 1):
        c = 0.5 * (
            p_objective.clearanceCost(states[i])
            + p_objective.clearanceCost(states[i + 1])
        )
        c_cost += c * si.getStateSpace().distance(states[i], states[i + 1])
    c_cost = p_objective.clearance_weight * c_cost

    print(f"Path Cost: {p_cost}")
    print(f"Terminal Cost: {t_cost}")
    print(f"Clearance Cost: {c_cost}")

    # Plot state
    for i in range(len(states)):
        states[i] = [states[i].getX(), states[i].getY(), states[i].getYaw()]
    plot_path(
        states, goal, goal_radius, [-1, 1], obstacle_poses, obstacle_rads
    )


def plot_path(
    states, goal, goal_radius, bounds, obstacle_poses=(), obstacle_rads=()
):
    """
    Plot the SE(2) trajectory with orientation arrows, start/goal markers,
    and the goal region circle.

    Args:
        states (np.ndarray): Nx3 array of [x, y, yaw] states.
        goal (ob.State): OMPL goal state.
        goal_radius (float): Radius of the goal region.
        bounds (list): [low, high] plotting bounds for x and y.
    """
    states = np.array(states)
    goal = np.array(goal)
    xs, ys, yaws = states[:, 0], states[:, 1], states[:, 2]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, "-", linewidth=2, label="trajectory", alpha=0.2)

    # Sample fewer arrows for clarity
    step = max(1, len(states) // 20)
    ax.quiver(
        xs[::step],
        ys[::step],
        np.cos(yaws[::step]),
        np.sin(yaws[::step]),
        width=0.005,
        scale=15,
        label="heading",
    )

    # Obstacles
    for pose, rad in zip(obstacle_poses, obstacle_rads):
        circle = plt.Circle(pose, rad, fill=True, color="gray", alpha=0.5)
        ax.add_artist(circle)

    # Start and goal markers
    ax.scatter(xs[0], ys[0], s=80, marker="o", label="start")
    ax.scatter(goal[0], goal[1], s=120, marker="*", label="goal", zorder=3)
    ax.quiver(
        goal[0],
        goal[1],
        np.cos(goal[2]),
        np.sin(goal[2]),
        width=0.005,
        scale=15,
        color="red",
    )

    # Goal region
    goal_circle = plt.Circle(
        goal, goal_radius, fill=False, linestyle="--", alpha=0.8
    )
    ax.add_artist(goal_circle)

    # Aesthetics
    ax.set_aspect("equal", "box")
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.grid(True, linestyle=":")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SE(2) Trajectory")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_graph(pd, output_file="graph.png"):
    """Visualize OMPL PlannerData using matplotlib instead of graph-tool."""
    # Extract vertices (positions) and edges (connectivity)
    vertices = []
    for i in range(pd.numVertices()):
        s = pd.getVertex(i).getState()
        x, y = s.getX(), s.getY()
        vertices.append([x, y])
    vertices = np.array(vertices)

    edges = []
    for i in range(pd.numVertices()):
        for j in range(pd.numVertices()):
            if pd.edgeExists(i, j):
                edges.append([vertices[i], vertices[j]])
    edges = np.array(edges)

    # Plot the graph
    fig, ax = plt.subplots(figsize=(6, 6))
    for (x0, y0), (x1, y1) in edges:
        ax.plot([x0, x1], [y0, y1], color="gray", linewidth=0.5, alpha=0.6)
    # if len(edges) > 0:
    #     lc = LineCollection(edges, colors="gray", linewidths=0.5, alpha=0.6)
    #     ax.add_collection(lc)

    # Draw vertices
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        s=15,
        color="yellow",
        edgecolors="black",
    )

    # Highlight start and goal
    start_ids = [i for i in range(pd.numVertices()) if pd.isStartVertex(i)]
    goal_ids = [i for i in range(pd.numVertices()) if pd.isGoalVertex(i)]
    if start_ids:
        ax.scatter(
            vertices[start_ids, 0],
            vertices[start_ids, 1],
            s=40,
            color="cyan",
            label="Start",
        )
    if goal_ids:
        ax.scatter(
            vertices[goal_ids, 0],
            vertices[goal_ids, 1],
            s=40,
            color="green",
            label="Goal",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_title("OMPL Planner Data Visualization")
    ax.legend()
    plt.tight_layout()

    # plt.show()
    plt.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Graph written to {output_file}")


if __name__ == "__main__":
    from experiments.utils import set_seed

    set_seed(42)
    env = generate_car_env()
    # envs = np.load("data/planning_car_envs.npy")
    # env = envs[0]
    visualize_car_env(env, [env["start"]], draw_car_shape=True)
    plt.show()
