import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt

from ompl import base as ob
from ompl import control as oc
from ompl import util as ou

from geometry.pose import angle_diff, wrap_to_pi
from geometry.car_dynamics import CAR_WHEELBASE, CAR_SIZE, propagate_analytical
from geometry.car_dynamics import process_cov_body, propagate_cov_linear

from planning.planning_utils import vec_to_cov, cov_to_vec, world_2d_cov
from planning.planning_utils import approx_rect_to_circles
from planning.planning_utils import circles_in_collision
from planning.planning_utils import circles_collision_risk
from planning.planning_utils import draw_rect, draw_circle, draw_cov_ellipse
from planning.planning_utils import draw_gradient_circle

########## Pushing Environment ##########
# four obstacles to represent two cars
OBSTACLES = np.zeros((3, 4))
OBSTACLES[:, 0] = np.array([3.75, 3.75, 5.25]) * CAR_SIZE[1]
OBSTACLES[:, 1] = np.array([0.75, 3.75, 3.75]) * CAR_SIZE[0]
OBSTACLES[:, 2] = 1.5 * CAR_SIZE[1]
OBSTACLES[:, 3] = np.array([1.5, 1.5, 7.5]) * CAR_SIZE[0]
POS_RANGES = [[0.0, 4.5 * CAR_SIZE[1]], [0.0, 7.5 * CAR_SIZE[0]]]
SAFE_RANGE = CAR_SIZE[0]  # np.linalg.norm(CAR_SIZE)


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
    # Obstacles are rectangles
    obstacles = np.asarray(obstacles)
    # Free region to sample from
    free = [r.copy() for r in POS_RANGES]
    free[0][0] = 0.5 / 3.0 * POS_RANGES[0][1]
    free[0][1] = 1.5 / 3.0 * POS_RANGES[0][1]
    free[1][0] = 0.75 / 2.0 * POS_RANGES[1][1]
    free[1][1] = 1.5 / 2.0 * POS_RANGES[1][1]

    # Find a valid random start
    while True:
        # generate a random state
        pi2 = np.pi / 2
        start = get_random_se2_states(1, free, (pi2 - 0.2, pi2 + 0.2))[0]
        # check clearance
        dists = np.linalg.norm(start[:2] - obstacles[:, :2], axis=-1)
        clearance = dists - safe_start_range
        if clearance.min() > 0.0:
            break

    # Fixed Two Goal Regions (Two parking spots)
    goals_center = np.zeros((2, 3))
    goals_center[:, 0] = 3.75 * CAR_SIZE[1]
    goals_center[:, 1] = np.array([5.25, 2.25]) * CAR_SIZE[0]
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
                # cov_world = world_2d_cov(yaw, vec_to_cov(state[3:]))
                cov_world = vec_to_cov(state[3:])
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
        # control_bounds = ((-0.5, 1.0), (-0.3, 0.3))
        control_bounds = ((-0.3, 0.3), (-0.3, 0.3))
        self.c_dim = len(control_bounds)
        self.belief = belief
        self.terminal_weight = terminal_weight

        # For collision checking
        self.car_shape = car_shape
        self.car_circles = approx_rect_to_circles((0, 0, *car_shape), -0.05)
        self.obstacles = np.asarray(obstacles)
        self.obstacles_circles = self.approx_obstacles(obstacles)

        # Initialize the spaces and set up the planner
        self.space = self.init_state_space(bounds)
        self.control_space = self.init_control_space(control_bounds)
        self.si = oc.SpaceInformation(self.space, self.control_space)
        self.si.setPropagationStepSize(0.5)
        self.si.setMinMaxControlDuration(2, 2)
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
            ob.StateValidityCheckerFn(self.is_state_valid)
        )

        # State propagator (always propagate belief)
        propagator = CarPropagator(self.si, belief=True)
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
            planner.setPruningRadius(0.03)
        self.ss.setPlanner(planner)

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
    def approx_obstacles(self, obstacles):
        """Approximate the rectangles obstacles to circles"""
        obs_circles = []
        for i, obstacle in enumerate(obstacles):
            circle = approx_rect_to_circles(obstacle, -0.08)
            obs_circles.extend(circle.tolist())
        return np.array(obs_circles)

    def is_state_valid(self, state):
        """Check if the state is in the bounds"""
        # In bounds
        in_bounds = self.si.satisfiesBounds(state)
        if not in_bounds:
            return False

        # In collision
        if len(self.obstacles) == 0:
            return False
        pose = np.array([state.getX(), state.getY(), state.getYaw()])

        # chance constrained
        if self.belief:
            cov = vec_to_cov([state.getCovariance(i) for i in range(6)])
            risk = circles_collision_risk(
                pose, cov, self.car_circles, self.obstacles_circles
            )
            return risk <= 0.05

        # regular collision checking
        else:
            in_collision = circles_in_collision(
                pose, self.car_circles, self.obstacles_circles
            )
            return not in_collision

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
        if isinstance(start, np.ndarray):
            start = start.tolist()
        start_state = self.get_belief_state(start[:3])
        self.ss.setStartState(start_state)

        # Set goal
        self.ss.setGoal(SE2CarGoals(self.si, goals, goal_size))
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
                if t_i > 0:
                    states_over_time[t_i] = states_over_time[t_i - 1]
                    controls_over_time[t_i] = controls_over_time[t_i - 1]
                    costs_over_time[t_i] = costs_over_time[t_i - 1]
                else:
                    states_over_time[t_i] = [
                        start[:3] + [1e-6, 0, 0, 1e-6, 0, 1e-6]
                    ]
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
class CarPropagator(oc.StatePropagator):
    """Car propagator with/without belief"""

    def __init__(self, si, belief=True):
        """Initialize the car propagator"""
        super().__init__(si)
        self.si = si
        self.belief = belief

    def propagate(self, state, control, duration, result):
        """Propagate the car state"""
        v, phi = control[0], control[1]
        init_state = (state.getX(), state.getY(), state.getYaw())

        # Propagate the state
        next_state = propagate_analytical((v, phi), duration, init_state)
        result.setX(next_state[0])
        result.setY(next_state[1])
        result.setYaw(next_state[2])

        if not self.belief:
            return

        # We consider uncertainty
        cov = vec_to_cov([state.getCovariance(i) for i in range(6)])
        # Linear model of process covariance
        process_cov = process_cov_body((v, phi), duration)
        cov = propagate_cov_linear(
            (v, phi), duration, init_state, cov, process_cov
        )
        cov_vec = cov_to_vec(cov)
        for i in range(6):
            result.setCovariance(i, float(cov_vec[i]))
        return result


class SE2CarGoals(ob.GoalStates):
    """Goal (represented as a ellipsoid) region for car task"""

    def __init__(self, si, goals, goal_size, rot_weight=0.2):
        """Initialize the goal for car task"""
        super().__init__(si)
        self.goals = np.asarray(goals)  # (n_goals, 3)
        self.goal_size = goal_size
        self.rot_w = rot_weight

        # for GoalStates
        for goal in goals:
            goal_state = ob.State(si.getStateSpace())
            goal_state().setX(float(goal[0]))
            goal_state().setY(float(goal[1]))
            goal_state().setYaw(float(goal[2]))
            self.addState(goal_state)
        self.setThreshold(goal_size)

    def distanceGoal(self, state):
        """Compute the distance to an ellipsoidal goal"""
        x = float(state.getX())
        y = float(state.getY())
        yaw = float(state.getYaw())

        dx = x - self.goals[:, 0]
        dy = y - self.goals[:, 1]
        dyaw = angle_diff(yaw, self.goals[:, 2])

        d2 = dx**2 + dy**2 + (self.rot_w * dyaw) ** 2
        d = float(np.sqrt(d2.min()))
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

    def motionCost(self, s1, s2):
        """Compute the cost of the motion from s1 to s2"""
        # Regular SE2 distance
        s1_x, s1_y, s1_yaw = s1.getX(), s1.getY(), s1.getYaw()
        s2_x, s2_y, s2_yaw = s2.getX(), s2.getY(), s2.getYaw()

        # TODO
        # cov1 = vec_to_cov([s1.getCovariance(i) for i in range(6)])
        # cov2 = vec_to_cov([s2.getCovariance(i) for i in range(6)])
        # d1, d2 = self.se2_distance(
        #     [s1_x, s1_y, s1_yaw], [s2_x, s2_y, s2_yaw], cov1, cov2, self.weight
        # )
        # if self.belief:
        #     return ob.Cost(d2)
        # else:
        #     return ob.Cost(d1)

        if self.belief:
            cov1 = vec_to_cov([s1.getCovariance(i) for i in range(6)])
            cov2 = vec_to_cov([s2.getCovariance(i) for i in range(6)])
        else:
            cov1 = None
            cov2 = None

        dist = self.se2_distance(
            [s1_x, s1_y, s1_yaw], [s2_x, s2_y, s2_yaw], cov1, cov2, self.weight
        )
        return ob.Cost(dist)

    # Not implemented for multiple goals
    # def costToGo(self, state, goal):
    #     """
    #     Compute the cost to goal from the current state to the goal region
    #     This needs to be admissible (under-estimate the true cost)
    #     """
    #     threshold = goal.getThreshold()
    #     goal_state = goal.getState()
    #     # just skip covariance in the distance computation
    #     # it will for sure under-estimate the true cost
    #     dist_to_goal = self.se2_distance(
    #         [state.getX(), state.getY(), state.getYaw()],
    #         [goal_state.getX(), goal_state.getY(), goal_state.getYaw()],
    #         weight=self.weight,
    #     )
    #     return ob.Cost(max(dist_to_goal - threshold, 0))

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

        # TODO
        # cov = vec_to_cov([state.getCovariance(i) for i in range(6)])
        # d1, d2 = self.se2_distance(
        #     [x, y, yaw], self.goal, cov, weight=self.weight
        # )
        # if self.belief:
        #     return ob.Cost(self.terminal_weight * d2)
        # else:
        #     return ob.Cost(self.terminal_weight * d1)

        # if in belief space, include covariance for Wasserstein distance
        if self.belief:
            cov = vec_to_cov([state.getCovariance(i) for i in range(6)])
        else:
            cov = None

        # regular SE2 distance
        dist = self.se2_distance(
            [x, y, yaw], self.goal, cov, weight=self.weight
        )
        return ob.Cost(self.terminal_weight * dist)

    @staticmethod
    def se2_distance(
        s1, s2, cov1=None, cov2=None, weight=np.diag([1.0, 1.0, 0.2])
    ):
        """
        Compute the Linearized SE2 distance between two states

        When no covariance is provided, this is regular SE2 distance.
        When both covariances are provided, this is Wasserstein distance.
        When only one covariance is provided, this is Wasserstein distance
        from a distribution to a dirac measure.
        """
        if weight.shape == (3,):
            weight = np.diag(weight)
        elif weight.shape == (3, 3):
            pass
        else:
            raise ValueError(
                f"Invalid SE2 weight shape: {weight.shape}, "
                + "expected (3,) or (3, 3)"
            )

        # Linearized SE2 distance (at local s1 frame)
        dx = s1[0] - s2[0]
        dy = s1[1] - s2[1]
        dyaw = angle_diff(s1[2], s2[2])
        # apply weights
        dist_vec = weight @ np.array([dx, dy, dyaw])
        # distance squared
        dist2 = dist_vec.T @ dist_vec

        # No belief state
        if cov1 is None:
            return np.sqrt(dist2)
        cov1 = weight @ cov1 @ weight.T

        # Belief Wasserstein distance to a dirac measure at s2
        if cov2 is None:
            # TODO
            # return np.sqrt(dist2), np.sqrt(dist2 + np.trace(cov1))
            return np.sqrt(dist2 + np.trace(cov1))
        cov2 = weight @ cov2 @ weight.T

        # TODO
        # return np.sqrt(dist2), np.sqrt(
        #     dist2 + SE2CarOptimizationObjective.bures(cov1, cov2)
        # )

        # Wasserstein distance from one distribution to another
        return np.sqrt(dist2 + SE2CarOptimizationObjective.bures(cov1, cov2))

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


if __name__ == "__main__":
    from experiments.utils import set_seed
    from simulation.car_sim import Sim

    set_seed(42)
    env = generate_car_env()
    # envs = np.load("data/planning_car_envs.npy", allow_pickle=True)
    # env = envs[10]
    visualize_car_env(env, [env["start"]], draw_car_shape=True)
    plt.show()

    # Plan
    algo = "sst"
    belief = False
    planner = SE2CarPlanner(
        env["obstacles"],
        CAR_SIZE,
        belief=belief,
        algo=algo,
        terminal_weight=5.0,
    )
    total_time = 30
    times = list(np.linspace(1.0, total_time, total_time))
    states, controls, costs = planner.plan(
        env["start"], env["goals"], env["goal_size"], 0, times
    )
    for i in range(len(times)):
        print(f"{times[i]:.2f}: {costs[i][0]:.2f}, {costs[i][1]:.2f}")
    # print risk
    for i in range(len(states[-1])):
        risk = circles_collision_risk(
            states[-1][i][:3],
            vec_to_cov(states[-1][i][3:]),
            planner.car_circles,
            planner.obstacles_circles,
        )
        print(f"State: {i}: {states[-1][i][:3]}: Risk: {risk}")

    # Execution
    u = [np.array(controls[-1])]
    t = [np.ones(len(controls[-1]))]
    par_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml = open(os.path.join(par_dir, "simulation/car_sim.xml")).read()
    sim = Sim(xml, n_envs=10, dt=0.01, visualize=True)
    sim.set_car_init_states(states[-1][0])
    sim.reset()
    exec_states, exec_inter_states = sim.execute_controls(
        u, t, wait_time=0.5, return_intermediate=True
    )

    # Visualization
    visualize_car_env(env, states[-1], exec_states[-1], CAR_SIZE, True)
    plt.show()
