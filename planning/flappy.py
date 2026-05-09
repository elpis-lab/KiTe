import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
import matplotlib.pyplot as plt

import ompl.base as ob
import ompl.control as oc
import ompl.util as ou

from planning.planning_utils import point_in_rects, point_dist_to_rects
from planning.planning_utils import draw_circle, draw_rect, draw_gradient_rect


########## Flappy Bird Environment ##########
# Flappy Bird dynamics constants
# X constant horizontal velocity (px/frame)
VX = 5.0
# Y gravitational downward acceleration (px/frame^2)
G = 1.0
# Y FLAP upward acceleration when controlu = 1
FLAP = 9.0
# reasonable vertical velocity bounds
VY_MIN, VY_MAX = -10.0, 10.0
# world bounds
WIDTH, HEIGHT = 720, 480


def generate_flappy_env(
    pipe_gap=120 - 24, pipe_width=52 + 32, pipe_spacing=192
):
    """Generate a flappy bird environment"""
    # empty space between pipes
    empty = pipe_spacing - pipe_width

    # Compute how many pipes fit (centers)
    pipe_centers = []
    cx = empty + pipe_width / 2.0
    while True:
        right = cx + pipe_width / 2.0
        if (WIDTH - right) >= empty:
            pipe_centers.append(cx)
            cx += pipe_spacing
        else:
            break
    if not pipe_centers:
        raise ValueError(
            "No pipes fit: increase WIDTH or decrease pipe_width/pipe_spacing."
        )

    # 2 * len(pipe_centers) Rectangles (upper + lower pipes)
    obstacles = []
    base_y = HEIGHT
    gap_top_min = int(base_y * 0.2)
    gap_top_max = int(base_y * 0.8 - pipe_gap)
    if gap_top_max <= gap_top_min:
        raise ValueError("pipe_gap too large for HEIGHT.")
    # Generate random gap for each pipe
    for cx in pipe_centers:
        gap_top = np.random.randint(gap_top_min, gap_top_max)
        gap_bottom = gap_top + pipe_gap
        # lower pipe: y in [0, gap_top]
        lower_cy = 0.5 * gap_top
        obstacles.append((cx, lower_cy, pipe_width, gap_top))
        # upper pipe: y in [gap_bottom, HEIGHT]
        upper_h = HEIGHT - gap_bottom
        upper_cy = gap_bottom + 0.5 * upper_h
        obstacles.append((cx, upper_cy, pipe_width, upper_h))

    # Compute start / goal
    # fixed start
    start = (10.0, HEIGHT / 2.0, 0.0)
    # goal
    last_right = pipe_centers[-1] + pipe_width / 2.0
    last_empty_size = WIDTH - last_right
    goal_center = (last_right + last_empty_size / 2.0, HEIGHT / 2.0)
    goal_size = last_empty_size

    return {
        "start": start,
        "goal": goal_center,
        "goal_size": goal_size,
        "obstacles": obstacles,
    }


########## Visualization ##########
def visualize_flappy_env(
    env,
    paths=None,
    title="Flappy Bird Environment",
    text_size=None,
    auto_legend=True,
    env_legend_labels=True,
):
    """Visualize Flappy Bird environment.

    Parameters
    ----------
    text_size : int, optional
        Font size for axis labels, title, and ticks (legend uses ``text_size - 1``
        when ``auto_legend`` is True).
    auto_legend : bool
        If True, call ``ax.legend`` with default placement.
    env_legend_labels : bool
        If False, do not attach legend labels to goal, obstacles, or start (for a
        custom figure legend built by the caller).
    """
    obstacles = env["obstacles"]
    start = env["start"]
    goal = env["goal"]
    goal_size = env["goal_size"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal", adjustable="box")

    g_lab = "Goal Region" if env_legend_labels else None
    draw_gradient_rect(ax, *goal, goal_size, goal_size, label=g_lab)
    for i, (cx, cy, w, h) in enumerate(obstacles):
        label = ("Obstacle" if i == 0 else None) if env_legend_labels else None
        draw_rect(ax, cx, cy, w, h, 0, "black", alpha=0.8, label=label)

    if paths:
        for item in paths:
            path = item.get("path", None)
            if path is None or len(path) == 0:
                continue
            xs = [s[0] for s in path]
            ys = [s[1] for s in path]
            ax.plot(
                xs,
                ys,
                label=item.get("label", None),
                linestyle=item.get("linestyle", "-"),
                linewidth=item.get("linewidth", 2.5),
                alpha=item.get("alpha", 0.9),
                color=item.get("color", None),
            )

    sx, sy, _ = start
    s_lab = "Start" if env_legend_labels else None
    draw_circle(ax, sx, sy, 10.0, "C3", 1.0, label=s_lab)

    if auto_legend:
        leg_kw = {"loc": "upper left"}
        if text_size is not None:
            leg_kw["fontsize"] = text_size - 1
        ax.legend(**leg_kw)
    if text_size is not None:
        ax.set_xlabel("x", fontsize=text_size)
        ax.set_ylabel("y", fontsize=text_size)
        if title:
            ax.set_title(title, fontsize=text_size)
        ax.tick_params(axis="both", labelsize=text_size - 1)
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)
    return fig, ax


########## OMPL Main ##########
class FlappyPlanner:
    def __init__(self, obstacles, planner, terminal_weight):
        """Initialize the planner for flappy bird"""
        self.space = self.init_state_space(
            [(0, WIDTH), (0, HEIGHT), (VY_MIN, VY_MAX)]
        )
        self.control_space = self.init_control_space([(0, 1)])
        self.obstacles = np.asarray(obstacles)
        self.terminal_weight = terminal_weight

        # Initialize the spaces and set up the planner
        self.si = oc.SpaceInformation(self.space, self.control_space)
        self.ss = oc.SimpleSetup(self.si)
        self.pdef = self.ss.getProblemDefinition()
        self.set_up_planner(planner)

    def init_state_space(self, bounds):
        """Initialize the state space"""
        # 3D state: (x, y, vy)
        space = ob.RealVectorStateSpace(3)
        state_bounds = ob.RealVectorBounds(3)
        for i in range(3):
            state_bounds.setLow(i, bounds[i][0])
            state_bounds.setHigh(i, bounds[i][1])
        space.setBounds(state_bounds)
        return space

    def init_control_space(self, cbounds):
        """Initialize the control space"""
        # Control: 1D in [0, 1] (sampler will make it binary)
        control_space = oc.RealVectorControlSpace(self.space, 1)
        control_bounds = ob.RealVectorBounds(1)
        control_bounds.setLow(0, cbounds[0][0])
        control_bounds.setHigh(0, cbounds[0][1])
        control_space.setBounds(control_bounds)
        return control_space

    def set_up_planner(self, planner):
        """Initialize the planner"""
        # State validity checker
        self.ss.setStateValidityChecker(self.is_state_valid)

        # State propagator
        propagator = FlappyPropagator(self.si)
        self.ss.setStatePropagator(propagator.propagate)

        # Control sampler
        control_sampler = lambda c_space: FlappyControlSampler(c_space, 0.5)
        self.control_space.setControlSamplerAllocator(control_sampler)

        # Optimization objective (set later with goal)
        # objective = FlappyOptimizationObjective(
        #     self.si, self.clearance, goal, self.terminal_weight
        # )
        # self.pdef.setOptimizationObjective(objective)

        # Planner algorithm
        if planner == "sst":
            algo = oc.SST(self.si)
            algo.setSelectionRadius(10.0)
            algo.setPruningRadius(5.0)
        elif planner == "aorrt":
            algo = oc.AORRT(self.si)
        self.ss.setPlanner(algo)
        self.si.setPropagationStepSize(1.0)
        self.si.setMinMaxControlDuration(1, 1)

    # Validity checkers
    def is_state_valid(self, state):
        """Check if the state is in the bounds"""
        # Not in bounds
        in_bounds = self.si.satisfiesBounds(state)
        if not in_bounds:
            return False
        # In collision with obstacles (pines)
        if point_in_rects(state[0], state[1], self.obstacles):
            return False
        return True

    # Planner
    def plan(
        self,
        start,
        goal,
        goal_size,
        planning_times=(1.0, 2.0),
        verbose=True,
    ):
        """Plan with given start and goal given a list of planning budgets"""
        for i in range(1, len(planning_times)):
            if planning_times[i] < planning_times[i - 1]:
                raise ValueError("planning_time must be non-decreasing")

        # Set start
        start_state = self.si.allocState()
        start_state[0] = float(start[0])
        start_state[1] = float(start[1])
        start_state[2] = float(start[2])
        self.ss.setStartState(start_state)

        # Set goal
        self.ss.setGoal(FlappyGoal(self.si, goal, goal_size))
        # Set optimization objective
        self.obj = FlappyOptimizationObjective(
            self.si, self.obstacles, goal, self.terminal_weight
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
            ompl_states.append(s)
            states.append([float(s[0]), float(s[1]), float(s[2])])

        # Controls
        controls = []
        for i in range(path.getControlCount()):
            u = path.getControl(i)
            controls.append([u[0]])

        # Costs
        running_cost = 0.0
        for i in range(len(ompl_states) - 1):
            running_cost += self.obj.motionCost(
                ompl_states[i], ompl_states[i + 1]
            ).value()
        terminal_cost = self.obj.terminalCost(ompl_states[-1]).value()
        # terminal_cost = self.obj.dist_to_goal(ompl_states[-1])

        return states, controls, [running_cost, terminal_cost]


########## OMPL Components ##########
class FlappyPropagator:
    """Flappy propagator"""

    def __init__(self, si):
        """Initialize the propagator"""
        self.si = si

    def propagate(self, state, control, duration, result):
        """
        Propagate Flappy dynamics over frame:
            x_dot  = VX (constant)
            y_dot  = vy
            vy_dot = -G + FLAP * u (flap or not)
        """
        x = state[0]
        y = state[1]
        vy = state[2]
        u = 1.0 if control[0] >= 0.5 else 0.0
        dt = duration

        # y acceleration
        ay = -G + FLAP * u
        # new state
        result[0] = x + VX * dt
        result[1] = y + vy * dt + 0.5 * ay * dt * dt
        result[2] = vy + ay * dt


class FlappyControlSampler(oc.ControlSampler):
    def __init__(self, control_space, control_prob=0.5):
        """Initialize the control sampler"""
        super().__init__(control_space)
        self.control_prob = control_prob

    def sample(self, control):
        """Sample u ∈ {0, 1} with a given probability."""
        control[0] = 1.0 if np.random.rand() < self.control_prob else 0.0


class FlappyGoal(ob.GoalState):
    def __init__(self, si, goal, goal_size):
        """Initialize the goal for flappy bird task"""
        super().__init__(si)
        self.goal = goal  # (x, y)
        self.goal_size = goal_size
        goal_state = si.allocState()
        goal_state[0], goal_state[1] = float(goal[0]), float(goal[1])
        self.setState(goal_state)
        self.setThreshold(0.01)

    def distanceGoal(self, state):
        """Compute the distance to the square goal"""
        dx = max(abs(state[0] - self.goal[0]) - self.goal_size / 2, 0.0)
        dy = max(abs(state[1] - self.goal[1]) - self.goal_size / 2, 0.0)
        d = math.hypot(dx, dy)
        return d


class StateCostIntegralObjective(ob.OptimizationObjective):
    """Original State Cost Integral Objective implementation in python"""

    def __init__(self, si, enable_motion_cost_interpolation):
        super().__init__(si)
        self.si = si
        self.ss = si.getStateSpace()
        self.interpolation = enable_motion_cost_interpolation

    def motionCost(self, s1, s2):
        """Compute the cost of the motion from s1 to s2"""
        if self.interpolation:
            total_cost = self.identityCost()
            nd = self.ss.validSegmentCount(s1, s2)

            temp1 = self.si.cloneState(s1)
            temp2 = self.si.allocState()

            prev_cost = self.stateCost(temp1)
            for j in range(1, nd + 1):
                if j < nd:
                    t = float(j) / float(nd)
                    self.ss.interpolate(s1, s2, t, temp2)
                    curr = temp2
                else:
                    curr = s2

                curr_cost = self.stateCost(curr)
                seg_cost = self.trapezoid(
                    prev_cost, curr_cost, self.si.distance(temp1, curr)
                )
                total_cost = self.combineCosts(total_cost, seg_cost)

                if j < nd:
                    self.si.copyState(temp1, temp2)
                prev_cost = curr_cost

            return total_cost

        # No interpolation
        else:
            return self.trapezoid(
                self.stateCost(s1),
                self.stateCost(s2),
                self.si.distance(s1, s2),
            )

    def motionCostBestEstimate(self, s1, s2):
        return self.trapezoid(
            self.stateCost(s1), self.stateCost(s2), self.si.distance(s1, s2)
        )

    @staticmethod
    def trapezoid(c1, c2, dist):
        return ob.Cost(0.5 * dist * (c1.value() + c2.value()))

    def isMotionCostInterpolationEnabled(self):
        return self.interpolation


class FlappyOptimizationObjective(StateCostIntegralObjective):
    """
    Optimization objective for flappy bird.
    Integral clearance cost:
        ∫ 1 / clearance(x) ds
    Terminal cost:
        Φ(x) = terminal_weight * distance_to_goal
    """

    def __init__(
        self, si, obstacles, goal, terminal_weight=1.0, min_clearance=1e-2
    ):
        """Initialize the optimization objective for flappy bird"""
        super().__init__(si, True)
        self.si = si
        self.ss = si.getStateSpace()

        self.obstacles = obstacles
        self.goal = goal
        self.terminal_weight = terminal_weight
        self.min_clearance = min_clearance

    def stateCost(self, s):
        """Compute the cost of the state"""
        c = self.clearance(s)
        inv_clearance = 1.0 / max(c, self.min_clearance)
        return ob.Cost(inv_clearance)

    def clearance(self, state):
        """Check the clearance to the nearest obstacle"""
        x = float(state[0])
        y = float(state[1])
        # Distance to world boundaries
        d_bound = min(x - 0, WIDTH - x, y - 0, HEIGHT - y)
        d_bound = float("inf")
        # Distance to closest obstacle
        d_obs = point_dist_to_rects(x, y, self.obstacles)
        # return d_obs
        return min(d_bound, d_obs)

    def terminalCost(self, s):
        """Compute the terminal cost, considering only y-coordinate"""
        d = self.dist_to_goal(s)
        cost = self.terminal_weight * d
        return ob.Cost(cost)

    def dist_to_goal(self, pos):
        dx = float(pos[0]) - self.goal[0]
        dy = float(pos[1]) - self.goal[1]
        return math.hypot(dx, dy)


def test():
    seed = 100
    ou.RNG.setSeed(seed)
    np.random.seed(10)

    env = generate_flappy_env()
    # envs = np.load("data/planning_flappy_envs.npy")
    # env = envs[0]
    visualize_flappy_env(env)
    plt.show()

    planner = FlappyPlanner(env["obstacles"], "sst", terminal_weight=1.0)
    # times = list(np.linspace(0.1, 10.0, 100))
    times = [0.1, 0.4, 10.0]
    states, controls, costs = planner.plan(
        env["start"], env["goal"], env["goal_size"], times
    )
    for i in range(len(times)):
        print(f"{times[i]:.2f}: {costs[i][0]:.2f}, {costs[i][1]:.2f}")

    visualize_flappy_env(
        env,
        # [{"path": states[0], "color": "red"}],
        [{"path": states[i]} for i in range(len(times))],
    )
    plt.show()


if __name__ == "__main__":
    test()
