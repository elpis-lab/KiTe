import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from ompl import base as ob
from ompl import control as oc
from ompl import util as ou

from planning.planning_utils import rects_circles_in_collision


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


class ClearanceObjective(ob.StateCostIntegralObjective):
    """
    Objective function that maximizes the clearance to the nearest obstacle.
    """

    def __init__(
        self, si, obstacle_poses, obstacle_rads, clearance_weight=0.1
    ):
        super().__init__(si, True)
        self.obstacle_poses = np.asarray(obstacle_poses)
        self.obstacle_rads = np.asarray(obstacle_rads)
        self.clearance_weight = clearance_weight

    def stateCost(self, state):
        return ob.Cost(self.clearance_weight * 1.0 / self.clearance(state))

    def clearance(self, state):
        """Compute the minimum distance to the nearest obstacle"""
        if len(self.obstacle_poses) == 0:
            return 0

        dists = np.linalg.norm(
            np.array([state.getX(), state.getY()]) - self.obstacle_poses,
            axis=1,
        )
        return np.min(dists)


def isStateValid(spaceInformation, obstacles, state):
    # perform collision checking or check if other constraints are
    # satisfied
    """Check if the state is in the bounds"""
    # In bounds
    in_bounds = spaceInformation.satisfiesBounds(state)
    if not in_bounds:
        return False

    # In collision
    if len(obstacles) > 0:
        pose = np.array([state.getX(), state.getY(), state.getYaw()])
        in_collision = rects_circles_in_collision(
            pose, [0.02, 0.02], obstacles
        )
        if in_collision:
            return False

    return True


def propagate(start, control, duration, state):
    state.setX(start.getX() + control[0] * duration * np.cos(start.getYaw()))
    state.setY(start.getY() + control[0] * duration * np.sin(start.getYaw()))
    state.setYaw(start.getYaw() + control[1] * duration)


def angle_diff(a: float, b: float) -> float:
    """Compute the signed angle difference between two angles"""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


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
    if len(edges) > 0:
        lc = LineCollection(edges, colors="gray", linewidths=0.5, alpha=0.6)
        ax.add_collection(lc)

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
    ou.RNG.setSeed(0)
    plan()
