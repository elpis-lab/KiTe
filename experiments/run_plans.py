import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from models.torch_loss_se2 import mse_se2_loss
from geometry.pose import SE2Pose, angle_diff


def get_delta_states(states):
    """Get the intermediate delta states from the states"""
    delta_states = []
    for j in range(len(states) - 1):
        p1 = SE2Pose([states[j][0], states[j][1]], states[j][2])
        p2 = SE2Pose([states[j + 1][0], states[j + 1][1]], states[j + 1][2])
        delta_pose = p1.invert() @ p2
        delta_states.append([*delta_pose.pr()[0], delta_pose.pr()[1]])
    return delta_states


def run_plans(evaluate_fn, plan_states, plan_controls, reps_in_states=5):
    """Run the plans and collect the results"""
    # Execute Plans
    exec_states, hits, reacheds = evaluate_fn(plan_states, plan_controls)

    # Ge t plan delta states
    plan_delta_states = []
    for states in plan_states:
        plan_delta_states.append(get_delta_states(states))
    exec_delta_states = []
    for states in exec_states:
        exec_delta_states.append(get_delta_states(states))

    # Check path success
    # Reach goal before hitting obstacles?
    success = np.zeros(len(exec_states), dtype=bool)
    for i, (hit_seq, reached_seq) in enumerate(zip(hits, reacheds)):
        hit = np.asarray(hit_seq)
        reached = np.asarray(reached_seq)

        # never reached goal
        if not reached.any():
            success[i] = False
            continue
        # never hit obstacles
        if not hit.any():
            success[i] = True
            continue
        # reach goal before hitting obstacles?
        first_hit_idx = np.argmax(hit)
        first_reached_idx = np.argmax(reached)
        success[i] = first_reached_idx < first_hit_idx

    # Reduce to a single boolean value
    # if hit obstacles during execution
    hits = np.array([np.any(hit) for hit in hits])
    # if reached the goal
    reacheds = np.asarray([np.any(reached) for reached in reacheds])

    # Check path deviation
    errors = np.zeros(len(exec_states))
    pos_errors = np.zeros(len(exec_states))
    rot_errors = np.zeros(len(exec_states))
    for i, (delta_state1, delta_state2) in enumerate(
        zip(plan_delta_states, exec_delta_states)
    ):
        s1, s2 = np.array(delta_state1), np.array(delta_state2)
        # Compuate SE2 distance
        # (mse_se2_loss mean over x, y, theta, *3 to get original SE2 distance)
        mse_loss = 3 * mse_se2_loss(torch.from_numpy(s1), torch.from_numpy(s2))
        error = np.sqrt(mse_loss)
        pos_error = np.mean(np.linalg.norm(s1[:, :2] - s2[:, :2], axis=1))
        rot_error = np.mean(np.abs(angle_diff(s1[:, 2], s2[:, 2])))
        errors[i] = error
        pos_errors[i] = pos_error
        rot_errors[i] = rot_error
    # print(f"Error: {np.mean(errors)}")
    # print(f"Position error: {np.mean(pos_errors)}")
    # print(f"Rotation error: {np.mean(rot_errors)}")

    # Save results
    results = np.zeros((len(exec_states), 7))
    # 0, Plan is successful (not hitted, reached goal)
    # 1, Plan hits obstacles
    # 2, Plan reaches goal
    # 3, SE2 distance
    # 4, Position error
    # 5, Rotation error
    # 6, Number of controls
    results[:, 0] = success
    results[:, 1] = hits
    results[:, 2] = reacheds
    results[:, 3] = errors
    results[:, 4] = pos_errors
    results[:, 5] = rot_errors
    results[:, 6] = np.array([len(controls) for controls in plan_controls])

    # reshape to match reps_in_states
    # assume it is listed as [rep1_plan1, rep1_plan2, ..., rep2_plan1, ...]
    results = results.reshape(reps_in_states, -1, results.shape[1])
    # results = results.mean(axis=1)
    return results, exec_states
