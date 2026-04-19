import os, sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from models.torch_loss_se2 import mse_se2_loss
from geometry.pose import SE2Pose, angle_diff


class RunPlans:
    """An abstract class for running the plans and get the analysis results"""

    def __init__(self):
        """Initialization"""
        return

    def run_plans(self, plan_states, plan_controls, **kwargs):
        """
        Run the plans and collect the results

        Expected to return:
        exec_states: list of executed states
        hits: list of boolean values indicating if hit obstacles
        reacheds: list of boolean values indicating if reached the goal
        exec_costs: list of executed costs
        """
        raise NotImplementedError

    def get_delta_states(self, states):
        """Get the intermediate delta states from the states"""
        delta_states = []
        for j in range(len(states) - 1):
            p1 = SE2Pose([states[j][0], states[j][1]], states[j][2])
            p2 = SE2Pose(
                [states[j + 1][0], states[j + 1][1]], states[j + 1][2]
            )
            delta_pose = p1.invert() @ p2
            delta_states.append([*delta_pose.pr()[0], delta_pose.pr()[1]])
        return delta_states

    def evaluate_plans(self, plan_states, plan_controls, plan_costs, **kwargs):
        """
        Run the plans and collect the results
        plan_states is a list of plans (may be different length),
        where each plan is a list/array of plan states.
        plan_controls is the corresponding controls.
        plan_costs is the corresponding costs.
        """
        # Execute multiple plans
        plan_costs = np.array(plan_costs, dtype=float)
        exec_states, hits, reacheds, exec_costs, unified_costs = (
            self.run_plans(plan_states, plan_controls, **kwargs)
        )

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

        # Get delta states to compare
        plan_deltas = [self.get_delta_states(states) for states in plan_states]
        exec_deltas = [self.get_delta_states(states) for states in exec_states]

        # Check path deviation (delta state error)
        errors = np.zeros(len(exec_states))
        pos_errors = np.zeros(len(exec_states))
        rot_errors = np.zeros(len(exec_states))
        for i, (deltas1, deltas2) in enumerate[tuple[list, list]](
            zip(plan_deltas, exec_deltas)
        ):
            s1 = np.array(deltas1, dtype=float)
            s2 = np.array(deltas2, dtype=float)
            if s1.shape[0] == 0 and s2.shape[0] == 0:
                continue
            # Compuate SE2 distance.
            # mse_se2_loss mean over x, y, theta
            # *3 to get original SE2 distance
            mse_loss = 3 * mse_se2_loss(
                torch.from_numpy(s1), torch.from_numpy(s2)
            )
            error = np.sqrt(mse_loss)
            pos_error = np.mean(np.linalg.norm(s1[:, :2] - s2[:, :2], axis=1))
            rot_error = np.mean(np.abs(angle_diff(s1[:, 2], s2[:, 2])))
            errors[i] = error
            pos_errors[i] = pos_error
            rot_errors[i] = rot_error

        # Save results
        results = np.zeros((len(exec_states), 13))
        # 0, Plan is successful (not hitted, reached goal)
        # 1, Plan hits obstacles
        # 2, Plan reaches goal
        # 3, SE2 distance
        # 4, Position error
        # 5, Rotation error
        # 6, Number of controls
        # 7, Planned running cost
        # 8, Planned terminal cost
        # 9, Executed running cost
        # 10, Executed terminal cost
        # 11, Unified running cost (in belief space)
        # 12, Unified terminal cost (in belief space)
        results[:, 0] = success
        results[:, 1] = hits
        results[:, 2] = reacheds
        results[:, 3] = errors
        results[:, 4] = pos_errors
        results[:, 5] = rot_errors
        results[:, 6] = np.array([len(controls) for controls in plan_controls])
        results[:, 7] = plan_costs[:, 0]
        results[:, 8] = plan_costs[:, 1]
        results[:, 9] = exec_costs[:, 0]
        results[:, 10] = exec_costs[:, 1]
        results[:, 11] = unified_costs[:, 0]
        results[:, 12] = unified_costs[:, 1]
        return results, exec_states

    def evaluate(self, plan_states, plan_controls, plan_costs):
        """
        Evaluate plans with shape (n_repetition, n_problems, n_timestamps).

        plan_states, plan_controls: numpy arrays with dtype object, shape
            (n_rep, n_problems, n_timestamps). Each cell is a plan (list of
            states) and corresponding list of controls.
        plan_costs: array of shape (n_rep, n_problems, n_timestamps, 2) with
            [running_cost, terminal_cost] per plan.
        """
        plan_states = np.asarray(plan_states, dtype=object)
        plan_controls = np.asarray(plan_controls, dtype=object)
        plan_costs = np.asarray(plan_costs, dtype=float)
        n_rep, n_problems, n_timestamps = plan_states.shape
        n_total = n_rep * n_problems * n_timestamps

        # When no new solution is found at a timestamp, the same plan
        # is stored for that timestamp. We deduplicate by plan identity
        # (id) so each distinct plan is only run once
        # Flatten to 1D
        states_flat = plan_states.reshape(-1)
        controls_flat = plan_controls.reshape(-1)
        # plan_costs: (n_rep, n_problems, n_timestamps, 2) -> (n_total, 2)
        costs_flat = plan_costs.reshape(-1, plan_costs.shape[-1])

        # Deduplicate by (id(states), id(controls))
        # so inherited plans are only run once
        seen = {}
        idx_to_grid_idx = {}
        unique_states = []
        unique_controls = []
        unique_costs = []
        flat_to_unique = np.zeros(n_total, dtype=np.intp)
        for i in range(n_total):
            key = (id(states_flat[i]), id(controls_flat[i]))
            if key not in seen:
                idx = len(unique_states)
                seen[key] = idx
                r, p, t = np.unravel_index(
                    i, (n_rep, n_problems, n_timestamps)
                )
                idx_to_grid_idx[idx] = (int(r), int(p), int(t))
                unique_states.append(states_flat[i])
                unique_controls.append(controls_flat[i])
                unique_costs.append(costs_flat[i])
            flat_to_unique[i] = seen[key]
        unique_costs = np.array(unique_costs, dtype=float)

        # Evaluate each unique plan once
        results_unique, exec_states_unique = self.evaluate_plans(
            unique_states,
            unique_controls,
            unique_costs,
            idx_flat_to_grid=idx_to_grid_idx,
        )
        # Map results back to flat index
        results_flat = results_unique[flat_to_unique]
        exec_states_flat = np.array(
            [exec_states_unique[j] for j in flat_to_unique], dtype=object
        )

        # Reshape to (n_rep, n_problems, n_timestamps, -1)
        results = results_flat.reshape(n_rep, n_problems, n_timestamps, -1)
        exec_states = exec_states_flat.reshape(n_rep, n_problems, n_timestamps)
        return results, exec_states
