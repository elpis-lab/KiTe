import numpy as np
import matplotlib.pyplot as plt


def data_plot(configs, config_names):
    """Plot the results of the planning car experiments."""
    # Get all the results
    folder = f"results/planning_car"
    all_results = {}
    for i, config in enumerate(configs):
        algo = config[0]
        belief = config[1]
        terminal_weight = config[2]
        name = f"car_{algo}_{belief}_{terminal_weight}"
        results = np.load(f"{folder}/{name}_results.npy")
        all_results[name] = {
            "results": results,
            "config": config,
            "config_name": config_names[i],
        }

    # Plot 1: Planned costs vs planning timestamp
    plot_costs_vs_timestamp(all_results, folder)

    # Plot 2: Success rate vs planning timestamp
    plot_success_rate_vs_timestamp(all_results, folder)

    # Plot 3: Reached and hits vs planning timestamp
    plot_reached_hits_vs_timestamp(all_results, folder)

    # Plot 4: Terminal distance and success rate vs terminal weight (AORRT only)
    plot_terminal_weight_analysis(all_results, folder)


def _backfill_cost_to_avoid_jump(cost):
    """For each (rep, problem), set cost at time t to the cost at the first valid
    timestamp >= t, so early timestamps show the first-found plan cost (no jump).
    cost: (n_rep, n_problems, n_timestamps), invalid = nan.
    """
    out = np.array(cost, dtype=float)
    n_rep, n_prob, n_t = cost.shape
    for r in range(n_rep):
        for p in range(n_prob):
            row = out[r, p, :]
            # Find first valid index
            valid = np.isfinite(row) & (row >= 0.0)
            if not np.any(valid):
                continue
            first_valid = np.argmax(valid)
            # Backfill: timestamps 0..first_valid get the cost at first_valid
            out[r, p, : first_valid + 1] = row[first_valid]
    return out


def plot_costs_vs_timestamp(all_results, folder):
    """Plot running cost (solid) and AORRT-T total cost (dotted) vs planning time.

    Uses only selected configs: SST 0, AORRT 0, AORRT 50 (l2 and w2).
    Costs are backfilled so that if a plan is first found at 2s, that cost is
    shown at 1s too, avoiding an upward jump in the curve.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    planning_times = np.arange(1.0, 31.0)  # 1.0, 2.0, ..., 30.0

    for name, data in all_results.items():
        results = data["results"]
        config = data["config"]
        config_name = data["config_name"]
        algo = config[0]
        terminal_weight = config[2]

        if not (
            (algo == "sst" and terminal_weight == 0.0)
            or (algo == "aorrt" and terminal_weight in (0.0, 50.0))
        ):
            continue

        planned_running_cost = results[:, :, :, 7]
        planned_terminal_cost = results[:, :, :, 8]
        valid_mask = (planned_running_cost >= 0.0) & (planned_terminal_cost >= 0.0)

        running_cost = np.where(valid_mask, planned_running_cost.astype(float), np.nan)
        running_cost = _backfill_cost_to_avoid_jump(running_cost)

        running_mean_per_rep = np.nanmean(running_cost, axis=1)
        running_mean = np.nanmean(running_mean_per_rep, axis=0)
        running_std = np.nanstd(running_mean_per_rep, axis=0)

        ax.plot(
            planning_times,
            running_mean,
            label=config_name,
            linestyle="-",
        )
        ax.fill_between(
            planning_times,
            running_mean - running_std,
            running_mean + running_std,
            alpha=0.2,
        )

        if terminal_weight > 0.0:
            total_cost = np.where(
                valid_mask,
                (planned_running_cost + planned_terminal_cost).astype(float),
                np.nan,
            )
            total_cost = _backfill_cost_to_avoid_jump(total_cost)
            total_mean_per_rep = np.nanmean(total_cost, axis=1)
            total_mean = np.nanmean(total_mean_per_rep, axis=0)
            total_std = np.nanstd(total_mean_per_rep, axis=0)
            ax.plot(
                planning_times,
                total_mean,
                label=f"{config_name} (Total)",
                linestyle=":",
            )
            ax.fill_between(
                planning_times,
                total_mean - total_std,
                total_mean + total_std,
                alpha=0.2,
            )

    ax.set_xlabel("Planning Time (s)")
    ax.set_ylabel("Planned Cost")
    ax.set_title("Planned Costs vs Planning Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder}/costs_vs_timestamp.png")
    plt.show()


def plot_success_rate_vs_timestamp(all_results, folder):
    """Plot success rate vs planning time."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    planning_times = np.arange(1.0, 31.0)  # 1.0, 2.0, ..., 30.0

    for name, data in all_results.items():
        results = data["results"]
        config = data["config"]
        config_name = data["config_name"]
        algo = config[0]
        terminal_weight = config[2]

        if not (
            (algo == "sst" and terminal_weight == 0.0)
            or (algo == "aorrt" and terminal_weight in (0.0, 50.0))
        ):
            continue

        success = results[:, :, :, 0].astype(float)
        planned_running_cost = results[:, :, :, 7]
        planned_terminal_cost = results[:, :, :, 8]
        valid_mask = (planned_running_cost >= 0.0) & (planned_terminal_cost >= 0.0)
        success = np.where(valid_mask, success, np.nan)

        success_mean_per_rep = np.nanmean(success, axis=1)
        success_mean_overall = np.nanmean(success_mean_per_rep, axis=0)
        success_std_overall = np.nanstd(success_mean_per_rep, axis=0)

        ax.plot(planning_times, success_mean_overall, label=config_name)
        ax.fill_between(
            planning_times,
            success_mean_overall - success_std_overall,
            success_mean_overall + success_std_overall,
            alpha=0.2,
        )

    ax.set_xlabel("Planning Time (s)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs Planning Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder}/success_rate_vs_timestamp.png")
    plt.show()


def plot_reached_hits_vs_timestamp(all_results, folder):
    """Plot % reached and % hits vs planning time (percentage out of n_problems).

    Reached = fraction of (valid) plans that reached the goal at some point.
    Success (plot 2) = fraction that reached before hitting; so reached % >= success %.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    planning_times = np.arange(1.0, 31.0)
    n_problems = 20  # use fixed 20 for percentage scale

    for name, data in all_results.items():
        results = data["results"]
        config = data["config"]
        config_name = data["config_name"]
        algo = config[0]
        terminal_weight = config[2]

        if not (
            (algo == "sst" and terminal_weight == 0.0)
            or (algo == "aorrt" and terminal_weight in (0.0, 50.0))
        ):
            continue

        reached = results[:, :, :, 2].astype(float)
        hits = results[:, :, :, 1].astype(float)
        planned_running_cost = results[:, :, :, 7]
        planned_terminal_cost = results[:, :, :, 8]
        valid_mask = (planned_running_cost >= 0.0) & (planned_terminal_cost >= 0.0)
        reached = np.where(valid_mask, reached, np.nan)
        hits = np.where(valid_mask, hits, np.nan)

        # Mean rate across problems (dim 1), then mean/std across reps (dim 0)
        reached_rate_per_rep = np.nanmean(reached, axis=1)  # (n_rep, n_timestamps)
        hits_rate_per_rep = np.nanmean(hits, axis=1)
        reached_pct = np.nanmean(reached_rate_per_rep, axis=0) * 100.0
        reached_std = np.nanstd(reached_rate_per_rep, axis=0) * 100.0
        hits_pct = np.nanmean(hits_rate_per_rep, axis=0) * 100.0
        hits_std = np.nanstd(hits_rate_per_rep, axis=0) * 100.0

        ax.plot(
            planning_times,
            reached_pct,
            label=f"{config_name} (Reached)",
            linestyle="-",
        )
        ax.fill_between(
            planning_times,
            reached_pct - reached_std,
            reached_pct + reached_std,
            alpha=0.2,
        )

        ax.plot(
            planning_times,
            hits_pct,
            label=f"{config_name} (Hits)",
            linestyle="--",
        )
        ax.fill_between(
            planning_times,
            hits_pct - hits_std,
            hits_pct + hits_std,
            alpha=0.2,
        )

    ax.set_xlabel("Planning Time (s)")
    ax.set_ylabel("Percentage (out of 20 problems)")
    ax.set_title("Reached and Hits vs Planning Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{folder}/reached_hits_vs_timestamp.png")
    plt.show()


def plot_terminal_weight_analysis(all_results, folder):
    """Plot % reached (left) and terminal distance (right) vs terminal weight (AORRT only, last plan).
    Uses reached (ignore obstacles). Includes terminal weight 0 for reached. Both l2 and w2.
    """
    # Filter AORRT configs; group by belief (l2 vs w2)
    aorrt_by_belief = {"l2": {}, "w2": {}}
    for name, data in all_results.items():
        config = data["config"]
        if config[0] != "aorrt":
            continue
        belief = config[1]
        if belief not in aorrt_by_belief:
            continue
        aorrt_by_belief[belief][name] = data

    if not aorrt_by_belief["l2"] and not aorrt_by_belief["w2"]:
        print("No AORRT results found for terminal weight analysis")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax2 = ax1.twinx()

    colors_reached = {"l2": "tab:blue", "w2": "tab:cyan"}
    colors_dist = {"l2": "tab:red", "w2": "tab:orange"}

    for belief in ("l2", "w2"):
        aorrt_results = aorrt_by_belief[belief]
        if not aorrt_results:
            continue

        sorted_items = sorted(
            aorrt_results.items(),
            key=lambda x: x[1]["config"][2],
        )

        tw_reached = []
        reached_pct_mean = []
        reached_pct_std = []
        tw_dist = []
        dist_mean_list = []
        dist_std_list = []

        for name, data in sorted_items:
            results = data["results"]
            config = data["config"]
            terminal_weight = config[2]

            last_plan_results = results[:, :, -1, :]
            reached = last_plan_results[:, :, 2].astype(float)  # reached (ignore obstacles)
            planned_running_cost_last = last_plan_results[:, :, 7]
            planned_terminal_cost = last_plan_results[:, :, 8]
            valid_mask = (planned_running_cost_last >= 0.0) & (
                planned_terminal_cost >= 0.0
            )
            reached = np.where(valid_mask, reached, np.nan)

            # Reached % (all terminal weights including 0)
            reached_rate_per_rep = np.nanmean(reached, axis=1)
            reached_pct_mean.append(np.nanmean(reached_rate_per_rep) * 100.0)
            reached_pct_std.append(np.nanstd(reached_rate_per_rep) * 100.0)
            tw_reached.append(terminal_weight)

            # Terminal distance only for tw > 0
            if terminal_weight == 0:
                continue
            planned_terminal_cost_masked = np.where(
                valid_mask, planned_terminal_cost.astype(float), np.nan
            )
            terminal_distance = planned_terminal_cost_masked / terminal_weight
            terminal_distance_mean_per_rep = np.nanmean(terminal_distance, axis=1)
            dist_mean_list.append(np.nanmean(terminal_distance_mean_per_rep))
            dist_std_list.append(np.nanstd(terminal_distance_mean_per_rep))
            tw_dist.append(terminal_weight)

        tw_reached = np.array(tw_reached)
        reached_pct_mean = np.array(reached_pct_mean)
        reached_pct_std = np.array(reached_pct_std)

        ax1.plot(
            tw_reached,
            reached_pct_mean,
            color=colors_reached[belief],
            marker="o",
            label=f"Reached % ({belief})",
        )
        ax1.fill_between(
            tw_reached,
            reached_pct_mean - reached_pct_std,
            reached_pct_mean + reached_pct_std,
            alpha=0.2,
            color=colors_reached[belief],
        )

        if len(tw_dist) > 0:
            tw_dist = np.array(tw_dist)
            dist_mean = np.array(dist_mean_list)
            dist_std = np.array(dist_std_list)
            ax2.plot(
                tw_dist,
                dist_mean,
                color=colors_dist[belief],
                marker="s",
                label=f"Terminal distance ({belief})",
            )
            ax2.fill_between(
                tw_dist,
                dist_mean - dist_std,
                dist_mean + dist_std,
                alpha=0.2,
                color=colors_dist[belief],
            )

    ax1.set_xlabel("Terminal Weight")
    ax1.set_ylabel("Reached (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_ylabel("Planned Terminal Distance to Goal", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(
        "Reached % and Terminal Distance vs Terminal Weight (AORRT, Last Plan)"
    )
    plt.tight_layout()
    plt.savefig(f"{folder}/terminal_weight_analysis.png")
    plt.show()


if __name__ == "__main__":
    # Configs (algo, use_var, terminal_weight)
    configs = [
        ("sst", "l2", 0.0),  # Vanilla
        ("sst", "w2", 0.0),  # Gaussian Belief Trees
        ("aorrt", "l2", 0.0),  # Vanilla
        ("aorrt", "w2", 0.0),  # Gaussian Belief Trees
        ("aorrt", "l2", 2.0),  # Proposed
        ("aorrt", "w2", 2.0),  # Proposed
        ("aorrt", "l2", 5.0),  # Proposed
        ("aorrt", "w2", 5.0),  # Proposed
        ("aorrt", "l2", 10.0),  # Proposed
        ("aorrt", "w2", 10.0),  # Proposed
        ("aorrt", "l2", 20.0),  # Proposed
        ("aorrt", "w2", 20.0),  # Proposed
        ("aorrt", "l2", 50.0),  # Proposed
        ("aorrt", "w2", 50.0),  # Proposed
        ("aorrt", "l2", 100.0),  # Proposed
        ("aorrt", "w2", 100.0),  # Proposed
        ("aorrt", "l2", 200.0),  # Proposed
        ("aorrt", "w2", 200.0),  # Proposed
    ]

    config_names = [
        "Vanilla (SST-l2)",
        "Gaussian Belief Trees (SST-w2)",
        "Vanilla (AORRT-l2)",
        "Gaussian Belief Trees (AORRT-w2)",
        "Proposed (AORRT-T2-l2)",
        "Proposed (AORRT-T2-w2)",
        "Proposed (AORRT-T5-l2)",
        "Proposed (AORRT-T5-w2)",
        "Proposed (AORRT-T10-l2)",
        "Proposed (AORRT-T10-w2)",
        "Proposed (AORRT-T20-l2)",
        "Proposed (AORRT-T20-w2)",
        "Proposed (AORRT-T50-l2)",
        "Proposed (AORRT-T50-w2)",
        "Proposed (AORRT-T100-l2)",
        "Proposed (AORRT-T100-w2)",
        "Proposed (AORRT-T200-l2)",
        "Proposed (AORRT-T200-w2)",
    ]
    data_plot(configs, config_names)
