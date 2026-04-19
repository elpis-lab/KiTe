import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"  # makes math look like Times
# avoid Type 3 fonts
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def backfill_cost_to_avoid_jump(cost):
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


def get_selected_car_plot_styles():
    """Selected methods for car plots: (label, color, linestyle)."""
    return {
        ("aorrt", "l2", 0.0): ("Base (AO-RRT, L2)", "C1", "-"),
        ("aorrt", "w2", 0.0): ("GBT (AO-RRT, W2)", "C2", "-"),
        ("aorrt", "l2", 50.0): ("KiTe (AO-RRT, L2, 50)", "C6", "-"),
        ("sst", "l2", 0.0): ("Base (SST, L2)", "C8", "-"),
        ("sst", "w2", 0.0): ("GBT (SST, W2)", "C9", "-"),
        ("aorrt", "w2", 50.0): ("KiTe (AO-RRT, W2, 50)", "C4", "-"),
    }


# Bottom method legend order matches get_selected_car_plot_styles() dict key order.
CAR_METHOD_KEYS_IN_LEGEND_ORDER = [
    ("aorrt", "l2", 0.0),
    ("aorrt", "w2", 0.0),
    ("aorrt", "l2", 50.0),
    ("sst", "l2", 0.0),
    ("sst", "w2", 0.0),
    ("aorrt", "w2", 50.0),
]

# Panel (d) bar order: indices 0, 3, 1, 4, 2, 5 relative to CAR_METHOD_KEYS_IN_LEGEND_ORDER.
CAR_DECOMPOSITION_BAR_KEYS = [
    ("aorrt", "l2", 0.0),
    ("sst", "l2", 0.0),
    ("aorrt", "w2", 0.0),
    ("sst", "w2", 0.0),
    ("aorrt", "l2", 50.0),
    ("aorrt", "w2", 50.0),
]


def plot_main_2x2(all_results, folder, text_size=14):
    """Create 2x2 panel:
    (a) running cost, (b) terminal cost, (c) success rate, (d) collisions/goals.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    planning_times = np.arange(1.0, 31.0)
    style_map = get_selected_car_plot_styles()
    n_methods = len(CAR_DECOMPOSITION_BAR_KEYS)
    bar_success = {k: np.nan for k in CAR_DECOMPOSITION_BAR_KEYS}
    bar_not_reached = {k: np.nan for k in CAR_DECOMPOSITION_BAR_KEYS}
    bar_collision = {k: np.nan for k in CAR_DECOMPOSITION_BAR_KEYS}

    line_by_key = {}

    kite_desired_goal_count = []
    other_desired_goal_count = []
    for _, data in all_results.items():
        results = data["results"]
        config = data["config"]
        key = (config[0], config[1], float(config[2]))
        if key not in style_map:
            continue

        label, color, linestyle = style_map[key]

        running = results[:, :, :, 11].astype(float)
        terminal = results[:, :, :, 12].astype(float)
        valid_mask = (running >= 0.0) & (terminal >= 0.0)

        # Count the number of desired goals reached
        terminal_last = terminal[:, :, -1].reshape(-1)
        is_desired = (terminal_last < 1).astype(int).tolist()
        if float(config[2]) > 0:
            kite_desired_goal_count.extend(is_desired)
        else:
            other_desired_goal_count.extend(is_desired)

        running_masked = np.where(valid_mask, running, np.nan)
        terminal_masked = np.where(valid_mask, terminal, np.nan)

        running_backfilled = backfill_cost_to_avoid_jump(running_masked)
        terminal_backfilled = backfill_cost_to_avoid_jump(terminal_masked)
        # evaluate based on terminal weight 50
        terminal_backfilled = terminal_backfilled * 50
        running_mean_per_rep = np.nanmean(running_backfilled, axis=1)
        terminal_mean_per_rep = np.nanmean(terminal_backfilled, axis=1)
        running_mean = np.nanmean(running_mean_per_rep, axis=0)
        running_std = np.nanstd(running_mean_per_rep, axis=0)
        terminal_mean = np.nanmean(terminal_mean_per_rep, axis=0)
        terminal_std = np.nanstd(terminal_mean_per_rep, axis=0)

        success = results[:, :, :, 0].astype(float)
        collisions = results[:, :, :, 1].astype(float)
        reached = results[:, :, :, 2].astype(float)

        n_rep, n_prob = success.shape[0], success.shape[1]
        all_cases = float(n_rep * n_prob)
        success_rate = (
            np.nansum(np.where(valid_mask, success, np.nan), axis=(0, 1))
            / all_cases
            * 100.0
        )

        line = ax_a.plot(
            planning_times,
            running_mean,
            color=color,
            linestyle=linestyle,
            label=label,
            linewidth=3.5,
        )[0]
        ax_a.fill_between(
            planning_times,
            running_mean - running_std,
            running_mean + running_std,
            color=color,
            alpha=0.2,
        )
        ax_b.plot(
            planning_times,
            terminal_mean,
            color=color,
            linestyle=linestyle,
            linewidth=3.5,
        )
        ax_b.fill_between(
            planning_times,
            terminal_mean - terminal_std,
            terminal_mean + terminal_std,
            color=color,
            alpha=0.2,
        )
        ax_c.plot(
            planning_times,
            success_rate,
            color=color,
            linestyle=linestyle,
            linewidth=3.5,
        )

        # Panel (d): decomposition at 30s as stacked bar
        valid_last = valid_mask[:, :, -1]
        hits_last = collisions[:, :, -1] > 0.5
        reached_last = reached[:, :, -1] > 0.5
        valid_count = np.sum(valid_last)

        if valid_count > 0:
            collision_count = np.sum(valid_last & hits_last)
            not_reached_count = np.sum(
                valid_last & (~hits_last) & (~reached_last)
            )
            success_count = np.sum(valid_last & (~hits_last) & reached_last)
            bar_collision[key] = collision_count / all_cases * 100.0
            bar_not_reached[key] = not_reached_count / all_cases * 100.0
            bar_success[key] = success_count / all_cases * 100.0

        line_by_key[key] = line

    print("Kite desired goal count: ", np.mean(kite_desired_goal_count))
    print("Other desired goal count: ", np.mean(other_desired_goal_count))

    legend_handles = []
    legend_labels = []
    for method_key in CAR_METHOD_KEYS_IN_LEGEND_ORDER:
        if method_key in line_by_key:
            label, _, _ = style_map[method_key]
            legend_handles.append(line_by_key[method_key])
            legend_labels.append(label)

    ax_a.set_title("(a) Running Cost", fontsize=text_size)
    ax_a.set_xlabel("Time (s)", fontsize=text_size)
    ax_a.set_ylabel("Cost", fontsize=text_size)
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_a.tick_params(axis="both", labelsize=text_size)
    ax_a.grid(True, alpha=0.3)

    ax_b.set_title("(b) Terminal Cost", fontsize=text_size)
    ax_b.set_xlabel("Time (s)", fontsize=text_size)
    ax_b.set_ylabel("Cost", fontsize=text_size)
    ax_b.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_b.tick_params(axis="both", labelsize=text_size)
    ax_b.grid(True, alpha=0.3)

    ax_c.set_title("(c) Success Rate", fontsize=text_size)
    ax_c.set_xlabel("Time (s)", fontsize=text_size)
    ax_c.set_ylabel(r"Success Rate (\%)", fontsize=text_size)
    ax_c.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_c.tick_params(axis="both", labelsize=text_size)
    ax_c.set_ylim(0.0, 102.0)
    ax_c.grid(True, alpha=0.3)

    x_pos = np.arange(n_methods)
    bar_success_plot = np.array(
        [bar_success[k] for k in CAR_DECOMPOSITION_BAR_KEYS]
    )
    bar_not_reached_plot = np.array(
        [bar_not_reached[k] for k in CAR_DECOMPOSITION_BAR_KEYS]
    )
    bar_collision_plot = np.array(
        [bar_collision[k] for k in CAR_DECOMPOSITION_BAR_KEYS]
    )
    bar_colors_plot = [style_map[k][1] for k in CAR_DECOMPOSITION_BAR_KEYS]
    bar_labels_plot = [style_map[k][0] for k in CAR_DECOMPOSITION_BAR_KEYS]
    x_pos = np.array(
        [0.9, 1.3, 1.7, 2.1, 2.5, 2.9]
    )  # tighter cluster near center
    bar_width = 0.22
    ax_d.bar(
        x_pos,
        bar_success_plot,
        width=bar_width,
        color=bar_colors_plot,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_d.bar(
        x_pos,
        bar_not_reached_plot,
        width=bar_width,
        bottom=bar_success_plot,
        color="0.6",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_d.bar(
        x_pos,
        bar_collision_plot,
        width=bar_width,
        bottom=bar_success_plot + bar_not_reached_plot,
        color="black",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_d.set_xlim(0.5, 3.3)

    ax_d.set_title("(d) Success Decomposition at 30s", fontsize=text_size)
    ax_d.set_ylabel(r"Plan Fraction (\%)", fontsize=text_size)
    ax_d.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_d.tick_params(axis="both", labelsize=text_size)
    # ax_d.set_xticks(x_pos)
    # ax_d.set_xticklabels(bar_labels_plot, rotation=20, ha="right")
    ax_d.set_xticks([])
    ax_d.set_xlabel("")
    ax_d.set_ylim(0.0, 102.0)
    ax_d.grid(True, alpha=0.3)

    method_legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.38, -0.015),
        ncol=2,
        frameon=True,
        fontsize=text_size,
    )
    decomp_handles = [
        Patch(facecolor="black", edgecolor="black", label="In Collision"),
        Patch(facecolor="0.6", edgecolor="black", label="Goal Not\nReached"),
    ]
    fig.legend(
        decomp_handles,
        [h.get_label() for h in decomp_handles],
        loc="lower center",
        bbox_to_anchor=(0.88, 0.0),
        ncol=1,
        frameon=True,
        fontsize=text_size,
    )
    fig.add_artist(method_legend)
    fig.tight_layout(rect=[0.0, 0.16, 1.0, 1.0])
    ax_d.xaxis.label.set_visible(False)
    plt.savefig(f"{folder}/car_result.pdf")


def plot_terminal_weight_analysis(all_results, folder, text_size=14):
    """Terminal-weight sweep using KiTe AO-RRT (l2/w2), at last timestamp only.
    Plots success rate (o, solid) and terminal cost (s, dashed).
    """
    by_belief = {"l2": [], "w2": []}
    for _, data in all_results.items():
        algo, belief, terminal_weight = data["config"]
        if algo != "aorrt" or belief not in by_belief:
            continue
        by_belief[belief].append(data)

    if not by_belief["l2"] and not by_belief["w2"]:
        print("No KiTe AO-RRT terminal-weight results found")
        return

    fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
    ax2 = ax1.twinx()
    colors = {"l2": "C6", "w2": "C4"}
    method_lines = []
    metric_lines = [
        Line2D(
            [0],
            [0],
            color="k",
            marker="o",
            linestyle="-",
            linewidth=3.5,
            label="Success Rate",
        ),
        Line2D(
            [0],
            [0],
            color="k",
            marker="s",
            linestyle="--",
            linewidth=3.5,
            label="Wasserstein\nDistance",
        ),
    ]

    for belief in ("l2", "w2"):
        items = sorted(by_belief[belief], key=lambda d: d["config"][2])
        if not items:
            continue

        terminal_weights = []
        success_rates = []
        wd_mean = []
        wd_std = []

        for data in items:
            results = data["results"]
            terminal_weight = data["config"][2]
            last_plan = results[:, :, -1, :]

            running_last = last_plan[:, :, 11]
            terminal_last = last_plan[:, :, 12]
            valid_mask = (running_last >= 0.0) & (terminal_last >= 0.0)

            success_last = np.where(
                valid_mask, last_plan[:, :, 2].astype(float), np.nan
            )
            terminal_cost_last = np.where(
                valid_mask, terminal_last.astype(float), np.nan
            )

            n_rep, n_prob = success_last.shape
            all_cases = float(n_rep * n_prob)
            success_rate = (
                np.nansum(success_last) / all_cases * 100.0
            )  # pure line, no shading

            terminal_wd = terminal_cost_last  # / terminal_weight
            terminal_wd_per_rep = np.nanmean(terminal_wd, axis=1)

            terminal_weights.append(terminal_weight)
            success_rates.append(success_rate)
            wd_mean.append(np.nanmean(terminal_wd_per_rep))
            wd_std.append(np.nanstd(terminal_wd_per_rep))

        method_line = ax1.plot(
            terminal_weights,
            success_rates,
            color=colors[belief],
            marker="o",
            linestyle="-",
            linewidth=3.5,
        )[0]
        ax2.plot(
            terminal_weights,
            wd_mean,
            color=colors[belief],
            marker="s",
            linestyle="--",
            linewidth=3.5,
        )
        ax2.fill_between(
            terminal_weights,
            np.array(wd_mean) - np.array(wd_std),
            np.array(wd_mean) + np.array(wd_std),
            color=colors[belief],
            alpha=0.2,
        )
        method_lines.append(
            Line2D(
                [0],
                [0],
                color=colors[belief],
                linestyle="-",
                linewidth=3.5,
                label=f"KiTe\n(AO-RRT, {belief.upper()})",
            )
        )

    ax1.set_xlabel("Terminal Weight", fontsize=text_size)
    ax1.set_ylabel(r"Goal-reaching Success Rate (\%)", fontsize=text_size)
    ax2.set_ylabel(
        "Terminal Wasserstein Distance \n to The Goal", fontsize=text_size
    )
    ax1.set_ylim(50.0, 102.0)
    # ax2.set_ylim(0.01, 0.1)
    # ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.tick_params(axis="both", labelsize=text_size)
    ax2.tick_params(axis="both", labelsize=text_size)
    ax1.set_xticks(np.arange(0, 210, 20))
    ax2.set_xticks(np.arange(0, 210, 20))
    ax1.grid(True, alpha=0.3)

    legend_handles = method_lines + metric_lines
    ax1.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
        frameon=True,
        fontsize=text_size,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.05, 1.0])
    plt.savefig(f"{folder}/car_terminal.pdf")


def data_plot(configs):
    """Plot the results of the planning car experiments."""
    main_plot_text_size = 28
    terminal_plot_text_size = 28

    # Get all the results
    folder = f"results/planning_car"
    all_results = {}
    for i, config in enumerate(configs):
        algo = config[0]
        belief = config[1]
        terminal_weight = config[2]
        name = f"car_{algo}_{belief}_{terminal_weight}"
        results = np.load(f"{folder}/{name}_results.npy")
        all_results[name] = {"results": results, "config": config}

    # Plot 1: 2x2 main panel
    plot_main_2x2(
        all_results,
        folder,
        text_size=main_plot_text_size,
    )
    # Plot 2: Terminal weight analysis (KiTe AO-RRT only)
    plot_terminal_weight_analysis(
        all_results,
        folder,
        text_size=terminal_plot_text_size,
    )
    plt.show()


if __name__ == "__main__":
    # Configs (algo, use_var, terminal_weight)
    configs = [
        ("aorrt", "l2", 0.0),  # Base
        ("aorrt", "w2", 0.0),  # Gaussian Belief Trees
        ("aorrt", "l2", 50.0),  # KiTe
        ("sst", "l2", 0.0),  # Base
        ("sst", "w2", 0.0),  # Gaussian Belief Trees
        ("aorrt", "w2", 50.0),  # KiTe
        #
        ("aorrt", "l2", 5.0),  # KiTe
        ("aorrt", "w2", 5.0),  # KiTe
        ("aorrt", "l2", 10.0),  # KiTe
        ("aorrt", "w2", 10.0),  # KiTe
        ("aorrt", "l2", 20.0),  # KiTe
        ("aorrt", "w2", 20.0),  # KiTe
        ("aorrt", "l2", 100.0),  # KiTe
        ("aorrt", "w2", 100.0),  # KiTe
        ("aorrt", "l2", 200.0),  # KiTe
        ("aorrt", "w2", 200.0),  # KiTe
    ]
    data_plot(configs)
