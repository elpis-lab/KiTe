import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from planning.flappy import visualize_flappy_env, generate_flappy_env
from planning.flappy import FlappyPlanner

import matplotlib as mpl

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"  # makes math look like Times
# avoid Type 3 fonts
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def process_cost(costs: np.ndarray, which: int, states: np.ndarray):
    """
    costs: (R, P, T, 2) with -1.0 for not valid
    which: 0=running, 1=terminal
    Returns:
      mean_curve: (T,)
      std_curve : (T,)  std across reps of (mean over problems)
    """
    # Goal
    envs = np.load("data/planning_flappy_envs.npy", allow_pickle=True)
    goal = np.array(envs[0]["goal"])

    # running cost
    if which == 0:
        x = costs[..., which]
    # terminal cost
    else:
        # x = costs[..., which]
        # Compute terminal distance to the goal
        r, p, t = states.shape
        terminals = np.array(
            [
                [[states[r, p, t][-1][:2] for t in range(t)] for p in range(p)]
                for r in range(r)
            ]
        )
        x = np.linalg.norm(terminals - goal, axis=-1)
    x = np.where(x < 0.0, np.nan, x)  # -1.0 for not valid

    # Per-rep mean over problems: (R,T)
    per_rep = np.nanmean(x, axis=1)

    # Across-rep mean+std: (T,)
    mean_curve = np.nanmean(per_rep, axis=0)
    std_curve = np.nanstd(per_rep, axis=0, ddof=1)  # 10 reps => ddof=1
    return mean_curve, std_curve


def plot_costs(
    title="Cost vs Planning Time (mean over problems; ±1 std across reps)",
    root="results/planning_flappy",
    show_error_band=True,
    error_band_alpha=0.18,
):
    # rep x n_problems x n_times x 2 (running, terminal)
    sst = np.load(root + "/sst_0.0_costs.npy")
    aorrt = np.load(root + "/aorrt_0.0_costs.npy")
    kite = np.load(root + "/aorrt_1.0_costs.npy")
    sst_s = np.load(root + "/sst_0.0_plan_states.npy", allow_pickle=True)
    aorrt_s = np.load(root + "/aorrt_0.0_plan_states.npy", allow_pickle=True)
    kite_s = np.load(root + "/aorrt_1.0_plan_states.npy", allow_pickle=True)

    # Running
    sst_run_mean, sst_run_std = process_cost(sst, 0, sst_s)
    aorrt_run_mean, aorrt_run_std = process_cost(aorrt, 0, aorrt_s)
    kite_run_mean, kite_run_std = process_cost(kite, 0, kite_s)
    # AORRT-T terminal
    sst_term_mean, sst_term_std = process_cost(sst, 1, sst_s)
    aorrt_term_mean, aorrt_term_std = process_cost(aorrt, 1, aorrt_s)
    kite_term_mean, kite_term_std = process_cost(kite, 1, kite_s)

    text_size = 18

    # Same method -> same color in both panels (matplotlib tab10 shorthand).
    col_aorrt, col_sst, col_kite = "C1", "C8", "C6"
    lbl_aorrt = "Base (AO-RRT)"
    lbl_sst = "Base (SST)"
    lbl_kite = "KiTe (AO-RRT, 1)"

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharex=True, sharey=False)

    planning_times = np.arange(0.1, 10.01, 0.1)
    axes[0].plot(
        planning_times,
        aorrt_run_mean,
        "-",
        linewidth=2.5,
        color=col_aorrt,
        label=lbl_aorrt,
    )
    axes[0].plot(
        planning_times,
        sst_run_mean,
        "-",
        linewidth=2.5,
        color=col_sst,
        label=lbl_sst,
    )
    axes[0].plot(
        planning_times,
        kite_run_mean,
        "-",
        linewidth=2.5,
        color=col_kite,
        label=lbl_kite,
    )

    if show_error_band:
        axes[0].fill_between(
            planning_times,
            aorrt_run_mean - aorrt_run_std,
            aorrt_run_mean + aorrt_run_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=col_aorrt,
        )
        axes[0].fill_between(
            planning_times,
            sst_run_mean - sst_run_std,
            sst_run_mean + sst_run_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=col_sst,
        )
        axes[0].fill_between(
            planning_times,
            kite_run_mean - kite_run_std,
            kite_run_mean + kite_run_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=col_kite,
        )

    axes[1].plot(
        planning_times,
        aorrt_term_mean,
        "-",
        linewidth=2.5,
        color=col_aorrt,
        label=lbl_aorrt,
    )
    axes[1].plot(
        planning_times,
        sst_term_mean,
        "-",
        linewidth=2.5,
        color=col_sst,
        label=lbl_sst,
    )
    axes[1].plot(
        planning_times,
        kite_term_mean,
        "-",
        linewidth=2.5,
        color=col_kite,
        label=lbl_kite,
    )

    if show_error_band:
        axes[1].fill_between(
            planning_times,
            aorrt_term_mean - aorrt_term_std,
            aorrt_term_mean + aorrt_term_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=col_aorrt,
        )
        axes[1].fill_between(
            planning_times,
            sst_term_mean - sst_term_std,
            sst_term_mean + sst_term_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=col_sst,
        )
        axes[1].fill_between(
            planning_times,
            kite_term_mean - kite_term_std,
            kite_term_mean + kite_term_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=col_kite,
        )

    # Running panel: y-range from data (do not force zero at bottom).
    run_lo = np.nanmin(
        np.stack(
            [
                sst_run_mean - (sst_run_std if show_error_band else 0.0),
                aorrt_run_mean - (aorrt_run_std if show_error_band else 0.0),
                kite_run_mean - (kite_run_std if show_error_band else 0.0),
            ]
        )
    )
    run_hi = np.nanmax(
        np.stack(
            [
                sst_run_mean + (sst_run_std if show_error_band else 0.0),
                aorrt_run_mean + (aorrt_run_std if show_error_band else 0.0),
                kite_run_mean + (kite_run_std if show_error_band else 0.0),
            ]
        )
    )
    run_pad = 0.06 * (run_hi - run_lo) if np.isfinite(run_hi - run_lo) else 1.0
    axes[0].set_ylim(run_lo - run_pad - 5, run_hi + run_pad + 5)
    axes[1].set_ylim(0, 220)

    axes[0].set_title("Running Cost", fontsize=text_size)
    axes[1].set_title(
        "Terminal Distance to the Goal Center", fontsize=text_size
    )

    axes[0].set_xlabel("Planning time", fontsize=text_size)
    axes[0].set_ylabel("Cost (px)", fontsize=text_size)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="both", labelsize=text_size - 1)

    axes[1].set_xlabel("Planning time", fontsize=text_size)
    # axes[1].set_ylabel("Cost (px)", fontsize=text_size)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis="both", labelsize=text_size - 1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=text_size - 1,
        frameon=True,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    fig.savefig(root + "/flappy_cost.pdf", dpi=300, bbox_inches="tight")

    return fig, axes


def plot_cases(
    root="results/planning_flappy",
    title="",
    text_size=20,
):
    """Single case-study figure: env + five trajectories, shared method colors."""
    np.random.seed(10)
    env = generate_flappy_env()

    s1 = np.load(root + "/aorrt_1.0_case_study_states.npy", allow_pickle=True)
    s2 = np.load(root + "/aorrt_0.0_case_study_states.npy", allow_pickle=True)
    s3 = np.load(root + "/sst_0.0_case_study_states.npy", allow_pickle=True)

    for i in range(len(s1)):
        if len(s1[i]) > 1:
            first_i = i
            break
    inter_i = min(first_i + int(1 / 0.01), len(s1) - 1)

    col_aorrt, col_sst, col_kite = "C0", "C9", "C6"
    lw_base = 2.5
    # KiTe time-slices: same color, solid lines; distinguish by alpha + linewidth.
    lw_kite_first, lw_kite_mid, lw_kite_final = 1.0, 1.6, lw_base
    alpha_kite_first, alpha_kite_mid, alpha_kite_final = 0.6, 0.8, 1.0

    paths = [
        {
            "path": s2[-1],
            "color": col_aorrt,
            "linestyle": "-",
            "linewidth": lw_base,
            "label": None,
        },
        {
            "path": s3[-1],
            "color": col_sst,
            "linestyle": "-",
            "linewidth": lw_base,
            "label": None,
        },
        {
            "path": s1[first_i],
            "color": col_kite,
            "linestyle": "-",
            "linewidth": lw_kite_first,
            "alpha": alpha_kite_first,
            "label": None,
        },
        {
            "path": s1[inter_i],
            "color": col_kite,
            "linestyle": "-",
            "linewidth": lw_kite_mid,
            "alpha": alpha_kite_mid,
            "label": None,
        },
        {
            "path": s1[-1],
            "color": col_kite,
            "linestyle": "-",
            "linewidth": lw_kite_final,
            "alpha": alpha_kite_final,
            "label": None,
        },
    ]

    fig, ax = visualize_flappy_env(
        env,
        paths,
        title=title,
        text_size=text_size,
        auto_legend=False,
        env_legend_labels=False,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="w",
            markerfacecolor="C3",
            markersize=9,
            linestyle="None",
            label="Start",
        ),
        Patch(
            facecolor="black",
            edgecolor="black",
            alpha=0.8,
            linewidth=0.5,
            label="Obstacles",
        ),
        Patch(
            facecolor="#2ca02c",
            edgecolor="#1f7a1f",
            alpha=0.55,
            linewidth=0.5,
            label="Goal",
        ),
        Line2D(
            [],
            [],
            color=col_aorrt,
            linestyle="-",
            linewidth=lw_base,
            label="Base (AO-RRT)",
        ),
        Line2D(
            [],
            [],
            color=col_sst,
            linestyle="-",
            linewidth=lw_base,
            label="Base (SST)",
        ),
        Line2D(
            [],
            [],
            color=col_kite,
            linestyle="-",
            linewidth=lw_kite_first,
            alpha=alpha_kite_first,
            label="KiTe (AO-RRT, 1)\nFirst",
        ),
        Line2D(
            [],
            [],
            color=col_kite,
            linestyle="-",
            linewidth=lw_kite_mid,
            alpha=alpha_kite_mid,
            label="KiTe (AO-RRT, 1)\nIntermediate",
        ),
        Line2D(
            [],
            [],
            color=col_kite,
            linestyle="-",
            linewidth=lw_kite_final,
            alpha=alpha_kite_final,
            label="KiTe (AO-RRT, 1)\nFinal",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=text_size - 1,
        frameon=True,
    )
    fig.tight_layout(rect=(0.03, 0, 0.97, 1))
    fig.savefig(root + "/flappy_demo.pdf", dpi=300, bbox_inches="tight")

    return fig, ax


if __name__ == "__main__":
    base = "results/planning_flappy"

    plot_costs(root=base, title="")
    plot_cases(root=base, title="")

    plt.show()
