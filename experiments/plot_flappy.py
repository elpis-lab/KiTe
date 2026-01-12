import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from planning.flappy import (
    visualize_flappy_env,
    generate_flappy_env,
    FlappyPlanner,
)
import ompl.util as ou


def process_cost(costs: np.ndarray, which: int):
    """
    costs: (R, P, T, 2) with -1.0 for not valid
    which: 0=running, 1=terminal
    Returns:
      mean_curve: (T,)
      std_curve : (T,)  std across reps of (mean over problems)
    """
    # (R,P,T)
    x = costs[..., which]
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
    sst = np.load(os.path.join(root, "sst_0.0_costs.npy"))
    aorrt = np.load(os.path.join(root, "aorrt_0.0_costs.npy"))
    at = np.load(os.path.join(root, "aorrt_1.0_costs.npy"))

    # Running
    sst_run_mean, sst_run_std = process_cost(sst, which=0)
    aorrt_run_mean, aorrt_run_std = process_cost(aorrt, which=0)
    at_run_mean, at_run_std = process_cost(at, which=0)
    # AORRT-T terminal
    at_term_mean, at_term_std = process_cost(at, which=1)

    # AORRT-T total (running + terminal) computed before stats
    at_total = (at[..., 0] + at[..., 1])[..., None]  # (R,P,T,1)
    at_total_pack = np.concatenate([at_total, at_total], -1)  # fake (R,P,T,2)
    at_tot_mean, at_tot_std = process_cost(at_total_pack, which=0)

    fig, ax = plt.subplots(figsize=(6, 6))

    planning_times = np.arange(0.1, 10.01, 0.1)
    (l_sst,) = ax.plot(
        planning_times, sst_run_mean, "-", linewidth=2.5, label="SST (running)"
    )
    (l_aorrt,) = ax.plot(
        planning_times,
        aorrt_run_mean,
        "-",
        linewidth=2.5,
        label="AORRT (running)",
    )
    (l_at,) = ax.plot(
        planning_times,
        at_run_mean,
        "-",
        linewidth=2.5,
        label="AORRT-T (running)",
    )

    if show_error_band:
        ax.fill_between(
            planning_times,
            sst_run_mean - sst_run_std,
            sst_run_mean + sst_run_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=l_sst.get_color(),
        )
        ax.fill_between(
            planning_times,
            aorrt_run_mean - aorrt_run_std,
            aorrt_run_mean + aorrt_run_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=l_aorrt.get_color(),
        )
        ax.fill_between(
            planning_times,
            at_run_mean - at_run_std,
            at_run_mean + at_run_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=l_at.get_color(),
        )

    c = l_at.get_color()
    ax.plot(
        planning_times,
        at_term_mean,
        ":",
        linewidth=2.5,
        color=c,
        label="AORRT-T (terminal)",
    )
    ax.plot(
        planning_times,
        at_tot_mean,
        "-.",
        linewidth=2.5,
        color=c,
        label="AORRT-T (total)",
    )

    if show_error_band:
        ax.fill_between(
            planning_times,
            at_term_mean - at_term_std,
            at_term_mean + at_term_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=c,
        )
        ax.fill_between(
            planning_times,
            at_tot_mean - at_tot_std,
            at_tot_mean + at_tot_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=c,
        )

    ax.set_xlabel("Planning time (s)")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(root + "/cost_curves.png", dpi=200, bbox_inches="tight")

    return fig, ax


def plot_cases(
    root="results/planning_flappy",
    title1="AORRT-T Planning Paths Over Time",
    title2="Planning Paths Comparison",
):
    np.random.seed(10)
    env = generate_flappy_env()
    start, goal, goal_size = env["start"], env["goal"], env["goal_size"]
    times = np.arange(0.1, 10.01, 0.1)

    # ou.RNG.setSeed(10)
    # p1 = FlappyPlanner(env["obstacles"], "aorrt", terminal_weight=1.0)
    # s1, _, _ = p1.plan(start, goal, goal_size, np.arange(0.01, 10.01, 0.01))
    # s1 = np.array(s1, dtype=object)
    # np.save(root + "/aorrt_1.0_case_study_states.npy", s1)

    # ou.RNG.setSeed(50)
    # p2 = FlappyPlanner(env["obstacles"], "aorrt", terminal_weight=0.0)
    # s2, _, _ = p2.plan(start, goal, goal_size, times)
    # s2 = np.array(s2, dtype=object)
    # np.save(root + "/aorrt_0.0_case_study_states.npy", s2)

    # ou.RNG.setSeed(10)
    # p3 = FlappyPlanner(env["obstacles"], "sst", terminal_weight=0.0)
    # s3, _, _ = p3.plan(start, goal, goal_size, times)
    # s3 = np.array(s3, dtype=object)
    # np.save(root + "/sst_0.0_case_study_states.npy", s3)

    s1 = np.load(root + "/aorrt_1.0_case_study_states.npy", allow_pickle=True)
    s2 = np.load(root + "/aorrt_0.0_case_study_states.npy", allow_pickle=True)
    s3 = np.load(root + "/sst_0.0_case_study_states.npy", allow_pickle=True)

    for i in range(len(s1)):
        if len(s1[i]) > 1:
            first_i = i
            break
    inter_i = first_i + int(1 / 0.01)
    fig, ax = visualize_flappy_env(
        env,
        [
            {"path": s1[-1], "linewidth": 2.5, "label": "AORRT-T Final"},
            {
                "path": s1[inter_i],
                "linewidth": 2.5,
                "label": "AORRT-T Intermediate",
            },
            {"path": s1[first_i], "linewidth": 2.5, "label": "AORRT-T First"},
        ],
        title=title1,
    )
    fig.savefig(root + "/aorrtt_planning.png", dpi=200, bbox_inches="tight")

    fig, ax = visualize_flappy_env(
        env,
        [
            {"path": s3[-1], "linewidth": 2.5, "label": "SST"},
            {"path": s2[-1], "linewidth": 2.5, "label": "AORRT"},
            {"path": s1[-1], "linewidth": 2.5, "label": "AORRT-T"},
        ],
        title=title2,
    )
    fig.savefig(root + "/comparison.png", dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    base = "results/planning_flappy"

    plot_costs(root=base, title="Flappy: SST vs AORRT vs AORRT-T (5x20 runs)")
    plot_cases(
        root=base,
        title1="AORRT-T Planning Paths Over Time",
        title2="Planning Paths Comparison",
    )

    plt.show()
