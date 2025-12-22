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


def load_costs(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(
            f"{path}: expected shape (rep, prob, time, 2), got {arr.shape}"
        )
    return arr


def mask_no_solution(costs: np.ndarray) -> np.ndarray:
    costs = costs.astype(float, copy=True)
    bad = (costs[..., 0] == -1.0) & (costs[..., 1] == -1.0)
    costs[bad, 0] = np.nan
    costs[bad, 1] = np.nan
    return costs


def mean_over_probs_then_std_over_reps(costs: np.ndarray, which: int):
    """
    costs: (R, P, T, 2) with NaNs for missing
    which: 0=running, 1=terminal
    Returns:
      mean_curve: (T,)
      std_curve : (T,)  std across reps of (mean over problems)
    """
    # (R,P,T)
    X = costs[..., which]

    # Per-rep mean over problems: (R,T)
    per_rep = np.nanmean(X, axis=1)

    # Across-rep mean+std: (T,)
    mean_curve = np.nanmean(per_rep, axis=0)
    std_curve = np.nanstd(per_rep, axis=0, ddof=1)  # 10 reps => ddof=1
    return mean_curve, std_curve


def plot_costs(
    planning_times,
    sst_costs_path,
    aorrt_costs_path,
    aorrt_t_costs_path,
    title="Cost vs Planning Time (mean over problems; ±1 std across reps)",
    save_path=None,
    show_error_band=True,
    error_band_alpha=0.18,
):
    planning_times = np.asarray(planning_times, dtype=float)

    sst = mask_no_solution(load_costs(sst_costs_path))
    aorrt = mask_no_solution(load_costs(aorrt_costs_path))
    aT = mask_no_solution(load_costs(aorrt_t_costs_path))

    # Running
    sst_run_mean, sst_run_std = mean_over_probs_then_std_over_reps(
        sst, which=0
    )
    aorrt_run_mean, aorrt_run_std = mean_over_probs_then_std_over_reps(
        aorrt, which=0
    )
    aT_run_mean, aT_run_std = mean_over_probs_then_std_over_reps(aT, which=0)

    # AORRT-T terminal
    aT_term_mean, aT_term_std = mean_over_probs_then_std_over_reps(aT, which=1)

    # AORRT-T total (running + terminal) computed before stats
    aT_total = (aT[..., 0] + aT[..., 1])[..., None]  # (R,P,T,1)
    aT_total_pack = np.concatenate([aT_total, aT_total], -1)  # fake (R,P,T,2)
    aT_tot_mean, aT_tot_std = mean_over_probs_then_std_over_reps(
        aT_total_pack, which=0
    )

    fig, ax = plt.subplots(figsize=(10, 6))

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
    (l_aT,) = ax.plot(
        planning_times,
        aT_run_mean,
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
            aT_run_mean - aT_run_std,
            aT_run_mean + aT_run_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=l_aT.get_color(),
        )

    c = l_aT.get_color()
    ax.plot(
        planning_times,
        aT_term_mean,
        ":",
        linewidth=2.5,
        color=c,
        label="AORRT-T (terminal)",
    )
    ax.plot(
        planning_times,
        aT_tot_mean,
        "-.",
        linewidth=2.5,
        color=c,
        label="AORRT-T (total)",
    )

    if show_error_band:
        ax.fill_between(
            planning_times,
            aT_term_mean - aT_term_std,
            aT_term_mean + aT_term_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=c,
        )
        ax.fill_between(
            planning_times,
            aT_tot_mean - aT_tot_std,
            aT_tot_mean + aT_tot_std,
            alpha=error_band_alpha,
            linewidth=0,
            color=c,
        )

    ax.set_xlabel("Planning time (s)")
    ax.set_ylabel("Cost")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax


def plot_case():
    ou.RNG.setSeed(100)
    np.random.seed(10)
    env = generate_flappy_env()
    start, goal, goal_size = env["start"], env["goal_center"], env["goal_size"]

    p1 = FlappyPlanner(env["obstacles"], "aorrt", terminal_weight=1.2)
    s1, _, _ = p1.plan(start, goal, goal_size, [0.092, 0.4, 10.0])
    p2 = FlappyPlanner(env["obstacles"], "aorrt", terminal_weight=0.0)
    s2, _, _ = p2.plan(start, goal, goal_size, [10.0])
    p3 = FlappyPlanner(env["obstacles"], "sst", terminal_weight=0.0)
    s3, _, _ = p3.plan(start, goal, goal_size, [10.0])

    visualize_flappy_env(
        env,
        [
            {"path": s1[0], "label": "AORRT-T First"},
            {"path": s1[1], "label": "AORRT-T Intermediate"},
            {"path": s1[2], "label": "AORRT-T Final"},
        ],
        title="AORRT-T Planning Paths Over Time",
        show=False,
    )

    visualize_flappy_env(
        env,
        [
            {"path": s3[-1], "label": "SST"},
            {"path": s2[-1], "label": "AORRT"},
            {"path": s1[-1], "label": "AORRT-T"},
        ],
        title="Planning Paths Comparison",
        show=False,
    )


if __name__ == "__main__":
    # Your time grid: 0.1 to 10.0 every 0.1 => 100 points
    planning_times = np.arange(0.1, 10.0 + 1e-9, 0.1)

    base = "results/planning_flappy"
    # Update names if needed
    sst_costs = os.path.join(base, "sst_0.0_costs.npy")
    aorrt_costs = os.path.join(base, "aorrt_0.0_costs.npy")
    aorrt_t_costs = os.path.join(base, "aorrt_1.2_costs.npy")

    plot_costs(
        planning_times,
        sst_costs_path=sst_costs,
        aorrt_costs_path=aorrt_costs,
        aorrt_t_costs_path=aorrt_t_costs,
        title="Flappy: SST vs AORRT vs AORRT-T (10×10 runs)",
        save_path=os.path.join(base, "cost_curves.png"),
        show_error_band=True,
    )

    plot_case()
    plt.show()
