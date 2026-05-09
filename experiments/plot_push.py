import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"  # makes math look like Times
# avoid Type 3 fonts
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def backfill_cost_to_avoid_jump(cost):
    """Backfill early invalid entries with first valid value."""
    out = np.array(cost, dtype=float)
    n_rep, n_prob, n_t = out.shape
    for r in range(n_rep):
        for p in range(n_prob):
            row = out[r, p, :]
            valid = np.isfinite(row) & (row >= 0.0)
            if not np.any(valid):
                continue
            first_valid = np.argmax(valid)
            out[r, p, : first_valid + 1] = row[first_valid]
    return out


def get_selected_push_plot_styles():
    """Selected methods for plots: (label, color, linestyle)."""
    return {
        ("aorrt", "l2", "random", 0.0): ("Base (AO-RRT, L2)", "C1", "-"),
        ("aorrt", "w2", "random", 0.0): ("GBT (AO-RRT, W2)", "C2", "-"),
        ("aorrt", "l2", "active", 0.0): ("AP (AO-RRT, L2 + MU)", "C5", "-"),
        ("aorrt", "l2", "random", 20.0): ("KiTe (AO-RRT, L2, 20)", "C6", "-"),
        ("sst", "l2", "random", 0.0): ("Base (SST, L2)", "C8", "-"),
        ("sst", "w2", "random", 0.0): ("GBT (SST, W2)", "C9", "-"),
        ("sst", "l2", "active", 0.0): ("AP (SST, L2 + MU)", "C7", "-"),
        ("aorrt", "w2", "random", 20.0): ("KiTe (AO-RRT, W2, 20)", "C4", "-"),
    }


TABLE_METHOD_ORDER = [
    ("sst", "l2", "random", 0.0),
    ("aorrt", "l2", "random", 0.0),
    ("sst", "w2", "random", 0.0),
    ("aorrt", "w2", "random", 0.0),
    ("sst", "l2", "active", 0.0),
    ("aorrt", "l2", "active", 0.0),
    ("aorrt", "l2", "random", 20.0),
    ("aorrt", "w2", "random", 20.0),
]

OBJECT_DISPLAY_NAMES = {
    "mustard_bottle_flipped": "Mustard Bottle (Sim.)",
    "master_chef_can_flipped": "Chef Can (Sim.)",
    "real_cracker_box_flipped": "Cracker Box (Real)",
    "real_trash_truck": "Trask Truck (Real)",
}


def get_result_name(obj_name, n_data, config, model_type="mlp"):
    algo, belief, active_sampling, terminal_weight = config
    use_var = 1 if belief == "w2" else 0
    return (
        f"{obj_name}_{model_type}_{use_var}_{n_data}_"
        f"{algo}_{belief}_{active_sampling}_{terminal_weight}"
    )


def load_all_results(obj_names, n_datas, configs, folder, model_type="mlp"):
    all_results = {}
    for obj_name in obj_names:
        all_results[obj_name] = {}
        for n_data in n_datas:
            all_results[obj_name][n_data] = {}
            for config in configs:
                name = get_result_name(
                    obj_name, n_data, config, model_type=model_type
                )
                path = f"{folder}/{name}_results.npy"
                if not os.path.exists(path):
                    print(f"[WARN] Missing result file: {path}")
                    continue
                all_results[obj_name][n_data][config] = np.load(path)
    return all_results


def compute_time_series_metrics(results):
    """Compute running/terminal/success over planning time using unified costs."""
    running = results[:, :, :, 11].astype(float)
    # scale terminal cost by 20.0 to match the pushing experiment
    terminal = results[:, :, :, 12].astype(float) * 20.0
    valid_mask = (running >= 0.0) & (terminal >= 0.0)

    running_masked = np.where(valid_mask, running, np.nan)
    terminal_masked = np.where(valid_mask, terminal, np.nan)
    running_backfilled = backfill_cost_to_avoid_jump(running_masked)
    terminal_backfilled = backfill_cost_to_avoid_jump(terminal_masked)

    running_per_rep = np.nanmean(running_backfilled, axis=1)
    terminal_per_rep = np.nanmean(terminal_backfilled, axis=1)
    running_mean = np.nanmean(running_per_rep, axis=0)
    running_std = np.nanstd(running_per_rep, axis=0)
    terminal_mean = np.nanmean(terminal_per_rep, axis=0)
    terminal_std = np.nanstd(terminal_per_rep, axis=0)

    success = results[:, :, :, 0].astype(float)
    n_rep, n_prob = success.shape[0], success.shape[1]
    all_cases = float(n_rep * n_prob)
    success_rate = (
        np.nansum(np.where(valid_mask, success, np.nan), axis=(0, 1))
        / all_cases
        * 100.0
    )

    return running_mean, running_std, terminal_mean, terminal_std, success_rate


def compute_last_plan_metrics(results):
    """Metrics on the final plan timestamp using unified costs."""
    last = results[:, :, -1, :].astype(float)
    success_last = last[:, :, 0]
    running_last = last[:, :, 11]
    terminal_last = last[:, :, 12]
    valid_last = (running_last >= 0.0) & (terminal_last >= 0.0)

    sr = np.mean(success_last) * 100.0
    cw = np.nanmean(np.where(valid_last, running_last, np.nan))
    cpsi = np.nanmean(np.where(valid_last, terminal_last, np.nan))
    return sr, cw, cpsi


def plot_object_2x2(
    obj_name,
    obj_results_by_ndata,
    n_datas,
    folder,
    text_size=20,
):
    style_map = get_selected_push_plot_styles()
    plot_n_data = n_datas[-1]
    if plot_n_data not in obj_results_by_ndata:
        print(
            f"[WARN] No data for {obj_name} at n_data={plot_n_data}, skipping plot"
        )
        return

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(3, 2)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])
    planning_times = np.arange(1.0, 31.0)
    line_handles = []
    line_labels = []
    decomp_success = []
    decomp_not_reached = []
    decomp_collision = []
    decomp_colors = []
    decomp_labels = []

    data_at_main_ndata = obj_results_by_ndata.get(plot_n_data, {})
    for method_key, style in style_map.items():
        if method_key not in data_at_main_ndata:
            continue
        label, color, linestyle = style
        results = data_at_main_ndata[method_key]
        (
            running_mean,
            running_std,
            terminal_mean,
            terminal_std,
            success_rate,
        ) = compute_time_series_metrics(results)

        line = ax_a.plot(
            planning_times,
            running_mean,
            color=color,
            linestyle=linestyle,
            linewidth=3.0,
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
            linewidth=3.0,
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
            linewidth=3.0,
        )
        line_handles.append(line)
        line_labels.append(label)

        last = results[:, :, -1, :].astype(float)
        running_last = last[:, :, 11]
        terminal_last = last[:, :, 12]
        valid_last = (running_last >= 0.0) & (terminal_last >= 0.0)
        collisions_last = last[:, :, 1] > 0.5
        reached_last = last[:, :, 2] > 0.5
        all_cases = float(last.shape[0] * last.shape[1])

        collision_count = np.sum(valid_last & collisions_last)
        not_reached_count = np.sum(
            valid_last & (~collisions_last) & (~reached_last)
        )
        success_count = np.sum(valid_last & (~collisions_last) & reached_last)
        decomp_success.append(success_count / all_cases * 100.0)
        decomp_not_reached.append(not_reached_count / all_cases * 100.0)
        decomp_collision.append(collision_count / all_cases * 100.0)
        decomp_colors.append(color)
        decomp_labels.append(label)

    for method_key, style in style_map.items():
        label, color, linestyle = style
        sr_list = []
        x_list = []
        for n_data in n_datas:
            obj_n_data_results = obj_results_by_ndata.get(n_data, {})
            if method_key not in obj_n_data_results:
                continue
            sr, _, _ = compute_last_plan_metrics(
                obj_n_data_results[method_key]
            )
            x_list.append(n_data)
            sr_list.append(sr)
        if len(x_list) == 0:
            continue
        ax_e.plot(
            x_list,
            sr_list,
            color=color,
            linestyle=linestyle,
            linewidth=3.0,
            marker="o",
        )

    ax_a.set_title("(a) Running Cost", fontsize=text_size)
    ax_a.set_xlabel("Time (s)", fontsize=text_size)
    ax_a.set_ylabel("Cost", fontsize=text_size)
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_a.tick_params(axis="both", labelsize=text_size)
    ax_a.grid(True, alpha=0.3)

    ax_b.set_title("(b) Terminal Cost", fontsize=text_size)
    ax_b.set_xlabel("Time (s)", fontsize=text_size)
    ax_b.set_ylabel(r"Cost", fontsize=text_size)
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

    x_pos = np.arange(len(decomp_labels))
    bar_width = 0.7
    ax_d.bar(
        x_pos,
        decomp_success,
        width=bar_width,
        color=decomp_colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax_d.bar(
        x_pos,
        decomp_not_reached,
        width=bar_width,
        bottom=decomp_success,
        color="0.6",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_d.bar(
        x_pos,
        decomp_collision,
        width=bar_width,
        bottom=np.array(decomp_success) + np.array(decomp_not_reached),
        color="black",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_d.set_title("(d) Success Decomposition at 30s", fontsize=text_size)
    ax_d.set_xlabel("")
    ax_d.set_xticks([])
    ax_d.set_ylabel(r"Plan Fraction (\%)", fontsize=text_size)
    ax_d.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_d.tick_params(axis="both", labelsize=text_size)
    ax_d.set_ylim(0.0, 102.0)
    ax_d.grid(True, alpha=0.3)

    ax_e.set_title(
        "(e) Success vs Number of Data (Last Plan)", fontsize=text_size
    )
    ax_e.set_xlabel("Number of Data", fontsize=text_size)
    ax_e.set_ylabel(r"Success Rate (\%)", fontsize=text_size)
    ax_e.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_e.tick_params(axis="both", labelsize=text_size)
    ax_e.set_ylim(0.0, 102.0)
    ax_e.grid(True, alpha=0.3)

    fig.legend(
        line_handles,
        line_labels,
        loc="lower center",
        bbox_to_anchor=(0.38, -0.01),
        ncol=2,
        frameon=True,
        fontsize=text_size - 2,
    )
    fig.legend(
        [
            Patch(facecolor="black", edgecolor="black", label="In Collision"),
            Patch(
                facecolor="0.6", edgecolor="black", label="Goal Not Reached"
            ),
        ],
        ["In Collision", "Goal Not Reached"],
        loc="lower center",
        bbox_to_anchor=(0.86, -0.01),
        ncol=1,
        frameon=True,
        fontsize=text_size - 2,
    )
    fig.suptitle(
        OBJECT_DISPLAY_NAMES.get(obj_name, obj_name), fontsize=text_size + 2
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])

    out_pdf = f"{folder}/push_{obj_name}_result.pdf"
    plt.savefig(out_pdf)
    # plt.close(fig)
    print(f"Saved plot: {out_pdf}")


def save_summary_table(all_results, obj_names, n_datas, folder):
    target_n_data = n_datas[-1]
    style_map = get_selected_push_plot_styles()

    row_labels = [style_map[k][0] for k in TABLE_METHOD_ORDER]
    big_columns = [OBJECT_DISPLAY_NAMES.get(obj, obj) for obj in obj_names]
    sub_columns = ["SR (%)", "Cw", "C\u03c8"]

    table_data = np.full((len(TABLE_METHOD_ORDER), len(obj_names) * 3), np.nan)
    for c_idx, obj_name in enumerate(obj_names):
        obj_results = all_results.get(obj_name, {})
        n_data_results = obj_results.get(target_n_data, {})
        for r_idx, method_key in enumerate(TABLE_METHOD_ORDER):
            if method_key not in n_data_results:
                continue
            sr, cw, cpsi = compute_last_plan_metrics(
                n_data_results[method_key]
            )
            table_data[r_idx, c_idx * 3 + 0] = sr
            table_data[r_idx, c_idx * 3 + 1] = cw
            table_data[r_idx, c_idx * 3 + 2] = cpsi * 20.0

    table_payload = {
        "target_n_data": target_n_data,
        "row_labels": row_labels,
        "big_columns": big_columns,
        "sub_columns": sub_columns,
        "values": table_data,
    }
    npy_path = f"{folder}/push_summary_table.npy"
    np.save(npy_path, table_payload, allow_pickle=True)
    print(f"Saved table npy: {npy_path}")

    csv_path = f"{folder}/push_summary_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header1 = [""]
        header2 = ["Method"]
        for big in big_columns:
            header1.extend([big, "", ""])
            header2.extend(sub_columns)
        writer.writerow(header1)
        writer.writerow(header2)

        for r_idx, row_name in enumerate(row_labels):
            row = [row_name]
            for v in table_data[r_idx]:
                if np.isnan(v):
                    row.append("")
                else:
                    row.append(f"{v:.2f}")
            writer.writerow(row)
    print(f"Saved table csv: {csv_path}")


def data_plot(obj_names, n_datas, configs):
    """Plot push planning results and save summary table."""
    folder = "results/planning_push"
    os.makedirs(folder, exist_ok=True)

    all_results = load_all_results(
        obj_names,
        n_datas,
        configs,
        folder,
        model_type="mlp",
    )

    for obj_name in obj_names:
        plot_object_2x2(
            obj_name,
            all_results.get(obj_name, {}),
            n_datas,
            folder,
            text_size=20,
        )
    plt.show()

    save_summary_table(all_results, obj_names, n_datas, folder)


if __name__ == "__main__":
    obj_names = [
        "mustard_bottle_flipped",
        "master_chef_can_flipped",
        "real_cracker_box_flipped",
        "real_trash_truck",
    ]
    model_type = "mlp"
    # n_datas = [200, 400, 600, 800, 1000]
    n_datas = [1000]

    configs = [
        ("aorrt", "l2", "random", 0.0),  # Base
        ("sst", "l2", "random", 0.0),  # Base
        ("aorrt", "w2", "random", 0.0),  # GBT
        ("sst", "w2", "random", 0.0),  # GBT
        ("aorrt", "l2", "active", 0.0),  # Active Pusher
        ("sst", "l2", "active", 0.0),  # Active Pusher
        ("aorrt", "l2", "random", 20.0),  # KiTe
        ("aorrt", "w2", "random", 20.0),  # KiTe
        #
        # ("aorrt", "l2", "random", 5.0),  # KiTe
        # ("aorrt", "w2", "random", 5.0),  # KiTe
        # ("aorrt", "l2", "random", 10.0),  # KiTe
        # ("aorrt", "w2", "random", 10.0),  # KiTe
        # ("aorrt", "l2", "random", 50.0),  # KiTe
        # ("aorrt", "w2", "random", 50.0),  # KiTe
        # ("aorrt", "l2", "random", 100.0),  # KiTe
        # ("aorrt", "w2", "random", 100.0),  # KiTe
    ]

    data_plot(obj_names, n_datas, configs)
