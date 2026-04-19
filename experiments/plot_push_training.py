import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"  # makes math look like Times
# avoid Type 3 fonts
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def main():
    """Plot active learning results for all objects in one 2x2 grid."""
    object_names = [
        "real_cracker_box_flipped",
        "real_trash_truck",
    ]
    object_titles = {
        "real_cracker_box_flipped": "Cracker Box (Real)",
        "real_trash_truck": "Trash Truck (Real)",
    }
    model_types = ("mlp",)
    use_vars = (0, 1)
    legend_names = {0: "MSE loss", 1: "NLL loss"}
    colors = {0: "C0", 1: "C4"}
    n_datas = np.arange(100, 1001, 100)
    n_exps = 5
    text_size = 18
    title_size = 18

    # height:width ~= 1:2
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 5), sharex=True, constrained_layout=False
    )

    legend_handles = []
    legend_labels = []

    for col, obj_name in enumerate(object_names):
        results = np.zeros(
            (n_exps, len(model_types), len(use_vars), len(n_datas), 2)
        )

        for r in range(n_exps):
            for i, model_type in enumerate(model_types):
                for j, use_var in enumerate(use_vars):
                    for k, n_data in enumerate(n_datas):
                        name = (
                            f"{obj_name}_{model_type}_{use_var}_{n_data}_{r}"
                        )
                        res = np.load(f"results/learning/loss_{name}.npy")
                        # nll_loss, rmse_loss, pos_error, rot_error,
                        # rmse_std, *std_error
                        results[r, i, j, k, 0] = res[1]
                        results[r, i, j, k, 1] = res[4]

        results_mean = np.mean(results, axis=0)
        results_std = np.std(results, axis=0)

        ax_top = axes[0, col]
        ax_bottom = axes[1, col]

        for i, model_type in enumerate(model_types):
            _ = model_type  # keep loop structure for compatibility
            for j, use_var in enumerate(use_vars):
                color = colors[use_var]
                label = legend_names[use_var]

                y_top = results_mean[i, j, :, 0]
                top_line = ax_top.plot(
                    n_datas, y_top, color=color, label=label, linewidth=2.5
                )[0]
                ax_top.fill_between(
                    n_datas,
                    y_top - results_std[i, j, :, 0],
                    y_top + results_std[i, j, :, 0],
                    alpha=0.2,
                    color=color,
                )

                y_bottom = results_mean[i, j, :, 1]
                # Show only NLL for sigma RMSE (no blue MSE curve).
                if use_var == 1:
                    ax_bottom.plot(
                        n_datas,
                        y_bottom,
                        color=color,
                        label=label,
                        linewidth=2.5,
                    )
                    ax_bottom.fill_between(
                        n_datas,
                        y_bottom - results_std[i, j, :, 1],
                        y_bottom + results_std[i, j, :, 1],
                        alpha=0.2,
                        color=color,
                    )

                if label not in legend_labels:
                    legend_handles.append(top_line)
                    legend_labels.append(label)

        for ax in (ax_top, ax_bottom):
            ax.set_xticks(np.arange(200, 1001, 200))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.tick_params(axis="both", labelsize=text_size)

        if col == 0:
            ax_top.set_ylabel(r"$\Delta x$ RMSE", fontsize=text_size)
            ax_bottom.set_ylabel(r"$\sigma$ RMSE", fontsize=text_size)

        ax_top.set_title(object_titles[obj_name], fontsize=title_size)
        ax_bottom.set_xlabel("Number of Data", fontsize=text_size)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        fontsize=text_size,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.24, hspace=0.08)

    plt.savefig(
        "results/learning/push_learn.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
