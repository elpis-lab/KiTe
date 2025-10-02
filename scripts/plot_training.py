import numpy as np
from itertools import product
import torch
import pandas as pd
import matplotlib.pyplot as plt

from train_model import load_model, get_push_physics, evaluate_results
from geometry.object_model import get_obj_shape
from lie_group.lie_se2 import se2_stats
from utils import DataLoader


def main():
    """Plot active learning results for all objects in 3x2 grid"""
    object_names = [
        "cracker_box_flipped",
        "mustard_bottle_flipped",
        "banana",
        "letter_t",
        "master_chef_can_flipped",
        "school_bus",
        "trash_truck",
        # "real_cracker_box_flipped",
        # "real_school_bus",
        # "real_trash_truck",
    ]
    model_types = ("mlp",)
    use_vars = (0.0, 1.0)
    n_datas = np.arange(100, 1001, 100)
    n_exps = 3

    for obj_name in object_names:
        # Seperate each object
        results = np.zeros(
            (n_exps, len(model_types), len(use_vars), len(n_datas), 2)
        )

        # Acquire results
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

        # Get mean and std of results
        results_mean = np.mean(results, axis=0)
        results_std = np.std(results, axis=0)
        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
        for i, model_type in enumerate(model_types):
            for j, use_var in enumerate(use_vars):
                y1 = results_mean[i, j, :, 0]
                ax1.plot(n_datas, y1, label=f"{model_type}_{use_var}")
                ax1.fill_between(
                    n_datas,
                    y1 - results_std[i, j, :, 0],
                    y1 + results_std[i, j, :, 0],
                    alpha=0.2,
                )

                if use_var == 0.0:
                    continue
                y2 = results_mean[i, j, :, 1]
                ax2.plot(n_datas, y2, label=f"{model_type}_{use_var}")
                ax2.fill_between(
                    n_datas,
                    y2 - results_std[i, j, :, 1],
                    y2 + results_std[i, j, :, 1],
                    alpha=0.2,
                )

        for ax, ylabel in zip((ax1, ax2), ("SE2 RMSE", "STD RMSE")):
            ax.set_xticks(n_datas)
            ax.set_xticklabels(n_datas, rotation=45)
            ax.set_ylabel(ylabel)
            ax.legend()
        ax2.set_xlabel("Number of Data")
        ax1.set_title(f"{obj_name}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
