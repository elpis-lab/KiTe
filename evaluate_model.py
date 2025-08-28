import numpy as np
from itertools import product
import torch
import pandas as pd
import matplotlib.pyplot as plt

from train_model import load_model, get_push_physics, evaluate_results
from geometry.object_model import get_obj_shape
from lie_group.lie_se2 import se2_stats
from utils import DataLoader


def run_evaluation(plot=False):
    # Define evaluation
    objs = (
        "cracker_box_flipped",
        "master_chef_can_flipped",
    )
    model_types = ("mlp",)
    use_vars = (0.0, 0.1)
    data_usages = (
        "100x1",
        "200x1",
        "300x1",
        "400x1",
        "500x1",
        "750x1",
        "1000x1",
    )
    lists = [model_types, use_vars, data_usages]

    for obj_name in objs:
        # Prepare data
        obj_data = obj_name + "_2000x10"
        data_loader = DataLoader(obj_data, val_size=1000)
        datasets = data_loader.load_data()
        y_val = np.array(
            [se2_stats(y) for y in datasets["y_val"]], dtype=np.float32
        )
        y_val_mean, y_val_var = y_val[:, 0], y_val[:, 1]

        # Evaluate different models
        for model_type, use_var, data_usage in product(*lists):
            name = f"{obj_name}_{model_type}_{use_var}_{data_usage}"
            print(f"Evaluating {name}")

            # Load model
            obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")
            physics_eq = get_push_physics(model_type, obj_shape[:2])
            model = load_model(model_type, physics_eq, use_var)
            model.load(f"results/models/{name}.pt")

            # Evaluate model
            pred = model.predict(datasets["x_val"])
            res = evaluate_results(pred, y_val_mean, y_val_var, plot)
            np.save(f"results/learning/{name}.npy", res)

            # Debug Plot
            if plot:
                y_val_std = np.sqrt(y_val_var)
                if pred.shape[1] == 3:
                    std = np.zeros_like(y_val_std)
                elif pred.shape[1] == 6:
                    logvar = pred[:, 3:]
                    std = np.sqrt(np.exp(logvar))
                elif pred.shape[1] == 12:
                    nu = pred[:, 3:6]
                    alpha = pred[:, 6:9]
                    beta = pred[:, 9:]
                    # Original
                    # aleatoric = beta / (alpha - 1.0)
                    # epistemic = aleatoric / nu
                    # Better aleatoric Proxy
                    # https://arxiv.org/pdf/2205.10060
                    aleatoric = beta * (1 + nu) / (alpha * nu)
                    epistemic = 0  # 1 / nu
                    total_var = aleatoric + epistemic
                    std = np.sqrt(total_var)

                # x = datasets["x_val"]
                # idx = np.where((x[:, 0] == 0) | (x[:, 0] == 0.5))[0]
                # for val in [0, 0.25, 0.5, 0.75]:
                #     idx = np.where(x[:, 0] == val)[0]
                #     print(np.mean(y_val_std[idx], axis=0))
                # std = std[idx]
                # y_val_std = y_val_std[idx]
                plt.plot(1000 * y_val_std[:, 0], 1000 * std[:, 0], "y.")
                plt.plot(1000 * y_val_std[:, 1], 1000 * std[:, 1], "g.")
                plt.plot(100 * y_val_std[:, 2], 100 * std[:, 2], "k.")
                plt.plot(
                    np.linspace(0, 10, 100), np.linspace(0, 10, 100), "--"
                )
                plt.gca().set_aspect("equal", adjustable="box")
                plt.xlabel("Actual Standard Deviation")
                plt.ylabel("Predicted Standard Deviation")
                plt.show()


def visualize_results():
    # Define evaluation
    objs = (
        "cracker_box_flipped",
        "master_chef_can_flipped",
    )
    model_types = ("mlp",)
    seeds = list(range(1, 11))
    use_vars = (0.0, 1.0, 2.0)
    data_usages = (
        "100x1",
        "200x1",
        "300x1",
        "400x1",
        "500x1",
        "600x1",
        "700x1",
        "800x1",
        "900x1",
        "1000x1",
    )

    for obj_name in objs:
        # Seperate each object
        results = np.zeros(
            (len(seeds), len(model_types), len(use_vars), len(data_usages), 2)
        )

        # Acquire results
        for s, seed in enumerate(seeds):
            for i, model_type in enumerate(model_types):
                for j, use_var in enumerate(use_vars):
                    for k, data_usage in enumerate(data_usages):
                        data_usage_seed = data_usage + f"x{seed}"
                        name = (
                            f"{obj_name}_{model_type}_{use_var}"
                            + f"_{data_usage_seed}"
                        )
                        res = np.load(f"results/learning/{name}.npy")
                        results[s, i, j, k, 0] = res[0]
                        results[s, i, j, k, 1] = res[3]

        # Get mean and std of results
        results_mean = np.mean(results, axis=0)
        results_std = np.std(results, axis=0)

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
        x = np.arange(len(data_usages))

        for i, model_type in enumerate(model_types):
            for j, use_var in enumerate(use_vars):
                y1 = results_mean[i, j, :, 0]
                y2 = results_mean[i, j, :, 1]
                ax1.plot(x, y1, label=f"{model_type}_{use_var}")
                ax1.fill_between(
                    x,
                    y1 - results_std[i, j, :, 0],
                    y1 + results_std[i, j, :, 0],
                    alpha=0.2,
                )
                ax2.plot(x, y2, label=f"{model_type}_{use_var}")
                ax2.fill_between(
                    x,
                    y2 - results_std[i, j, :, 1],
                    y2 + results_std[i, j, :, 1],
                    alpha=0.2,
                )
        for ax, ylabel in zip((ax1, ax2), ("SE2 RMSE", "STD RMSE")):
            ax.set_xticks(x)
            ax.set_xticklabels(data_usages, rotation=45)
            ax.set_ylabel(ylabel)
            ax.legend()
        ax2.set_xlabel("Data")
        plt.tight_layout()
        plt.show()


def visualize_training_loss():
    # Define evaluation
    objs = (
        "cracker_box_flipped",
        "master_chef_can_flipped",
    )
    model_types = ("mlp",)
    use_vars = (0.2, 0.3, 0.4)
    data_usages = (
        "100x1",
        "200x1",
        "300x1",
        "400x1",
        "500x1",
        "750x1",
        "1000x1",
    )

    for obj_name in objs:
        # Seperate each object
        results = np.zeros(
            (len(model_types), len(use_vars), len(data_usages), 2)
        )

        # Acquire results
        for i, model_type in enumerate(model_types):
            for j, use_var in enumerate(use_vars):
                for k, data_usage in enumerate(data_usages):
                    name = f"{obj_name}_{model_type}_{use_var}_{data_usage}"
                    res = np.load(f"results/models/loss_{name}.npy")
                    tr_losses = res[0]
                    val_losses = res[1]
                    plt.plot(
                        np.arange(len(tr_losses)),
                        tr_losses,
                        label="Training",
                    )
                    plt.plot(
                        np.arange(len(val_losses)),
                        val_losses,
                        label="Validation",
                    )
                    plt.legend()
                    plt.show()


if __name__ == "__main__":
    # visualize_training_loss()
    # run_evaluation()
    visualize_results()
