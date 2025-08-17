import numpy as np
from itertools import product
import torch
import pandas as pd
import matplotlib.pyplot as plt

from train_model import load_model, get_push_physics
from geometry.object_model import get_obj_shape
from geometry.pose import angle_diff
from lie_group.lie_se2 import se2_stats
from utils import set_seed, DataLoader
from models.torch_loss_se2 import mse_se2_loss


def evaluate_model(model, datasets, y_val_mean, y_val_var, use_var):
    # Evaluate point estimation (RMSE)
    pred = model.predict(datasets["x_val"])
    mu = pred[:, :3]
    mse_loss = mse_se2_loss(torch.tensor(mu), torch.tensor(y_val_mean)).item()
    rmse_loss = np.sqrt(mse_loss)
    pos_error = np.mean(np.linalg.norm(y_val_mean[:, :2] - mu[:, :2], axis=1))
    rot_error = np.mean(np.abs(angle_diff(y_val_mean[:, 2], mu[:, 2])))
    # print(
    #     f"RMSE Error: {rmse_loss}"
    #     + f" - Position Error: {pos_error}"
    #     + f" - Rotation Error: {rot_error}"
    # )

    # Evaluate variance estimation (RMSE on std)
    y_val_std = np.sqrt(y_val_var)
    if use_var == 0:
        std = np.zeros_like(y_val_std)
    elif use_var == 1:
        logvar = pred[:, 3:]
        std = np.sqrt(np.exp(logvar))
    elif use_var == 2:
        nu = pred[:, 3:6]
        alpha = pred[:, 6:9]
        beta = pred[:, 9:]
        aleatoric = beta / (alpha - 1.0)
        epistemic = aleatoric / nu
        total_var = aleatoric + epistemic
        std = np.sqrt(total_var)

    mse_std = mse_se2_loss(torch.tensor(std), torch.tensor(y_val_std)).item()
    rmse_std = np.sqrt(mse_std)
    std_error = np.mean(np.abs(std - y_val_std), axis=0)
    # print(
    #     f"RMSE Std Error: {rmse_std}"
    #     + f" - Position X Error: {std_error[0]}"
    #     + f" - Position Y Error: {std_error[1]}"
    #     + f" - Rotation Error: {std_error[2]}"
    # )

    # # Plot actual_std vs std
    # # find index with x[:, 0] is 0 or 0.5
    # x = datasets["x_val"]
    # idx = np.where((x[:, 0] == 0) | (x[:, 0] == 0.5))[0]
    # for val in [0, 0.25, 0.5, 0.75]:
    #     idx = np.where(x[:, 0] == val)[0]
    #     print(np.mean(y_val_std[idx], axis=0))
    # std = std[idx]
    # y_val_std = y_val_std[idx]
    # # plt.plot(1000 * y_val_std[:, 0], 1000 * std[:, 0], "r.")
    # # plt.plot(1000 * y_val_std[:, 1], 1000 * std[:, 1], "g.")
    # plt.plot(100 * y_val_std[:, 2], 100 * std[:, 2], "k.")
    # plt.plot(np.linspace(0, 10, 100), np.linspace(0, 10, 100), "--")
    # plt.gca().set_aspect("equal", adjustable="box")
    # plt.xlabel("Actual Variance")
    # plt.ylabel("Predicted Variance")
    # plt.show()

    return rmse_loss, pos_error, rot_error, rmse_std, *std_error


def main():
    data_file_suffix = "2000x10"

    # Define evaluation
    # "vehicle"
    objs = (
        "cracker_box_flipped",
        "master_chef_can_flipped",
        # "letter_t",
        # "mustard_bottle_flipped",
        # "banana",
    )
    model_types = ("mlp", "residual")
    use_vars = (0, 1, 2)
    data_usages = ("1000x1", "500x2", "333x3", "250x4", "200x5", "100x10")
    lists = [model_types, use_vars, data_usages]

    for obj_name in objs:
        # Prepare data
        obj_data = obj_name
        if data_file_suffix:
            obj_data += "_" + data_file_suffix
        data_loader = DataLoader(obj_data, val_size=1000)
        datasets = data_loader.load_data()
        y_val = np.array(
            [se2_stats(y) for y in datasets["y_val"]], dtype=np.float32
        )
        y_val_mean, y_val_var = y_val[:, 0], y_val[:, 1]

        # Evaluate different models
        results = []
        for model_type, use_var, data_usage in product(*lists):
            name = f"{obj_name}_{model_type}_{use_var}_{data_usage}"
            print(f"Evaluating {name}")

            # Load model
            obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")
            physics_eq = get_push_physics(model_type, obj_shape[:2])
            model = load_model(model_type, physics_eq, use_var, epochs=300)
            model.load(f"results/models/{name}.pt")

            # Evaluate model
            result = evaluate_model(
                model, datasets, y_val_mean, y_val_var, use_var
            )
            results.append(result)

        results = np.array(results).reshape(
            len(model_types), len(use_vars), len(data_usage), 7
        )
        np.save(f"results/learning/{obj_name}.npy", results)


def visualize_results():
    objs = (
        "cracker_box_flipped",
        "master_chef_can_flipped",
        # "letter_t",
        # "mustard_bottle_flipped",
        # "banana",
    )
    model_types = ("mlp", "residual")
    use_vars = (0, 1, 2)
    data_usages = ("1000x1", "500x2", "333x3", "250x4", "200x5", "100x10")
    lists = [model_types, use_vars, data_usages]

    for obj_name in objs:
        results = np.load(f"results/learning/{obj_name}.npy")
        # flatten to a table of shape (-1, 7)
        flat = results.reshape(-1, results.shape[-1])

        # build a MultiIndex of (model_type, use_var, data_usage)
        index = pd.MultiIndex.from_product(
            [model_types, use_vars, data_usages],
            names=["model_type", "use_var", "data_usage"],
        )
        df = pd.DataFrame(
            flat,
            index=index,
            columns=[
                "rmse_loss",
                "pos_err",
                "rot_err",
                "rmse_std",
                "std_x",
                "std_y",
                "std_rot",
            ],
        )

        for model_type in model_types:
            for use_var in use_vars:
                print(f"{obj_name} - {model_type} - use_var {use_var}")
                df_slice = df.xs(
                    (model_type, use_var), level=("model_type", "use_var")
                )
                print(df_slice)


if __name__ == "__main__":
    set_seed(42)
    main()
    visualize_results()
