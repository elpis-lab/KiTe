import numpy as np
import matplotlib.pyplot as plt

from utils import DataLoader, parse_args, set_seed
from geometry.object_model import get_obj_shape
from train_model import load_model, get_push_physics, evaluate_results


def main(obj_name, model_type, use_var=2, data_usage="1000x1", plot=True):
    """Train a model."""
    name = f"real_{obj_name}_{model_type}_{use_var}_{data_usage}"

    # Load data
    obj_data = "real_" + obj_name
    data_loader = DataLoader(obj_data, val_size=200)
    datasets = data_loader.load_data()
    # x_train, y_train = prepare_repetitive_data(datasets, data_usage)
    # y_val = np.array(
    #     [se2_stats(y) for y in datasets["y_val"]], dtype=np.float32
    # )
    # y_val_mean, y_val_var = y_val[:, 0], y_val[:, 1]

    # Load model
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")
    physics_eq = get_push_physics(model_type, obj_shape[:2])
    model = load_model(model_type, physics_eq, use_var, epochs=1000)

    # # Physics prediction
    # physics_eq = get_push_physics("physics", obj_shape[:2])
    # phy_model = load_model("physics", physics_eq, use_var, epochs=1000)
    # phy_model.fit(datasets["x_pool"], datasets["y_pool"])
    # pred = phy_model.predict(datasets["x_val"])
    # res = evaluate_results(
    #     pred, datasets["y_val"], np.zeros_like(datasets["y_val"]), plot
    # )

    # Train model
    x_train = datasets["x_pool"]
    y_train = datasets["y_pool"]
    x_val = datasets["x_val"]
    y_val = datasets["y_val"]
    tr_losses, val_losses = model.fit(
        x_train[:500], y_train[:500], x_val, y_val
    )
    if plot:
        plt.plot(np.arange(len(tr_losses)), tr_losses, label="Training")
        plt.plot(np.arange(len(val_losses)), val_losses, label="Validation")
        plt.legend()
        plt.show()
    # model.save(f"results/models/{name}.pt")
    # np.save(f"results/models/loss_{name}.npy", [tr_losses, val_losses])
    # model.load(f"results/models/{name}.pt")

    # Evaluate point estimation (RMSE)
    pred = model.predict(datasets["x_val"])
    res = evaluate_results(
        pred, datasets["y_val"], np.zeros_like(datasets["y_val"]), plot
    )
    # np.save(f"results/learning/{name}.npy", res)
    return res


if __name__ == "__main__":
    args = parse_args(
        [
            ("obj_name", "cracker_box_flipped"),
            ("model_type", "mlp"),
            ("use_var", 1, float),
            ("data_usage", "1000x1"),
            ("seed", 42, int),
        ]
    )
    set_seed(args.seed)

    main(args.obj_name, args.model_type, args.use_var, args.data_usage)
