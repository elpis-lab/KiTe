import numpy as np
import torch
import matplotlib.pyplot as plt

from lie_group.lie_se2 import se2_stats
from utils import DataLoader, parse_args, set_seed
from geometry.pose import angle_diff
from geometry.object_model import get_obj_shape
from models.physics import push_physics
from models.torch_model import MLP, MLPVar, MLPEvidential
from models.torch_model import Physics, ResidualPhysics
from models.torch_model import ResidualPhysicsVar, ResidualPhysicsEvidential
from models.torch_loss_se2 import mse_se2_loss, nll_se2_loss, beta_nll_se2_loss
from models.torch_loss_se2 import evidential_se2_loss
from models.model import TorchModel


def load_model(
    model_type="mlp",
    equation=None,  # physics equation
    use_var=0,  # 0: no variance, 1: variance, 2: evidential
    in_dim=3,
    out_dim=3,
    hidden=32,
    dropout=0,
    lr=1e-3,
    batch_size=16,
    epochs=1000,
    device=None,
):
    """
    Load model wrapper function
    Return a model with Sklearn style API for PyTorch
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Select which model class to use
    require_training = True
    if model_type == "mlp":
        if not use_var:
            model_class = lambda: MLP(in_dim, out_dim, hidden, dropout)
        elif use_var == 1:
            model_class = lambda: MLPVar(in_dim, out_dim, hidden, dropout)
        elif use_var == 2:
            model_class = lambda: MLPEvidential(
                in_dim, out_dim, hidden, dropout
            )
    elif model_type == "residual":
        if not use_var:
            model_class = lambda: ResidualPhysics(
                in_dim, out_dim, equation, hidden, dropout
            )
        elif use_var == 1:
            model_class = lambda: ResidualPhysicsVar(
                in_dim, out_dim, equation, hidden, dropout
            )
        elif use_var == 2:
            model_class = lambda: ResidualPhysicsEvidential(
                in_dim, out_dim, equation, hidden, dropout
            )
    elif model_type == "physics":
        model_class = lambda: Physics(equation)
        require_training = False
    else:
        raise ValueError(f"Model type {model_type} not supported")

    # Use NLL loss if variance is predicted
    if use_var == 0:
        loss_fn = mse_se2_loss
    elif use_var == 1:
        loss_fn = nll_se2_loss  # beta_nll_se2_loss
    elif use_var == 2:
        loss_fn = evidential_se2_loss
    score_fn = mse_se2_loss

    # Get a wrapper for the model
    model = TorchModel(
        model_class,
        optimizer=torch.optim.Adam,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        loss_fn=loss_fn,
        score_fn=score_fn,
        verbose=1,
        require_training=require_training,
        device=device,
    )
    return model


def get_push_physics(model_type, obj_size):
    """
    Get the push physics function with given object size
    if model_type requires physics.
    Return None otherwise.
    """

    def push_physics_with_size(param):
        """Push physics function with given object size."""
        return push_physics(param, obj_size, relative=True)

    if model_type == "residual" or model_type == "physics":
        return push_physics_with_size
    else:
        return None


def prepare_repetitive_data(datasets, data_usage):
    """Prepare repetitive data for training."""
    # Slice data
    n, m = map(int, data_usage.lower().split("x"))
    x_train = datasets["x_pool"][:n]
    y_train = datasets["y_pool"][:n, :m]

    # Reshape data
    x_train = np.repeat(x_train, m, axis=0)
    y_train = y_train.reshape(-1, y_train.shape[-1])
    return x_train, y_train


def main(
    obj_name,
    model_type,
    use_var=0,
    data_usage="250x4",
    data_file_suffix="2000x10",
):
    """Train a model."""
    name = f"{obj_name}_{model_type}_{use_var}_{data_usage}"

    # Load data
    obj_data = obj_name
    if data_file_suffix:
        obj_data += "_" + data_file_suffix
    data_loader = DataLoader(obj_data, val_size=1000)
    datasets = data_loader.load_data()
    x_train, y_train = prepare_repetitive_data(datasets, data_usage)
    y_val = np.array(
        [se2_stats(y) for y in datasets["y_val"]], dtype=np.float32
    )
    y_val_mean, y_val_var = y_val[:, 0], y_val[:, 1]

    # Load model
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")
    physics_eq = get_push_physics(model_type, obj_shape[:2])
    model = load_model(model_type, physics_eq, use_var, epochs=400)

    # Train model
    tr_losses, val_losses = model.fit(
        x_train, y_train, datasets["x_val"], datasets["y_val"][:, 0]
    )
    # plt.plot(np.arange(len(tr_losses)), tr_losses)
    # plt.plot(np.arange(len(val_losses)), val_losses)
    # plt.show()
    model.save(f"results/models/{name}.pt")
    model.load(f"results/models/{name}.pt")

    # Evaluate point estimation (RMSE)
    pred = model.predict(datasets["x_val"])
    mu = pred[:, :3]
    mse_loss = mse_se2_loss(torch.tensor(mu), torch.tensor(y_val_mean)).item()
    rmse_loss = np.sqrt(mse_loss)
    pos_error = np.mean(np.linalg.norm(y_val_mean[:, :2] - mu[:, :2], axis=1))
    rot_error = np.mean(np.abs(angle_diff(y_val_mean[:, 2], mu[:, 2])))
    print(
        f"RMSE Error: {rmse_loss}"
        + f" - Position Error: {pos_error}"
        + f" - Rotation Error: {rot_error}"
    )

    # Evaluate variance estimation (RMSE on std)
    y_val_std = np.sqrt(y_val_var)
    if use_var == 0:
        return
    elif use_var == 1:
        logvar = pred[:, 3:]
        std = np.sqrt(np.exp(logvar))
    elif use_var == 2:
        nu = pred[:, 3:6]
        alpha = pred[:, 6:9]
        beta = pred[:, 9:]
        aleatoric = beta / (alpha - 1.0)
        epistemic = aleatoric / nu
        # total_var = aleatoric + epistemic
        std = np.sqrt(epistemic)

    mse_std = mse_se2_loss(torch.tensor(std), torch.tensor(y_val_std)).item()
    rmse_std = np.sqrt(mse_std)
    std_error = np.mean(np.abs(std - y_val_std), axis=0)
    print(
        f"RMSE Std Error: {rmse_std}"
        + f" - Position X Error: {std_error[0]}"
        + f" - Position Y Error: {std_error[1]}"
        + f" - Rotation Error: {std_error[2]}"
    )

    np.save(
        f"results/learning/{name}.npy",
        (rmse_loss, pos_error, rot_error, rmse_std, *std_error),
    )
    return rmse_loss, pos_error, rot_error, rmse_std, std_error


if __name__ == "__main__":
    args = parse_args(
        [
            ("obj_name", "master_chef_can_flipped"),
            ("model_type", "mlp"),
            ("use_var", 2, int),
            ("data_usage", "1000x1"),
            ("data_file_suffix", "2000x10"),
        ]
    )
    set_seed(1)

    main(
        args.obj_name,
        args.model_type,
        args.use_var,
        args.data_usage,
        args.data_file_suffix,
    )
