import numpy as np
import torch
import torch.nn.functional as F
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
        # TODO
        if not use_var:
            model_class = lambda: MLP(in_dim, out_dim, hidden, dropout)
        elif use_var == 1:
            model_class = lambda: MLPVar(in_dim, out_dim, hidden, dropout)
        elif use_var == 2:
            model_class = lambda: MLPEvidential(
                in_dim, out_dim, hidden, dropout
            )
        # model_class = lambda: MLPVar(in_dim, out_dim, hidden, dropout)
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
    # TODO
    if use_var == 0:
        loss_fn = mse_se2_loss
    elif use_var == 1:
        loss_fn = nll_se2_loss
    elif use_var == 2:
        loss_fn = evidential_se2_loss
    # def get_beta_nll_se2_loss(use_var):
    #     def beta_nll(y_pred, y_true):
    #         return beta_nll_se2_loss(y_pred, y_true, use_var)

    #     return beta_nll

    # loss_fn = get_beta_nll_se2_loss(use_var)
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
    nxm = list(map(int, data_usage.lower().split("x")))
    x_train = datasets["x_pool"][: nxm[0]]
    y_train = datasets["y_pool"][: nxm[0], : nxm[1]]

    # Reshape data
    x_train = np.repeat(x_train, nxm[1], axis=0)
    y_train = y_train.reshape(-1, y_train.shape[-1])
    return x_train, y_train


def evaluate_results(pred, val_mean, val_var, plot=False):
    """Evaluate results"""
    val_std = np.sqrt(val_var)
    pred_mean = pred[:, :3]
    if pred.shape[1] == 3:
        pred_std = np.zeros_like(val_std)
    elif pred.shape[1] == 6:
        logvar = pred[:, 3:]
        pred_std = np.sqrt(np.exp(logvar))
    elif pred.shape[1] == 12:
        nu = pred[:, 3:6]
        alpha = pred[:, 6:9]
        beta = pred[:, 9:]
        # Original pred_std
        # aleatoric = beta / (alpha - 1.0)
        # epistemic = aleatoric / nu
        # Better aleatoric Proxy
        # https://arxiv.org/pdf/2205.10060
        aleatoric = beta * (1 + nu) / (alpha * nu)
        epistemic = 0  # 1 / nu
        total_var = aleatoric + epistemic
        pred_std = np.sqrt(total_var)

    # SE2 RMSE
    mse_loss = mse_se2_loss(
        torch.tensor(pred_mean), torch.tensor(val_mean)
    ).item()
    rmse_loss = np.sqrt(mse_loss)
    pos_error = np.mean(
        np.linalg.norm(val_mean[:, :2] - pred_mean[:, :2], axis=1)
    )
    rot_error = np.mean(np.abs(angle_diff(val_mean[:, 2], pred_mean[:, 2])))

    # STD RMSE
    mse_std = np.mean((pred_std - val_std) ** 2)
    rmse_std = np.sqrt(mse_std)
    std_error = np.mean(np.abs(pred_std - val_std), axis=0)

    if plot:
        print(
            f"RMSE Error: {rmse_loss}"
            + f" - Position Error: {pos_error}"
            + f" - Rotation Error: {rot_error}"
        )
        print(
            f"RMSE Std Error: {rmse_std}"
            + f" - Position X Error: {std_error[0]}"
            + f" - Position Y Error: {std_error[1]}"
            + f" - Rotation Error: {std_error[2]}"
        )

    return rmse_loss, pos_error, rot_error, rmse_std, *std_error


def main(obj_name, model_type, use_var=2, data_usage="1000x1", plot=True):
    """Train a model."""
    name = f"{obj_name}_{model_type}_{use_var}_{data_usage}"

    # Load data
    obj_data = obj_name + "_2000x10"
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
    model = load_model(model_type, physics_eq, use_var, epochs=1000)

    # Train model
    tr_losses, val_losses = model.fit(
        x_train, y_train, datasets["x_val"], datasets["y_val"][:, 0]
    )
    if plot:
        plt.plot(np.arange(len(tr_losses)), tr_losses, label="Training")
        plt.plot(np.arange(len(val_losses)), val_losses, label="Validation")
        plt.legend()
        plt.show()
    model.save(f"results/models/{name}.pt")
    np.save(f"results/models/loss_{name}.npy", [tr_losses, val_losses])
    # model.load(f"results/models/{name}.pt")

    # Evaluate point estimation (RMSE)
    pred = model.predict(datasets["x_val"])
    res = evaluate_results(pred, y_val_mean, y_val_var, plot)
    np.save(f"results/learning/{name}.npy", res)
    return res


if __name__ == "__main__":
    args = parse_args(
        [
            ("obj_name", "cracker_box_flipped"),
            ("model_type", "mlp"),
            ("use_var", 1, float),
            ("data_usage", "200x1"),
            ("seed", 42, int),
        ]
    )
    set_seed(args.seed)

    main(args.obj_name, args.model_type, args.use_var, args.data_usage)
