import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import DataLoader, parse_args, set_seed, get_names
from geometry.pose import angle_diff
from geometry.object_model import get_obj_shape
from models.physics import push_physics
from models.torch_model import MLP, MLPVar, MLPEvidential
from models.torch_model import Physics, ResidualPhysics
from models.torch_model import ResidualPhysicsVar, ResidualPhysicsEvidential
from models.torch_loss_se2 import (
    se2_split_loss,
    mse_se2_loss,
    nll_se2_loss,
)
from models.torch_loss_se2 import beta_nll_se2_loss, evidential_se2_loss
from models.model import TorchModel


def load_model(
    model_type="mlp",
    obj_shape=None,
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
    # deprecated for this project
    elif model_type == "residual":
        equation = get_push_physics(model_type, obj_shape[:2])
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
        equation = get_push_physics(model_type, obj_shape[:2])
        model_class = lambda: Physics(equation)
        require_training = False
    else:
        raise ValueError(f"Model type {model_type} not supported")

    # Use NLL loss if variance is predicted
    if use_var == 0:
        loss_fn = mse_se2_loss  # se2_split_loss
    elif use_var == 1:
        loss_fn = nll_se2_loss  # beta_nll_se2_loss
    elif use_var == 2:
        loss_fn = evidential_se2_loss
    score_fn = mse_se2_loss  # se2_split_loss

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
        return push_physics(param, obj_size[:2], relative=True)

    if model_type == "residual" or model_type == "physics":
        return push_physics_with_size
    else:
        return None


def evaluate_results(pred, true, verbose=False):
    """Evaluate results"""
    # Prepare
    pred = np.asarray(pred)
    true = np.asarray(true)
    true_std = true[:, 3:]
    # if pred is 3d (no variance), append zeros to make it 6d
    if pred.shape[1] == 3:
        pred = np.concatenate(
            [pred, np.zeros((pred.shape[0], 3), dtype=pred.dtype)], axis=1
        )
    pred_torch = torch.from_numpy(pred)
    true_mean_torch = torch.from_numpy(true[:, :3])

    # Process variance
    if pred.shape[1] == 3:
        # no variance prediction
        pred_std = np.zeros_like(true_std)
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
        epistemic = 1 / nu
        total_var = aleatoric + epistemic
        pred_std = np.sqrt(aleatoric)

    # NLL loss
    nll_loss = nll_se2_loss(pred_torch, true_mean_torch)

    # SE2 RMSE
    mse_loss = mse_se2_loss(pred_torch, true_mean_torch)
    rmse_loss = np.sqrt(mse_loss)
    pos_error = np.mean(np.linalg.norm(true[:, :2] - pred[:, :2], axis=1))
    rot_error = np.mean(np.abs(angle_diff(true[:, 2], pred[:, 2])))

    # STD RMSE
    mse_std = np.mean((pred_std - true_std) ** 2)
    rmse_std = np.sqrt(mse_std)
    std_error = np.mean(np.abs(pred_std - true_std), axis=0)

    if verbose:
        print(f"NLL Loss: {nll_loss}")
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

    return nll_loss, rmse_loss, pos_error, rot_error, rmse_std, *std_error


def main(obj_name, model_type, use_var=1, n_data=1000, seed=42, plot=False):
    """Train a model."""
    set_seed(seed)
    name = f"{obj_name}_{model_type}_{use_var}_{n_data}_{seed}"

    # Load data
    model_name, data_name, rep_data_name = get_names(obj_name)
    # train
    data_loader = DataLoader(data_name)
    datasets = data_loader.load_data()
    x_train, y_train = datasets["x_pool"], datasets["y_pool"]
    perm = np.random.permutation(1000)
    x_train, y_train = x_train[perm], y_train[perm]
    # eval
    # data_loader = DataLoader(rep_data_name, val_size=900)
    data_loader = DataLoader(rep_data_name)
    datasets = data_loader.load_data()
    x_eval = datasets["x_pool"]
    y_eval = datasets["y_pool"]

    # Load model
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")
    model = load_model(model_type, obj_shape, use_var, epochs=1000)
    tr_losses, val_losses = model.fit(
        x_train[:n_data], y_train[:n_data], x_eval, y_eval
    )
    plt.plot(np.arange(len(tr_losses)), tr_losses, label="Training")
    plt.plot(np.arange(len(val_losses)), val_losses, label="Validation")
    plt.legend()
    plt.savefig(f"results/learning/loss_{name}.jpg")
    if plot:
        plt.show()
    # model.load(f"results/models/{name}.pth")
    # Results
    pred = model.predict(x_eval)
    res = evaluate_results(pred, y_eval, verbose=True)

    # Save the model
    np.save(f"results/learning/idx_used_{name}.npy", perm[:n_data])
    np.save(f"results/learning/loss_{name}.npy", res)
    model.save(f"results/models/{name}.pth")
    return res


if __name__ == "__main__":
    args = parse_args(
        [
            ("obj_name", "cracker_box_flipped"),
            ("model_type", "mlp"),
            ("use_var", 0, float),
            ("n_data", 1000, int),
            ("seed", 42, int),
        ]
    )
    main(args.obj_name, args.model_type, args.use_var, args.n_data, args.seed)
