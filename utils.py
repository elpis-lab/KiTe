import warnings
import numpy as np
import torch
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # for better printing
    # np.set_printoptions(precision=4, suppress=True)


def parse_args(args):
    """
    A simple wrapper for argument parser
    args is a list of arguments, each argument is
    a tuple of (name, default(optional), type(optional))
    """
    parser = argparse.ArgumentParser()
    for arg in args:
        kwargs = {"nargs": "?"}
        if len(arg) > 1:
            kwargs["default"] = arg[1]
        if len(arg) > 2:
            kwargs["type"] = arg[2]
        parser.add_argument(arg[0], **kwargs)

    args = parser.parse_args()
    return args


class DataLoader:
    """Class for loading data and splitting them"""

    def __init__(
        self,
        object_name: str,
        folder: str = "data",
        val_size: int = 1000,
        invert_xy: bool = False,
        shuffle: bool = False,
    ):
        """Initialize with data files, split sizes, and some options"""
        self.data_x_file = folder + "/x_" + object_name + ".npy"
        self.data_y_file = folder + "/y_" + object_name + ".npy"
        self.val_size = val_size

        # Load data
        x = np.load(self.data_x_file).astype(np.float32)
        y = np.load(self.data_y_file).astype(np.float32)
        # Pre-process data
        if invert_xy:
            x, y = y, x
        if shuffle:
            idx = np.random.permutation(len(x))
            x, y = x[idx], y[idx]
        self.x = x
        self.y = y

        # Check
        self.pool_size = len(self.x) - self.val_size
        if self.pool_size < 0:
            raise ValueError("Pool size is negative")

    def load_data(self, verbose=1):
        """Load all data as a dictionary"""
        # Split data
        x_pool = self.x[: self.pool_size]
        y_pool = self.y[: self.pool_size]
        x_val = self.x[-self.val_size :]
        y_val = self.y[-self.val_size :]
        if verbose:
            print("Loading data")
            print(f"Pool data points: {x_pool.shape[0]}")
            print(f"Validation data points: {x_val.shape[0]}")

        datasets = dict()
        datasets["x_pool"] = x_pool
        datasets["y_pool"] = y_pool
        datasets["x_val"] = x_val
        datasets["y_val"] = y_val
        return datasets
