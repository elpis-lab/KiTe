import numpy as np
import matplotlib.pyplot as plt


def main(obj_name, model, n_datas, env, configs, config_names):
    # Get all the results
    results = None
    folder = f"results/planning_{env}"
    for i, config in enumerate(configs):
        for j, n_data in enumerate(n_datas):
            name = (
                f"{obj_name}_{model}_{config[0]}_{n_data}"
                + f"_{config[1]}_{config[2]}_{config[3]}"
            )
            res = np.load(f"{folder}/{name}_results.npy")
            if results is None:
                results = np.zeros(
                    (len(configs), len(n_datas), res.shape[0], 2)
                )
            results[i, j] = res[:, [0, 2]]

    # Get mean and std
    success_mean = np.mean(results[:, :, :, 0], axis=2)
    success_std = np.std(results[:, :, :, 0], axis=2)
    path_error_mean = np.mean(results[:, :, :, 1], axis=2)
    path_error_std = np.std(results[:, :, :, 1], axis=2)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    x = np.arange(len(n_datas))

    for j, config in enumerate(configs):
        y1 = success_mean[j, :]
        y1_std = success_std[j, :]
        y2 = path_error_mean[j, :]
        y2_std = path_error_std[j, :]
        ax1.plot(x, y1, label=f"{config_names[j]}")
        ax1.fill_between(x, y1 - y1_std, y1 + y1_std, alpha=0.2)
        ax2.plot(x, y2, label=f"{config_names[j]}")
        ax2.fill_between(x, y2 - y2_std, y2 + y2_std, alpha=0.2)
    for ax, ylabel in zip((ax1, ax2), ("Success Rate", "Path Error")):
        ax.set_xticks(x)
        ax.set_xticklabels(n_datas, rotation=45)
        ax.set_ylabel(ylabel)
        ax.legend()
    ax2.set_xlabel("Data")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    obj_names = [
        "cracker_box_flipped",
        # "mustard_bottle_flipped",
        # "banana",
        # "letter_t",
        "master_chef_can_flipped",
        "school_bus",
        # "trash_truck",
        # "real_cracker_box_flipped",
        # "real_mustard_bottle_flipped",
        # "real_master_chef_can_flipped",
        # "real_school_bus",
        # "real_trash_truck",
    ]
    model = "mlp"
    n_datas = np.arange(100, 1001, 100)
    envs = ["free", "obs"]

    # n_datas = [1000, 1000]
    obj_names = ["cracker_box_flipped", "master_chef_can_flipped"]
    envs = ["obs"]

    # Configs (use_var, belief_space, active_sampling, active_selection)
    configs = [
        # (1.0, False, False, False),  # Baseline SST
        # (0.0, False, True, False),  # Active Pusher
        # (1.0, True, False, False),  # Belief Random Tree
        # (1.0, True, True, False),  # AS
        # (1.0, True, False, True),  # AP
        # (1.0, True, True, True),  # Active Puna
        (1.0, "regular", "random", "cost"),  # Baseline SST
        (1.0, "belief", "random", "cost"),  # Belief Random Tree
        (0.0, "regular", "active", "cost"),  # Active Pusher
        (1.0, "belief", "active", "cost"),  # AS
        (1.0, "belief", "random", "prob"),  # AP
        (1.0, "belief", "active", "prob"),  # Active Puna
    ]
    config_names = [
        "SST",
        "BeliefRandomTree",
        "ActivePusher",
        "ActiveSampling",
        "ActivePathSelection",
        "ActivePuna",
    ]

    for obj_name in obj_names:
        for env in envs:
            main(obj_name, model, n_datas, env, configs, config_names)
