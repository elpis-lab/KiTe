import numpy as np
import matplotlib.pyplot as plt


def data_plot(obj_name, n_datas, configs, config_names):
    # Get all the results
    results = None
    folder = f"results/planning_push"
    for i, n_data in enumerate(n_datas):
        for j, config in enumerate(configs):
            algo = config[0]
            belief = config[1]
            use_var = 1 if belief == "w2" else 0
            active_sampling = config[2]
            terminal_weight = config[3]
            name = (
                f"{obj_name}_{model_type}_{use_var}_{n_data}"
                + f"_{algo}_{belief}_{active_sampling}_{terminal_weight}"
            )
            res = np.load(f"{folder}/{name}_results.npy")

            if results is None:
                results = np.zeros(
                    (len(n_datas), len(configs), res.shape[0], 2)
                )
            # Mean over all problems (for success rate and path error)
            results[i, j] = res[:, :, [0, 3]].mean(axis=1)

    # Get mean and std
    success_mean = np.mean(results[:, :, :, 0], axis=2)
    success_std = np.std(results[:, :, :, 0], axis=2)
    path_error_mean = np.mean(results[:, :, :, 1], axis=2)
    path_error_std = np.std(results[:, :, :, 1], axis=2)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True)
    x = np.arange(len(n_datas))

    for j, config in enumerate(configs):
        y1 = success_mean[:, j]
        y1_std = success_std[:, j]
        y2 = path_error_mean[:, j]
        y2_std = path_error_std[:, j]
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
        # "banana",
        "master_chef_can_flipped",
        "trash_truck",
        # "real_cracker_box_flipped",
        # "real_trash_truck",
    ]
    model_type = "mlp"
    n_datas = [100, 500, 1000]
    # n_datas = [100, 1000]

    # Configs (algo, use_var, active_sampling, terminal_weight)
    configs = [
        ("aorrt", "l2", "active", 0.0),  # Active Pusher
        ("sst", "l2", "active", 0.0),  # Active Pusher
        ("aorrt", "w2", "random", 0.0),  # Belief Random Tree
        ("sst", "w2", "random", 0.0),  # Belief Random Tree
        ("aorrt", "l2", "random", 2.0),  # Proposed
        ("aorrt", "w2", "random", 2.0),  # Proposed
    ]
    config_names = [
        "ActivePusher-l2-AORRT",
        "ActivePusher-l2-SST",
        "BeliefRandomTree-w2-AORRT",
        "BeliefRandomTree-w2-SST",
        "Kite-l2-AORRT-T",
        "Kite-w2-AORRT-T",
    ]

    for obj_name in obj_names:
        data_plot(obj_name, n_datas, configs, config_names)
