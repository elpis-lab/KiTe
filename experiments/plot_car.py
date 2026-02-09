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
                    (len(n_datas), len(configs), res.shape[0], 3)
                )
            # Mean over all problems (for success rate and path error)
            print(res[:, :, 7])
            results[i, j] = res[:, :, [0, 7, 8]].mean(axis=1)

    # Get mean and std
    success_mean = np.mean(results[:, :, :, 0], axis=2)
    success_std = np.std(results[:, :, :, 0], axis=2)
    # path_error_mean = np.mean(results[:, :, :, 1], axis=2)
    # path_error_std = np.std(results[:, :, :, 1], axis=2)
    running_cost_mean = np.mean(results[:, :, :, 1], axis=2)
    running_cost_std = np.std(results[:, :, :, 1], axis=2)
    terminal_cost_mean = np.mean(results[:, :, :, 2], axis=2)
    terminal_cost_std = np.std(results[:, :, :, 2], axis=2)

    # Plot results: 1 x 3 (Success | Running Cost | Terminal Cost)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    ax_succ, ax_run, ax_term = axes
    x = np.arange(len(n_datas))

    for j, config in enumerate(configs):
        label = config_names[j]

        # Success rate
        y = success_mean[:, j]
        y_std = success_std[:, j]
        ax_succ.plot(x, y, label=label)
        ax_succ.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        # Running cost
        y = running_cost_mean[:, j]
        y_std = running_cost_std[:, j]
        ax_run.plot(x, y)
        ax_run.fill_between(x, y - y_std, y + y_std, alpha=0.2)

        # Terminal cost
        y = terminal_cost_mean[:, j]
        y_std = terminal_cost_std[:, j]
        ax_term.plot(x, y)
        ax_term.fill_between(x, y - y_std, y + y_std, alpha=0.2)

    # Axis labels & formatting
    ax_succ.set_ylabel("Success Rate")
    ax_run.set_ylabel("Running Cost")
    ax_term.set_ylabel("Terminal Cost")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(n_datas, rotation=45)
        ax.set_xlabel("Data")

    # Legend (put it once to avoid clutter)
    ax_succ.legend()

    plt.tight_layout()
    plt.savefig(f"results/planning_push/{obj_name}_data.png")
    plt.show()


if __name__ == "__main__":
    obj_names = [
        "cracker_box_flipped",
        "banana",
        "master_chef_can_flipped",
        "trash_truck",
        "real_cracker_box_flipped",
        "real_trash_truck",
    ]
    model_type = "mlp"
    # n_datas = [100, 1000]
    n_datas = [200, 400, 600, 800, 1000]

    # Configs (algo, use_var, active_sampling, terminal_weight)
    # configs = [
    #     ("aorrt", "l2", "active", 0.0),  # Active Pusher
    #     ("sst", "l2", "active", 0.0),  # Active Pusher
    #     ("aorrt", "w2", "random", 0.0),  # Gaussian Belief Trees
    #     ("sst", "w2", "random", 0.0),  # Gaussian Belief Trees
    #     ("aorrt", "l2", "random", 2.0),  # Proposed
    #     ("aorrt", "w2", "random", 2.0),  # Proposed
    # ]
    configs = [
        ("aorrt", "l2", "random", 0.0),  # Vanilla
        ("sst", "l2", "random", 0.0),  # Vanilla
        ("sst", "l2", "active", 0.0),  # Active Pusher
        ("sst", "w2", "random", 0.0),  # Gaussian Belief Trees
        ("aorrt", "l2", "random", 2.0),  # Proposed
        ("aorrt", "w2", "random", 2.0),  # Proposed
    ]
    # config_names = [
    #     "ActivePusher-l2-AORRT",
    #     "ActivePusher-l2-SST",
    #     "BeliefRandomTree-w2-AORRT",
    #     "BeliefRandomTree-w2-SST",
    #     "Kite-l2-AORRT-T",
    #     "Kite-w2-AORRT-T",
    # ]
    config_names = [
        "Vanilla (AORRT-l2)",
        "Vanilla (SST-l2)",
        "Active Pusher (SST-l2)",
        "Gaussian Belief Trees (SST-w2)",
        "Proposed (AORRT-T-l2)",
        "Proposed (AORRT-T-w2)",
    ]

    for obj_name in obj_names:
        data_plot(obj_name, n_datas, configs, config_names)
