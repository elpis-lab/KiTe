import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from geometry.object_model import get_obj_shape
from experiments.train_push_model import load_model
from experiments.utils import get_names
from planning.push import visualize_push_env

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def gen_path():
    # Load dynamics model
    m_id = 0
    # obj_name = "master_chef_can_flipped"
    obj_name = "trash_truck"
    model_type = "mlp"
    n_data = 1000
    model_name, data_name, rep_data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")

    # Load trained model (always use var=1 model for fair comparison)
    model = load_model(model_type, obj_shape, 1)
    name = f"{obj_name}_{model_type}_{1}_{n_data}_{m_id}"
    model.load(f"results/models/{name}.pth")

    # Load data
    from experiments.utils import DataLoader

    data_loader = DataLoader(data_name)
    dataset = data_loader.load_data()
    used_indices = np.load(f"results/learning/idx_used_{name}.npy")
    x_train = dataset["x_pool"][used_indices]
    y_train = dataset["y_pool"][used_indices]
    print(x_train[:10])
    print(y_train[:10])

    pred = model.predict(x_train[:10])
    pred[:, 3:] = np.exp(pred[:, 3:])
    print(pred)

    #
    controls = np.ones((4, 3))
    controls[:, 0] = controls[:, 0] * 0.75
    controls[:, 1] = 0.02
    controls[:, 2] = 0.05

    from lie_group.lie_se2 import to_se2_transform, to_se2_vec, log_se2
    from lie_group.propagation import propagate_cov
    from planning.planning_utils import vec_to_cov, cov_to_vec

    preds = model.predict(controls)
    unified_states = np.zeros((len(preds) + 1, 9))
    unified_states[0, :3] = [0, -0.7, 0]
    unified_states[0, 3:] = [1e-6, 0, 0, 1e-6, 0, 1e-6]
    # propagate states
    for j, delta in enumerate(preds):
        delta_s = delta[:3]
        var = np.exp(delta[3:])
        delta_cov = vec_to_cov([var[0], 0, 0, var[1], 0, var[2]])

        # get delta
        curr_state = unified_states[j, :3]
        curr_cov = vec_to_cov(unified_states[j, 3:])
        # state
        t_delta = to_se2_transform(delta_s)
        state = to_se2_transform(curr_state) @ t_delta
        # covariance
        delta = log_se2(t_delta)
        cov = propagate_cov(delta, curr_cov, delta_cov)
        unified_states[j + 1, :3] = to_se2_vec(state)
        unified_states[j + 1, 3:] = cov_to_vec(cov)

    return unified_states


def vis_master_chef_can():
    obj_name = "master_chef_can_flipped"

    # Load object shape
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")

    # Load environment
    env_i = 6
    envs = np.load("data/planning_push_envs.npy", allow_pickle=True)
    env = envs[env_i]
    # visualize_push_env(env, obj_shape=obj_shape)
    # plt.show()

    # Load path
    kite_path = np.load(
        f"results/planning_push/{obj_name}_mlp_1_1000_aorrt_w2_random_20.0_plan_states.npy",
        allow_pickle=True,
    )

    path = kite_path[3, env_i, -1]
    text_size = 30
    fig, ax = visualize_push_env(
        env, path, None, obj_shape, "C4", legend=False
    )
    fig.set_size_inches(8, 6)

    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-1.0, -0.4)
    ax.set_xlabel("X (m)", fontsize=text_size)
    ax.set_ylabel("Y (m)", fontsize=text_size)
    ax.tick_params(axis="both", labelsize=text_size)
    fig.tight_layout()
    fig.savefig(f"results/planning_push/kite_{obj_name}.png", dpi=300)
    plt.show()


def vis_real_trash_truck():
    obj_name = "real_trash_truck"

    # Load object shape
    model_name, data_name, rep_data_name = get_names(obj_name)
    obj_shape = get_obj_shape(f"assets/{model_name}/textured.obj")

    # Load environment
    # env_i = 12
    env_i = 14
    envs = np.load("data/planning_push_envs.npy", allow_pickle=True)
    env = envs[env_i]
    # visualize_push_env(env, obj_shape=obj_shape)
    # plt.show()

    # Load path
    kite_path = np.load(
        f"results/planning_push/{obj_name}_mlp_1_1000_aorrt_w2_random_20.0_plan_states.npy",
        allow_pickle=True,
    )
    ap_path = np.load(
        f"results/planning_push/{obj_name}_mlp_0_1000_aorrt_l2_active_0.0_plan_states.npy",
        allow_pickle=True,
    )

    path1 = kite_path[4, env_i, -1]
    path2 = ap_path[3, env_i, -1]
    # Clean to make the plot more clear
    path1 = path1[:4] + path1[-3:-2] + path1[-1:]
    path2 = path2[:-1]
    path2[-1][2] = path2[-1][2] / 4

    # Figures
    text_size = 30
    f1, ax1 = visualize_push_env(
        env, path1, None, obj_shape, "C4", legend=False
    )
    f2, ax2 = visualize_push_env(
        env, path2, None, obj_shape, "C5", legend=False
    )

    f1.set_size_inches(8, 6)
    f2.set_size_inches(8, 6)

    ax1.set_xlim(-0.4, 0.4)
    ax1.set_ylim(-1.0, -0.4)
    ax1.set_xlabel("X (m)", fontsize=text_size)
    ax1.set_ylabel("Y (m)", fontsize=text_size)
    ax1.tick_params(axis="both", labelsize=text_size)
    ax2.set_xlim(-0.4, 0.4)
    ax2.set_ylim(-1.0, -0.4)
    ax2.set_xlabel("X (m)", fontsize=text_size)
    ax2.set_ylabel("Y (m)", fontsize=text_size)
    ax2.tick_params(axis="both", labelsize=text_size)
    f1.tight_layout()
    f2.tight_layout()
    f1.savefig(f"results/planning_push/kite_{obj_name}.png", dpi=300)
    f2.savefig(f"results/planning_push/ap_{obj_name}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    from experiments.utils import set_seed

    set_seed(42)

    # vis_master_chef_can()
    vis_real_trash_truck()
    exit()

    obj_name = "real_trash_truck"

    # Load object shape
    obj_shape = get_obj_shape(f"assets/{obj_name}/textured.obj")

    # Load environment
    env_i = 0
    envs = np.load("data/planning_push_envs.npy", allow_pickle=True)
    env = envs[env_i]
    # visualize_push_env(env, obj_shape=obj_shape)
    # plt.show()

    # Load path
    kite_path = np.load(
        f"results/planning_push/{obj_name}_mlp_1_1000_aorrt_w2_random_20.0_plan_states.npy",
        allow_pickle=True,
    )
    # ap_path = np.load(
    #     f"results/planning_push/{obj_name}_mlp_0_1000_aorrt_l2_active_0.0_plan_states.npy",
    #     allow_pickle=True,
    # )
    # base_path = np.load(
    #     f"results/planning_push/{obj_name}_mlp_0_1000_aorrt_l2_random_0.0_plan_states.npy",
    #     allow_pickle=True,
    # )
    # path1 = kite_path[0, env_i, -1]
    # path2 = base_path[0, env_i, -1]
    # clean a bit for visualization

    # f1, ax1 = visualize_push_env(
    #     env, path1, obj_shape=obj_shape, title="KiTe", color="C4"
    # )
    # f2, ax2 = visualize_push_env(
    #     env, path2, obj_shape=obj_shape, title="Base", color="C7"
    # )
