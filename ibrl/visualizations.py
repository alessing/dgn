from train_rl import load_model
from train_bc import load_model as load_model_bc
import torch
import os
import pickle
import numpy as np
import common_utils
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting


RL_BASE_DIR = '/iris/u/aml2023/rl_refinement/ibrl/exps/rl_sweep5'
BC_BASE_DIR = '/iris/u/aml2023/rl_refinement/ibrl/exps/bc_sweep5'

#For each run model0.pt is the best checkpoint. model_step{step_num}.pt is the model saved at that step

# BC policy
BC_NO_DROP_RUN_PATH = os.path.join(BC_BASE_DIR, 'square_50demos_seed1_no_drop/model0.pt')
bc_policy, _, _ = load_model_bc(BC_NO_DROP_RUN_PATH, 'cuda')

BC_NO_DROP_ONE_EP_RUN_PATH = os.path.join(BC_BASE_DIR, 'square_50demos_seed1_no_drop_one_ep/model0.pt')
bc_policy_one_ep, _, _ = load_model_bc(BC_NO_DROP_RUN_PATH, 'cuda')

# RLPD policy
RUN_PATH_rlpd = os.path.join(RL_BASE_DIR, 'rlpd_bl_square_state_50demos_seed118')
agent_rlpd, _, _ = load_model(os.path.join(RUN_PATH_rlpd, 'model0.pt'), "cuda", load_dataset=True)
# # state_dict = torch.load(os.path.join(RUN_PATH, 'expl_module_ckpts', "cov_mlp_model_step50000.pt"))

# RFT policy
RUN_PATH_rft = os.path.join(RL_BASE_DIR, 'rft_bl_square_state_50demos_seed127')
agent_rft, _, _ = load_model(os.path.join(RUN_PATH_rft, 'model0.pt'), "cuda", load_dataset=True)

# Ours
RLPD_LC_RUN_PATH = os.path.join(RL_BASE_DIR, 'rlpd_learned_cov_hidSize128_square_state_50demos_seed100')
agent_rlpd_lc, _, _ = load_model(os.path.join(RLPD_LC_RUN_PATH, 'model0.pt'), "cuda", load_dataset=True)

# IBRL with no actor dropout
IBRL_NO_ACTOR_DROP_RUN_PATH = os.path.join(RL_BASE_DIR, 'ibrl_noBCDrop_noDropActor_bl_square_state_50demos_seed100')
agent_ibrl_no_actor_dropout, _, _ = load_model(os.path.join(IBRL_NO_ACTOR_DROP_RUN_PATH, 'model0.pt'), "cuda", load_dataset=True)

# IBRL with dropout
IBRL_RUN_PATH = os.path.join(RL_BASE_DIR, 'ibrl_noBCDrop_bl_square_state_50demos_seed130')
agent_ibrl, _, _ = load_model(os.path.join(IBRL_NO_ACTOR_DROP_RUN_PATH, 'model0.pt'), "cuda", load_dataset=True)

#How to get demo states
bc_dataset = agent_rlpd_lc.actor.explore_module.dataset
example_state_from_demos = agent_rlpd_lc.actor.explore_module.dataset[0][0]['state'].unsqueeze(0).cuda()

# Inference with actor
#explore_mode=True adds perturbations from learned cov model the actions returned when eval_mode=False. eval_mode=True turns off noise when sampling
with common_utils.ibrl_utils.eval_mode(agent_rlpd_lc):
    action = agent_rlpd_lc.act({'state': example_state_from_demos}, eval_mode=True, stddev=0., explore_mode=False)

    print(action)
    
# Inference with BC policy
with common_utils.ibrl_utils.eval_mode(bc_policy):
    action = bc_policy.act({'state': example_state_from_demos}, eval_mode=True)

    print(action)
    

def compute_kl_divergence(agent1, agent2, dataset, num_samples=100, bins=100, name=""):
    """
    Estimate the KL divergence between two policies by comparing their
    empirical action distributions over a dataset using histograms.
    """
    actions1 = []
    actions2 = []

    for i in range(min(num_samples, len(dataset))):
        state = dataset[i][np.random.choice(len(dataset[i]))]['state'].unsqueeze(0).cuda()

        with torch.no_grad():
            with common_utils.ibrl_utils.eval_mode(agent1):
                action1 = agent1.act({'state': state}, eval_mode=True, stddev=0., explore_mode=False).cpu().numpy()

            with common_utils.ibrl_utils.eval_mode(agent2):
                action2 = agent2.act({'state': state}, eval_mode=True).cpu().numpy()

        actions1.append(action1.squeeze())
        actions2.append(action2.squeeze())

    actions1 = np.stack(actions1)
    actions2 = np.stack(actions2)

    # Flatten to 1D if needed, or handle multi-dimensions separately
    if actions1.ndim == 2 and actions1.shape[1] > 1:
        # Sum KL divergences across each dimension
        total_kl = 0.
        for dim in range(actions1.shape[1]):
            hist1, bin_edges = np.histogram(actions1[:, dim], bins=bins, density=False)
            hist2, _ = np.histogram(actions2[:, dim], bins=bin_edges, density=False)

            hist1 = hist1.astype(float) + 1e-6
            hist2 = hist2.astype(float) + 1e-6

            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()

            kl = scipy.stats.entropy(hist1, hist2)
            total_kl += kl
        avg_kl = total_kl
    else:
        # 1D actions
        hist1, bin_edges = np.histogram(actions1, bins=bins, density=False)
        hist2, _ = np.histogram(actions2, bins=bin_edges, density=False)

        hist1 = hist1.astype(float) + 1e-6
        hist2 = hist2.astype(float) + 1e-6

        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()

        avg_kl = scipy.stats.entropy(hist1, hist2)
    print(f"Empirical KL divergence between action distributions of {name} and BC policy on Square: {avg_kl:.6f}")
    return avg_kl

# Usage example
# kl_ours_bc = compute_kl_divergence(agent_rlpd_lc, bc_policy, bc_dataset, num_samples=100, name="ours")
# kl_ibrl_bc = compute_kl_divergence(agent_ibrl_no_actor_dropout, bc_policy, bc_dataset, num_samples=100, name="ibrl")
# kl_rlpd_bc = compute_kl_divergence(agent_rlpd, bc_policy, bc_dataset, num_samples=100, name="rlpd")
# kl_rft_bc = compute_kl_divergence(agent_rft, bc_policy, bc_dataset, num_samples=100, name="rft")

def evaluate_kl_over_checkpoints(agent, checkpoint_dir, checkpoint_steps, bc_policy, dataset, label):
    kls = []
    for step in checkpoint_steps:
        ckpt_path = os.path.join(checkpoint_dir, f"model_step{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[{label}] Skipping missing checkpoint: {ckpt_path}")
            continue

        state_dict = torch.load(ckpt_path)
        agent.load_state_dict(state_dict)

        kl = compute_kl_divergence(agent, bc_policy, dataset, num_samples=100, name=label)
        print(f"[{label}] Step {step}: KL = {kl:.4f}")
        kls.append((step, kl))
    return kls


def plot_kl_trajectories(kl_dict):
    plt.figure()
    for label, kl_data in kl_dict.items():
        steps, kls = zip(*kl_data)
        plt.plot(steps, kls, label=label, marker='o')
    plt.xlabel("Checkpoint step")
    plt.ylabel("KL divergence vs. BC")
    plt.title("KL divergence across training steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/iris/u/asc8/workspace/bc_exploration/rl_refinement/ibrl/kl_divergence_plot.png")


# === Example Usage ===
# checkpoint_steps = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 325000, 350000, 375000, 400000, 425000, 450000, 475000, 500000]

checkpoint_steps = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]

# kl_results = {
#     "ours": evaluate_kl_over_checkpoints(agent_rlpd_lc, RLPD_LC_RUN_PATH, checkpoint_steps, bc_policy, bc_dataset, label="ours"),
#     "ibrl": evaluate_kl_over_checkpoints(agent_ibrl, IBRL_NO_ACTOR_DROP_RUN_PATH, checkpoint_steps, bc_policy, bc_dataset, label="ibrl"),
#     "rlpd": evaluate_kl_over_checkpoints(agent_rlpd, RUN_PATH_rlpd, checkpoint_steps, bc_policy, bc_dataset, label="rlpd"),
#     "rft": evaluate_kl_over_checkpoints(agent_rft, RUN_PATH_rft, checkpoint_steps, bc_policy, bc_dataset, label="rft"),
# }

# plot_kl_trajectories(kl_results)


def compute_and_plot_perturbation_magnitudes(agent, base_path, checkpoint_steps, dataset, num_samples=100):
    """
    For each checkpoint, load the weights, compute perturbations across demo states,
    and plot average perturbation magnitude (L2 norm).
    
    Parameters:
    - agent: the agent object to reuse for loading checkpoints
    - base_path: directory containing the checkpoints
    - checkpoint_steps: list of step numbers corresponding to filenames like model_step{step}.pt
    - dataset: demo dataset (e.g., agent.actor.explore_module.dataset)
    - num_samples: number of demo states to sample per checkpoint
    """
    magnitudes = []

    for step in checkpoint_steps:
        ckpt_path = os.path.join(base_path, f"model_step{step}.pt")

        # Load checkpoint weights
        state_dict = torch.load(ckpt_path)
        agent.load_state_dict(state_dict)

        total_norm = 0.0
        count = 0

        # Compute perturbations over sampled demo states
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                for j in range(len(dataset[i])):
                    # Assuming dataset[i] is a tuple of (state, action)
                    # and we only need the state for perturbation
                    state = dataset[i][j]['state'].unsqueeze(0).cuda()
                    dist = agent.actor.explore_module._forward(state, inference=True)
                    pert = dist.sample()
                    total_norm += pert.norm(p=2).item()
                    count += 1

        avg_norm = total_norm / count
        magnitudes.append(avg_norm)
        print(f"Step {step}: Avg. perturbation magnitude = {avg_norm:.4f}")

    # Plotting
    plt.figure()
    plt.plot(checkpoint_steps, magnitudes, marker='o')
    plt.xlabel("Checkpoint step")
    plt.ylabel("Average perturbation magnitude (L2 norm)")
    plt.title("Perturbation magnitude across checkpoints")
    plt.grid(True)
    plt.savefig("/iris/u/asc8/workspace/bc_exploration/rl_refinement/ibrl/perturbation_magnitude_plot.png")

    return checkpoint_steps, magnitudes

# checkpoint_steps = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 325000, 350000, 375000, 400000, 425000, 450000, 475000, 500000]
# base_path = os.path.join(RLPD_LC_RUN_PATH)  # or wherever your checkpoints are
# dataset = agent_rlpd_lc.actor.explore_module.dataset
# agent_rlpd_lc.actor.explore_module.cov_mlp.eval()

# compute_and_plot_perturbation_magnitudes(agent_rlpd_lc, base_path, checkpoint_steps, dataset)


# Compute perturbation magnitudes from replay buffers
def compute_perturbation_from_replay(agent, replay_dir, epochs, key='delta_actions'):
    """
    Loads replay buffer trajectory files and computes perturbation magnitudes
    using agent.actor.explore_module._forward for all states.
    
    Parameters:
        agent: the policy agent with explore_module
        replay_dir: directory with trajectory{epoch}.npz files
        epochs: list of epoch numbers to process
        key: the key in npz files to load states from (default: 'states')
    """
    perturbation_mags = []
    all_epochs = []

    for epoch in epochs:
        path = os.path.join(replay_dir, f"trajectory{epoch}.npz")
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        data = np.load(path)
        if key not in data:
            print(f"Key '{key}' not found in {path}")
            continue

        delta_actions = torch.tensor(data[key], dtype=torch.float32).cuda()  # shape [T, action_delta_shape]
        total_norm = 0.0
        total_steps = 0
        all_mags = []
        with torch.no_grad():
            for i in range(1, delta_actions.shape[0]):
                mag = delta_actions[i].norm(p=2).item()
                if not np.isnan(mag):
                    total_norm += mag
                    total_steps += 1
                    all_mags.append(mag)
                else:
                    print(f"NaN encountered at index {i} for epoch {epoch}")
        
        avg_norm = total_norm / total_steps  # Exclude the first state
        all_epochs.append(epoch)
        median_mag = np.median(all_mags)

        print(f"Processed epoch {epoch}: {delta_actions.shape[0]} delta_actions", 
              f"Avg. perturbation magnitude = {avg_norm:.4f}")
        perturbation_mags.append(median_mag)

    # Plotting
    plt.figure()
    plt.plot(all_epochs, perturbation_mags)
    plt.xlabel("Epoch")
    plt.ylabel("Perturbation magnitude (L2 norm)")
    plt.title("Perturbation magnitude across checkpoints")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/iris/u/asc8/workspace/bc_exploration/rl_refinement/ibrl/perturbation_magnitude_replay_plot.png")

    return perturbation_mags, all_epochs

epochs = [25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 325000, 350000, 375000, 400000, 425000, 450000, 475000, 500000]
replay_dir = os.path.join(RLPD_LC_RUN_PATH)

# compute_perturbation_from_replay(agent_rlpd_lc, replay_dir, epochs, key='delta_actions')


def load_states_from_replay(replay_dir, epochs, key='states'):
    all_states = []
    for epoch in epochs:
        path = os.path.join(replay_dir, f"trajectory{epoch}.npz")
        if not os.path.exists(path):
            continue
        data = np.load(path)
        if key not in data:
            continue
        states = data[key]  # shape [T, state_dim]
        all_states.append(states)
    if not all_states:
        return None
    return np.concatenate(all_states, axis=0)  # [N, state_dim]


def extract_demo_states(dataset, max_samples=1000):
    states = []
    for i in range(min(max_samples, len(dataset))):
        for j in range(len(dataset[i])):
            # Assuming dataset[i] is a tuple of (state, action)
            # and we only need the state for perturbation
            state = dataset[i][j]['state'].unsqueeze(0).cpu().numpy()
            states.append(state)
        # states.append(dataset[i][0]['state'].cpu().numpy())
    return np.concatenate(states)  # shape [N, state_dim]


def compute_kl_state_distribution(method_states, demo_states, bins=100):
    """
    Compute empirical KL divergence between two state distributions.
    """
    if method_states.shape[1] > 1:
        total_kl = 0.0
        for dim in range(method_states.shape[1]):
            hist_demo, bin_edges = np.histogram(demo_states[:, dim], bins=bins, density=False)
            hist_method, _ = np.histogram(method_states[:, dim], bins=bin_edges, density=False)

            hist_demo = hist_demo.astype(float) + 1e-6
            hist_method = hist_method.astype(float) + 1e-6

            hist_demo /= hist_demo.sum()
            hist_method /= hist_method.sum()

            kl = scipy.stats.entropy(hist_demo, hist_method)
            total_kl += kl
        return total_kl
    else:
        hist_demo, bin_edges = np.histogram(demo_states, bins=bins, density=False)
        hist_method, _ = np.histogram(method_states, bins=bin_edges, density=False)

        hist_demo = hist_demo.astype(float) + 1e-6
        hist_method = hist_method.astype(float) + 1e-6

        hist_demo /= hist_demo.sum()
        hist_method /= hist_method.sum()

        return scipy.stats.entropy(hist_demo, hist_method)


def compute_state_kl_vs_demo_over_epochs(methods, replay_dirs, demo_dataset, epochs, bins=100):
    """
    For each method, loop through trajectory{epoch}.npz files and compute KL divergence
    between that epoch's states and the demonstration states.

    Args:
        methods: list of method names (str)
        replay_dirs: list of directory paths corresponding to methods
        demo_dataset: demo dataset object
        epochs: list of epoch integers to consider
        bins: histogram bins for KL

    Returns:
        kl_results: dict mapping method name -> list of (epoch, kl)
    """
    demo_states = extract_demo_states(demo_dataset)
    kl_results = {}

    for method, replay_dir in zip(methods, replay_dirs):
        kl_list = []
        for epoch in epochs:
            path = os.path.join(replay_dir, f"trajectory{epoch}.npz")
            if not os.path.exists(path):
                print(f"Skipping missing file: {path}")
                continue

            data = np.load(path)
            # ['observations', 'actions', 'rewards', 'terminals', 'steps', 'delta_actions', 'episodes', 'obs_dim', 'action_dim', 'dtype']
            # observations: (25083, 69), stacked the last 3 states, 23 per step x 3 steps = 69
            # actions: (25083, 7)
            states = data['observations']
            kl = compute_kl_state_distribution(states, demo_states, bins=bins)
            kl_list.append((epoch, kl))

        if kl_list:
            kl_results[method] = kl_list
            print(f"Done with {method}")

    # Plotting
    plt.figure()
    for method, kl_data in kl_results.items():
        steps, kls = zip(*kl_data)
        plt.plot(steps, kls, label=method, marker='o')

    plt.xlabel("Epoch / Checkpoint")
    plt.ylabel("KL divergence vs. Demo")
    plt.title("State Distribution KL vs. Demo (Over Training)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/iris/u/asc8/workspace/bc_exploration/rl_refinement/ibrl/state_kl_divergence_plot.png")

    return kl_results


methods = ['ours', 'ibrl', 'rlpd', 'rft']
replay_dirs = [
    os.path.join(RLPD_LC_RUN_PATH),
    os.path.join(IBRL_NO_ACTOR_DROP_RUN_PATH),
    os.path.join(RUN_PATH_rlpd),
    os.path.join(RUN_PATH_rft)
]
epochs = [50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000]

# compute_state_kl_vs_demo_over_epochs(methods, replay_dirs, agent_rlpd_lc.actor.explore_module.dataset, epochs)



def plot_xy_trajectories_with_delta_actions(npz_path, num_trajectories=3, skip=1, epoch=0):
    """
    Loads observation and delta_action from a trajectory file, and plots x-y positions
    with arrows indicating delta action directions.

    Args:
        npz_path: path to trajectory{epoch}.npz file
        num_trajectories: number of separate trajectory segments to plot
        skip: plot every `skip`th point (for clearer arrows)
    """
    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}")
        return

    data = np.load(npz_path)
    if 'observations' not in data or 'delta_actions' not in data:
        print("File must contain 'observations' and 'delta_actions'")
        return

    obs = data['observations']  # shape [T, obs_dim]
    actions = data['delta_actions']  # shape [T, action_dim]
    rewards = data['rewards']  # shape [T, 1]
    terminals = data['terminals']  # shape [T, 1]

    assert obs.shape[0] == actions.shape[0], "Mismatched lengths between obs and actions"

    starts = [i + 2 for i in np.where(terminals == 1)[0][::2]]
    starts.insert(0, 0)  # Start from the beginning
    
    T = obs.shape[0]
    # segment_length = 500 # T // num_trajectories
    # import pdb; pdb.set_trace()
    # starts = np.where
    
    # array([  496,   594,   670,  1354,  1432,  1573,  2258,  2365,  2491,
    #     2577,  2657,  2759,  3166,  3243,  3330,  3441,  3512,  4548,
    #     4725,  5111,  5884,  7198,  7289,  7376,  7751,  7838,  7948,
    #     8108, 10348, 10423, 10580, 11261, 11341, 11719, 11841, 12534,
    #    12609, 12708, 12791, 12920, 12998, 13714, 13785, 13873, 13982,
    #    14056, 14151, 14404, 14481, 14602, 14691, 14790, 14891, 14981,
    #    15366, 15454, 15551, 15628, 15730, 16147, 16248, 17054, 17782,
    #    17854, 17970, 18679, 18755, 18841, 18950, 19044, 19169, 19322,
    #    20315, 20415, 20791, 20897, 20977, 21535, 21935, 22026, 22445,
    #    22530, 22961, 23027, 23109, 23359, 23449, 23572, 23670, 23805,
    #    23911, 24320, 24700, 24784, 24873, 25110]),)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(2, 3):
        start = starts[i]
        end = starts[i+1] - 1
        print("Start:", start, "End:", end)
        o = obs[start:end:skip]
        a = actions[start:end:skip]

        x = o[:, 0]
        y = o[:, 1]
        z = o[:, 2]
        dx = a[:, 0] * 0.05  # Scale down for better visualization
        dy = a[:, 1] * 0.05
        dz = a[:, 2] * 0.05
        # x -= dx
        # y -= dy
        # z -= dz

        ax.quiver(x-dx, y-dy, z-dz, dx, dy, dz, length=1.0, normalize=False, linewidth=0.8, arrow_length_ratio=0.1, color='black')
        ax.plot(x, y, z, label=f"Trajectory {i+1} original", linestyle='--')
        # ax.plot(x+dx, y+dy, z+dz, label=f"Trajectory {i+1} corrected", linestyle='-', color='green')

    ax.scatter(x[0], y[0], z[0], c='red', s=80, marker='*', label='Start')
    ax.scatter(o[0,9], o[0,10], o[0,11], c='purple', s=80, marker='*', label='Object')
    print(f"Start: {x[0]}, {y[0]}, {z[0]}")
    print(f"Object: {o[0,:23]} {o[1,:23]} {o[2,:23]}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectories with delta_action Arrows")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"/iris/u/asc8/workspace/bc_exploration/rl_refinement/ibrl/trajectory_plot{epoch}.png")

epoch = 400000
npz_path = os.path.join(RLPD_LC_RUN_PATH, f"trajectory{epoch}.npz")
plot_xy_trajectories_with_delta_actions(npz_path, num_trajectories=1, skip=1, epoch=epoch)

    
# import pdb; pdb.set_trace()


# RLPD LC128, No Dropout — rlpd_learned_cov_hidSize128_square_state_50demos_seed100
# RLPD BL, No Dropout — rlpd_bl_square_state_50demos_seed118
# RFT BL — rft_bl_square_state_50demos_seed127
# IBRL, No Actor Dropout — ibrl_noBCDrop_noDropActor_bl_square_state_50demos_seed100
# IBRL, with Actor Drop — ibrl_noBCDrop_bl_square_state_50demos_seed130

# #TODO
# IBRL_NO_ACTOR_DROP_RUN_PATH = os.path.join('/iris/u/aml2023/rl_refinement/ibrl/exps/rl_sweep5', 'rlpd_learned_cov_hidSize128_square_state_50demos_seed100')
# agent_ibrl_no_actor_dropout, _, _ = load_model(os.path.join(RLPD_LC_RUN_PATH, 'model_step50000.pt'), "cuda", load_dataset=True)

# state_dict = torch.load(os.path.join(RUN_PATH, 'expl_module_ckpts', "cov_mlp_model_step50000.pt"))


# # action distributions
# action = agent.act(obs, eval_mode=eval_mode)

# #Inference with learned covariance model

# agent.actor.explore_module.cov_mlp.eval()
# with torch.no_grad():
#     dist = agent.actor.explore_module._forward(example_state_from_demos.unsqueeze(0).cuda(), inference=True)

# print(dist)

# #Cholesky matrix
# print("Chol", dist.scale_tril)

# #Cov matrix
# print("Cov", dist.covariance_matrix)

