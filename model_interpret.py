import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import torch as th
import os
from stable_baselines3 import SAC
from c_elegans_env_v4 import CEMazeEnv, TrainingStatsCallback # type: ignore
import pickle
from stable_baselines3.common.utils import get_device
import glob
from scipy.stats import ttest_ind
import pickle

# Load the model
model_path = "RL_models/trial11_sac_500k_stdhyps_no_bonus_penalty"
model = SAC.load(model_path)


#-----------------------------------------------------------------------------------------------------------
# POLICY LANDSCAPING
# Generate a grid over 2 of the 3 input features
# We'll fix prev_p_reorient = 0.5 and vary prev_conc and curr_conc
prev_p_reorient_fixed = 0.5
prev_p_walk_fixed = 0.5

# Ranges for prev_conc and curr_conc
prev_conc_vals = np.linspace(0.0, 1.5, 50)
curr_conc_vals = np.linspace(0.0, 1.5, 50)

# Create meshgrid
P, C = np.meshgrid(prev_conc_vals, curr_conc_vals)
A1 = np.zeros_like(P)  # action[0] = k_w2r
A2 = np.zeros_like(P)  # action[1] = k_r2w

def rescale_action(raw_action, low, high):
    # raw_action is in [-1, 1] from tanh
    return low + 0.5 * (raw_action + 1.0) * (high - low)

# Loop over grid and compute mean action from actor
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        obs = np.array([
            P[i, j],       # prev_conc
            C[i, j],       # curr_conc
            prev_p_reorient_fixed,  # fixed
            prev_p_walk_fixed
        ], dtype=np.float32)

        # Build obs_tensor correctly
        obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0).to(model.device)

        # Get action distribution from policy
        obs = np.array([P[i, j], C[i, j], prev_p_reorient_fixed, 0.5], dtype=np.float32)  # Include 4th obs feature
        obs_tensor = th.tensor(obs, dtype=th.float32).unsqueeze(0).to(model.device)

        with th.no_grad():
            latent_pi = model.policy.actor.latent_pi(obs_tensor)
            mu = model.policy.actor.mu(latent_pi)
            squashed = th.tanh(mu).squeeze()

            # Rescale
            low = th.tensor(model.action_space.low, device=mu.device)
            high = th.tensor(model.action_space.high, device=mu.device)
            rescaled = rescale_action(squashed, low, high).cpu().numpy()

        A1[i, j] = rescaled[0]
        A2[i, j] = rescaled[1]


# Plot heatmaps for both outputs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cs1 = axes[0].contourf(P, C, A1, cmap='viridis')
axes[0].set_title("Policy Output: k_w2r (action[0])")
axes[0].set_xlabel("prev_conc")
axes[0].set_ylabel("curr_conc")
fig.colorbar(cs1, ax=axes[0])

cs2 = axes[1].contourf(P, C, A2, cmap='plasma')
axes[1].set_title("Policy Output: k_r2w (action[1])")
axes[1].set_xlabel("prev_conc")
axes[1].set_ylabel("curr_conc")
fig.colorbar(cs2, ax=axes[1])

plt.tight_layout()
plt.savefig("policy_landscape.png", dpi=300)
plt.show(block=False)


#-----------------------------------------------------------------------------------------------------------
# SALIENCY MAP
# Sample input state
# Format: [prev_conc, curr_conc, prev_p_reorient, prev_p_walk]
example_state = np.array([0.2, 0.6, 0.4, 0.5], dtype=np.float32)
device = model.device

# Create leaf tensor directly on the correct device
state_tensor = th.tensor(example_state, dtype=th.float32, requires_grad=True, device=device)
state_tensor_unsq = state_tensor.unsqueeze(0)  # Do ops after

# Forward pass through actor
latent_pi = model.policy.actor.latent_pi(state_tensor_unsq)
mu = model.policy.actor.mu(latent_pi)

# Let's say we want saliency for action[0] (k_w2r)

# Zero out any old gradients
if state_tensor.grad is not None:
    state_tensor.grad.zero_()

mu[0][0].backward(retain_graph = True)  # Backprop through first action component

saliency1 = state_tensor.grad.detach().cpu().numpy()
feature_names = ["prev_conc", "curr_conc", "prev_p_reorient", "prev_p_walk"]

# Print feature saliency
print("Saliency for action[0] (k_w2r):")
for name, val in zip(feature_names, saliency1):
    print(f"  {name}: {val:.4f}")


# Now suppose we want saliency for action[1] (k_r2w)

# Zero gradients again
state_tensor.grad.zero_()

mu[0][1].backward()  # Backprop through second action component
saliency2 = state_tensor.grad.detach().cpu().numpy()

# Print feature saliency
print("Saliency for action[1] (k_r2w):")
for name, val in zip(feature_names, saliency2):
    print(f"  {name}: {val:.4f}")


#-----------------------------------------------------------------------------------------------------------
# REPLAY BUFFER MINING

# Load replay buffer (from 'data/replay_buffer.pkl')
with open("RL_models/trial10_replay_buffer_pt2.pkl", "rb") as f:
    buffer = pickle.load(f)

# Gather all obs + actions
observations = buffer.observations[:buffer.size()]
actions = buffer.actions[:buffer.size()]
n_samples = observations.shape[0]

# Group by concentration trend: pos vs neg gradient
pos_grad_actions = []
neg_grad_actions = []
# print(type(observations[i]), observations[i])
# print(type(actions[i]), actions[i])


for i in range(n_samples):
    obs = observations[i]
    if obs.ndim > 1:
        obs = obs[0]  # flatten batch dim if needed

    prev_conc, curr_conc, *_ = obs
    delta = curr_conc - prev_conc

    action = actions[i]
    if action.ndim > 1:
        action = action[0]  # flatten batch dim if needed

    if delta > 0:
        pos_grad_actions.append(action)
    elif delta < 0:
        neg_grad_actions.append(action)

pos_grad_actions = np.array(pos_grad_actions)
neg_grad_actions = np.array(neg_grad_actions)

# Compare means
print("Average action when conc increasing (delta > 0):")
print("  k_w2r:", np.mean(pos_grad_actions[:, 0]))
print("  k_r2w:", np.mean(pos_grad_actions[:, 1]))

print("\nAverage action when conc decreasing (delta < 0):")
print("  k_w2r:", np.mean(neg_grad_actions[:, 0]))
print("  k_r2w:", np.mean(neg_grad_actions[:, 1]))


#-----------------------------------------------------------------------------------------------------------
# SIDE-BY-SIDE TRAJECTORY COMPARISON

# Load models
model1 = SAC.load("/content/trial14_atmp1_sac_100k_stdhyps_no_bonus_penalty.zip")
model2 = SAC.load("/content/trial17_atmp1_sac_100k_stdhyps_no_bonus_penalty.zip")

# Setup envs with same seed for fair comparison
seed = 42
env1 = CEMazeEnv()
env2 = CEMazeEnv()

obs1, _ = env1.reset(seed=seed)
obs2, _ = env2.reset(seed=seed)
target = env1.targets[0]  # same for both

traj1 = [env1.agent_pos.copy()]
traj2 = [env2.agent_pos.copy()]
dirs1, dirs2 = [], []

# Rollout
for _ in range(400):
    a1, _ = model1.predict(obs1, deterministic=True)
    a2, _ = model2.predict(obs2, deterministic=True)

    obs1, _, done1, trunc1, _ = env1.step(a1)
    obs2, _, done2, trunc2, _ = env2.step(a2)

    traj1.append(env1.agent_pos.copy())
    traj2.append(env2.agent_pos.copy())

    dirs1.append(env1.theta)
    dirs2.append(env2.theta)

    if done1 or trunc1 or done2 or trunc2:
        break

traj1 = np.array(traj1)
traj2 = np.array(traj2)

# Plot
plt.figure(figsize=(7, 7))
ax = plt.gca()
ax.set_xlim(0, env1.size)
ax.set_ylim(0, env1.size)

# Plot gradient field
x = np.linspace(0, env1.size, 100)
y = np.linspace(0, env1.size, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for tgt in env1.targets:
    Z += np.exp(-((X - tgt[0])**2 + (Y - tgt[1])**2) / (2 * env1.gradient_sigma**2))

plt.imshow(Z, extent=[0, env1.size, 0, env1.size], origin='lower', cmap='Greens', alpha=0.4)

# Draw target
plt.scatter(*target, c='red', marker='*', s=200, label='Target Center')

# Plot trajectories
plt.plot(traj1[:, 0], traj1[:, 1], label='Reward = conc', color='blue', linewidth=2)
plt.plot(traj2[:, 0], traj2[:, 1], label='Reward = conc - prev_conc', color='orange', linewidth=2)

# Add direction arrows every N steps
arrow_stride = 10
arrow_len = 0.3

for i in range(0, len(traj1) - 1, arrow_stride):
    x, y = traj1[i]
    dx = arrow_len * np.cos(dirs1[i])
    dy = arrow_len * np.sin(dirs1[i])
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.05, color='blue'))

for i in range(0, len(traj2) - 1, arrow_stride):
    x, y = traj2[i]
    dx = arrow_len * np.cos(dirs2[i])
    dy = arrow_len * np.sin(dirs2[i])
    ax.add_patch(FancyArrow(x, y, dx, dy, width=0.05, color='orange'))

plt.title("Side-by-Side Agent Trajectories with Direction Arrows")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig("side_by_side_directional_rollout.png", dpi=300)
plt.show()

#-----------------------------------------------------------------------------------------------------------
# STATISTICAL ANALYSIS

# Path to your pickle file
file_path = '/content/dir_for_reward_conc/metrics_trial17_atmp1_20250714_235947.pkl'

# Load the pickle file
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Check the type of data
print(f"Type of data: {type(data)}")

# If it's a tensor, print its shape and content
if isinstance(data, th.Tensor):
    print(f"Tensor shape: {data.shape}")
    print(data)

# If it's a dict (common for model checkpoints), print keys
elif isinstance(data, dict):
    print("Dictionary keys:")
    for key in data.keys():
        print(f"- {key}")

    # Optionally inspect one key
    sample_key = list(data.keys())[0]
    print(f"\nData under '{sample_key}':\n{data[sample_key]}")
    print(len(data[sample_key]))

# Handle lists or other types
else:
    print(data)



def load_final_metric_from_dict(folder_path, key="episode_rewards", last_n=50):
    metric_averages = []

    pkl_files = glob.glob(os.path.join(folder_path, "metrics_trial*_atmp*.pkl"))
    print(f"Found {len(pkl_files)} pickle files in {folder_path}")

    for file in pkl_files:
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)

            if key not in data:
                print(f"Skipping {file}: key '{key}' not found.")
                continue

            values = np.array(data[key], dtype=np.float32)
            if len(values) < last_n:
                print(f"Skipping {file}: only {len(values)} entries for '{key}'")
                continue

            avg_final = np.mean(values[-last_n:])
            metric_averages.append(avg_final)

        except Exception as e:
            print(f"Error loading {file}: {e}")

    return np.array(metric_averages)



# --- Define your folders ---
folder_conc = "/content/dir_for_reward_conc"
folder_delta = "/content/dir_for_reward_conc_minus_prev"

# --- Choose the metric you want to compare ---
metric_key = "episode_rewards"
# --- Load and compare ---
conc_vals = load_final_metric_from_dict(folder_conc, key=metric_key, last_n=50)
delta_vals = load_final_metric_from_dict(folder_delta, key=metric_key, last_n=50)

print("\n--- Final 50-Episode Averages ---")
print(f"Reward = conc: {conc_vals.mean():.3f} ± {conc_vals.std():.3f}")
print(f"Reward = conc - prev_conc: {delta_vals.mean():.3f} ± {delta_vals.std():.3f}")



# Welch's t-test
t_stat, p_val = ttest_ind(conc_vals, delta_vals, equal_var=False)
print(f"\nWelch's t-test for '{metric_key}': t = {t_stat:.4f}, p = {p_val:.4f}")
if p_val < 0.05:
    print("→ Statistically significant difference ✅")
else:
    print("→ No significant difference ❌")


# Boxplot
plt.figure(figsize=(8, 5))
plt.boxplot([conc_vals, delta_vals], labels=["Reward = conc", "Reward = Δconc"])
plt.ylabel(f"Avg {metric_key} (last 50 episodes)")
plt.title(f"Comparison of {metric_key}")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{metric_key}_comparison_boxplot.png", dpi=300)
plt.show()

#-----------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------