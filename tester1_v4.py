import gym
from c_elegans_env_v4 import CEMazeEnv, TrainingStatsCallback # type: ignore
import matplotlib.pyplot as plt
import time
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import numpy as np
from collections import Counter
import os
import pickle
from datetime import datetime

# SET THE RANDOM SEED
trial_seed = 1
np.random.seed(trial_seed)
th.manual_seed(trial_seed)
env = CEMazeEnv()
env.seed(trial_seed)

# --- ENVIRONMENT ---
env = CEMazeEnv()

# --- LOAD SAVED MODEL AND CONTINUE TRAINING ---
model_path = "RL_models/trial11_atmp2_sac_200k_stdhyps_no_bonus_penalty"
model = SAC.load(model_path, env=env, verbose=1)

# Use a new callback to collect fresh training stats
callback = TrainingStatsCallback(verbose=1)

# Continue training for another 300k steps
model.learn(total_timesteps=300000, callback=callback)

# Save updated model
model.save("RL_models/trial11_atmp2_sac_500k_stdhyps_no_bonus_penalty")

with open("RL_models/trial11_atmp2_replay_buffer_pt2.pkl", "wb") as f:
    pickle.dump(model.replay_buffer, f)
    

# PLOTTING TRAINING STATISTICS
# Convert to arrays (safely)
if len(callback.mean_actions) > 0:
    mean_actions = np.vstack(callback.mean_actions)
    stddevs = np.vstack(callback.stddevs)
    entropies = np.array(callback.entropies)
else:
    print("No mean_actions collected. Skipping action-related plots.")
    mean_actions = np.empty((0, 2))
    stddevs = np.empty((0, 2))
    entropies = np.empty((0,))

episode_rewards = np.array(callback.episode_rewards)

print(mean_actions.shape)  # Should be (n_steps, 3)

print("Episode rewards collected:", episode_rewards.shape)
print("Sample rewards:", episode_rewards[:5])  # Optional: peek at a few


# DIRECTORY TO SAVE TRAINING DATA
save_dir = "training_plots_v2"
os.makedirs(save_dir, exist_ok=True)


#-----------------------------------------------------------------------------------------------------------
# PLOT 4 TRAINING METRICS- MEAN, STD DEV, ENTROPY, EPISODIC RETURN
plt.figure(figsize=(14, 8))

# Mean Rate Constants
plt.subplot(2, 2, 1)
plt.plot(mean_actions[:, 0], label='Mean k_w2r', linewidth=0.8)
plt.plot(mean_actions[:, 1], label='Mean k_r2w', linewidth=0.8)
plt.title("Mean Rate Constants Over Time")
plt.xlabel("Step")
plt.ylabel("Mean Value")
plt.grid(True)
plt.legend()

# Std Dev of Rate Constants
plt.subplot(2, 2, 2)
plt.plot(stddevs[:, 0], label='Stddev k_w2r', linewidth=0.8)
plt.plot(stddevs[:, 1], label='Stddev k_r2w', linewidth=0.8)
plt.title("Stddev of Rate Constants Over Time")
plt.xlabel("Step")
plt.ylabel("Stddev")
plt.grid(True)
plt.legend()

# Entropy of Policy
plt.subplot(2, 2, 3)
plt.plot(entropies, label='Entropy', color='green', linewidth=0.8)
plt.title("Policy Entropy Over Time")
plt.xlabel("Step")
plt.ylabel("Entropy")
plt.grid(True)
plt.legend()

# Episodic Return
plt.subplot(2, 2, 4)
plt.plot(episode_rewards, label='Raw Episodic Return', alpha=0.4, linewidth=0.7, color='purple')
if len(episode_rewards) >= 100:
    rolling_avg = np.convolve(episode_rewards, np.ones(100) / 100, mode='valid')
    plt.plot(range(99, len(episode_rewards)), rolling_avg, color='red', label='100-Episode Rolling Avg', linewidth=1.5)
plt.title("Episodic Return Over Training")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "policy_outputs_and_rewards.png"))
plt.close()


#-----------------------------------------------------------------------------------------------------------
# PLOT TARGET HIT RATE PER EPISODE (BINARY) AND NUMBER OF TARGETs HIT PER EPISODE
hits = np.array([1 if e == "hit" else 0 for e in callback.episode_end_types])
target_hits = np.array(callback.target_hits)
window_size = 100

# Compute Rolling Averages
if len(hits) >= window_size:
    rolling_hit_rate = np.convolve(hits, np.ones(window_size)/window_size, mode='valid')
else:
    rolling_hit_rate = []

if len(target_hits) >= window_size:
    rolling_hits = np.convolve(target_hits, np.ones(window_size)/window_size, mode='valid')
else:
    rolling_hits = []

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: Target Hit Rate (Binary)
axs[0].plot(hits, color='green', alpha=0.3, label='Binary Hit (1=Hit, 0=Timeout)')
if len(rolling_hit_rate):
    axs[0].plot(range(window_size - 1, len(hits)), rolling_hit_rate, color='green', label=f'{window_size}-Episode Rolling Avg')
axs[0].set_title("Target Hit Rate per Episode")
axs[0].set_ylabel("Fraction of Hits")
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Target Hits per Episode
axs[1].plot(target_hits, color='blue', alpha=0.3, label='Raw Target Hits per Episode')
if len(rolling_hits):
    axs[1].plot(range(window_size - 1, len(target_hits)), rolling_hits, color='blue', label=f'{window_size}-Episode Rolling Avg')
axs[1].set_title("Number of Target Hits per Episode")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Hit Count")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "hit_rate_and_total_hits.png"))
plt.close()


#-----------------------------------------------------------------------------------------------------------
# PLOT THE MOTION STATE HISTOGRAM
motion_counts = Counter(callback.all_motion_history)
states = list(motion_counts.keys())
counts = list(motion_counts.values())

plt.figure(figsize=(6, 4))
plt.bar(states, counts, color=["blue", "orange"])
plt.title("Motion Type Frequency (Walk vs Reorient)")
plt.xlabel("Motion State")
plt.ylabel("Count")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "motion_state_histogram.png"))
plt.close()

#-----------------------------------------------------------------------------------------------------------
# PLOT THE SAMPLED CONCENTRATION TRACES ACROSS TRAINING

# Define the bucket size (e.g., 100 episodes per bucket)
bucket_size = 100
max_steps = 400  # Or the maximum number of steps per episode
window_size = 30  # Smoothing window for the concentration traces

# Calculate the total number of buckets based on the number of episodes available
num_buckets = len(callback.episode_concentrations) // bucket_size

# Initialize an array to hold the average concentration for each bucket
average_concentrations = []

# Loop over each bucket
for bucket_idx in range(num_buckets):
    # Get the episodes in the current bucket
    start_idx = bucket_idx * bucket_size
    end_idx = (bucket_idx + 1) * bucket_size
    bucket_concentrations = []

    # For each episode in the current bucket, retrieve the concentration trace
    for episode_idx in range(start_idx, end_idx):
        conc_trace = np.array(callback.episode_concentrations[episode_idx])
        
        # If the concentration trace is shorter than max_steps, pad it with NaN
        if len(conc_trace) < max_steps:
            conc_trace = np.pad(conc_trace, (0, max_steps - len(conc_trace)), constant_values=np.nan)
        
        # Append to the list of concentrations for the current bucket
        bucket_concentrations.append(conc_trace)

    # Stack the concentrations for all episodes in the bucket and calculate the average
    bucket_concentrations = np.vstack(bucket_concentrations)
    avg_conc_per_step = np.nanmean(bucket_concentrations, axis=0)  # Ignore NaNs while calculating the mean
    average_concentrations.append(avg_conc_per_step)

# Now plot the average concentrations for each bucket
average_concentrations = np.array(average_concentrations)

# Plot the smoothed concentration traces across buckets
plt.figure(figsize=(12, 6))
for bucket_idx in range(average_concentrations.shape[0]):
    smoothed = np.convolve(average_concentrations[bucket_idx], np.ones(window_size) / window_size, mode='valid')
    plt.plot(range(window_size - 1, max_steps), smoothed, label=f'Bucket {bucket_idx+1} ({bucket_idx*bucket_size + 1}-{(bucket_idx+1)*bucket_size})', linewidth=1.5)

plt.title("Smoothed Concentration Traces Across Episode Buckets")
plt.xlabel("Step")
plt.ylabel("Average Concentration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "concentration_traces.png"))
plt.close()


#-----------------------------------------------------------------------------------------------------------
# PLOT 2 TRAINING METRICS BASED ON DISTANCES TO PROVE AGENTIC LEARNING
# Plotting Avg Distance and Delta Distance to Target on the left
plt.figure(figsize=(14, 4))

# Avg distance to target
plt.subplot(1, 2, 1)
plt.plot(callback.episode_avg_distances, label="Avg Dist to Target", color='blue')
plt.title("Avg Distance to Target per Episode")
plt.xlabel("Episode")
plt.ylabel("Distance")
plt.grid(True)

# Delta distance to target
plt.subplot(1, 2, 2)
plt.plot(callback.episode_avg_delta_distances, label="Avg ΔDist to Target", color='purple')
plt.title("Avg Δ Distance to Target per Episode")
plt.xlabel("Episode")
plt.ylabel("Delta Distance")
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "distance_metrics.png"))
plt.close()

#-----------------------------------------------------------------------------------------------------------

# SAVE TRAINING METRICS
# Create trial-specific filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_filename = f"RL_models/metrics_trial11_atmp2_{timestamp}.pkl"

# Prepare dict of metrics
metrics = {
    "episode_rewards": callback.episode_rewards,
    "episode_avg_distances": callback.episode_avg_distances,
    "episode_avg_delta_distances": callback.episode_avg_delta_distances,
    "target_hits": callback.target_hits,
    "episode_end_types": callback.episode_end_types,
    "mean_actions": callback.mean_actions,
    "stddevs": callback.stddevs,
    "entropies": callback.entropies
}

# Save
with open(metrics_filename, "wb") as f:
    pickle.dump(metrics, f)

print(f"[INFO] Saved metrics to {metrics_filename}")


#-----------------------------------------------------------------------------------------------------------

# EVALUATION-----------
# Reset and evaluate
# obs, _ = env.reset()

# # SAC Evaluation code
# for _ in range(400):
#     action, _ = model.predict(obs, deterministic=True)

#     # Convert observation to tensor and add batch dimension
#     obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(model.device)

#     # Compute mean and std using SAC's actor network
#     latent_pi = model.policy.actor.latent_pi(obs_tensor)
#     mean_tensor = model.policy.actor.mu(latent_pi)
#     log_std_tensor = model.policy.actor.log_std(latent_pi)
#     std_tensor = th.exp(log_std_tensor)

#     mean = mean_tensor.detach().cpu().numpy()[0]
#     std = std_tensor.detach().cpu().numpy()[0]

#     # Take step in the environment
#     obs, reward, terminated, truncated, _ = env.step(action)
#     env.render()

#     if truncated:
#         break

#     time.sleep(0.05)

# env.close()
# plt.ioff()
# plt.show()