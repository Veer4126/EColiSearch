import gym
from c_elegans_env_v2 import CEMazeEnv, TrainingStatsCallback # type: ignore
import matplotlib.pyplot as plt
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import numpy as np
import os



# --- ENVIRONMENT ---
env = CEMazeEnv()

# --- LOAD SAVED MODEL AND CONTINUE TRAINING ---
model_path = "RL_models/trial5_ppo_200k_stdhyps_05bonus20_penalty5"
model = SAC.load(model_path, env=env, verbose=1)

# Use a new callback to collect fresh training stats
callback = TrainingStatsCallback(verbose=1)

# Continue training for another 300k steps
model.learn(total_timesteps=300000, callback=callback)

# Save updated model
model.save("RL_models/trial5_ppo_500k_stdhyps_05bonus20_penalty5")



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
save_dir = "training_plots_pt2"
os.makedirs(save_dir, exist_ok=True)


#-----------------------------------------------------------------------------------------------------------
# PLOT 4 TRAINING METRICS- MEAN, STD DEV, ENTROPY, EPISODIC RETURN
plt.figure(figsize=(14, 8))

# Mean actions
plt.subplot(2, 2, 1)
plt.plot(mean_actions[:, 0], label='Mean of dX', linewidth=0.8)
plt.plot(mean_actions[:, 1], label='Mean of dY', linewidth=0.8)
plt.title("Mean Action Over Time")
plt.xlabel("Step")
plt.ylabel("Mean")
plt.grid(True)
plt.legend()

# Stddevs
plt.subplot(2, 2, 2)
plt.plot(stddevs[:, 0], label='Stddev of dX', linewidth=0.8)
plt.plot(stddevs[:, 1], label='Stddev of dY', linewidth=0.8)
plt.title("Action Stddev Over Time")
plt.xlabel("Step")
plt.ylabel("Stddev")
plt.grid(True)
plt.legend()

# Entropy
plt.subplot(2, 2, 3)
plt.plot(entropies, label='Entropy', color='green', linewidth=0.8) # Policy entropy measures how random the policy is (exploration)
plt.title("Policy Entropy Over Time")
plt.xlabel("Step")
plt.ylabel("Entropy")
plt.grid(True)
plt.legend()


# Episodic return (with smoothing)
plt.subplot(2, 2, 4)
plt.plot(episode_rewards, label='Raw Episode Return', color='purple', alpha=0.4, linewidth=0.7)

# Rolling average
window_size = 100  # You can tune this
if len(episode_rewards) >= window_size:
    rolling_avg = np.convolve(
        episode_rewards, np.ones(window_size) / window_size, mode='valid'
    )
    plt.plot(range(window_size - 1, len(episode_rewards)), rolling_avg,
             label=f'{window_size}-Episode Rolling Avg', color='red', linewidth=1.5)

plt.title("Episodic Return Over Training")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "mean_std_entropy_rewards.png"))
plt.close()


#-----------------------------------------------------------------------------------------------------------
# PLOT TARGET HIT RATE PER EPISODE (BINARY) AND NUMBER OF TARGETS HIT PER EPISODE
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
# PLOT 2 MORE TRAINING METRICS BASED ON COSINE SIMILARITY TO PROVE AGENTIC LEARNING
# Directional accuracy plot with the regular cosine similarity and averaged over buckets
cosine_sim = np.array(callback.episode_directional_accuracies)

# Create figure with two subplots
plt.figure(figsize=(14, 5))

# --- Plot 1: Raw directional accuracy per episode
plt.subplot(1, 2, 1)
plt.plot(cosine_sim, label="Episode Cosine Similarity", color="green", alpha=0.7)
plt.title("Directional Accuracy (Cosine Similarity) per Episode")
plt.xlabel("Episode")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.legend()

# --- Plot 2: Smoothed bucketed cosine similarity (avg over 100-episode windows)
# Params
bucket_size = 100
max_steps = 400  # Assumes each episode is max 400 steps

# Convert to array of arrays (pad with np.nan for shorter episodes)
episode_traces = []
for ep_trace in callback.all_episode_directional_traces:
    if len(ep_trace) < max_steps:
        padded = np.pad(ep_trace, (0, max_steps - len(ep_trace)), constant_values=np.nan)
    else:
        padded = np.array(ep_trace[:max_steps])
    episode_traces.append(padded)

episode_traces = np.vstack(episode_traces)  # shape = (num_episodes, max_steps)

# Compute bucket averages (across episodes, for each step)
num_episodes = episode_traces.shape[0]
num_buckets = num_episodes // bucket_size

plt.subplot(1, 2, 2)

for i in range(num_buckets):
    start = i * bucket_size
    end = (i + 1) * bucket_size
    bucket = episode_traces[start:end]  # shape = (bucket_size, max_steps)

    # Average across episodes (axis=0), ignore nans
    bucket_mean = np.nanmean(bucket, axis=0)

    plt.plot(bucket_mean, label=f"Episodes {start + 1}-{end}")

plt.title("Step-wise Average Cosine Similarity across Episode Buckets")
plt.xlabel("Step")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "directional_accuracies.png"))
plt.close()
#-----------------------------------------------------------------------------------------------------------


# EVALUATION-----------
# Reset and evaluate
obs, _ = env.reset()

# Clear step lengths before evaluation
env.step_lengths = []

# SAC Evaluation code
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
#     env.render(mean_action=mean)

#     if truncated:
#         break

#     time.sleep(0.05)


# PPO Evaluation code
for _ in range(400):
    # Use deterministic=True for evaluation
    action, _ = model.predict(obs, deterministic=True)

    # Convert observation to tensor and add batch dimension
    obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(model.device)

    try:
        # Get action distribution from PPO policy
        dist = model.policy.get_distribution(obs_tensor)

        mean = dist.distribution.mean.detach().cpu().numpy()[0]
        std = dist.distribution.stddev.detach().cpu().numpy()[0]
    except Exception as e:
        print(f"Error getting action stats during eval: {e}")
        mean = np.zeros_like(action)
        std = np.zeros_like(action)

    # Take step in the environment
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render(mean_action=mean)

    if terminated or truncated:
        break

    time.sleep(0.05)

# Plot step lengths from the raw env
env.plot_step_length_distribution()
env.close()
plt.ioff()
plt.show()
