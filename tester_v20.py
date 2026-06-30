import gymnasium as gym
from gymnasium import spaces
from c_elegans_env_v20 import CEMazeEnv, TrainingStatsCallback, ModelAndMetricsCheckpointCallback # type: ignore
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import numpy as np
from collections import Counter
import os
import pickle
from datetime import datetime


# DEFAULT MODEL HYPERPARAMETERS FOR SAC

# self: default=<class 'inspect._empty'>
# policy: default=<class 'inspect._empty'>
# env: default=<class 'inspect._empty'>
# learning_rate: default=0.0003
# buffer_size: default=1000000
# learning_starts: default=100
# batch_size: default=256
# tau: default=0.005
# gamma: default=0.99
# train_freq: default=1
# gradient_steps: default=1
# action_noise: default=None
# replay_buffer_class: default=None
# replay_buffer_kwargs: default=None
# optimize_memory_usage: default=False
# n_steps: default=1
# ent_coef: default=auto
# target_update_interval: default=1
# target_entropy: default=auto
# use_sde: default=False
# sde_sample_freq: default=-1
# use_sde_at_warmup: default=False
# stats_window_size: default=100
# tensorboard_log: default=None
# policy_kwargs: default=None
# verbose: default=0
# seed: default=None
# device: default=auto
# _init_setup_model: default=True


# Define seeds and trial number
trial_number = 36
seeds = [17]

# Create main output directories
os.makedirs("RL_models", exist_ok=True)
os.makedirs("training_plots", exist_ok=True)

for attempt, trial_seed in enumerate(seeds, start=1):
    print(f"\n[INFO] Starting Trial {trial_number} Attempt {trial_seed} with seed {trial_seed}...\n")

    # Set seeds
    np.random.seed(trial_seed)
    th.manual_seed(trial_seed)

    # Set up environment
    env = CEMazeEnv()
    env.reset(seed=trial_seed)

    # Define SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=2,
        learning_rate=3e-4,
        buffer_size=1_000_000, # 150K for 500K iterations
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.90,
        train_freq=1,
        gradient_steps=2,
        ent_coef="auto"
    )

    callback = TrainingStatsCallback(verbose=1)

    # Create checkpoint callback to save every 300K steps
    checkpoint_callback = ModelAndMetricsCheckpointCallback(
        save_freq=30_000,
        save_path="RL_models",
        trial_number=trial_number,
        trial_seed=trial_seed,
        stats_callback=callback,  # <-- this is your TrainingStatsCallback
        verbose=1
    )

    # Combine with your existing training stats callback
    from stable_baselines3.common.callbacks import CallbackList
    combined_callback = CallbackList([callback, checkpoint_callback])

    model.learn(total_timesteps=3_000_000, callback=combined_callback)

    # Save model
    model_filename = f"RL_models/trial{trial_number}_atmp{trial_seed}_sac_3M"
    model.save(model_filename)
    print(model.actor)
    for name, layer in model.actor.named_modules():
        print(name, layer)


    with open(f"RL_models/trial{trial_number}_atmp{trial_seed}_replay_buffer.pkl", "wb") as f:
        pickle.dump(model.replay_buffer, f)

    # ==== PLOTTING AND SAVING METRICS ====
    save_dir = f"training_plots/trial{trial_number}_atmp{trial_seed}"
    os.makedirs(save_dir, exist_ok=True)

    # Collect and validate callback data
    if len(callback.mean_actions) > 0:
        mean_actions = np.vstack(callback.mean_actions)
        stddevs = np.vstack(callback.stddevs)
        entropies = np.array(callback.entropies)
    else:
        mean_actions = np.empty((0, 2))
        stddevs = np.empty((0, 2))
        entropies = np.empty((0,))

    episode_rewards = np.array(callback.episode_rewards)

    def plot_metrics():

        #-----------------------------------------------------------------------------------------------------------
        # PLOT 4 TRAINING METRICS- MEAN, STD DEV, ENTROPY, EPISODIC RETURN
        plt.figure(figsize=(14, 8))

        # Mean Rate Constants
        plt.subplot(2, 2, 1)
        plt.plot(mean_actions[:, 0], label='Mean p_w2w', linewidth=0.8)
        plt.title("Mean Probability Over Time")
        plt.xlabel("Step")
        plt.ylabel("Mean Value")
        plt.grid(True)
        plt.legend()

        # Std Dev of Rate Constants
        plt.subplot(2, 2, 2)
        plt.plot(stddevs[:, 0], label='Stddev p_w2w', linewidth=0.8)
        plt.title("Stddev of Probability Over Time")
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
        window = 100
        if len(episode_rewards) >= window:
            rolling_avg = np.convolve(episode_rewards, np.ones(100) / 100, mode='valid')
            plt.plot(range(window-1, len(episode_rewards)), rolling_avg, color='red', label=f'{window}-Episode Rolling Avg', linewidth=1.5)
        plt.title("Episodic Return Over Training")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "policy_outputs_and_rewards.png"))
        plt.close()



        #-----------------------------------------------------------------------------------------------------------
        # PLOT TARGET HIT RATE PER EPISODE (BINARY) AND NUMBER OF TARGETS HIT PER EPISODE
        hits = np.array([1 if e == "hit" else 0 for e in callback.episode_end_types])
        target_hits = np.array(callback.target_hits)
        window = 100

        # Compute Rolling Averages
        if len(hits) >= window:
            rolling_hit_rate = np.convolve(hits, np.ones(window)/window, mode='valid')
        else:
            rolling_hit_rate = []

        if len(target_hits) >= window:
            rolling_hits = np.convolve(target_hits, np.ones(window)/window, mode='valid')
        else:
            rolling_hits = []

        # Plot
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Subplot 1: Target Hit Rate (Binary)
        axs[0].plot(hits, color='green', alpha=0.3, label='Binary Hit (1=Hit, 0=Timeout)')
        if len(rolling_hit_rate):
            axs[0].plot(range(window - 1, len(hits)), rolling_hit_rate, color='green', label=f'{window}-Episode Rolling Avg')
        axs[0].set_title("Target Hit Rate per Episode")
        axs[0].set_ylabel("Fraction of Hits")
        axs[0].legend()
        axs[0].grid(True)

        # Subplot 2: Target Hits per Episode
        axs[1].plot(target_hits, color='blue', alpha=0.3, label='Raw Target Hits per Episode')
        if len(rolling_hits):
            axs[1].plot(range(window - 1, len(target_hits)), rolling_hits, color='blue', label=f'{window}-Episode Rolling Avg')
        axs[1].set_title("Number of Target Hits per Episode")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Hit Count")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "hit_rate_and_total_hits.png"))
        plt.close()


        #-----------------------------------------------------------------------------------------------------------
        # ONLY USED THIS PLOT FOR THE 2 STATE AGENT
        # # PLOT THE MOTION STATE HISTOGRAM
        # motion_counts = Counter(callback.all_motion_history)
        # states = list(motion_counts.keys())
        # counts = list(motion_counts.values())

        # plt.figure(figsize=(6, 4))
        # plt.bar(states, counts, color=["blue", "orange"])
        # plt.title("Motion Type Frequency (Walk vs Reorient)")
        # plt.xlabel("Motion State")
        # plt.ylabel("Count")
        # plt.grid(True, axis='y')
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, "motion_state_histogram.png"))
        # plt.close()

        #-----------------------------------------------------------------------------------------------------------
        # PLOT WALK VS REORIENT PER EPISODE
        if len(callback.walk_vs_reorient_ratio) > 0:
            walk_percentages, reorient_percentages = zip(*callback.walk_vs_reorient_ratio)

            plt.figure(figsize=(10, 4))
            plt.plot(walk_percentages, label="Walk %", color="blue", alpha=0.6)
            plt.plot(reorient_percentages, label="Reorient %", color="orange", alpha=0.6)

            window = 100
            if len(walk_percentages) >= window:
                walk_smooth = np.convolve(walk_percentages, np.ones(window)/window, mode='valid')
                reorient_smooth = np.convolve(reorient_percentages, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(walk_percentages)), walk_smooth, color="blue", label="Walk % (rolling avg)")
                plt.plot(range(window-1, len(reorient_percentages)), reorient_smooth, color="orange", label="Reorient % (rolling avg)")

            plt.title("Walk vs Reorient Percentages Per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Percentage")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "walk_vs_reorient_percentages.png"))
            plt.close()


        #-----------------------------------------------------------------------------------------------------------
        # PLOT 2 TRAINING METRICS BASED ON DISTANCES TO PROVE AGENTIC LEARNING (Wrapped & Unwrapped Avg Distance)
        plt.figure(figsize=(18, 10))

        # 1. Wrapped Avg Distance to Target
        plt.subplot(1, 2, 1)
        plt.plot(callback.episode_avg_wrapped_distances, label="Wrapped Avg Dist to Target", color='blue')
        plt.title("Wrapped Avg Distance to Target per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.legend()

        # 2. Unwrapped Avg Distance to Target
        plt.subplot(1, 2, 2)
        plt.plot(callback.episode_avg_unwrapped_distances, label="Unwrapped Avg Dist to Target", color='green')
        plt.title("Unwrapped Avg Distance to Target per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Distance")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "distance_metrics_separated.png"))
        plt.close()


        #-----------------------------------------------------------------------------------------------------------
        # DWELL TIMES
        plt.figure(figsize=(10, 4))
        plt.plot(callback.episode_peak_dwell_times, color='blue', alpha=0.4, label="Dwell Time at Peak")

        window = 100
        if len(callback.episode_peak_dwell_times) >= window:
            rolling_dwell = np.convolve(callback.episode_peak_dwell_times, np.ones(window)/window, mode='valid')
            plt.plot(range(window - 1, len(callback.episode_peak_dwell_times)), rolling_dwell, label=f"{window}-Episode Rolling Avg", color='navy')

        plt.title("Dwell Time at Peak Concentration per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps at Peak (1 ≥ Dist to target)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "dwell_time_at_peak.png"))
        plt.close()


        #-----------------------------------------------------------------------------------------------------------
        # STANDARD DEVIATION
        plt.figure(figsize=(6, 4))
        # Plot the stddev for each probability separately (since stddevs is 2D)
        plt.plot([std[0] for std in callback.stddevs], label="Stddev p_w2w", linewidth=0.8)
        
        window = 100
        if len(callback.stddevs) >= window:
            rolling_std_k_w2w = np.convolve([std[0] for std in callback.stddevs], np.ones(window)/window, mode='valid')
            plt.plot(range(window - 1, len(callback.stddevs)), rolling_std_k_w2w, label=f"{window}-Episode Rolling Avg (p_w2w)", color='red', linestyle='--')

        plt.title("Standard Deviation of Probabilities Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Standard Deviation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "stddev_probabilities_over_time.png"))
        plt.close()


        #-----------------------------------------------------------------------------------------------------------
        # FIRST TARGET HIT STEP
        plt.figure(figsize=(10, 4))
        plt.plot(callback.first_hit_steps, color='green', alpha=0.4, label="First Hit Step")

        window = 100
        if len(callback.first_hit_steps) >= window:
            rolling_first_hit = np.convolve(callback.first_hit_steps, np.ones(window)/window, mode='valid')
            plt.plot(range(window - 1, len(callback.first_hit_steps)), rolling_first_hit, label=f"{window}-Episode Rolling Avg", color='darkgreen')

        plt.title("First Target Hit Step per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Step Number")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "first_target_hit_steps.png"))
        plt.close()


        #-----------------------------------------------------------------------------------------------------------
        # PLOT TIME SINCE HIT METRICS (Average & Maximum)
        plt.figure(figsize=(18, 10))

        # Average Time Since Hit per Episode
        plt.subplot(1, 2, 1)
        plt.plot(callback.episode_mean_time_since_hit, label="Avg Time Since Hit", color='purple', alpha=0.7)
        window = 100
        if len(callback.episode_mean_time_since_hit) >= window:
            rolling_avg_time = np.convolve(callback.episode_mean_time_since_hit, np.ones(window)/window, mode='valid')
            plt.plot(range(window - 1, len(callback.episode_mean_time_since_hit)), rolling_avg_time,
                     label=f"{window}-Episode Rolling Avg", color='darkviolet', linestyle='--')
        plt.title("Average Time Since Last Target Hit per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)
        plt.legend()

        # Maximum Time Since Hit per Episode
        plt.subplot(1, 2, 2)
        plt.plot(callback.episode_max_time_since_hit, label="Max Time Since Hit", color='orange', alpha=0.7)
        if len(callback.episode_max_time_since_hit) >= window:
            rolling_max_time = np.convolve(callback.episode_max_time_since_hit, np.ones(window)/window, mode='valid')
            plt.plot(range(window - 1, len(callback.episode_max_time_since_hit)), rolling_max_time,
                     label=f"{window}-Episode Rolling Avg", color='darkorange', linestyle='--')
        plt.title("Maximum Time Since Last Target Hit per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "time_since_hit_metrics.png"))
        plt.close()


    plot_metrics()

    # Save metrics as .pkl
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"RL_models/metrics_trial{trial_number}_atmp{trial_seed}_{timestamp}.pkl"
    metrics = {
        "episode_rewards": callback.episode_rewards,
        "episode_avg_wrapped_distances": callback.episode_avg_wrapped_distances,
        "episode_avg_unwrapped_distances": callback.episode_avg_unwrapped_distances,
        "episode_mean_time_since_hit": callback.episode_mean_time_since_hit,
        "episode_max_time_since_hit": callback.episode_max_time_since_hit,        
        "target_hits": callback.target_hits,
        "episode_end_types": callback.episode_end_types,
        "mean_actions": callback.mean_actions,
        "stddevs": callback.stddevs,
        "entropies": callback.entropies,
        "first_hit_steps": callback.first_hit_steps,
        "all_motion_history": callback.all_motion_history,
        "episode_peak_dwell_times": callback.episode_peak_dwell_times,
    }

    with open(metrics_filename, "wb") as f:
        pickle.dump(metrics, f)

    print(f"[INFO] Trial {trial_number} Attempt {trial_seed} complete. Saved to {metrics_filename}")

print("\n[INFO] All training attempts completed.\n")