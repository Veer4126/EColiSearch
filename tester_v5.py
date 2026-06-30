import gym
from c_elegans_env_v5 import CEMazeEnv, TrainingStatsCallback # type: ignore
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import numpy as np
from collections import Counter
import os
import pickle
from datetime import datetime



class EpsilonGreedyEnvWrapper(gym.Wrapper):
    def __init__(self, env, initial_epsilon=0.9, final_epsilon=0.01, decay_duration=1_550_000): # CHANGE DECAY DURATION TO NUMBER OF ITERATIONS
        super().__init__(env)
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_duration = decay_duration
        self.step_count = 0
        self.random_action_count = 0
        self.random_actions_this_episode = 0  # NEW
        self.current_step_in_episode = 0  # Initialize the step count for each episode        

    def step(self, action):
        self.step_count += 1
        self.current_step_in_episode += 1  # Increment the step number for the current episode        
        current_epsilon = self._get_current_epsilon()

        if np.random.rand() < current_epsilon:
            action = self.action_space.sample()
            self.random_action_count += 1
            self.random_actions_this_episode += 1  # NEW

        obs, reward, done, truncated, info = self.env.step(action)
        info = dict(info)  # make a copy
        info["random_action"] = self.random_actions_this_episode  # pass count via info
        return obs, reward, done, truncated, info

    def _get_current_epsilon(self):
        decay_ratio = min(self.step_count / self.decay_duration, 1.0)
        return self.initial_epsilon * (1 - decay_ratio) + self.final_epsilon * decay_ratio
    
    def reset(self, **kwargs):
        self.random_actions_this_episode = 0  # Reset random actions counter per episode
        self.current_step_in_episode = 0  # Reset step counter for the new episode
        return self.env.reset(**kwargs)


# Define seeds and trial number
trial_number = 20
seeds = [1]

# Create main output directories
os.makedirs("RL_models", exist_ok=True)
os.makedirs("training_plots", exist_ok=True)

for attempt, trial_seed in enumerate(seeds, start=1):
    print(f"\n[INFO] Starting Trial {trial_number} Attempt {trial_seed} with seed {trial_seed}...\n")

    # Set seeds
    np.random.seed(trial_seed)
    th.manual_seed(trial_seed)

    base_env = CEMazeEnv()
    base_env.seed(trial_seed)

    # Wrap with epsilon-greedy
    env = EpsilonGreedyEnvWrapper(
        base_env,
        initial_epsilon=0.9,  # Starts more random
        final_epsilon=0.01,   # Almost greedy by the end
        decay_duration=1_550_000  # Epsilon fully decayed after 150k steps # CHANGE to 350K for 300K iterations
    )


    # Define SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000, # CHANGE BUFFER TO 100K
        learning_starts=1000,
        batch_size=256,
        tau=0.01,
        gamma=0.99,
        train_freq=1,
        gradient_steps=2,
        ent_coef=0.0
    )

    callback = TrainingStatsCallback(verbose=1)

    # from stable_baselines3.common.base_class import BaseAlgorithm

    # def count_model_parameters(model: BaseAlgorithm):
    #     return sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    # # Example usage
    # print(f"Total trainable parameters: {count_model_parameters(model):,}")

    model.learn(total_timesteps=1_500_000, callback=callback) # CHANGE TIMESTEPS

    # Save model
    model_filename = f"RL_models/trial{trial_number}_atmp{trial_seed}_sac_1p5M_sparse_eg" # CHANGE NAME
    model.save(model_filename)

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


    plot_metrics()

    # Save metrics as .pkl
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"RL_models/metrics_trial{trial_number}_atmp{trial_seed}_{timestamp}.pkl"
    metrics = {
        "episode_rewards": callback.episode_rewards,
        "episode_avg_distances": callback.episode_avg_distances,
        "episode_avg_delta_distances": callback.episode_avg_delta_distances,
        "target_hits": callback.target_hits,
        "episode_end_types": callback.episode_end_types,
        "mean_actions": callback.mean_actions,
        "stddevs": callback.stddevs,
        "entropies": callback.entropies,
        "first_hit_steps": callback.first_hit_steps,
        "final_distances": callback.final_distances,
        "random_actions_per_episode": callback.random_actions_per_episode,
        "step_lengths": callback.step_lengths,
        "all_motion_history": callback.all_motion_history,
    }

    with open(metrics_filename, "wb") as f:
        pickle.dump(metrics, f)

    print(f"[INFO] Trial {trial_number} Attempt {trial_seed} complete. Saved to {metrics_filename}")

print("\n[INFO] All training attempts completed.\n")