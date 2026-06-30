from c_elegans_env_v6 import CEMazeEnv, TrainingStatsCallback # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
import numpy as np
import os


# Set up the environment
env = CEMazeEnv()
obs, _ = env.reset()

# Load Trained Agent
model = SAC.load("RL_models/atmp3_sac_500k_stdhyps_05bonus20_penalty", env=env)
os.makedirs("Evaluation_Plots", exist_ok=True)

# Define main evaluation function
def evaluate_agent(model_path, num_episodes=10, render=False):
    env = CEMazeEnv()
    model = SAC.load(model_path, env=env)

    all_positions = []
    all_delta_concs = []
    all_rewards = []
    gradient_following_ratios = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        episode_positions = []
        episode_delta_concs = []
        episode_reward = 0
        steps_climbing = 0
        total_steps = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            # Tracking data
            pos = env.agent_pos  # or whatever your env exposes
            delta_conc = info.get("delta_conc", 0)  # adjust if needed

            episode_positions.append(pos)
            episode_delta_concs.append(delta_conc)
            episode_reward += reward

            if delta_conc > 0:
                steps_climbing += 1
            total_steps += 1

            if render:
                env.render()

        all_positions.extend(episode_positions)
        all_delta_concs.extend(episode_delta_concs)
        all_rewards.append(episode_reward)

        if total_steps > 0:
            ratio = steps_climbing / total_steps
        else:
            ratio = 0
        gradient_following_ratios.append(ratio)

    env.close()
    
    return {
        "positions": np.array(all_positions),
        "delta_concs": np.array(all_delta_concs),
        "rewards": np.array(all_rewards),
        "gradient_following_ratios": np.array(gradient_following_ratios)
    }


# Position Heatmap
def plot_position_heatmap(positions, title="Agent Position Heatmap"):
    x = positions[:, 0]
    y = positions[:, 1]
    plt.figure(figsize=(6, 5))
    sns.kdeplot(x=x, y=y, fill=True, cmap="viridis")
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("Evaluation_Plots/position_heatmap.png", dpi=300)
    plt.close()


# Concentration Histogram
def plot_delta_conc_hist(delta_concs, title="Delta Conc Distribution"):
    plt.figure(figsize=(6, 4))
    plt.hist(delta_concs, bins=50, color='orange', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel("Delta Conc")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("Evaluation_Plots/concentration_histogram.png", dpi=300)
    plt.close()

# Gradient Following Ratio
def print_gradient_following_stats(ratios):
    mean_ratio = np.mean(ratios)
    print(f"Gradient-following ratio (avg over episodes): {mean_ratio:.3f}")


# Compare Agents
trained_results1 = evaluate_agent("RL_models/trial", num_episodes=20)
trained_results2 = evaluate_agent("RL_models/untrained", num_episodes=20)

# Compare plots
plot_position_heatmap(trained_results1["positions"], title="Trained Agent 1")
plot_position_heatmap(trained_results2["positions"], title="Trained Agent 2")

plot_delta_conc_hist(trained_results1["delta_concs"], title="Trained Delta Conc 1")
plot_delta_conc_hist(trained_results2["delta_concs"], title="Trained Delta Conc 2")

print("Trained Agent:")
print_gradient_following_stats(trained_results1["gradient_following_ratios"])

print("Untrained Agent:")
print_gradient_following_stats(trained_results2["gradient_following_ratios"])





# from c_elegans_env_v4 import CEMazeEnv, TrainingStatsCallback
# from stable_baselines3 import SAC
# import time
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# # EVALUATION-----------
# # Reset and evaluate
# env = CEMazeEnv()
# model = SAC.load("RL_models/trial", env=env)
# obs, _ = env.reset()

# # Clear step lengths before evaluation
# env.step_lengths = []
# frames = []

# for _ in range(40):
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

#     # Render as image and store frame
#     env.render()
#     env.fig.canvas.draw()
#     canvas = FigureCanvas(env.fig)
#     canvas.draw()

#     frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
#     frame = frame.reshape(canvas.get_width_height()[::-1] + (4,))
#     frame = frame[:, :, :3]  # Drop alpha if you want RGB only
#     frames.append(frame)

#     if truncated:
#         break

#     time.sleep(0.05)

# # Plot step lengths from the raw env
# env.plot_step_length_distribution()
# env.close()

# # Save animation
# import imageio
# imageio.mimsave('/content/eval_run.gif', frames, fps=10)
# from IPython.display import Image
# Image(open('/content/eval_run.gif','rb').read())
