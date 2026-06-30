import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
from matplotlib.patches import FancyArrow
import os

os.makedirs("more_training_plots", exist_ok=True)


class CEMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=6, max_steps=500, k_w2r=0.5, k_r2w=0.5, alpha=0.5):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.max_targets = 1
        self.targets = [np.array([self.size / 2, self.size / 2])] # initializing the target location to be the center of the grid
        self.steps = 0
        self.motion_state = "walk"  # or "reorient"
        self.k_w2r = k_w2r
        self.k_r2w = k_r2w
        self.alpha = alpha
        self.dt = 1.0  # fixed time step
        self.theta = 0.0
        self.agent_pos = np.zeros(2)
        self.prev_p_reorient = 0.5  # Initial value for obs
        self.prev_p_walk = 0.5
        self.total_raw_reward = 0.0

        self.action_space = gym.spaces.Box(
        low=np.array([1e-4, 1e-4], dtype=np.float32),   # min allowed values
        high=np.array([1.0, 1.0], dtype=np.float32),    # max allowed values
        dtype=np.float32)


        # [prev_p_reorient, prev_p_walk]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.fig = None

    
    def _get_observation(self):
        return np.array([
            self.prev_p_reorient,
            self.prev_p_walk
        ], dtype=np.float32)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent position
        self.steps = 0
        self.target_hits = 0
        self.agent_pos = np.random.uniform(0, self.size, size=(2,))
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.motion_state = "walk"
        self.total_raw_reward = 0.0

        # Randomize location of the target(s)
        self.targets = []
        for _ in range(self.max_targets):
            while True:
                new_target = np.random.uniform(0, self.size, size=2)
                if np.linalg.norm(new_target - self.agent_pos) > 2:  # Avoid very close starts
                    break
            self.targets.append(new_target)

        self.prev_p_reorient = 0.5
        self.prev_p_walk = 0.5
        self.curr_reward = 0.0

        return self._get_observation(), {}


    def step(self, action=None):
        if action is not None:
            self.k_w2r, self.k_r2w = action

        prev_pos = self.agent_pos.copy()

        # Transition probabilities from rate constants
        p_reorient = 1 - np.exp(-self.k_w2r * self.dt)
        p_walk = 1 - np.exp(-self.k_r2w * self.dt)

        # Decide motion state
        if self.motion_state == "walk":
            if np.random.rand() < p_reorient:
                self.motion_state = "reorient"
                self.theta += np.random.uniform(-np.pi, np.pi)
        elif self.motion_state == "reorient":
            if np.random.rand() < p_walk:
                self.motion_state = "walk"

        # Move if walking
        if self.motion_state == "walk":
            dx = self.alpha * np.cos(self.theta)
            dy = self.alpha * np.sin(self.theta)
            self.agent_pos += np.array([dx, dy])

        # Keep within bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.size)
        self.steps += 1

        # Reward: Sparse reward only if near target
        reward = 0.0
        hit = False
        if np.linalg.norm(self.agent_pos - self.targets[0]) < 1.0:
            self.target_hits += 1
            hit = True
            reward = 10.0  # reward only if target reached

        self.curr_reward = reward
        self.total_raw_reward += reward

        obs = self._get_observation()
        terminated = False
        truncated = self.steps >= self.max_steps

        self.prev_p_reorient = p_reorient
        self.prev_p_walk = p_walk

        return obs, reward, terminated, truncated, {
            "motion_state": self.motion_state,
            "target_hits": self.target_hits,
            "target_hit": hit,
            "reward": reward
        }

    
    def render(self, mode='human'):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)

        # Draw scalar field
        x = np.linspace(0, self.size, 100)
        y = np.linspace(0, self.size, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        self.ax.imshow(Z, extent=[0, self.size, 0, self.size], origin='lower', cmap='Greens', alpha=0.6) # alpha is the image transparency

        self.ax.plot(*self.targets[0], 'ro', markersize=6)

        arrow = FancyArrow(
            x=self.agent_pos[0],
            y=self.agent_pos[1],
            dx=0.3 * np.cos(self.theta),
            dy=0.3 * np.sin(self.theta),
            width=0.05,
            color='blue'
        )
        self.ax.add_patch(arrow)

        self.ax.set_title(
            f"Step {self.steps} | State: {self.motion_state} | k_w2r: {self.k_w2r:.3f}, k_r2w: {self.k_r2w:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
            )
        plt.pause(0.01)


    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None



class TrainingStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.reset_episode_stats()
    
        # Storage across episodes
        self.episode_rewards = [] # to check if episodic returns are improving with more training
        self.step_lengths = []

        # Tracking target hit rate per episode
        self.target_hits = []
        self.episode_end_types = []

        # Tracking average distance to target per episode
        self.episode_avg_distances = [] # Across episodes

        # Tracking the change in distance to the target from one step to the next per episode
        self.episode_avg_delta_distances = [] # Across episodes

        # Track motion states over training
        self.all_motion_history = []

        # Policy stats
        self.mean_actions = []
        self.stddevs = []
        self.entropies = []  #Tracking entropy of the policy over time to quantify exploration

        # Track first hit in an episode
        self.first_hit_steps = []         # Step at which first target is hit
        self.first_hit_step_this_episode = None     # Flag
        self.current_step_in_episode = 0        # Step counter per episode

        self.final_distances = []        # Final distance to target at episode end
        self.random_actions_per_episode = []        # Number of random actions per episode.
        
        # Track walks and reorients in an episode
        self.walk_vs_reorient_ratio = []        # Percentage of walks and reorients per episode
        motion_walk_count = 0
        motion_reorient_count = 0


    def reset_episode_stats(self):
        self.current_episode_reward = 0.0 # to check if episodic returns are improving with more training

        # Tracking target hits per episode
        self.current_target_hits = 0  # Counter
        self.did_hit_target_this_episode = False # remove this to allow termination

        # Tracking average distance to target per episode
        self.distances_to_target = [] # For current episode

        # Tracking the change in distance to the target from one step to the next per episode
        self.delta_distances = [] # Step-wise deltas

        # Tracking step lengths with training
        self.prev_distance = None
        self.prev_pos = None

        # Track motion states over training per episode
        self.motion_history = []

        # Track current step in each episode
        self.current_step_in_episode = 0

        # Track counts for walk and reorient motions
        self.motion_walk_count = 0
        self.motion_reorient_count = 0

        # Track when the agent first hits the target in an episode
        self.first_hit_step_this_episode = None


    # PPO/SAC Code
    def _on_step(self) -> bool:
        # Track the current step
        self.current_step_in_episode += 1

        # Get underlying raw env
        env = self.training_env.envs[0]
        raw_env = env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env

        # Track info
        info = self.locals.get("infos", [{}])[0]

        # Reward
        reward = self.locals.get("rewards", [0.0])[0]
        self.current_episode_reward += reward

        # Target hit
        if info.get("target_hit", False):
            self.current_target_hits += 1
            if not self.did_hit_target_this_episode:
                self.first_hit_step_this_episode = self.current_step_in_episode  # Track it temporarily
                self.did_hit_target_this_episode = True

        # Step length
        curr_pos = raw_env.agent_pos.copy()
        if self.prev_pos is not None:
            step_length = np.linalg.norm(curr_pos - self.prev_pos)
            self.step_lengths.append(step_length)
        self.prev_pos = curr_pos

        # Motion state
        if hasattr(raw_env, "motion_state"):
            self.motion_history.append(raw_env.motion_state)
            self.all_motion_history.append(raw_env.motion_state)
            info["all_motion_history"] = self.motion_history  # Send the full history of motion states for the episode

        # Distance to target
        target = raw_env.targets[0]
        vec_to_target = target - curr_pos
        dist_to_target = np.linalg.norm(vec_to_target)
        self.distances_to_target.append(dist_to_target)

        # Delta distance
        if self.prev_distance is not None:
            delta = self.prev_distance - dist_to_target
            self.delta_distances.append(delta)
        self.prev_distance = dist_to_target

        info["motion_state"] = raw_env.motion_state

        # Count walks vs reorients
        state = info.get("motion_state")
        if state == "walk":
            self.motion_walk_count += 1
        elif state == "reorient":
            self.motion_reorient_count += 1

        # Policy stats
        try:
            obs_tensor = th.as_tensor(
                self.model.rollout_buffer.observations[-1]
                if hasattr(self.model, "rollout_buffer")
                else self.locals.get("new_obs", [None])[0],
                dtype=th.float32
            ).unsqueeze(0).to(self.model.device)

            if hasattr(self.model.policy, "get_distribution"):
                dist = self.model.policy.get_distribution(obs_tensor)
                mean = dist.distribution.mean.detach().cpu().numpy()[0]
                std = dist.distribution.stddev.detach().cpu().numpy()[0]
                entropy = dist.distribution.entropy().detach().cpu().numpy().mean()
            elif hasattr(self.model.policy, "actor"):  # SAC-style
                latent_pi = self.model.policy.actor.latent_pi(obs_tensor)
                mean = self.model.policy.actor.mu(latent_pi).detach().cpu().numpy()[0]
                log_std = self.model.policy.actor.log_std(latent_pi)
                std = th.exp(log_std).detach().cpu().numpy()[0]
                entropy = np.mean(log_std.detach().cpu().numpy())
            else:
                mean, std, entropy = None, None, None

            if mean is not None:
                self.mean_actions.append(mean)
                self.stddevs.append(std)
                self.entropies.append(entropy)

        except Exception as e:
            if self.verbose:
                print(f"Skipping policy stat logging: {e}")

        # End of episode
        done = self.locals.get("dones", [False])[0]
        if done:
            # Store final episode data
            self.episode_rewards.append(self.current_episode_reward)
            self.target_hits.append(self.current_target_hits)
            self.episode_end_types.append("hit" if self.did_hit_target_this_episode else "timeout")

            # Store step-wise episode data
            if self.distances_to_target:
                self.episode_avg_distances.append(np.mean(self.distances_to_target))
            else:
                self.episode_avg_distances.append(0)
            
            if self.delta_distances:
                self.episode_avg_delta_distances.append(np.mean(self.delta_distances))
            else:
                self.episode_avg_delta_distances.append(0)
            
            # Final distance to target
            if self.distances_to_target:
                self.final_distances.append(self.distances_to_target[-1])
            else:
                self.final_distances.append(0)

            # Random actions (if applicable)
            if "random_action" in info:
                self.random_actions_per_episode.append(info["random_action"])
            else:
                self.random_actions_per_episode.append(0)
            
            # Compute walk vs reorient percentages ONCE per episode
            total_steps = self.motion_walk_count + self.motion_reorient_count
            if total_steps > 0:
                percent_walk = (self.motion_walk_count / total_steps) * 100
                percent_reorient = (self.motion_reorient_count / total_steps) * 100
            else:
                percent_walk = percent_reorient = 0.0
            self.walk_vs_reorient_ratio.append((percent_walk, percent_reorient))

            # First Target Hit Step
            if self.first_hit_step_this_episode is not None:
                self.first_hit_steps.append(self.first_hit_step_this_episode)
            else:
                self.first_hit_steps.append(0)  # Or np.nan if you want to exclude it from averages

            # Reset flags (after episode)
            self.did_hit_target_this_episode = False

            # Reset episode-specific stats
            self.reset_episode_stats()

        return True
    

    def _on_training_end(self):
        if self.step_lengths:
            plt.figure(figsize=(6, 4))
            min_val, max_val = min(self.step_lengths), max(self.step_lengths)
            bins = np.linspace(min_val, max_val, 50)
            plt.hist(self.step_lengths, bins=bins, edgecolor='black', alpha=0.7)
            plt.xlabel("Step Length")
            plt.ylabel("Frequency")
            plt.title("Training Step Length Distribution")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("more_training_plots/step_length_distribution.png", dpi=300)
            plt.close()


        # === First Target Hit Step ===
        if self.first_hit_steps:
            plt.figure(figsize=(10, 4))
            plt.plot(self.first_hit_steps, color='green', alpha=0.4, label="First Hit Step")

            window = 100
            if len(self.first_hit_steps) >= window:
                rolling_first_hit = np.convolve(self.first_hit_steps, np.ones(window)/window, mode='valid')
                plt.plot(range(window - 1, len(self.first_hit_steps)), rolling_first_hit, label="100-Episode Rolling Avg", color='darkgreen')

            plt.title("First Target Hit Step per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Step Number")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("more_training_plots/first_target_hit_steps.png", dpi=300)
            plt.close()


        # === Final Distance to Target ===
        if self.final_distances:
            plt.figure(figsize=(10, 4))
            plt.plot(self.final_distances, color='orange', alpha=0.4, label="Final Distance")

            window = 100
            if len(self.final_distances) >= window:
                rolling_dist = np.convolve(self.final_distances, np.ones(window)/window, mode='valid')
                plt.plot(range(window - 1, len(self.final_distances)), rolling_dist, label=f"{window}-Episode Rolling Avg", color='red')

            plt.title("Final Distance to Target per Episode")
            plt.xlabel("Episode")
            plt.ylabel("Distance")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("more_training_plots/final_dist_to_target.png", dpi=300)
            plt.close()


        # === Random Actions Per Episode ===
        if self.random_actions_per_episode:
            plt.figure(figsize=(10, 4))
            plt.plot(self.random_actions_per_episode, label="Random Actions", alpha=0.4, color='gray')

            window = 100
            if len(self.random_actions_per_episode) >= window:
                rolling_random = np.convolve(self.random_actions_per_episode, np.ones(window)/window, mode='valid')
                plt.plot(range(window - 1, len(self.random_actions_per_episode)), rolling_random, label="100-Episode Rolling Avg", color='black')
            
            plt.title("Random Actions Per Episode (Epsilon-Greedy)")
            plt.xlabel("Episode")
            plt.ylabel("Count")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("more_training_plots/random_actions_per_episode.png", dpi=300)
            plt.close()


        # === Plot standard deviation over time ===
        if self.stddevs:
            plt.figure(figsize=(6, 4))
            # Plot the stddev for each rate constant separately (since stddevs is 2D)
            plt.plot([std[0] for std in self.stddevs], label="Stddev k_w2r", linewidth=0.8)
            plt.plot([std[1] for std in self.stddevs], label="Stddev k_r2w", linewidth=0.8)
            
            window = 100
            if len(self.stddevs) >= window:
                rolling_std_k_w2r = np.convolve([std[0] for std in self.stddevs], np.ones(window)/window, mode='valid')
                rolling_std_k_r2w = np.convolve([std[1] for std in self.stddevs], np.ones(window)/window, mode='valid')
                plt.plot(range(window - 1, len(self.stddevs)), rolling_std_k_w2r, label=f"{window}-Episode Rolling Avg (k_w2r)", color='red', linestyle='--')
                plt.plot(range(window - 1, len(self.stddevs)), rolling_std_k_r2w, label=f"{window}-Episode Rolling Avg (k_r2w)", color='blue', linestyle='--')

            plt.title("Standard Deviation of Rate Constants Over Time")
            plt.xlabel("Episode")
            plt.ylabel("Standard Deviation")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("more_training_plots/stddev_rate_constants_over_time.png", dpi=300)
            plt.close()