# Sparse reward, no gradient, state vector with [prev_p_w2r, prev_p_r2w], with curriculum learning

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
from matplotlib.patches import FancyArrow
import os
import pickle


class CEMazeEnv1(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=10, max_steps=2500, p_w2r=0.5, p_r2w=0.5, alpha=0.1, gradient_sigma=2.0):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.max_targets = 1
        self.targets = [np.array([self.size / 2, self.size / 2])]
        self.gradient_sigma = gradient_sigma # Slightly smaller spread for multiple sources
        self.steps = 0
        self.motion_state = "walk"  # or "reorient"
        self.p_w2r = p_w2r
        self.p_r2w = p_r2w
        self.alpha = alpha
        self.dt = 1.0  # fixed time step
        self.theta = 0.0
        self.agent_pos = np.zeros(2)
        self.true_agent_pos = np.zeros(2)
        self.prev_p_reorient = 0.5  # Initial value for obs
        self.prev_p_walk = 0.5
        self.total_raw_reward = 0.0

        # Init for speed and dwell
        self.peak_threshold = 0.95
        self.prev_conc = 0.0            # For computing delta concentration per step
        self.delta_conc = 0.0
        self.gradient_rates = []        # List of ΔC per step
        self.peak_times = []            # List of 1s/0s if agent is at peak per step


        self.action_space = gym.spaces.Box(
        low=np.array([0.0, 0.0], dtype=np.float32),   # min allowed values
        high=np.array([1.0, 1.0], dtype=np.float32),    # max allowed values
        dtype=np.float32)


        # [curr_conc, prev_p_w2r, prev_p_r2w]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0]),
            high=np.array([np.inf, 1.0, 1.0]),
            dtype=np.float32
        )

        self.fig = None


    def get_concentration(self, pos): # get the concentration at each position to inform the state of the agent and determine the next step
        # Scalar field: Gaussian centered at targets. ADD UP the scalar field values for each target
        return sum(
            np.exp(-np.linalg.norm(pos - tgt)**2 / (2 * self.gradient_sigma**2))
            for tgt in self.targets
        )
    
    
    def _get_observation(self, current_conc=None):
        if current_conc is None:
            current_conc = self.get_concentration(self.agent_pos)

        return np.array([
            current_conc,
            self.prev_p_w2r,
            self.prev_p_r2w
        ], dtype=np.float32)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent position
        self.steps = 0
        self.target_hits = 0
        self.agent_pos = np.random.uniform(0, self.size, size=(2,))
        self.true_agent_pos = self.agent_pos.copy()
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.motion_state = "walk"
        self.total_raw_reward = 0.0

        # Resets for speed and dwell
        self.prev_conc = self.get_concentration(self.agent_pos)
        self.delta_conc = 0.0
        self.gradient_rates = []
        self.peak_times = []

        # Randomize target(s)
        self.targets = [np.array([self.size / 2, self.size / 2])]

        self.prev_p_w2r = 0.5
        self.prev_p_r2w = 0.5
        self.curr_reward = 0.0

        return self._get_observation(), {}


    def step(self, action=None):
        if action is not None:
            self.p_w2r, self.p_r2w = action

        prev_pos = self.agent_pos.copy()
        
        # Initialize dx and dy to 0 in case the agent does not move
        dx, dy = 0.0, 0.0

        # Decide motion state
        if self.motion_state == "walk":
            if np.random.rand() < self.p_w2r:
                self.motion_state = "reorient"
                self.theta += np.random.uniform(-np.pi, np.pi)
        elif self.motion_state == "reorient":
            if np.random.rand() < self.p_r2w:
                self.motion_state = "walk"

        # Move if walking
        if self.motion_state == "walk":
            dx = self.alpha * np.cos(self.theta)
            dy = self.alpha * np.sin(self.theta)
            self.agent_pos += np.array([dx, dy])

        # Apply periodic boundary conditions (wrap-around effect)
        self.true_agent_pos += np.array([dx, dy])  # never wrapped
        self.agent_pos = self.true_agent_pos % self.size  # wrapped around both x and y coordinates for simulation
        
        self.steps += 1

        curr_conc = self.get_concentration(self.agent_pos) # This gives the third component of the state vector.
        delta_conc = curr_conc - self.prev_conc

        # Log gradient climbing
        self.gradient_rates.append(delta_conc)

        # Log peak dwelling
        self.peak_times.append(int(curr_conc > self.peak_threshold))

        # Reward: Sparse reward only if near target
        reward = 0.0
        hit = False
        if np.linalg.norm(self.agent_pos - self.targets[0]) < 0.5:
            self.target_hits += 1
            hit = True
            reward = 1.0  # reward only if target reached

        self.curr_reward = reward
        self.total_raw_reward += reward

        obs = self._get_observation(curr_conc) # This is where the 2 probabilities, and curr conc are added to the state vector.
        terminated = False
        truncated = self.steps >= self.max_steps

        self.prev_p_w2r = self.p_w2r
        self.prev_p_r2w = self.p_r2w
        
        # Update previous concentration for next step
        self.prev_conc = curr_conc

        return obs, reward, terminated, truncated, {
            "concentration": curr_conc,
            "delta_conc": delta_conc,
            "action": action,
            "motion_state": self.motion_state,
            "target_hits": self.target_hits,
            "target_hit": hit,
            "reward": reward,
            "wrapped_dist_to_target": np.linalg.norm(self.agent_pos - self.targets[0]),
            "unwrapped_dist_to_target": np.linalg.norm(self.true_agent_pos - self.targets[0])
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

        for tgt in self.targets:
            Z += np.exp(-((X - tgt[0])**2 + (Y - tgt[1])**2) / (2 * self.gradient_sigma**2)) # The gradient is centered at the target and the decay varies with the std dev from the center

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
            f"Step {self.steps} | State: {self.motion_state} | p_w2r: {self.p_w2r:.3f}, p_r2w: {self.p_r2w:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
            )
        plt.pause(0.01)

    
    def render_static(self, mode='human'):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)

        x = np.linspace(0, self.size, 100)
        y = np.linspace(0, self.size, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for tgt in self.targets:
            Z += np.exp(-((X - tgt[0])**2 + (Y - tgt[1])**2) / (2 * self.gradient_sigma**2))

        self.ax.imshow(Z, extent=[0, self.size, 0, self.size], origin='lower', cmap='Greens', alpha=0.6)
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
            f"Step {self.steps} | State: {self.motion_state} | p_w2r: {self.p_w2r:.3f}, p_r2w: {self.p_r2w:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
        )
        # Note: no plt.pause() here


    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None


class CEMazeEnv2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=10, max_steps=500, p_w2r=0.5, p_r2w=0.5, alpha=0.5):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.max_targets = 1
        self.targets = [np.array([self.size / 2, self.size / 2])]
        self.steps = 0
        self.motion_state = "walk"  # or "reorient"
        self.p_w2r = p_w2r
        self.p_r2w = p_r2w
        self.alpha = alpha
        self.dt = 1.0  # fixed time step
        self.theta = 0.0
        self.agent_pos = np.zeros(2)
        self.true_agent_pos = np.zeros(2)
        self.prev_p_reorient = 0.5  # Initial value for obs
        self.prev_p_walk = 0.5
        self.total_raw_reward = 0.0

        # Init for dwell times
        self.peak_times = []            # List of 1s/0s if agent is at peak per step
        self.dwell_distance_thresh = 1.0


        self.action_space = gym.spaces.Box(
        low=np.array([0.0, 0.0], dtype=np.float32),   # min allowed values
        high=np.array([1.0, 1.0], dtype=np.float32),    # max allowed values
        dtype=np.float32)


        # [prev_p_w2r, prev_p_r2w]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.fig = None
    
    
    def _get_observation(self):
        return np.array([
            self.prev_p_w2r,
            self.prev_p_r2w
        ], dtype=np.float32)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent position
        self.steps = 0
        self.target_hits = 0
        self.agent_pos = np.random.uniform(0, self.size, size=(2,))
        self.true_agent_pos = self.agent_pos.copy()
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.motion_state = "walk"
        self.total_raw_reward = 0.0

        # Reset for dwell time
        self.peak_times = []

        # Randomize target(s)
        self.targets = [np.array([self.size / 2, self.size / 2])]

        self.prev_p_w2r = 0.5
        self.prev_p_r2w = 0.5
        self.curr_reward = 0.0

        return self._get_observation(), {}


    def step(self, action=None):
        if action is not None:
            self.p_w2r, self.p_r2w = action

        prev_pos = self.agent_pos.copy()
        
        # Initialize dx and dy to 0 in case the agent does not move
        dx, dy = 0.0, 0.0

        # Decide motion state
        if self.motion_state == "walk":
            if np.random.rand() < self.p_w2r:
                self.motion_state = "reorient"
                self.theta += np.random.uniform(-np.pi, np.pi)
        elif self.motion_state == "reorient":
            if np.random.rand() < self.p_r2w:
                self.motion_state = "walk"

        # Move if walking
        if self.motion_state == "walk":
            dx = self.alpha * np.cos(self.theta)
            dy = self.alpha * np.sin(self.theta)
            self.agent_pos += np.array([dx, dy])

        # Apply periodic boundary conditions (wrap-around effect)
        self.true_agent_pos += np.array([dx, dy])  # never wrapped
        self.agent_pos = self.true_agent_pos % self.size  # wrapped around both x and y coordinates for simulation
        
        self.steps += 1

        # Log peak dwelling
        dist_to_target = np.linalg.norm(self.agent_pos - self.targets[0])
        self.peak_times.append(int(dist_to_target < self.dwell_distance_thresh))

        # Reward: Sparse reward only if near target
        reward = 0.0
        hit = False
        if np.linalg.norm(self.agent_pos - self.targets[0]) < 0.5:
            self.target_hits += 1
            hit = True
            reward = 1.0  # reward only if target reached

        self.curr_reward = reward
        self.total_raw_reward += reward

        obs = self._get_observation() # This is where the 2 probabilities, and curr conc are added to the state vector.
        terminated = False
        truncated = self.steps >= self.max_steps

        self.prev_p_w2r = self.p_w2r
        self.prev_p_r2w = self.p_r2w
        

        return obs, reward, terminated, truncated, {
            "action": action,
            "motion_state": self.motion_state,
            "target_hits": self.target_hits,
            "target_hit": hit,
            "reward": reward,
            "peak_time": self.peak_times[-1],
            "wrapped_dist_to_target": np.linalg.norm(self.agent_pos - self.targets[0]), # dist between agent's position on the (0, 0) to (10, 10) grid and the target at (5, 5)
            "unwrapped_dist_to_target": np.linalg.norm(self.true_agent_pos - self.targets[0]) # dist between agent's position and target (5, 5). Can be a measure of how many grids the agent covers
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
            f"Step {self.steps} | State: {self.motion_state} | p_w2r: {self.p_w2r:.3f}, p_r2w: {self.p_r2w:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
            )
        plt.pause(0.01)

    
    def render_static(self, mode='human'):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)

        x = np.linspace(0, self.size, 100)
        y = np.linspace(0, self.size, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        self.ax.imshow(Z, extent=[0, self.size, 0, self.size], origin='lower', cmap='Greens', alpha=0.6)
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
            f"Step {self.steps} | State: {self.motion_state} | p_w2r: {self.p_w2r:.3f}, p_r2w: {self.p_r2w:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
        )
        # Note: no plt.pause() here


    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None


class TrainingStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        # Storage across episodes
        self.episode_rewards = []

        # Tracking target hit rate per episode
        self.target_hits = []
        self.episode_end_types = []

        # Distance metrics
        self.wrapped_distances_to_target = []
        self.unwrapped_distances_to_target = []
        self.episode_avg_wrapped_distances = []
        self.episode_avg_unwrapped_distances = []

        # Track motion states over training
        self.all_motion_history = []

        # Policy stats
        self.mean_actions = []
        self.stddevs = []
        self.entropies = []

        # Track first hit in an episode
        self.first_hit_steps = []
        self.first_hit_step_this_episode = None
        self.current_step_in_episode = 0

        # Track walks and reorients in an episode
        self.walk_vs_reorient_ratio = []
        self.motion_walk_count = 0
        self.motion_reorient_count = 0

        # Track speed and dwell
        self.episode_peak_dwell_times = []

        # Now that everything is initialized, it's safe to reset episode stats
        self.reset_episode_stats()



    def reset_episode_stats(self):
        self.current_episode_reward = 0.0 # to check if episodic returns are improving with more training

        # Tracking target hits per episode
        self.current_target_hits = 0  # Counter
        self.did_hit_target_this_episode = False # remove this to allow termination

        # Tracking dwell times
        self.curr_dwell_time = 0

        # Tracking average distance to target per episode
        self.wrapped_distances_to_target = []
        self.unwrapped_distances_to_target = []
        
        self.prev_wrapped_pos = None
        self.prev_unwrapped_pos = None

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

        # Motion state
        if hasattr(raw_env, "motion_state"):
            self.motion_history.append(raw_env.motion_state)
            self.all_motion_history.append(raw_env.motion_state)
            info["all_motion_history"] = self.motion_history  # Send the full history of motion states for the episode

        # --- Distance to target ---
        target = raw_env.targets[0]

        # Unwrapped distance to target (across global space)
        unwrapped_dist = np.linalg.norm(raw_env.true_agent_pos - target)
        self.unwrapped_distances_to_target.append(unwrapped_dist)

        # Wrapped distance (min distance accounting for wraparound)
        wrapped_dist = np.linalg.norm(raw_env.agent_pos - target)
        self.wrapped_distances_to_target.append(wrapped_dist)  # used for learning/analysis
        
        # --- Dwell Times ---
        peak_time = info.get("peak_time", 0)
        self.curr_dwell_time += peak_time

        # --- Store current positions for next step ---
        curr_pos = raw_env.agent_pos.copy()
        self.prev_wrapped_pos = curr_pos
        self.prev_unwrapped_pos = raw_env.true_agent_pos.copy()

        self.prev_distance = wrapped_dist

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
            self.episode_peak_dwell_times.append(self.curr_dwell_time)

            # --- Store average distances (wrapped & unwrapped) ---
            if self.wrapped_distances_to_target:
                self.episode_avg_wrapped_distances.append(np.mean(self.wrapped_distances_to_target))
            else:
                self.episode_avg_wrapped_distances.append(0.0)

            if self.unwrapped_distances_to_target:
                self.episode_avg_unwrapped_distances.append(np.mean(self.unwrapped_distances_to_target))
            else:
                self.episode_avg_unwrapped_distances.append(0.0)

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

            # Reset flags and stats
            self.did_hit_target_this_episode = False
            self.prev_distance = None
            self.reset_episode_stats()

        return True
    

    def _on_training_end(self):
        pass



class ModelAndMetricsCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, trial_number, trial_seed, stats_callback, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.trial_number = trial_number
        self.trial_seed = trial_seed
        self.stats_callback = stats_callback

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model
            model_filename = f"{self.save_path}/trial{self.trial_number}_atmp{self.trial_seed}_step{self.n_calls}_model"
            self.model.save(model_filename)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved model to {model_filename}")

            # Save training metrics
            metrics = {
                "episode_rewards": self.stats_callback.episode_rewards,
                "episode_avg_wrapped_distances": self.stats_callback.episode_avg_wrapped_distances,
                "episode_avg_unwrapped_distances": self.stats_callback.episode_avg_unwrapped_distances,
                "target_hits": self.stats_callback.target_hits,
                "episode_end_types": self.stats_callback.episode_end_types,
                "mean_actions": self.stats_callback.mean_actions,
                "stddevs": self.stats_callback.stddevs,
                "entropies": self.stats_callback.entropies,
                "first_hit_steps": self.stats_callback.first_hit_steps,
                "all_motion_history": self.stats_callback.all_motion_history,
                "episode_peak_dwell_times": self.stats_callback.episode_peak_dwell_times,
            }

            metrics_filename = f"{self.save_path}/trial{self.trial_number}_atmp{self.trial_seed}_step{self.n_calls}_metrics.pkl"
            with open(metrics_filename, "wb") as f:
                pickle.dump(metrics, f)

            if self.verbose > 0:
                print(f"[Checkpoint] Saved metrics to {metrics_filename}")

        return True