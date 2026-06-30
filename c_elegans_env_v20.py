# Single State Agent, Multiple sparse rewards, no gradient, state vector WITH memory [prev_p_w2w, time_since_hit]

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
from matplotlib.patches import FancyArrow
import os
import pickle


def segment_circle_first_intersection(p1, p2, center, radius):
    """Returns distance along segment to first intersection, or None if no hit"""
    d = p2 - p1         # direction vector from 
    f = p1 - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    t = (-b - np.sqrt(discriminant)) / (2*a)
    if 0 <= t <= 1:
        return t        # fraction along segment, multiply by segment length for distance
    return None


class CEMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=200, max_steps=2000, step_size=1.0, hit_radius=0.25):
        super().__init__()

        # Set environment parameters
        self.size = size
        self.max_steps = max_steps
        self.hit_radius = hit_radius 
        self.n_targets = 100

        # Set agent parameters
        self.steps = 0
        self.motion_state = "walk"  # or "reorient"
        self.reoriented = False
        self.step_size = step_size
        self.theta = np.random.uniform(-np.pi, np.pi)        

        # Init observation vector and action vector
        self.p_w2w = 0.5
        self.prev_p_w2w = 0.5
        self.time_since_hit = 0  # counts steps since last target hit        

        # Init agent position        
        self.agent_pos = np.zeros(2) # wrapped position
        self.true_agent_pos = np.zeros(2) # unwrapped position
        
        # Init rewards
        self.curr_reward = 0.0
        self.total_raw_reward = 0.0       

        # Init for dwell times
        self.peak_times = []            # List of 1s/0s if agent is at peak per step
        self.dwell_distance_thresh = 1.0

        # Observation: [prev_p_w2w, (time_since_hit / max_steps)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0]),       # min allowed value
            high=np.array([1.0, 1.0]),      # max allowed value
            dtype=np.float32
        )

        # Action: [p_w2w]
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),    # min allowed value
            high=np.array([1.0], dtype=np.float32),   # max allowed value
        dtype=np.float32)        

        self.fig = None # 
    
    
    def _get_observation(self):
        return np.array([
            self.prev_p_w2w,
            (self.time_since_hit / self.max_steps) # normalize to [0, 1]
        ], dtype=np.float32)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agent state and orientation
        self.steps = 0
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.motion_state = "walk"
        self.reoriented = False

        # Reset stats for the new episode
        self.target_hits = 0    
        self.has_hit = False
        self.first_hit_step = None

        # Reset rewards
        self.curr_reward = 0.0
        self.total_raw_reward = 0.0     

        # Reset observation vector for step 0 of the new episode
        self.prev_p_w2w = 0.5
        self.time_since_hit = 0         

        # Reset agent position
        self.agent_pos = np.random.uniform(0, self.size, size=(2,))
        self.true_agent_pos = self.agent_pos.copy()

        # Reset for dwell time
        self.peak_times = []

        self.targets = [
            np.random.uniform(0, self.size, size=2)
            for _ in range(self.n_targets)
        ]

        return self._get_observation(), {}


    def step(self, action=None):
        prev_true_pos = self.true_agent_pos.copy() # unwrapped position

        if action is not None:
            self.p_w2w = float(np.clip(action, 0.0, 1.0).item())
        
        # Initialize dx and dy to 0
        dx, dy = 0.0, 0.0

        # Always walking
        self.motion_state = "walk"

        # Did we tumble this step?
        reoriented = False
        if np.random.rand() > self.p_w2w:
            reoriented = True
            self.theta = np.random.uniform(-np.pi, np.pi)

        # Walk (always counted as walk)
        dx = self.step_size * np.cos(self.theta)
        dy = self.step_size * np.sin(self.theta)

        proposed_end = prev_true_pos + np.array([dx, dy])

        # Precompute all lifted targets at once
        targets_array = np.array(self.targets)  # shape (n_targets, 2)
        lifted_targets = targets_array + self.size * np.round(
            (prev_true_pos - targets_array) / self.size
        )

        # Vectorized distance check
        dists = np.linalg.norm(prev_true_pos - lifted_targets, axis=1)  # shape (n_targets,)

        # Only check targets within range
        candidate_mask = (dists > self.hit_radius) & (dists <= self.hit_radius + self.step_size) # original position of agent was outside the target and after taking the step, it is within range
        # (helps prevent counting of hits when agent's prev position is inside the target)

        candidate_indices = np.where(candidate_mask)[0]

        # Collect ALL hits along segment ordered by distance
        all_hits = []  # list of (t, lifted_target_pos)

        for i in candidate_indices:
            t = segment_circle_first_intersection(
                prev_true_pos, proposed_end, lifted_targets[i], self.hit_radius
            )
            if t is not None:
                all_hits.append((t, lifted_targets[i].copy()))

        # Sort by t so we process in order of encounter
        all_hits.sort(key=lambda x: x[0])

        reward = 0.0
        if all_hits:
            reward = 1.0 # float(len(all_hits))  # reward = number of targets hit
            self.target_hits += 1

            # Relocate to FIRST hit target
            _, first_hit_pos = all_hits[0]
            self.true_agent_pos = first_hit_pos.copy()
            self.time_since_hit = 0

            if not self.has_hit:
                self.has_hit = True
                self.first_hit_step = self.steps
        else:
            self.true_agent_pos = proposed_end
            self.time_since_hit += 1

        # Wrap position
        self.agent_pos = self.true_agent_pos % self.size

        # --- Dwell detection: reuse dists, but use FINAL position ---
        # Note: dists was computed from prev_true_pos, but agent has now moved. Recompute dists from final position for accurate dwell detection
        final_lifted_targets = targets_array + self.size * np.round(
            (self.true_agent_pos - targets_array) / self.size
        )
        final_dists = np.linalg.norm(self.true_agent_pos - final_lifted_targets, axis=1)
        at_peak = int(np.any(final_dists < self.dwell_distance_thresh))
        self.peak_times.append(at_peak)        

        self.curr_reward = reward
        self.total_raw_reward += reward

        self.reoriented = reoriented

        self.steps += 1

        obs = self._get_observation() # This is where the probability and time_since_hit are added to the state
        terminated = False
        truncated = self.steps >= self.max_steps

        self.prev_p_w2w = self.p_w2w
        

        return obs, reward, terminated, truncated, {
            "action": action,
            "motion_state": self.motion_state, # always 'walk' (only a temporary placeholder)
            "reoriented_boolean": reoriented, # use this to get the motion state histogram
            "target_hits": self.target_hits, # number of targets hit this so far in the episode
            "target_hit": len(all_hits) > 0, # has the agent hit the target on this step or not
            "reward": reward,
            "time_since_hit": self.time_since_hit,
            "first_hit_step": self.steps,
            "peak_time": self.peak_times[-1]
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
            f"Step {self.steps} | State: {self.motion_state} | p_w2w: {self.p_w2w:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
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
            f"Step {self.steps} | Reoriented: {self.reoriented} | p_w2w: {self.p_w2w:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
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

        # Memory metrics
        self.time_since_hit_history = []           # per-step tracker
        self.episode_mean_time_since_hit = []      # per-episode mean
        self.episode_max_time_since_hit = []       # per-episode max

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

        # Reset memory metric
        self.time_since_hit_history = []

    
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

        # Log the new time_since_hit observation
        time_since_hit = getattr(raw_env, "time_since_hit", None)
        if time_since_hit is not None:
            self.time_since_hit_history.append(time_since_hit)

        # Target hit
        if info.get("target_hit", False):
            self.current_target_hits += 1
            if not self.did_hit_target_this_episode:
                self.first_hit_step_this_episode = self.current_step_in_episode  # Track it temporarily
                self.did_hit_target_this_episode = True

        # Motion state
        if hasattr(raw_env, "reoriented_boolean"):
            self.motion_history.append(raw_env.reoriented)
            self.all_motion_history.append(raw_env.reoriented)
            info["all_motion_history"] = self.motion_history  # Send the full history of motion states for the episode

        # --- Distance to target ---
        targets_array = np.array(raw_env.targets)
        dists_to_all_unwrapped = np.linalg.norm(raw_env.true_agent_pos - targets_array, axis=1)
        dists_to_all_wrapped = np.linalg.norm(raw_env.agent_pos - targets_array, axis=1)

        # Unwrapped distance to target (across global space)
        unwrapped_dist = np.min(dists_to_all_unwrapped)
        self.unwrapped_distances_to_target.append(unwrapped_dist)

        # Wrapped distance (min distance accounting for wraparound)
        wrapped_dist = np.min(dists_to_all_wrapped)
        self.wrapped_distances_to_target.append(wrapped_dist)  # used for learning/analysis
        
        # --- Dwell Times ---
        peak_time = info.get("peak_time", 0)
        self.curr_dwell_time += peak_time

        # --- Store current positions for next step ---
        curr_pos = raw_env.agent_pos.copy()
        self.prev_wrapped_pos = curr_pos
        self.prev_unwrapped_pos = raw_env.true_agent_pos.copy()

        self.prev_distance = wrapped_dist

        info["reoriented_boolean"] = raw_env.reoriented


        # Count walks vs reorients
        state = info.get("reoriented_boolean")
        if state == False:
            self.motion_walk_count += 1
        elif state == True:
            self.motion_reorient_count += 1
            self.motion_walk_count += 1

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
            elif hasattr(self.model.policy, "actor"): # the action-selection neural net inside the sac
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

        # At the end of the episode
        done = self.locals.get("dones", [False])[0]
        if done:
            # Store final episode data
            self.episode_rewards.append(self.current_episode_reward)
            self.target_hits.append(self.current_target_hits)
            self.episode_end_types.append("hit" if self.did_hit_target_this_episode else "timeout")
            self.episode_peak_dwell_times.append(self.curr_dwell_time)

            # Store average distances (wrapped & unwrapped)
            if self.wrapped_distances_to_target:
                self.episode_avg_wrapped_distances.append(np.mean(self.wrapped_distances_to_target))
            else:
                self.episode_avg_wrapped_distances.append(0.0)

            if self.unwrapped_distances_to_target:
                self.episode_avg_unwrapped_distances.append(np.mean(self.unwrapped_distances_to_target))
            else:
                self.episode_avg_unwrapped_distances.append(0.0)

            # Compute walk vs reorient percentages ONCE per episode
            total_steps = self.motion_walk_count
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

            # New memory-related episode metrics
            if self.time_since_hit_history:
                self.episode_mean_time_since_hit.append(np.mean(self.time_since_hit_history))
                self.episode_max_time_since_hit.append(np.max(self.time_since_hit_history))
            else:
                self.episode_mean_time_since_hit.append(0)
                self.episode_max_time_since_hit.append(0)


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

    def _on_training_start(self) -> None:
        model_filename = f"{self.save_path}/trial{self.trial_number}_atmp{self.trial_seed}_step0_model"
        self.model.save(model_filename)
        if self.verbose > 0:
            print(f"[Checkpoint] Saved initial model to {model_filename}")        

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # Save model
            model_filename = f"{self.save_path}/trial{self.trial_number}_atmp{self.trial_seed}_step{self.num_timesteps}_model"
            self.model.save(model_filename)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved model to {model_filename}")

            # # Save training metrics
            # metrics = {
            #     "episode_rewards": self.stats_callback.episode_rewards,
            #     "episode_avg_wrapped_distances": self.stats_callback.episode_avg_wrapped_distances,
            #     "episode_avg_unwrapped_distances": self.stats_callback.episode_avg_unwrapped_distances,
            #     "episode_mean_time_since_hit": self.stats_callback.episode_mean_time_since_hit,
            #     "episode_max_time_since_hit": self.stats_callback.episode_max_time_since_hit,
            #     "target_hits": self.stats_callback.target_hits,
            #     "episode_end_types": self.stats_callback.episode_end_types,
            #     "mean_actions": self.stats_callback.mean_actions,
            #     "stddevs": self.stats_callback.stddevs,
            #     "entropies": self.stats_callback.entropies,
            #     "first_hit_steps": self.stats_callback.first_hit_steps,
            #     "all_motion_history": self.stats_callback.all_motion_history,
            #     "episode_peak_dwell_times": self.stats_callback.episode_peak_dwell_times,
            # }

            # metrics_filename = f"{self.save_path}/trial{self.trial_number}_atmp{self.trial_seed}_step{self.n_calls}_metrics.pkl"
            # with open(metrics_filename, "wb") as f:
            #     pickle.dump(metrics, f)

            # if self.verbose > 0:
            #     print(f"[Checkpoint] Saved metrics to {metrics_filename}")

        return True