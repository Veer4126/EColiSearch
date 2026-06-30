# Single State Power Law Agent, Sparse reward, no gradient, state vector WITHOUT memory [prev_alpha]

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


def segment_circle_hit(p0, p1, c, r): # p1, p0, and c are vectors. r is a number
    """
    Returns True if the line segment p0 -> p1 intersects
    the circle centered at c with radius r.
    """
    d = p1 - p0 # direction vector of the agent's step
    f = p0 - c

    a = np.dot(d, d)
    b = 2.0 * np.dot(f, d)
    c_ = np.dot(f, f) - r * r

    disc = b * b - 4 * a * c_
    if disc < 0:
        return False  # no intersection

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


class CEMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=20, max_steps=500, alpha=0.5, hit_radius=0.25):
        super().__init__()


        self.size = size # doesn't reset every episode
        self.max_steps = max_steps # doesn't reset every episode
        self.alpha = alpha # doesn't reset every episode
        self.hit_radius = hit_radius
        self.total_raw_reward = 0.0

        self.agent_pos = np.zeros(2) # wrapped
        self.true_agent_pos = np.zeros(2) # unwrapped

        self.theta = np.random.uniform(-np.pi, np.pi)
        self.motion_state = "walk"
        self.reoriented = False
        self.steps = 0
        self.targets = [np.array([self.size / 2, self.size / 2])] # always at the grid center, doesn't reset every episode        

        # Init for dwell times
        self.peak_times = [] # List of 1s/0s if agent is at peak per step
        self.dwell_distance_thresh = 0.5 # doesn't reset every episode

        # --- Run–tumble (explicit run-length sampling) ---
        self.prev_alpha = 2.0 # measured from original naive agent
        self.curr_alpha = 2.0 # measured from original naive agent
        self.L_max = 100
        # self.max_run_length = 40
        # self.min_run_length = 1

        # Action: [curr_alpha]
        self.action_space = gym.spaces.Box(
        low=np.array([0.1], dtype=np.float32),   # min allowed value
        high=np.array([20.0], dtype=np.float32),    # max allowed value
        dtype=np.float32)


        # Observation: [prev_alpha]
        self.observation_space = spaces.Box(
            low=np.array([0.1]),
            high=np.array([20.0]),
            dtype=np.float32
        )

        self.fig = None
    
    
    def _get_observation(self):
        return np.array([
            self.prev_lambda
        ], dtype=np.float32)

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0 # resets every episode

        self.agent_pos = np.random.uniform(0, self.size, size=(2,)) # anywhere between [0, 0] & [size, size]
        self.true_agent_pos = self.agent_pos.copy() # resets every episode     

        self.theta = np.random.uniform(-np.pi, np.pi) # also in init, resets every episode
        self.motion_state = "walk" # also in init
        self.reoriented = False # also in init
        self.total_raw_reward = 0.0

        self.prev_alpha = 2.0 # measured from original naive agent
        self.curr_alpha = 2.0 # measured from original naive agent        

        # Reset for dwell time
        self.peak_times = [] # also in init

        # Resets for hits
        self.has_hit = False
        self.first_hit_step = None # resets every episode
        self.target_hits = 0

        # Target location always the same
        self.targets = [np.array([self.size / 2, self.size / 2])]

        # # --- Pre-sample full episode run lengths ---
        # remaining = self.max_steps
        # self.run_lengths = []

        # while remaining > 0:
        #     L = np.random.exponential(self.curr_lambda)
        #     L = int(np.ceil(L))
        #     # L = np.clip(L, self.min_run_length, self.max_run_length)

        #     if L > remaining:
        #         L = remaining

        #     self.run_lengths.append(L)
        #     remaining -= L

        # self.current_run_index = 0
        # assert sum(self.run_lengths) == self.max_steps

        return self._get_observation(), {}
    


    def step(self, action=None):

        if action is not None:
            self.curr_alpha = float(action)

        self.curr_alpha = np.clip(self.curr_alpha, 0.1, 20.0)# just in case             
                
        # Sample ONE run length

        def sample_powerlaw(alpha, xmin):
            u = np.random.rand()
            return xmin * (1 - u) ** (-1.0 / (alpha - 1.0))
        
        while True:
            L = sample_powerlaw(alpha=2, xmin=0.1) # REJECTION SAMPLING: only use if run length is 0.1 < L < 100
            if L <= self.L_max:
                break


        # Prevent overshooting episode length
        if self.steps + L > self.max_steps:
            L = self.max_steps - self.steps

        prev_true_pos = self.true_agent_pos.copy() # unwrapped position
        
        # Initialize dx and dy to 0
        dx, dy = 0.0, 0.0

        # Always walking
        self.motion_state = "walk"

        # New orientation for this run
        theta = np.random.uniform(-np.pi, np.pi)

        # length of the walk in the arena's coordinate space.
        dx = self.alpha * L * np.cos(theta)
        dy = self.alpha * L * np.sin(theta)

        proposed_end = prev_true_pos + np.array([dx, dy])

        # Lift target into correct unwrapped copy
        lifted_target = self.targets[0] + self.size * np.round(
            (prev_true_pos - self.targets[0]) / self.size
        )        

        # If already inside the target AFTER RELOCATION, skip hit detection
        dist_to_target = np.linalg.norm(prev_true_pos - lifted_target)

        # Log peak dwelling
        self.peak_times.append(int(dist_to_target < self.dwell_distance_thresh))        

        # --- Check for intersection in UNWRAPPED space ---
        if dist_to_target < self.hit_radius: # if agent has been relocated to the target center in the previous walk, do not count the current walk as a hit
            hit = False
        else:
            hit = segment_circle_hit(
                prev_true_pos, # previous unwrapped position of the agent
                proposed_end, # current unwrapped position of the agent
                lifted_target, # unwrapped position of target
                self.hit_radius
            )

        reward = 0.0

        if hit:
            reward = 1.0

            # Relocate agent to target center (unwrapped)
            self.true_agent_pos = lifted_target.copy()

            # self.time_since_hit = 0
            self.target_hits += 1

            if not self.has_hit:
                self.has_hit = True
                self.first_hit_step = self.steps # first hit step is the start of the run, not when the hit happened

        else:
            # No hit: move normally
            self.true_agent_pos = proposed_end
            # self.time_since_hit += L
        
        
        # Wrap position
        self.agent_pos = self.true_agent_pos % self.size

        self.steps += L
        self.total_raw_reward += reward

        # self.current_run_index += 1

        # Reorientation ALWAYS occurs when a walk finishes
        reoriented = True

        self.prev_lambda = self.curr_lambda

        obs = self._get_observation() # This is where the prev_lambda is added to the state

        # truncated = self.current_run_index >= len(self.run_lengths)
        truncated = self.steps >= self.max_steps # truncation now depends on steps, not number of walks
        terminated = False     

        return obs, reward, terminated, truncated, {
            "action": action,
            "motion_state": self.motion_state, # always 'walk'
            "reoriented_boolean": reoriented,
            "run_length": L,
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
            f"Step {self.steps} | State: {self.motion_state} | Current Lambda: {self.curr_lambda:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
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
            f"Step {self.steps} | Reoriented: {self.reoriented} | Current Lambda: {self.curr_lambda:.3f} | Hits: {self.target_hits} | Reward: {self.total_raw_reward:.3f}"
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

        # Episode Length
        self.episode_length = 500

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

        # Track dwell times
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

        # # Reset memory metric
        # self.time_since_hit_history = []

    
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

        # # Log the new time_since_hit observation
        # time_since_hit = getattr(raw_env, "time_since_hit", None)
        # if time_since_hit is not None:
        #     self.time_since_hit_history.append(time_since_hit)

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

        info["reoriented_boolean"] = raw_env.reoriented


        # Count walks vs reorients
        state = info.get("reoriented_boolean") # this should always be true when we sample walks beforehand
        # if state == False:
        #     self.motion_walk_count += 1
        # elif state == True:
        #     self.motion_reorient_count += 1
        #     self.motion_walk_count += 1
        if state == True:
            self.motion_reorient_count += 1
        self.motion_walk_count = self.episode_length

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
            total_steps = self.episode_length
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
                self.first_hit_steps.append(0)  # Or np.nan to exclude it from averages

            # # New memory-related episode metrics
            # if self.time_since_hit_history:
            #     self.episode_mean_time_since_hit.append(np.mean(self.time_since_hit_history))
            #     self.episode_max_time_since_hit.append(np.max(self.time_since_hit_history))
            # else:
            #     self.episode_mean_time_since_hit.append(0)
            #     self.episode_max_time_since_hit.append(0)


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
                # "episode_mean_time_since_hit": self.stats_callback.episode_mean_time_since_hit,
                # "episode_max_time_since_hit": self.stats_callback.episode_max_time_since_hit,
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