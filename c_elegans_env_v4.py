import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from stable_baselines3.common.callbacks import BaseCallback
import torch as th
from matplotlib.patches import FancyArrow



class CEMazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=8, max_steps=400, k_w2r=1.0, k_r2w=0.5, alpha=0.5, gradient_sigma=2.0):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.max_targets = 1
        self.targets = [np.array([self.size / 2, self.size / 2])]
        self.gradient_sigma = gradient_sigma # Slightly smaller spread for multiple sources
        self.steps = 0
        self.gamma = 0.95 # for temporally discounting rewards
        self.motion_state = "walk"  # or "reorient"
        self.k_w2r = k_w2r
        self.k_r2w = k_r2w
        self.alpha = alpha
        self.dt = 1.0  # fixed time step
        self.theta = 0.0
        self.agent_pos = np.zeros(2)
        self.prev_conc = 0.0
        self.prev_p_reorient = 0.5  # Initial value for obs
        self.total_raw_reward = 0.0

        self.action_space = gym.spaces.Box(
        low=np.array([1e-4, 1e-4], dtype=np.float32),   # min allowed values
        high=np.array([1.0, 1.0], dtype=np.float32),    # max allowed values
        dtype=np.float32)


        # [prev_conc, curr_conc, prev_p_reorient, prev_p_walk]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, 0.0, 0.0]),
            high=np.array([np.inf, np.inf, 1.0, 1.0]),
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
            self.prev_conc,
            current_conc,
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

        # Randomize target(s)
        self.targets = []
        for _ in range(self.max_targets):
            while True:
                new_target = np.random.uniform(0, self.size, size=2)
                if np.linalg.norm(new_target - self.agent_pos) > 2:  # Avoid very close starts
                    break
            self.targets.append(new_target)

        self.prev_conc = self.get_concentration(self.agent_pos)
        self.prev_p_reorient = 0.5
        self.prev_p_walk = 0.5
        self.curr_reward = 0.0

        return self._get_observation(current_conc=self.prev_conc), {} # In the reset, the curr_conc is set to prev_conc for the first step of every episode


    def step(self, action=None):
        if action is not None:
            self.k_w2r, self.k_r2w = action

        prev_pos = self.agent_pos.copy()

        # Transition probabilities from rate constants
        # These are 2 of the 4 components of the state vector
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

        curr_conc = self.get_concentration(self.agent_pos) # This gives the third component of the state vector. 
        reward = curr_conc
        self.curr_reward = reward
        self.total_raw_reward += reward

        # Target hit check (within radius)
        hit = False
        if np.linalg.norm(self.agent_pos - self.targets[0]) < 0.5:
            self.target_hits += 10
            hit = True

        obs = self._get_observation(curr_conc) # This is where the 2 probabilities, and curr conc are added to the state vector.
        # Additionally, the prev_conc, computed from the prev step, is added to this vector.
        terminated = False
        truncated = self.steps >= self.max_steps

        self.prev_conc = curr_conc # This is the 4th component of the state vector but it is used in the NEXT step as prev_conc
        self.prev_p_reorient = p_reorient
        self.prev_p_walk = p_walk

        return obs, reward, terminated, truncated, {
            "concentration": curr_conc,
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

        # Tracking raw concentration values
        self.episode_concentrations = [] # List of lists (one per episode)

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

    def reset_episode_stats(self):
        self.current_episode_reward = 0.0 # to check if episodic returns are improving with more training

        # Tracking target hits per episode
        self.current_target_hits = 0  # Counter
        self.did_hit_target_this_episode = False # remove this to allow termination

        # Tracking raw concentration values
        self.current_episode_concs = [] # Concentrations for the current episode

        # Tracking average distance to target per episode
        self.distances_to_target = [] # For current episode

        # Tracking the change in distance to the target from one step to the next per episode
        self.delta_distances = [] # Step-wise deltas

        # Tracking step lengths with training
        self.prev_distance = None
        self.prev_pos = None

        # Track motion states over training per episode
        self.motion_history = []


    # PPO/SAC Code
    def _on_step(self) -> bool:
        # Get underlying raw env
        env = self.training_env.envs[0]
        raw_env = env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env

        # Track info
        info = self.locals.get("infos", [{}])[0]

        # Concentration
        conc = info.get("concentration", None)
        if conc is not None:
            self.current_episode_concs.append(conc)

        # Reward
        reward = self.locals.get("rewards", [0.0])[0]
        self.current_episode_reward += reward

        # Target hit
        if info.get("target_hit", False):
            self.current_target_hits += 1
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
            self.episode_rewards.append(self.current_episode_reward)
            self.target_hits.append(self.current_target_hits)
            self.episode_end_types.append("hit" if self.did_hit_target_this_episode else "timeout")
            self.episode_concentrations.append(self.current_episode_concs)

            if self.distances_to_target:
                self.episode_avg_distances.append(np.mean(self.distances_to_target))
            else:
                self.episode_avg_distances.append(0)

            if self.delta_distances:
                self.episode_avg_delta_distances.append(np.mean(self.delta_distances))
            else:
                self.episode_avg_delta_distances.append(0)

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
            plt.show(block=False)