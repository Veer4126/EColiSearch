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

    def __init__(self, size=8, max_steps=400, max_targets=1):
        super().__init__()

        self.size = size
        self.max_steps = max_steps
        self.max_targets = max_targets
        self.gradient_sigma = 2  # Slightly smaller spread for multiple sources
        self.targets = [np.array([self.size / 2, self.size / 2])]
        self.steps = 0
        self.gamma = 0.95 # for temporally discounting rewards
        self.total_raw_reward = 0.0
        self.total_discounted_reward = 0.0
        self.motion_state = "walk"  # or "reorient"


        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Action = [reorient_prob (0 to 1), theta (-π to π), alpha (step size, 0 to 1)]
        self.action_space = spaces.Box(low=np.array([0.0, -np.pi, 0.0]),
                                high=np.array([1.0, np.pi, 1.0]),
                                shape=(3,),
                                dtype=np.float32)


        self.observation_space = spaces.Box(
        low=np.array([-np.inf, -np.inf, -np.pi, 0.0, 0.0]), # [prev_conc, curr_conc, prev_dtheta, prev_alpha, prev_p_reorient]
        high=np.array([np.inf, np.inf, np.pi, 1.0, 1.0]),
        shape=(5,),
        dtype=np.float32)

        self.step_lengths = []
        self.fig = None  # For rendering

    def _generate_random_targets(self): # ensure that targets are at a different location each time
        num_targets = np.random.randint(1, self.max_targets + 1)
        return [np.random.uniform(1, self.size - 1, size=(2,)) for _ in range(num_targets)]

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
            self.prev_dtheta,
            self.prev_alpha,
            self.prev_p_reorient
        ], dtype=np.float32)
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent position
        self.agent_pos = np.random.uniform(0, self.size, size=(2,))
        self.steps = 0
        self.step_lengths = []
        self.total_raw_reward = 0.0
        self.total_discounted_reward = 0.0

        # Randomize target(s)
        self.targets = []
        for _ in range(self.max_targets):
            while True:
                new_target = np.random.uniform(0, self.size, size=2)
                if np.linalg.norm(new_target - self.agent_pos) > 2:  # Avoid very close starts
                    break
            self.targets.append(new_target)

        # Initialize internal motor state
        self.theta = np.random.uniform(-np.pi, np.pi)  # initial direction
        self.alpha = 0.0
        self.motion_state = "walk"

        # Initialize state memory
        self.prev_conc = self.get_concentration(self.agent_pos)
        self.prev_alpha = 0.0
        self.prev_dtheta = 0.0
        self.prev_p_reorient = 0.5  # neutral

        return self._get_observation(current_conc=self.prev_conc), {}


    def step(self, action):
        prev_pos = self.agent_pos.copy()

        # Sample motion state
        p_reorient, dtheta, alpha = action
        p_reorient = np.clip(p_reorient, 0.0, 1.0)  # Ensure it's a valid probability

        # Sample whether to reorient or walk
        is_reorient = np.random.rand() < p_reorient

        if is_reorient:
            self.motion_state = "reorient"
            self.theta += dtheta  # Turn in place
            dx, dy = 0.0, 0.0
            self.alpha = 0.0
            self.dtheta = dtheta
        else:
            self.motion_state = "walk"
            dx = alpha * np.cos(self.theta)
            dy = alpha * np.sin(self.theta)
            self.agent_pos += np.array([dx, dy])
            self.theta = self.theta  # theta stays the same
            self.alpha = alpha
            self.dtheta = 0.0  # No turning during walk

        # Clip position to remain in bounds and update counter
        self.agent_pos = np.clip(self.agent_pos, 0, self.size)
        self.steps += 1

        # Track step length
        step_length = np.linalg.norm(self.agent_pos - prev_pos)
        self.step_lengths.append(step_length)

        # Get concentration at new position
        curr_conc = self.get_concentration(self.agent_pos)

        # Reward: change in concentration
        # reward = curr_conc - self.prev_conc
        
        # Distance to (first) target

        tgt = self.targets[0]
        dist_to_target = np.linalg.norm(self.agent_pos - tgt)

        # Max possible distance: diagonal of square grid
        max_dist = np.sqrt(2) * self.size
        normalized_dist = dist_to_target / max_dist  # Between 0 and 1

        # Reward: higher when closer to target
        reward = 1.5 * (1.0 - (normalized_dist)) ** 2

        # Bonus: Target hit ---
        target_hit = False
        if np.linalg.norm(self.agent_pos - self.targets[0]) < 0.5:
        #     reward += 20.0
             target_hit = True

        # # Penalty: Wall hugging ---
        # if np.any(self.agent_pos <= 0.1) or np.any(self.agent_pos >= self.size - 0.1):
        #     reward -= 5.0

        # # Penalty: zero movement (walk with very small alpha)
        # if not is_reorient and alpha < 0.1:
        #     reward -= 5.0

        # Update cumulative reward
        self.total_raw_reward += reward
        self.total_discounted_reward += (self.gamma ** self.steps) * reward

        # Get observation (with current concentration)
        obs = self._get_observation(current_conc=curr_conc)

        # Termination is disabled; only truncation
        terminated = False
        truncated = self.steps >= self.max_steps

        # Update internal memory for observation
        self.prev_conc = curr_conc
        self.prev_p_reorient = p_reorient
        self.prev_dtheta = dtheta
        self.prev_alpha = alpha

        return obs, reward, terminated, truncated, {
            "target_hit": target_hit,
            "distance": np.linalg.norm(self.agent_pos - self.targets[0]),
            "concentration": curr_conc 
        }


    def render(self, mode='human', mean_action=None):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)

        # Render the scalar field as a background image
        x = np.linspace(0, self.size, 100) # define the resolution of the scalar field grid
        y = np.linspace(0, self.size, 100) # define the resolution of the scalar field grid
        X, Y = np.meshgrid(x, y) # create a 2D grid based on the possible x and y values
        Z = np.zeros_like(X)

        for tgt in self.targets:
            Z += np.exp(-((X - tgt[0])**2 + (Y - tgt[1])**2) / (2 * self.gradient_sigma**2)) # The gradient is centered at the target and the decay varies with the std dev from the center

        self.ax.imshow(Z, extent=[0, self.size, 0, self.size], origin='lower', cmap='Greens', alpha=0.6) # alpha is the image transparency

        # Plot targets and agent
        for tgt in self.targets:
            self.ax.plot(*tgt, 'ro', markersize=6)

        # self.ax.plot(*self.agent_pos, 'bo', label='Agent')
        # self.ax.plot(*self.agent_pos, marker=(3, 0, np.degrees(self.theta)), 
        #      color='blue', markersize=10, label='Agent')
        
        arrow = FancyArrow(
            x=self.agent_pos[0],
            y=self.agent_pos[1],
            dx=0.2 * np.cos(self.theta),
            dy=0.2 * np.sin(self.theta),
            width=0.05,
            color='blue'
        )
        self.ax.add_patch(arrow)

        # Add action arrow if provided
        if mean_action is not None:
            arrow_scale = 1  # optional: scale for visualization
            dx = arrow_scale * np.cos(self.theta)
            dy = arrow_scale * np.sin(self.theta)
            self.ax.arrow(
                self.agent_pos[0], self.agent_pos[1],
                arrow_scale * dx, arrow_scale * dy,
                head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='Direction'
            )

        self.ax.set_title(f"Step {self.steps} | State: {self.motion_state} | Targets: {len(self.targets)} | Raw Reward: {self.total_raw_reward:.2f} | Discounted Reward: {self.total_discounted_reward:.3f}")
        self.ax.legend()
        plt.pause(0.01)


    def plot_step_length_distribution(self):
        if not self.step_lengths:
            print("No step length data to plot!")
            return
        plt.figure(figsize=(6, 4))
        plt.hist(self.step_lengths, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel("Step Length")
        plt.ylabel("Frequency")
        plt.title("Distribution of Agent's Step Lengths")
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)


    def close(self):
        if self.fig:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None



class TrainingStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        # Mean/std/entropy of action distributions
        self.mean_actions = []
        self.stddevs = []
        self.entropies = [] #Tracking entropy of the policy over time to quantify exploration

        # Tracking episodic reward with training
        self.episode_rewards = [] # to check if episodic returns are improving with more training
        self.current_episode_reward = 0.0 # to check if episodic returns are improving with more training

        # Tracking step lengths with training
        self.step_lengths = []
        self.prev_pos = None
        
        # Tracking target hit rate per episode
        self.target_hits = []
        self.episode_end_types = []  # "hit" or "timeout"

        # Tracking target hits per episode
        self.current_target_hits = 0  # Counter
        self.did_hit_target_this_episode = False # remove this to allow termination
        
        # Tracking raw concentration values
        self.episode_concentrations = []  # List of lists (one per episode)
        self.current_episode_concs = []   # Concentrations for the current episode

        # Tracking directional accuracies per episode
        self.directional_accuracies = []  # For current episode
        self.all_episode_directional_traces = []  # New: stores full trace per episode
        self.episode_directional_accuracies = []  # Across episodes

        # Tracking average distance to target per episode
        self.distances_to_target = []     # For current episode
        self.episode_avg_distances = []   # Across episodes

        # Tracking the change in distance to the target from one step to the next per episode
        self.delta_distances = [] # Step-wise deltas
        self.episode_avg_delta_distances = [] # Across episodes
        self.prev_distance = None  # For delta tracking
    
        # Tracking walk vs reorient over time
        self.motion_history = []

        # Tracing agent movements with time
        self.alphas = []
        self.thetas = []
        self.dthetas = []

    # PPO/SAC Code
    def _on_step(self) -> bool:
        # --- Get underlying raw env ---
        env = self.training_env.envs[0]
        raw_env = env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env

        # --- Concentration tracking ---
        info = self.locals.get("infos", [{}])[0]
        concentration = info.get("concentration", None)
        if concentration is not None:
            self.current_episode_concs.append(concentration)

        # --- Reward tracking ---
        reward = self.locals.get("rewards", [0.0])[0]
        self.current_episode_reward += reward

        # --- Target hit tracking ---
        if info.get("target_hit", False):
            self.current_target_hits += 1
            self.did_hit_target_this_episode = True

        # --- Step length tracking ---
        curr_pos = raw_env.agent_pos.copy()
        if self.prev_pos is not None:
            step_length = np.linalg.norm(curr_pos - self.prev_pos)
            self.step_lengths.append(step_length)
        self.prev_pos = curr_pos

        # --- Motion state tracking ---
        if hasattr(raw_env, "motion_state"):
            self.motion_history.append(raw_env.motion_state)

        # --- Motor variables tracking ---
        if hasattr(raw_env, "alpha"):
            self.alphas.append(raw_env.alpha)
        if hasattr(raw_env, "theta"):
            self.thetas.append(raw_env.theta)
        if hasattr(raw_env, "dtheta"):
            self.dthetas.append(raw_env.dtheta)

        # --- Distance to target tracking ---
        target = raw_env.targets[0]
        vector_to_target = target - curr_pos
        distance_to_target = np.linalg.norm(vector_to_target)
        self.distances_to_target.append(distance_to_target)

        if distance_to_target > 1e-8:
            vector_to_target /= distance_to_target  # Normalize

        # --- Delta distance to target tracking ---
        if self.prev_distance is not None:
            delta = self.prev_distance - distance_to_target
            self.delta_distances.append(delta)
        self.prev_distance = distance_to_target

        # --- Directional accuracy ---
        if raw_env.motion_state == "walk" and raw_env.alpha > 1e-4:
            heading_vector = np.array([np.cos(raw_env.theta), np.sin(raw_env.theta)])
            directional_accuracy = np.dot(vector_to_target, heading_vector)
            self.directional_accuracies.append(directional_accuracy)


        # --- Get Policy Statistics ---
        try:
            obs_tensor = th.as_tensor(
                self.model.rollout_buffer.observations[-1] if hasattr(self.model, "rollout_buffer") else self.locals.get("new_obs", [None])[0],
                dtype=th.float32
            ).unsqueeze(0).to(self.model.device)

            if isinstance(self.model.policy, th.nn.Module):
                # PPO or other on-policy algorithm
                if hasattr(self.model, "rollout_buffer"):
                    dist = self.model.policy.get_distribution(obs_tensor)
                    mean = dist.distribution.mean.detach().cpu().numpy()[0]
                    std = dist.distribution.stddev.detach().cpu().numpy()[0]
                    entropy = dist.distribution.entropy().detach().cpu().numpy().mean()
                # SAC or similar off-policy algorithm
                elif hasattr(self.model.policy, "actor"):
                    latent_pi = self.model.policy.actor.latent_pi(obs_tensor)
                    mean_tensor = self.model.policy.actor.mu(latent_pi)
                    log_std_tensor = self.model.policy.actor.log_std(latent_pi)
                    std_tensor = th.exp(log_std_tensor)

                    mean = mean_tensor.detach().cpu().numpy()[0]
                    std = std_tensor.detach().cpu().numpy()[0]
                    entropy = np.mean(log_std_tensor.detach().cpu().numpy())

                self.mean_actions.append(mean)
                self.stddevs.append(std)
                self.entropies.append(entropy)

        except Exception as e:
            if self.verbose:
                print(f"Skipping stat logging this step: {e}")


        # --- Episode termination ---
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.target_hits.append(self.current_target_hits)

            # self.episode_end_types.append("hit" if info.get("target_hit", False) else "timeout")
            self.episode_end_types.append("hit" if self.did_hit_target_this_episode else "timeout")
            self.episode_concentrations.append(self.current_episode_concs)

            if self.directional_accuracies:
                self.all_episode_directional_traces.append(self.directional_accuracies.copy())
                self.episode_directional_accuracies.append(np.mean(self.directional_accuracies))
            else:
                self.all_episode_directional_traces.append([np.nan] * 400)
                self.episode_directional_accuracies.append(0)

            if self.distances_to_target:
                self.episode_avg_distances.append(np.mean(self.distances_to_target))
            else:
                self.episode_avg_distances.append(0)

            if self.delta_distances:
                self.episode_avg_delta_distances.append(np.mean(self.delta_distances))
            else:
                self.episode_avg_delta_distances.append(0)

            # Reset per-episode trackers
            self.current_episode_reward = 0.0
            self.current_target_hits = 0
            self.did_hit_target_this_episode = False
            self.current_episode_concs = []
            self.directional_accuracies = []
            self.distances_to_target = []
            self.delta_distances = []
            self.prev_distance = None

        return True

    def _on_training_end(self):
        if self.step_lengths:
            plt.figure(figsize=(6, 4))
            min_val, max_val = min(self.step_lengths), max(self.step_lengths)
            bins = np.linspace(min_val, max_val, 50)  # Higher-resolution bins

            plt.hist(self.step_lengths, bins=bins, edgecolor='black', alpha=0.7)
            plt.xlabel("Step Length")
            plt.ylabel("Frequency")
            plt.title("Training Step Length Distribution")
            plt.grid(True)
            plt.tight_layout()
            plt.show(block=False)
