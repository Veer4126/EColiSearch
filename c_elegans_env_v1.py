import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from stable_baselines3.common.callbacks import BaseCallback
import torch as th



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


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,),  # [prev_conc, curr_conc, prev_dx, prev_dy]
        dtype=np.float32
        )

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset agent position
        self.agent_pos = np.random.uniform(0, self.size, size=(2,))
        self.steps = 0
        self.step_lengths = []
        self.total_raw_reward = 0
        self.total_discounted_reward = 0

        # Randomize target(s)
        self.targets = []
        for _ in range(self.max_targets):
            while True:
                new_target = np.random.uniform(0, self.size, size=2)
                if np.linalg.norm(new_target - self.agent_pos) > 2:  # Avoid very close starts
                    break
            self.targets.append(new_target)

        # self.targets = [np.array([self.size / 2, self.size / 2])]

        # Reset previous memory if you're using it
        self.prev_conc = 0.0
        self.prev_action = np.array([0.0, 0.0], dtype=np.float32)

        return self._get_observation(current_conc=0.0), {}


    def step(self, action):
        prev_pos = self.agent_pos.copy()

        # Scale action to limit step size
        self.agent_pos += action
        self.agent_pos = np.clip(self.agent_pos, 0, self.size)
        self.steps += 1

        # Track step length
        step_length = np.linalg.norm(self.agent_pos - prev_pos)
        self.step_lengths.append(step_length)

        # Distance to (first) target
        tgt = self.targets[0]
        dist_to_target = np.linalg.norm(self.agent_pos - tgt)

        # Max possible distance: diagonal of square grid
        max_dist = np.sqrt(2) * self.size
        normalized_dist = dist_to_target / max_dist  # Between 0 and 1

        # Reward: higher when closer to target
        reward = 1.5 * (1.0 - (dist_to_target / max_dist)) ** 2

        # Bonus if target is reached
        target_hit = False
        if dist_to_target < 0.5:
            reward += 50.0
            target_hit = True

        # Slight penalty to reduce bouncing
        reward -= 0.01 * np.linalg.norm(action)

        # Update total raw reward
        self.total_raw_reward += reward

        # Get new observation
        obs = self._get_observation()
        # terminated = target_hit or self.steps >= self.max_steps
        terminated = False
        truncated = self.steps >= self.max_steps

        # Update memory
        self.prev_conc = self.get_concentration(self.agent_pos)
        self.prev_action = action

        return obs, reward, terminated, truncated, { # info["target_hit"] == True only on the step where the agent hits the target.
            "target_hit": target_hit,
            "distance": dist_to_target
        }

    

    def _get_observation(self, current_conc=None):
        if current_conc is None:
            current_conc = self.get_concentration(self.agent_pos)
        return np.array([
            self.prev_conc,
            current_conc,
            *self.prev_action
        ], dtype=np.float32)


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
        self.ax.plot(*self.agent_pos, 'bo', label='Agent')

        # Add action arrow if provided
        if mean_action is not None:
            arrow_scale = 0.8  # optional: scale for visualization
            dx, dy = mean_action
            self.ax.arrow(
                self.agent_pos[0], self.agent_pos[1],
                arrow_scale * dx, arrow_scale * dy,
                head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='Mean Action'
            )

        self.ax.set_title(f"Step {self.steps} | Targets: {len(self.targets)} | Raw Reward: {self.total_raw_reward:.2f} | Discounted Reward: {self.total_discounted_reward:.3f}")
        self.ax.legend()
        plt.pause(0.01)


    def plot_step_length_distribution(self):
        # if not self.step_lengths:
        #     print("No step length data to plot!")
        #     return
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
        self.mean_actions = []
        self.stddevs = []
        self.entropies = [] # to quantify exploration
        self.episode_rewards = [] # to check if episodic returns are improving with more training
        self.target_hits = []  # New list
        self.episode_end_types = []  # "hit" or "timeout"
        self.step_lengths = []
        self.current_episode_reward = 0.0 # to check if episodic returns are improving with more training      
        self.current_target_hits = 0  # Counter
        self.prev_pos = None
        self.did_hit_target_this_episode = False # remove this to allow termination

    

    def _on_step(self) -> bool:
        # --- Get underlying raw env ---
        env = self.training_env.envs[0]
        raw_env = env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env

        # --- Reward tracking ---
        reward = self.locals.get("rewards", [0.0])[0]
        self.current_episode_reward += reward

        # --- Target hit tracking ---
        info = self.locals.get("infos", [{}])[0]
        if info.get("target_hit", False):
            self.current_target_hits += 1
            self.did_hit_target_this_episode = True # remove this to allow termination

        # --- Episode termination ---
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.target_hits.append(self.current_target_hits)
            # self.episode_end_types.append("hit" if info.get("target_hit", False) else "timeout")
            self.episode_end_types.append("hit" if self.did_hit_target_this_episode else "timeout")
            self.current_episode_reward = 0.0
            self.current_target_hits = 0
            self.did_hit_target_this_episode = False

        # --- Step length tracking ---
        curr_pos = raw_env.agent_pos.copy()
        if self.prev_pos is not None:
            step_length = np.linalg.norm(curr_pos - self.prev_pos)
            self.step_lengths.append(step_length)
        self.prev_pos = curr_pos

        # --- Action stats ---
        try:
            obs = self.locals.get("new_obs", [None])[0]
            if obs is not None:
                obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(self.model.device)

                # For SAC: actor outputs mean and log_std explicitly
                latent_pi = self.model.policy.actor.latent_pi(obs_tensor)
                mean_tensor = self.model.policy.actor.mu(latent_pi)
                log_std_tensor = self.model.policy.actor.log_std(latent_pi)
                std_tensor = th.exp(log_std_tensor)

                mean = mean_tensor.detach().cpu().numpy()[0]
                std = std_tensor.detach().cpu().numpy()[0]

                # SAC entropy (approximate via log_std)
                entropy = np.mean(log_std_tensor.detach().cpu().numpy())  # Not exact entropy, but good proxy

                self.mean_actions.append(mean)
                self.stddevs.append(std)
                self.entropies.append(entropy)
        except Exception as e:
            if self.verbose:
                print(f"Skipping stat logging this step: {e}")

        return True

    
    # def _on_step(self) -> bool:
    #     # Access raw env
    #     env = self.training_env.envs[0]
    #     raw_env = env
    #     while hasattr(raw_env, 'env'):
    #         raw_env = raw_env.env

    #     # Track reward from env
    #     reward = self.locals.get("rewards", [0.0])[0]
    #     self.current_episode_reward += reward

    #     # Track target hit flag (from env info)
    #     info = self.locals.get("infos", [{}])[0]
    #     if info.get("target_hit", False):
    #         self.current_target_hits += 1

    #     # Check if episode ended
    #     done = self.locals.get("dones", [False])[0]
    #     if done:
    #         self.episode_rewards.append(self.current_episode_reward)
    #         self.target_hits.append(self.current_target_hits)

    #         # Track type of termination: target hit vs timeout
    #         if info.get("target_hit", False):
    #             self.episode_end_types.append("hit")
    #         else:
    #             self.episode_end_types.append("timeout")

    #         self.current_episode_reward = 0.0
    #         self.current_target_hits = 0

    #     # Step length tracking
    #     curr_pos = raw_env.agent_pos.copy()
    #     if self.prev_pos is not None:
    #         step_length = np.linalg.norm(curr_pos - self.prev_pos)
    #         self.step_lengths.append(step_length)
    #     self.prev_pos = curr_pos

    #     # Action stats and entropy
    #     try:
    #         obs_tensor = th.as_tensor(
    #             self.model.rollout_buffer.observations[-1], dtype=th.float32
    #         ).to(self.model.device)
    #         if len(obs_tensor.shape) == 1:
    #             obs_tensor = obs_tensor.unsqueeze(0)

    #         dist = self.model.policy.get_distribution(obs_tensor)
    #         mean = dist.distribution.mean.detach().cpu().numpy()
    #         stddev = dist.distribution.stddev.detach().cpu().numpy()
    #         entropy = dist.distribution.entropy().detach().cpu().numpy().mean()

    #         # Ensure consistent shape: always (action_dim,)
    #         mean = np.atleast_1d(mean)
    #         stddev = np.atleast_1d(stddev)

    #         self.mean_actions.append(mean)
    #         self.stddevs.append(stddev)
    #         self.entropies.append(entropy)
    #     except Exception as e:
    #         if self.verbose:
    #             print(f"Skipping stat logging this step: {e}")
    #     return True


    
    # def _on_training_end(self):
    #     if self.step_lengths:
    #         plt.figure(figsize=(6, 4))
    #         min_val, max_val = min(self.step_lengths), max(self.step_lengths)
    #         bins = np.linspace(min_val, max_val, 50)  # Higher-resolution bins

    #         plt.hist(self.step_lengths, bins=bins, edgecolor='black', alpha=0.7)
    #         plt.xlabel("Step Length")
    #         plt.ylabel("Frequency")
    #         plt.title("Training Step Length Distribution")
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.show(block=False)




    # PPO Function

#    def _on_step(self) -> bool:
#         # Access raw env
#         env = self.training_env.envs[0]
#         raw_env = env
#         while hasattr(raw_env, 'env'):
#             raw_env = raw_env.env

#         # Track reward from env
#         reward = self.locals.get("rewards", [0.0])[0]
#         self.current_episode_reward += reward

#         # Track target hit flag (from env info)
#         info = self.locals.get("infos", [{}])[0]
#         if info.get("target_hit", False):
#             self.current_target_hits += 1

#         # Check if episode ended
#         done = self.locals.get("dones", [False])[0]
#         if done:
#             self.episode_rewards.append(self.current_episode_reward)
#             self.target_hits.append(self.current_target_hits)

#             # Track type of termination: target hit vs timeout
#             if info.get("target_hit", False):
#                 self.episode_end_types.append("hit")
#             else:
#                 self.episode_end_types.append("timeout")

#             self.current_episode_reward = 0.0
#             self.current_target_hits = 0

#         # Step length tracking
#         curr_pos = raw_env.agent_pos.copy()
#         if self.prev_pos is not None:
#             step_length = np.linalg.norm(curr_pos - self.prev_pos)
#             self.step_lengths.append(step_length)
#         self.prev_pos = curr_pos

#         # Action stats and entropy
#         try:
#             obs_tensor = th.as_tensor(
#                 self.model.rollout_buffer.observations[-1], dtype=th.float32
#             ).to(self.model.device)
#             if len(obs_tensor.shape) == 1:
#                 obs_tensor = obs_tensor.unsqueeze(0)

#             dist = self.model.policy.get_distribution(obs_tensor)
#             mean = dist.distribution.mean.detach().cpu().numpy()[0]
#             stddev = dist.distribution.stddev.detach().cpu().numpy()[0]
#             entropy = dist.distribution.entropy().detach().cpu().numpy().mean()

#             self.mean_actions.append(mean)
#             self.stddevs.append(stddev)
#             self.entropies.append(entropy)
#         except Exception as e:
#             if self.verbose:
#                 print(f"Skipping stat logging this step: {e}")

#         return True


# if you want both:
# try:
#     obs = self.locals.get("new_obs", [None])[0]
#     if obs is not None:
#         obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0).to(self.model.device)

#         if hasattr(self.model.policy, "actor"):  # SAC
#             dist = self.model.policy.actor.get_dist(obs_tensor)
#             mean = dist.mean.detach().cpu().numpy()[0]
#             stddev = dist.stddev.detach().cpu().numpy()[0]
#             entropy = dist.entropy().mean().item()

#         elif hasattr(self.model.policy, "get_distribution"):  # PPO
#             dist = self.model.policy.get_distribution(obs_tensor)
#             mean = dist.distribution.mean.detach().cpu().numpy()[0]
#             stddev = dist.distribution.stddev.detach().cpu().numpy()[0]
#             entropy = dist.distribution.entropy().detach().cpu().numpy().mean()

#         else:
#             raise ValueError("Unsupported policy type for extracting action statistics.")

#         self.mean_actions.append(mean)
#         self.stddevs.append(stddev)
#         self.entropies.append(entropy)
# except Exception as e:
#     if self.verbose:
#         print(f"Skipping stat logging this step: {e}")
