import gym
import gymnasium
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display, clear_output # Import clear_output
import pygame
from pygame import gfxdraw

# Use human mode to open a render window
env = gymnasium.make("Walker2d-v5", render_mode='human')
env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        env.reset()

env.close()
