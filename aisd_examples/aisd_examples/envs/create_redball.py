import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class CreateRedBallEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode='None'):
        self.observation_space = spaces.Discrete(10)
        self.action_space = spaces.Discrete(5)
        self.reset()

    def reset(self, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
	    next_state = self.observation_space.sample()
	    reward = np.random.randn()
	    done = False
	    info = {}
	    return next_state, reward, done, False, info
	
    def render(self):
        pass
