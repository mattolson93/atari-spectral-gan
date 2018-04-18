# Generate some frames of Pong
import random
import sys
import numpy as np
import gym
from scipy.misc import imresize
from imutil import show

import torch


class AtariDataloader():
    def __init__(self, name, batch_size):
        self.environments = []
        for i in range(batch_size):
            env = gym.make(name)
            env.reset()
            # Start each environment at a random time
            for _ in range(random.randint(1, 100)):
                env.step(env.action_space.sample())
            self.environments.append(env)

    def __iter__(self):
        return self

    def __next__(self):
        observations = []
        for env in self.environments:
            obs, r, done, info = env.step(env.action_space.sample())
            if done:
                env.reset()
            pixels = obs[34:194].mean(2)
            pixels = imresize(pixels, (80,80))
            pixels = pixels.astype(np.float32).reshape(1, 80, 80) / 255.
            pixels = np.concatenate((pixels, pixels, pixels))
            observations.append(pixels)
        target = None
        return torch.Tensor(np.array(observations)), None
