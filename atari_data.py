# Generate some frames of Pong
import random
import sys
import numpy as np
import gym
import random
from scipy.misc import imresize

from imutil import show

import torch

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

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
            pixels = self.step_env(env)
            
            observations.append(pixels)
        # Standard API: next() returns two tensors (x, y)
        return torch.Tensor(np.array(observations)), None
        
    def step_env(self, env):
        obs, r, done, info = env.step(env.action_space.sample())
        if r:
            # After a point is scored, reset
            env.reset()
            # Wait until the ball appears
            for _ in range(20):
                env.step(env.action_space.sample())
                
        return prepro(obs)
        '''pixels = obs[35:195]
        pixels = imresize(pixels, (80,80)).astype(np.float32)
        pixels = (pixels - 128) / 128
        # Output batch x channels x height x width
        
        return pixels.transpose((2,0,1))'''
        
    
    def get_img(self):
        return self.step_env(random.choice(self.environments))
        
