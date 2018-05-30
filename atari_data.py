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
            env.seed(i)
            env.reset()
            self.environments.append(env)

    def __iter__(self):
        return self
    
    def __next__(self):
    
        #import pdb; pdb.set_trace()
        observations = []
        for env in self.environments:

            vid = self.step_env(env)
            observations.append(vid)
        # Standard API: next() returns two tensors (x, y)
        return torch.Tensor(np.array(observations)), None
        
    def step_env(self, env):
        ret = []
        episode_length = 0
        done = False
        while episode_length <= (1e3 - 1):
            obs, r, done, info = env.step(env.action_space.sample())
            ret.append(prepro(obs))
            episode_length += 1
            #done = done or episode_length >= 1e3 #TODO: allow for arbitrary game lengths
        print(episode_length)
        env.reset()
        return ret
        
    def get_img(self):
        env, steps = random.choice(self.environments)
        return self.step_env(env, steps)
        