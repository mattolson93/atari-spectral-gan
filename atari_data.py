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
            steps = random.randint(1, 100)
            for _ in range(steps):
                env.step(env.action_space.sample())
            self.environments.append((env,steps))

    def __iter__(self):
        return self
    
    def __next__(self):
    
        #import pdb; pdb.set_trace()
        observations = []
        for (env,steps) in self.environments:
            vid = self.step_env(env,steps)
            
            observations.append(vid)
        # Standard API: next() returns two tensors (x, y)
        return torch.Tensor(np.array(observations)), None
        
    def step_env(self, env, episode_length):
        ret = []
        for _ in range(4):
            obs, r, done, info = env.step(env.action_space.sample())
            ret.append(obs)
            done = done or episode_length >= 1e4
            if done:
                # After a point is scored, reset
                env.reset()
                obs, r, done, info = env.step(env.action_space.sample())
                
                ret = []
                ret.append(obs*4)
                break
                
        return [prepro(x) for x in ret]
        '''pixels = obs[35:195]
        pixels = imresize(pixels, (80,80)).astype(np.float32)
        pixels = (pixels - 128) / 128
        # Output batch x channels x height x width
        
        return pixels.transpose((2,0,1))'''
        
    
    def get_img(self):
        env, steps = random.choice(self.environments)
        return self.step_env(env, steps)
        
