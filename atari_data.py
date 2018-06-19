# Generate some frames of Pong
import random
import sys
import numpy as np
import gym
import random
from scipy.misc import imresize


import torch
import random
from multiprocessing import Pool

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

class AtariDataloader():
    def __init__(self, name, batch_size):
        self.environments = []
        self.batch_size = batch_size

        for i in range(batch_size):
            env = gym.make(name)
            env.seed(i)
            env.reset()
            self.environments.append(env)

    def _start_threads(self):
        pool = Pool(processes = self.batch_size)
        self.ending = random.randint(0, 20)
        ret = pool.map(self._get_frames, (self.environments))

        pool.close()

        return ret

    def __iter__(self):
        return self


    def _get_frames(self, env):
        ret = []
        episode_length = 0
        done = False
        total_length = 1e3 - 1 + self.ending
        while episode_length <= (total_length):
            obs, r, done, info = env.step(env.action_space.sample())
            ret.append(prepro(obs))
            episode_length += 1

        env.reset()
        return ret  

    def __next__(self):
        #return the obs
        return torch.Tensor(np.array(self._start_threads())), None


        
    def get_img(self):
        env, steps = random.choice(self.environments)
        return self.step_env(env, steps)
    
def main():
    import time
    loader = AtariDataloader('Pong-v0', batch_size=8)
    for batch_idx, (data, target) in enumerate(loader):
        print("{} sleeping for 5 seconds".format( batch_idx))
        time.sleep(5)
        print("done sleeping")
        
        if batch_idx == 2: import pdb; pdb.set_trace()
        if batch_idx == 5: break


if __name__ == '__main__':
    main()