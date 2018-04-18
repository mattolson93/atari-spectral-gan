# Generate some frames of Pong
import sys
import numpy as np
import gym
from scipy.misc import imresize
from imutil import show

import torch


env = gym.make('Pong-v0')

class AtariDataloader():
    def __init__(self, name, batch_size):
        self.environments = []
        for i in range(batch_size):
            env = gym.make(name)
            self.environments.append(env)
        for env in self.environments:
            env.reset()

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
            pixels = pixels > 0
            pixels = pixels.astype(np.float32).reshape(1, 80, 80)
            pixels = np.concatenate((pixels, pixels, pixels))
            observations.append(pixels)
        target = None
        return torch.Tensor(np.array(observations)), None

"""
import asyncio
class AsyncIteratorExecutor:
    def __init__(self, iterator, loop=None, executor=None):
        self.__iterator = iterator
        self.__loop = loop or asyncio.get_event_loop()
        self.__executor = executor

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.__loop.run_in_executor(
            self.__executor, next, self.__iterator, self)
        if value is self:
            raise StopAsyncIteration
        return value
"""


"""
def reset(env):
    env.reset()
    for _ in range(50):
        env.step(env.action_space.sample())


def step(env, action):
    obs, r, done, info = env.step(action)
    if r:
        print("Reset after reward {}".format(r))
        reset(env)
        obs, r, done, info = env.step(action)
    return obs, r, done, info


def get_pong_frame():
    action = env.action_space.sample()
    obs, reward, done, info = step(env, action)
    pixels = imresize(obs[34:194].max(2), (80,80)).astype(np.float32).reshape(1, 80, 80) / 255.
    return pixels


if __name__ == '__main__':
    output_dir = sys.argv[1]
    reset(env)
    for i in range(1000 * 100):
        frame = get_pong_frame()
        filename = '{}/pong_{:08d}.jpg'.format(output_dir, i)
        show(frame, filename=filename)
"""
