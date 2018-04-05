# Generate some frames of Pong
import numpy as np
import gym
from scipy.misc import imresize
from imutil import show

env = gym.make('Pong-v0')
env.reset()

def reset(env):
    env.reset()
    for _ in range(30):
        env.step(env.action_space.sample())


def step(env, action):
    obs, r, done, info = env.step(action)
    if r:
        reset(env)
    return obs, r, done, info


def get_pong_frame():
    action = env.action_space.sample()
    obs, reward, done, info = step(env, action)
    pixels = imresize(obs[35:195].mean(2), (80,80)).astype(np.float32).reshape(1, 80, 80) / 255.
    return pixels


for _ in range(10):
    frame = get_pong_frame()
    show(frame)
