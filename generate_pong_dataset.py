# Generate some frames of Pong
import sys
import numpy as np
import gym
from scipy.misc import imresize
from imutil import show


env = gym.make('Pong-v0')


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
