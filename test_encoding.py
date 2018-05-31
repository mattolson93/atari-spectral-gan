import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch import autograd
from torch.autograd import Variable
import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import imutil
from scipy.misc import imsave
from scipy.misc import imresize
import gym

print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--img_dir', type=str, default='eval_imgs')

parser.add_argument('--latent_size', type=int, default=10)
parser.add_argument('--gpu', type=int, default=7)
parser.add_argument('--env_name', type=str, default='Pong-v0')
parser.add_argument('--enc_file', type=str, default=None)
parser.add_argument('--gen_file', type=str, default=None)
parser.add_argument('--game_length', type=int, default=200)

args = parser.parse_args()
prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.


def main():
    #load models
    #load up an atari game
    #run (and save) every frame of the game

    #run every model on all frames (4*n frames))
    print('Building model...')

    torch.cuda.set_device(args.gpu)
    Z_dim = args.latent_size
    #number of updates to discriminator for every update to generator
    disc_iters = 5

    generator = model.Generator(Z_dim).cuda()
    encoder = model.Encoder(Z_dim).cuda()

    map_loc = {
            'cuda:0': 'cuda:0',
            'cuda:1': 'cuda:0',
            'cuda:2': 'cuda:0',
            'cuda:3': 'cuda:0',
            'cuda:4': 'cuda:0',
            'cuda:5': 'cuda:0',
            'cuda:6': 'cuda:0',
            'cuda:7': 'cuda:0',
            'cpu': 'cpu',
    }

    if args.enc_file == None or args.gen_file == None:
        print("Need to load models for the gen and enc")
        exit()

    encoder.load_state_dict(torch.load(args.enc_file))#, map_location=map_loc))
    generator.load_state_dict(torch.load(args.gen_file))
    print('finished loading models')

    env = gym.make(args.env_name)
    env.seed(558423)
    env.reset()
    env.reset()

    os.makedirs(args.img_dir, exist_ok=True)

    print("running the game for {} frames".format(args.game_length))
    frames = []
    frames.append(prepro(env.render(mode='rgb_array')) * 4)
    for i in range(args.game_length+1):
        #imsave(args.img_dir + "/real_" + str(i) + ".png", np.reshape(frames[-1], (80,80)) * 255)
        obs, r, done, info = env.step(env.action_space.sample())
        frames.append(prepro(obs))

    hx = Variable(torch.zeros(1, args.latent_size)).cuda()
    cx = Variable(torch.zeros(1, args.latent_size)).cuda()

    frames = Variable(torch.Tensor(np.array(frames)).cuda())
    #TODO: check if this is off by 1 compared to real frames
    print("parsing the game frames through the network")
    for i in range(len(frames)):
        #import pdb; pdb.set_trace()
        x = frames[i].view([1,1,80,80])
        hx, cx = encoder(x,(hx,cx))
        #pass hx (z) to decoder
        if i < 4: continue
        reconstructed, prev = generator(hx) 
        reconstructed_prev, _ = generator(prev) 

        #import pdb; pdb.set_trace()
        fake = np.hstack([reconstructed_prev.cpu().data.numpy()[0][0], reconstructed.cpu().data.numpy()[0][0]])
        real = np.hstack(torch.cat([frames[i-1],frames[i]]).cpu().data.numpy())
        output = np.vstack([real, fake]) * 255
        #output = np.hstack([frames[i][0].cpu().data.numpy(), reconstructed.cpu().data.numpy()[0][0]]) * 255
        
        imsave(args.img_dir + "/output" + str(i)+".png", output)




if __name__ == '__main__':
    main()
