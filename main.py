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
import model_resnet
import model

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import imutil
from scipy.misc import imsave


print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--latent_size', type=int, default=10)
parser.add_argument('--model', type=str, default='dcgan')
parser.add_argument('--gpu', type=int, default=7)
parser.add_argument('--env_name', type=str, default='Pong-v0')
parser.add_argument('--enc_file', type=str, default=None)
parser.add_argument('--gen_file', type=str, default=None)
parser.add_argument('--starting_epoch', type=int, default=0)


args = parser.parse_args()


from atari_data import AtariDataloader
print('Initializing OpenAI environment...')
loader = AtariDataloader(args.env_name, batch_size=args.batch_size)
print('Environment initialized')


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

if args.enc_file != None:
    encoder.load_state_dict(torch.load(args.enc_file))#, map_location=map_loc))
if args.gen_file != None:
    generator.load_state_dict(torch.load(args.gen_file))#, map_location=map_loc))

#encoder.load_state_dict(torch.load("./checkpoints/enc_49"))
#generator.load_state_dict(torch.load("./checkpoints/gen_49"))

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0,0.9))
optim_enc = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=args.lr, betas=(0.0,0.9))

# use an exponentially decaying learning rate
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_e = optim.lr_scheduler.ExponentialLR(optim_enc, gamma=0.99)
print('finished building model')

def sample_z(batch_size, z_dim):
    # Normal Distribution
    z = torch.randn(batch_size, z_dim)
    return Variable(z.cuda())

def train(epoch, max_batches=100):
    #import pdb; pdb.set_trace()
    for batch_idx, (data, target) in enumerate(loader):
        
        if len(data.size()) != 5 or data.size()[0] != args.batch_size:
            #TODO: add debug here to find out why the data is bad
            continue
        data = Variable(data.cuda())
        
        data = data.squeeze(2)
        
        # reconstruct images
        optim_enc.zero_grad()
        optim_gen.zero_grad()
        #import pdb; pdb.set_trace()
        reconstructed = generator(encoder(data))

        aac_loss = torch.sum((reconstructed - data)**2 )#> .1)
        aac_loss.backward()

        optim_enc.step()
        optim_gen.step()

        if batch_idx % 10 == 0:
            #print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])
            print("Autoencoder loss: {:.3f}".format(aac_loss.data[0]))
        if batch_idx >= max_batches:
            print('Training completed {} batches, ending epoch'.format(max_batches))
            break
    scheduler_e.step()
    scheduler_g.step()
    


fixed_z = sample_z(args.batch_size, Z_dim)
def evaluate(epoch):
    samples = generator(fixed_z).cpu().data.numpy()[:64]
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
    scores = []
    for i in range(10):
        scores.append(evaluate_fit(epoch, i))
    avg_mse = np.array(scores).mean()
    print("Epoch {} Avg Encoding MSE:{:.4f}".format(epoch, avg_mse))

    smoothness_scores = []
    for i in range(3):
        distance = evaluate_smoothness(epoch, i)
        smoothness_scores.append(distance)
    avg_smoothness = np.array(smoothness_scores).mean()
    print('Epoch {} smoothness measure: {:.6f}'.format(epoch, avg_smoothness))

    print('Epoch {} MSE {:.4f} smoothness {:.6f}'.format(
        epoch, avg_mse, avg_smoothness))


def evaluate_fit(epoch, idx=0):
    # Get a random Atari frame, try to fit it by gradient descent
    frames, _ = next(loader)
    frame = frames[idx]
    frame = Variable(frame.cuda())

    # TODO: Instead of standard gradient descent, this should be
    #  projected gradient descent on eg. the unit sphere if the
    #  behavior of sample_z is changed
    z = Variable(torch.randn(1, Z_dim).cuda(), requires_grad=True)

    speed = .01
    for _ in range(100):
        encoded = generator(z)[0]
        mse = (frame - encoded) ** 2
        loss = mse.sum()
        df_dz = autograd.grad(loss, z, loss)[0]
        z = z - speed * df_dz
        speed *= .99  # annealing schedule

    if idx == 0:
        filename = 'fit_{:03d}_{:04d}.png'.format(epoch, idx)
        comparison = torch.cat((frame.expand(1,-1,-1,-1), encoded.expand((1, -1, -1, -1))))
        imutil.show(comparison, filename=filename)
    return loss.data[0]


def evaluate_smoothness(epoch, idx=0):
    # Get two consecutive, similar Atari frames
    # How far apart are their representations?
    first_frame = next(loader)[0][idx]
    second_frame = next(loader)[0][idx]

    f0 = Variable(first_frame.cuda())
    f1 = Variable(second_frame.cuda())
    distance = (encode(f0) - encode(f1)) ** 2
    return distance.mean().data[0]


def encode(frame):
    speed = .01
    z = Variable(torch.randn(1, Z_dim).cuda(), requires_grad=True)
    for _ in range(100):
        encoded = generator(z)[0]
        mse = (frame - encoded) ** 2
        loss = mse.sum()
        df_dz = autograd.grad(loss, z, loss)[0]
        z = z - speed * df_dz
        speed *= .99  # annealing schedule
    return encoded



fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
fixed_zprime = Variable(torch.randn(args.batch_size, Z_dim).cuda())
def make_video(output_video_name):
    v = imutil.VideoMaker(output_video_name)
    for i in range(400):
        theta = abs(i - 200) / 200.
        z = theta * fixed_z + (1 - theta) * fixed_zprime
        #z = z[:args.batch_size]
        samples = generator(z).cpu().data.numpy()
        pixels = samples.transpose((0,2,3,1)) * 0.5 + 0.5
        v.write_frame(pixels)
    v.finish()


def main():
    print('creating checkpoint directory')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        epoch += args.starting_epoch
        print('starting epoch {}'.format(epoch))
        train(epoch)
        #import pdb; pdb.set_trace()
        test_img = loader.get_img()
        #print (test_img.shape)
        imsave("./imgs/real_img"+str(epoch)+".png", np.reshape(test_img[0], (80,80)) * 255)
        list = []
        list.append(test_img)
        z = encoder(Variable(torch.Tensor(np.array(list)).cuda()).squeeze(2))
        samples = generator(z).cpu().data.numpy()[0][0]
        pixels = np.reshape(samples, (80,80)) * 255.0
        imutil.show(pixels, filename="./imgs/fake_img"+str(epoch)+".png")
       
        #torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
        torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(epoch)))
        torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, 'enc_{}'.format(epoch)))


if __name__ == '__main__':
    main()
