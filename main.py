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


print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--latent_size', type=int, default=1000)
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

def save_real_fake(fake, real, img_dir, i, epoch):
    fake_pixels = np.hstack(fake.cpu().data.numpy()[0]) * 255.0
    imsave("{}/fake_img{}_{}.png".format(img_dir, epoch, i), fake_pixels)
    imsave("{}/real_img{}_{}.png".format(img_dir, epoch, i), real[0][0].cpu().data.numpy() * 255)
                
'''
imsave("real.png", np.hstack((data[i-4:i]).squeeze(2).cpu().data.numpy()[:,0])*255)
imsave("fake.png", np.hstack(np.reshape((reconstructed.cpu().data.numpy()[:,0]), (4, 80,80))) * 255.0 )
'''
def train(epoch, img_dir, max_batches=1):

    for batch_idx, (data, target) in enumerate(loader):
        
        if len(data.size()) != 5 or data.size()[0] != args.batch_size:
            #TODO: add debug here to find out why the data is bad
            continue
        data = Variable(data.cuda())
        #data = data.squeeze(2)
        
        # reconstruct images
        
        #import pdb; pdb.set_trace()

        hx = Variable(torch.zeros(args.batch_size, args.latent_size)).cuda()
        cx = Variable(torch.zeros(args.batch_size, args.latent_size)).cuda()

        lstm_steps = 20
        cur_loss = 0
        aac_loss = []
        ret = None
        data = torch.transpose(data, 0,1)
        for i in range(len(data)):
            (hx, cx) = encoder(data[i],(hx,cx))
            #pass hx (z) to decoder
            if i < 4: continue

            reconstructed = generator(hx) 
            #if epoch == 7:
            #    imsave("test{}.png".format(i), reconstructed.cpu().data.numpy()[0][-1] * 255)
            real = torch.cat([data[i-3],data[i-2],data[i-1],data[i]],1)
            cur_loss = torch.sum((reconstructed - real)**2 )
            aac_loss.append(cur_loss)#> .1)

            # 4995.7231
            #[torch.cuda.FloatTensor of size 1 (GPU 1)]

            if i % lstm_steps == 0:
                print("Autoencoder loss: {:.3f}".format(cur_loss.data[0]))
                #print("hx: {:.3f}".format(torch.sum(hx[0].cpu().data)))
                #ret = (reconstructed[-1][0] , data[i][0][0])
                save_real_fake(reconstructed, data[i], img_dir, i, epoch)
                optim_enc.zero_grad()
                optim_gen.zero_grad()
                #probably do this 
                #torch.nn.utils.clip_grad_norm(model.parameters(), 40)
                full_loss = torch.sum(torch.cat(aac_loss))#[0]
                #import pdb; pdb.set_trace()
                full_loss.backward()

                #for param in generator.parameters():
                #    param.grad.data.clamp_(-0.1, 0.1)
                #for param in encoder.parameters():
                #    param.grad.data.clamp_(-0.1, 0.1)
                aac_loss = []

                optim_enc.step()
                optim_gen.step()

                hx = Variable(hx.data).cuda()
                cx = Variable(cx.data).cuda()

        scheduler_e.step()
        scheduler_g.step()
        break

        #if batch_idx % 2 == 0:
            #print('disc loss', disc_loss.data[0], 'gen loss', gen_loss.data[0])


def main():
    print('creating directories')
    img_dir = "./imgs_gpu" + str(args.gpu) + "_" + str(args.latent_size)
    args.checkpoint_dir +=   str(args.gpu) + "_" + str(args.latent_size)
    os.makedirs(args.checkpoint_dir , exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        epoch += args.starting_epoch
        print('starting epoch {}'.format(epoch))
        train(epoch, img_dir)
        #print (test_img.shape)
        #imsave("{}/real_img{}.png".format(img_dir, epoch), np.reshape(real.cpu().data.numpy(), (80,80)) * 255)
        #pixels = np.reshape(fake.cpu().data.numpy(), (80,80)) * 255.0
        #imutil.show(pixels, filename="{}/fake_img{}.png".format(img_dir, epoch))
       
        #torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(epoch)))
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, 'gen_{}'.format(args.epochs + args.starting_epoch)))
    torch.save(encoder.state_dict(), os.path.join(args.checkpoint_dir, 'enc_{}'.format(args.epochs + args.starting_epoch)))


if __name__ == '__main__':
    main()
