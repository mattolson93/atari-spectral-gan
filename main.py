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

from collections import deque

print('Parsing arguments')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
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
    #import pdb; pdb.set_trace()
    fake_pixels = fake[0][0].cpu().data.numpy()* 255.0
    imsave("{}/fake_img{}_{}.png".format(img_dir, epoch, i), fake_pixels)
    imsave("{}/real_img{}_{}.png".format(img_dir, epoch, i), real[0][0].cpu().data.numpy() * 255)
                
'''
imsave("real.png", np.hstack((data[i-4:i]).squeeze(2).cpu().data.numpy()[:,0])*255)
imsave("fake.png", np.hstack(np.reshape((reconstructed.cpu().data.numpy()[:,0]), (4, 80,80))) * 255.0 )
'''
def train(epoch, img_dir, max_batches=1):

    for batch_idx, (data, target) in enumerate(loader):
        #print("also here")
        #import pdb; pdb.set_trace()
        if len(data.size()) != 5 or data.size()[0] != args.batch_size:
            #TODO: add debug here to find out why the data is bad
            continue
        data = Variable(data.cuda())
        #data = data.squeeze(2)
        
        # reconstruct images
        
        #import pdb; pdb.set_trace()

        hx = Variable(torch.zeros(args.batch_size, args.latent_size)).cuda()
        prev_hx = Variable(torch.zeros(args.batch_size, args.latent_size)).cuda()
        cx = Variable(torch.zeros(args.batch_size, args.latent_size)).cuda()

        lstm_steps = 10

        hxs = [hx]
        main_losses = []
        k=0
        ret = None
        data = torch.transpose(data, 0,1)
        for i in range(len(data)):
            prev_hx = hx.clone()
            (hx, cx) = encoder(data[i],(hx,cx))
            hxs.append(hx.clone())
            if i==0:continue

            reconstructed, predicted_prev_z = generator(hx)
            recon_loss = torch.sum((reconstructed - data[i])**2 )
            pred_loss = torch.sum((prev_hx - predicted_prev_z)**2 )

            #optim_enc.zero_grad()
            #optim_gen.zero_grad()
            if epoch > 10000:
                main_loss = torch.sum((recon_loss)) + torch.sum((pred_loss))
                main_losses.append(main_loss)
            else:
                optim_enc.zero_grad()
                optim_gen.zero_grad()
                main_loss = torch.sum((recon_loss)) + torch.sum((pred_loss))
                main_loss.backward()
                optim_enc.step()
                optim_gen.step()

                hx = Variable(hx.data).cuda()
                cx = Variable(cx.data).cuda()
                if i % 20 == 0:
                    print("Autoencoder loss: {:.3f}, pred_loss: {:.3f}".format(recon_loss.data[0], pred_loss.data[0]))
                

            if k + lstm_steps <= i and epoch > 100000:
                k = i
                #lstm_steps = random.randint(2, 12)
                #import pdb; pdb.set_trace()
                recon_losses = []
                pred_losses = []
                hxs = list(reversed(hxs))
                for j in range(len(hxs)-1):
                    _, predicted_prev_z = generator(predicted_prev_z)

                    #save_real_fake(reconstructed, data[i-j], img_dir, i-j, epoch)

                    #recon_losses.append(torch.sum((reconstructed - data[i-j])**2 ) )
                    pred_losses.append(torch.sum(torch.abs(predicted_prev_z - hxs[j+1])))

                #if len(recon_losses) == 0:
                #    import pdb; pdb.set_trace()
                    
                #print("hx: {:.3f}".format(torch.sum(hx[0].cpu().data)))
                #ret = (reconstructed[-1][0] , data[i][0][0])
                #save_real_fake(reconstructed, data[i-9], img_dir, i, epoch)
                optim_enc.zero_grad()
                optim_gen.zero_grad()
                #probably do this 
                #torch.nn.utils.clip_grad_norm(model.parameters(), 40)
                pred_l = (torch.sum(torch.cat(pred_losses)) / 10)
                full_loss = pred_l + torch.sum(torch.cat(main_losses))
                print("Autoencoder loss: {:.3f}, pred_1_loss: {:.3f}, pred_long_loss: {:.3f}".format(recon_loss.data[0], pred_loss.data[0],pred_l.data[0] ))

                #import pdb; pdb.set_trace()
                full_loss.backward()
                main_losses = []
                hxs = []

                #this needs to go before non-linearity of each gate of LSTM
                #for param in generator.parameters():
                #    param.grad.data.clamp_(-0.1, 0.1)
                #for param in encoder.parameters():
                #    param.grad.data.clamp_(-0.1, 0.1)

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
