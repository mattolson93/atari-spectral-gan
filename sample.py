import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
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


print('running argparse stuff')
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
parser.add_argument('--image_dir', type=str, default='input/')
parser.add_argument('--video_name', type=str, default='sample')
parser.add_argument('--model', type=str, default='resnet')

args = parser.parse_args()

print('building datasets.ImageFolder')
dataset = datasets.ImageFolder(root=args.image_dir,
   transform=transforms.Compose([
       transforms.Resize((80,80)),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ]))
print('building DataLoader...')
loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
print('finished building DataLoader')


print('building model...')
Z_dim = 4

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training
if args.model == 'resnet':
    discriminator = model_resnet.Discriminator().cuda()
    generator = model_resnet.Generator(Z_dim).cuda()
else:
    discriminator = model.Discriminator().cuda()
    generator = model.Generator(Z_dim).cuda()

print('Loading model...')
discriminator.load_state_dict(torch.load('checkpoints/disc_34'))
generator.load_state_dict(torch.load('checkpoints/gen_34'))
print('Loaded model')

output_video_name = args.video_name
output_count = 1
fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
fixed_zprime = Variable(torch.randn(args.batch_size, Z_dim).cuda())
def evaluate():
    v = imutil.VideoMaker(output_video_name)
    for i in range(400):
        theta = abs(i - 200) / 200.
        z = theta * fixed_z + (1 - theta) * fixed_zprime
        print(z.cpu().data)
        samples = generator(z[output_count]).cpu().data.numpy()
        samples = samples.transpose((0,2,3,1))
        v.write_frame(samples)
    v.finish()


def main():
    print('creating checkpoint directory')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    evaluate()


if __name__ == '__main__':
    main()
