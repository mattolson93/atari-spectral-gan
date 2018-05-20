# DCGAN-like generator and discriminator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from spectral_normalization import SpectralNorm

channels = 4
leak = 0.1


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        # 80 x 80
        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1,1)))
        # 80 x 80
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        # 40 x 40
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        # 40 x 40
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        # 20 x 20
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        # 20 x 20
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        # 10 x 10
        self.conv7 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        # 5 x 5
        self.conv8 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(0,0)))
        # 3 x 3

        self.fc = SpectralNorm(nn.Linear(3 * 3 * 512, latent_size))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        m = nn.LeakyReLU(leak)(self.conv8(m))

        return self.fc(m.view(-1, 3 * 3 * 512))


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(0,0)), # 10
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)), # 20
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=(1,1)), # 40
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),  # 80
            nn.BatchNorm2d(64),
            nn.ReLU())
        # Generate a grid to help learn absolute position
        grid = np.zeros((1, 80, 80))
        grid[:,:,::5] = 1
        self.placement_grid = Variable(torch.Tensor(grid).cuda())
        self.conv_to_rgb = nn.ConvTranspose2d(64 + 1, channels, 3, stride=1, padding=(1,1))

    def forward(self, z):
        batch_size = z.shape[0]
        z_in = z.view(-1, self.z_dim, 1, 1)
        # Run the model except for the last layer
        x1 = self.model(z_in)
        # Create a fixed pattern for orientation
        x2 = self.placement_grid.expand(batch_size, 1, 80, 80)
        # Combine them along the channels axis
        x = torch.cat((x1, x2), dim=1)
        x = nn.Tanh()(self.conv_to_rgb(x))
        return x
