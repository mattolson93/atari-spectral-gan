# DCGAN-like generator and discriminator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from spectral_normalization import SpectralNorm

channels = 1
leak = 0.1

CUDA = True

class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r
'''
class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, 8)
        self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)'''

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.training = True
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

        self.fc_mu = SpectralNorm(nn.Linear(3 * 3 * 512, latent_size))
        self.fc_lvar = SpectralNorm(nn.Linear(3 * 3 * 512, latent_size))

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

        mu, logvar = self.fc_mu(m.view(-1, 3 * 3 * 512)), self.fc_lvar(m.view(-1, 3 * 3 * 512))
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            std = logvar.mul(0.5).exp_()  # type: Variable
            # - std.data is the [128,ZDIMS] tensor that is wrapped by std
            # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
            #   and stddev 1 normal distribution that is 128 samples
            #   of random ZDIMS-float vectors
            eps = Variable(std.data.new(std.size()).normal_())
            # - sample from a normal distribution with standard
            #   deviation = std and mean = mu by multiplying mean 0
            #   stddev 1 sample with desired std and mu, see
            #   https://stats.stackexchange.com/a/16338
            # - so we have 128 sets (the batch) of random ZDIMS-float
            #   vectors sampled from normal distribution with learned
            #   std and mu for the current input
            return eps.mul(std).add_(mu)

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu



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
        self.placement_grid = Variable(torch.Tensor(grid)).cuda()
        self.conv_to_rgb = nn.ConvTranspose2d(64 + 1, channels, 3, stride=1, padding=(1,1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        batch_size = z.shape[0]
        z_in = z.view(-1, self.z_dim, 1, 1)
        # Run the model except for the last layer
        x1 = self.model(z_in)
        # Create a fixed pattern for orientation
        x2 = self.placement_grid.expand(batch_size, 1, 80, 80)
        # Combine them along the channels axis
        x = torch.cat((x1, x2), dim=1)
        x = self.sigmoid(self.conv_to_rgb(x))
        return x

# What is w_g supposed to be?
w_g = 3
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

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

        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

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

        return self.fc(m.view(-1,w_g * w_g * 512))

