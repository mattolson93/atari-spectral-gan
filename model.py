# DCGAN-like generator and discriminator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import torch.nn.init as weight_init


channels = 1
leak = 0.1


class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(channels, 64, 8, 2, (0, 1))
        self.conv2 = nn.Conv2d(64, 128, 6, 2, (1, 1))
        self.conv3 = nn.Conv2d(128, 128, 6, 2, (1, 1))
        self.conv4 = nn.Conv2d(128, 128, 4, 2, (0, 0))

        self.hidden_units = 128 * 2 * 3
        # 3 x 3
        self.fc = nn.Linear(self.hidden_units, latent_size)
        self.lstm = nn.LSTMCell(latent_size, latent_size)

        self.init_weights()

    #
    #hx, cx = self.lstm(x.view(-1, 32 * 5 * 5), (hx, cx))
    #return self.critic_linear(hx), self.actor_linear(hx), (hx, cx)
    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform(layer.weight.data)
                layer.bias.data.fill_(0)
        nn.init.uniform(self.fc.weight.data, -1, 1)
        

    def forward(self, x, memory):
        (hx, cx) = memory
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view((-1, self.hidden_units))
        x = F.relu(self.fc(x))

        hx, cx = self.lstm(x, (hx, cx))
        return  (hx, cx)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.hidden_units = 128 * 2 * 3 * 4
        self.prev = nn.Linear(z_dim, z_dim)

        self.fc = nn.Linear(z_dim, self.hidden_units)
        self.deconv1 = nn.ConvTranspose2d(self.hidden_units, 512, 4, stride=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(0,0)) # 10
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)) #20
        self.deconv4 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=(1,1)) #40
        self.deconv5 = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=(1,1))

        self.init_weights()

        '''self.deconv1 = nn.ConvTranspose2d(128, 128, 4, 2)
        self.deconv2 = nn.ConvTranspose2d(128, 128, 6, 2, (1, 1))
        self.deconv3 = nn.ConvTranspose2d(128, 128, 6, 2, (1, 1))
        self.deconv4 = nn.ConvTranspose2d(128, 1, 8, 2, (0, 1)) #[1, 1, 66, 80]

        self.test1 = nn.ConvTranspose2d(128, 1, 6, 2, (0, 1)) # [1, 1, 64, 78]
        self.test2 = nn.ConvTranspose2d(128, 4, 6, 2, (0, 1)) # [1, 4, 64, 78]
        self.test3 = nn.ConvTranspose2d(128, 4, 8, 2, (0, 1)) # [1, 4, 66, 80]
        
        self.test4 = nn.ConvTranspose2d(128, 4, 8, 2, (1, 1)) # [1, 4, 64, 80]
        self.test5 = nn.ConvTranspose2d(128, 4, 8, 2, (1, 0)) # [1, 4, 64, 82]
        self.test6 = nn.ConvTranspose2d(128, 4, 8, 2, (0, 0)) # [1, 4, 66, 82]

        self.test7 = nn.ConvTranspose2d(128, 4, 16, 2, (1, 1)) '''

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform(layer.weight.data)
                layer.bias.data.fill_(0)
        nn.init.uniform(self.fc.weight.data, -1, 1)


    def forward(self, x):
        prev_z = self.prev(x)
        x = F.relu(self.fc(x))
        x = x.view((-1, self.hidden_units, 1, 1))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
        x = self.deconv5(x)

        return x, prev_z
