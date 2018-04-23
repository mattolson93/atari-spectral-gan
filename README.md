# Atari Spectral Normalization GAN

A fork of https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

Requires `gym[atari]` and `ffmpeg`.
Takes an additional `--env` parameter to specify the Atari environment.

# SN-GAN (spectral normalization GAN) in PyTorch
Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida

ICLR 2018 preprint:
https://openreview.net/forum?id=B1QRgziT-

## Implementation Details
This code implements both DCGAN-like and ResNet GAN architectures. In addition, training with standard, Wasserstein, and hinge losses is possible.

To get ResNet working, initialization (Xavier/Glorot) turned out to be very important.

## Training
Train ResNet generator and discriminator with hinge loss: `python main.py --model resnet --loss hinge`

Train ResNet generator and discriminator with wasserstein loss: `python main.py --model resnet --loss wasserstein`

Train DCGAN generator and discriminator with cross-entropy loss: `python main.py --model dcgan --loss bce`
