import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


class DownBlock(nn.Module):
    def __init__(self, in_ch : int, out_ch : int):
        super(DownBlock, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch : int, out_ch : int):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.up(x)
    

class Encoder(nn.Module):
    def __init__(self, w : int, h : int, in_channels : int, out_channels : int, ch_mult : int = 8, latent_space_dim: int = 32 * 32 * 3):
        super(Encoder, self).__init__()
        self.w = w
        self.h = h

        factor_in = in_channels * ch_mult
        self.encoder = nn.Sequential(
            DownBlock(in_channels, factor_in),
            DownBlock(factor_in ,2 * factor_in),
            DownBlock(2 * factor_in, 4 * factor_in),
            DownBlock(4 * factor_in, 8 * factor_in),
            nn.Flatten(),
            nn.Linear(8 * factor_in * (w//(2 ** 4)) *(h//(2**4)), 16 * factor_in),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, w : int, h : int, in_channels : int, out_channels : int, ch_mult : int = 8, latent_space_dim: int = 32 * 32 * 3):
        super(Decoder, self).__init__()
        self.w = w
        self.h = h

        factor_in = in_channels * ch_mult
        self.factor_in = factor_in

        factor_out = out_channels * ch_mult

        self.linear =  nn.Sequential(
            nn.Linear(16 * factor_in, 8 * factor_in*(w//(2 ** 4))*(h//(2**4))),
            nn.LeakyReLU(),
            )
        self.decoder = nn.Sequential(
            UpBlock(8 * factor_in , 4 * factor_in),
            UpBlock(4 * factor_in , 2 * factor_in),
            UpBlock(2 * factor_in , factor_in),
            UpBlock(factor_in, factor_out),
            nn.Conv2d(
                in_channels= factor_out,
                out_channels= out_channels,
                kernel_size=1
            )
        )

    def forward(self, x):

        B = x.shape[0]
        C = 8 * self.factor_in

        x = self.linear(x)
        x = torch.reshape(x,(B, C, self.h//(2 ** 4), self.w//(2 ** 4)))
        return self.decoder(x)


class AutoEncoder(nn.Module):
    """
    Includes encoder and decoder methods for an autoencoder.
    """
    def __init__(self, w : int, h : int, in_channels : int, out_channels : int, ch_mult : int = 8, latent_space_dim : int = 32 * 32 * 3):
        super(AutoEncoder, self).__init__()

        factor_in = in_channels * ch_mult
        factor_out = out_channels * ch_mult

        self.encoder = Encoder(w = w, h = h, in_channels=in_channels, out_channels=out_channels, ch_mult=ch_mult, latent_space_dim=latent_space_dim)
        self.decoder = Decoder(w = w, h = h, in_channels=in_channels, out_channels=out_channels, ch_mult=ch_mult, latent_space_dim=latent_space_dim)

    def forward(self, x):
        # print(f"x shape before encoding = {x.shape}")
        z = self.encoder(x)
        # print(f"x shape after encoding = {z.shape}")
        y = self.decoder(z)
        return y