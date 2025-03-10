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
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.down(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch : int, out_ch : int):
        super(UpBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        return x


class AutoEncoder(nn.Module):
    """
    Includes encoder and decoder methods for an autoencoder
    """
    def __init__(self, w : int, h : int, in_channels : int, out_channels : int):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            DownBlock(in_channels, 32),
            DownBlock(32,64),
            DownBlock(64,128),
            DownBlock(128,256),
            nn.Flatten(),
            nn.Linear(256*(w//(2 ** 4))*(h//(2**4)), 1024),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256*16*16),
            nn.ReLU(True),
            UpBlock(256,128),
            UpBlock(128,64),
            UpBlock(64,32),
            UpBlock(32,1),
            nn.Conv2d(
                in_channels= 1,
                out_channels= out_channels,
                kernel_size=1
            )
        )

    def forward(self, x):
        x = self.encoder(x)     
        x = self.decoder(x)
        return x