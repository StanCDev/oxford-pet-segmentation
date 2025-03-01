import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from mlp import MLP
from ... import main

class ConvBlock(nn.Module):
    """
    Convolutional block that applies two consecutive convolutions and ReLUs
    """
    def __init__(self, in_ch : int, out_ch : int):
        super().__init__(ConvBlock, self)
        self.padding_mode = "edge"

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels= in_ch,
                out_channels= out_ch,
                kernel_size=(3,3),
                padding=1,
                padding_mode=self.padding_mode
            ),
            nn.ReLU()
            ,
            nn.Conv2d(
                in_channels= out_ch,
                out_channels= out_ch,
                kernel_size=(3,3),
                padding=1,
                padding_mode=self.padding_mode
            ),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Compute forward pass of convolutions

        Arguments:
            x (tensor): input batch of shape (N, in_ch, H, W)
        Returns:
            y (tensor): output batch of shape (N, out_ch, H, W)
        """
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, w : int, h : int, ch : int):
        super().__init__(UNet, self)
        self.w = w
        self.h = h
        self.ch = ch

        self.down1 = ConvBlock(ch,64*ch)
        self.max_pool1 = nn.MaxPool2d(
            kernel_size=(2,2)
        )

        self.down2 = ConvBlock(64*ch, 128*ch)
        self.max_pool2 = nn.MaxPool2d(
            kernel_size=(2,2)
        )

        self.down3 = ConvBlock(128*ch, 256*ch)
        self.max_pool3 = nn.MaxPool2d(
            kernel_size=(2,2)
        )

        self.down4 = ConvBlock(256*ch, 512*ch)
        self.max_pool4 = nn.MaxPool2d(
            kernel_size=(2,2)
        )

        self.down5 = ConvBlock(w,h,512*ch, 1024*ch)

        self.conv_trans5 = nn.ConvTranspose2d(
            in_channels= 1024 * ch,
            out_channels= 512 * ch,
            kernel_size= (2,2)
        )
        self.up5 = ConvBlock(1024*ch, 512*ch)

        self.conv_trans4 = nn.ConvTranspose2d(
            in_channels= 512 * ch,
            out_channels= 256 * ch,
            kernel_size= (2,2)
        )
        self.up4 = ConvBlock(512*ch, 256*ch)

        self.conv_trans3 = nn.ConvTranspose2d(
            in_channels= 256 * ch,
            out_channels= 128 * ch,
            kernel_size= (2,2)
        )
        self.up3 = ConvBlock(256*ch, 128*ch)

        self.conv_trans2 = nn.ConvTranspose2d(
            in_channels= 128 * ch,
            out_channels= 64 * ch,
            kernel_size= (2,2)
        )
        self.up2 = ConvBlock(128*ch, ch)

    def forward(self, x):
        x1 = self.down1(x)
        y1 = self.max_pool1(x1)

        x2 = self.down2(y1)
        y2 = self.max_pool2(x2)

        x3 = self.down3(y2)
        y3 = self.max_pool3(x3)

        x4 = self.down4(y3)
        y4 = self.max_pool4(x4)

        y5 = self.down5(y4)

        ## dim=0 means needs to have same x dimensions, dim=1 means needs to have same y dims
        x4_y5 = torch.cat((x4, self.conv_trans5(y5)),dim=1)
        y6 = self.conv_trans5(x4_y5)

        x3_y6 = torch.cat((x3,self.conv_trans4(y6)),dim=1)
        y7 = self.conv_trans4(x3_y6)

        x2_y7 = torch.cat((x2,self.conv_trans3(y7)),dim=1)
        y8 = self.conv_trans3(x2_y7)

        x1_y8 = torch.cat((x1,self.conv_trans3(y8)),dim=1)
        y9 = self.conv_trans2(x1_y8)

        return y9