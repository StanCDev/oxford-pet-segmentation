import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class ConvBlock(nn.Module):
    """
    Convolutional block that applies two consecutive convolutions and ReLUs
    """
    def __init__(self, in_ch : int, out_ch : int):
        super(ConvBlock, self).__init__()
        self.padding_mode = 'replicate'

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
    def __init__(self, w : int, h : int, ch : int, ch_mult : int = 16):
        super(UNet, self).__init__()
        self.w = w
        self.h = h
        self.ch = ch

        factor = ch_mult*ch

        self.conv1 = ConvBlock(ch, factor)
        self.max_pool1 = nn.MaxPool2d(
            kernel_size=(2,2)
        )

        self.conv2 = ConvBlock(factor, 2 * factor)
        self.max_pool2 = nn.MaxPool2d(
            kernel_size=(2,2)
        )

        self.conv3 = ConvBlock(2 * factor, 4 * factor)
        self.max_pool3 = nn.MaxPool2d(
            kernel_size=(2,2)
        )

        self.conv4 = ConvBlock(4*factor, 8*factor)
        self.max_pool4 = nn.MaxPool2d(
            kernel_size=(2,2)
        )

        self.conv5 = ConvBlock(8*factor, 16*factor)

        self.conv_trans5 = nn.ConvTranspose2d(
            in_channels= 16 * factor,
            out_channels= 8 * factor,
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        self.conv6 = ConvBlock(16 * factor, 8 * factor)

        self.conv_trans4 = nn.ConvTranspose2d(
            in_channels= 8 * factor,
            out_channels= 4 * factor,
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        self.conv7 = ConvBlock(8 * factor, 4*factor)

        self.conv_trans3 = nn.ConvTranspose2d(
            in_channels= 4 * factor,
            out_channels= 2 * factor,
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        self.conv8 = ConvBlock(4 * factor, 2 * factor)

        self.conv_trans2 = nn.ConvTranspose2d(
            in_channels= 2 * factor,
            out_channels= factor,
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        self.conv9 = ConvBlock(2 * factor, factor)
        self.conv10 = ConvBlock(factor, ch)

    def forward(self, x):
        """
        x has shape as follows: (N, ch, w, h)
        - N = batch size
        """
        x1 = self.conv1(x)
        y1 = self.max_pool1(x1)

        x2 = self.conv2(y1)
        y2 = self.max_pool2(x2)

        x3 = self.conv3(y2)
        y3 = self.max_pool3(x3)

        x4 = self.conv4(y3)
        y4 = self.max_pool4(x4)

        y5 = self.conv5(y4)

        ## dim=0 means needs to have diff N dimensions but all other dims the same. dim=1 means ... ch ...
        ###print(f"Here is the shape of x4 and y5: {x4.shape} , {y5.shape}")
        y5_up = self.conv_trans5(y5)
        ###print(f"Here is the shape of x4 and y5_up: {x4.shape} , {y5_up.shape}")
        x4_y5 = torch.cat((x4, y5_up),dim=1)
        y6 = self.conv6(x4_y5)

        x3_y6 = torch.cat((x3,self.conv_trans4(y6)),dim=1)
        y7 = self.conv7(x3_y6)

        x2_y7 = torch.cat((x2,self.conv_trans3(y7)),dim=1)
        y8 = self.conv8(x2_y7)

        x1_y8 = torch.cat((x1,self.conv_trans2(y8)),dim=1)
        y9 = self.conv9(x1_y8)

        y10 = self.conv10(y9)

        return y10