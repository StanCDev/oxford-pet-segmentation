import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader




class AutoEncoder(nn.Module):
    """
    Includes encoder and decoder methods for an autoencoder
    """
    def __init__(self):
        super(self).__init__()

        self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                     nn.ReLU(True),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Flatten(),
                                     nn.Linear(256*16*16, 1024),
                                     )
        self.decoder = nn.Sequential(nn.Linear(1024, 256*16*16),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.ReLU(True),
                                     nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                                     nn.Sigmoid()
                                     )
    def forward(self, x):
        x = self.encoder(x)     
        x = self.decoder(x)
        return x

''' Once the decoder has been trained to match the encoder, the encoder can be used to extract features from the input data. 
This can the be used to train a segmentation model using the extracted features and masks as the target. '''