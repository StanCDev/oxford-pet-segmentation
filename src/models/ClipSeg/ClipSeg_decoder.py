import torch
import torch.nn as nn
import math

class Decoder(nn.Module):
    # Decoder for the transformer model. REduces the dimensionality of the features and applies FiLM layers
    # to the activations of the encoder. The activations are then passed through a series of MHA blocks and
    # upsampled to the original image size.
    # Args:
    # reduce_dim: The dimensionality to reduce the features to.
    # cond_layer: The layer to apply FiLM conditioning to. If None, FiLM conditioning is applied to all layers.
    # extract_layers: The layers to extract features from the encoder.
    # mha_heads: The number of heads for the MHA blocks.
    # Returns: The reconstructed image.
    
    def __init__(self, reduce_dim=128, cond_layer = None,
                 extract_layers=[8, 9, 10, 11], mha_heads=4):
        super(Decoder, self).__init__()
        
        self.cond_layer = cond_layer
        # FiLM layers used to apply feature-wise transformations to the activations from the encoder (mul is for scaling and add is used for shifting)
        self.film_mul = nn.Linear(512, reduce_dim) 
        self.film_add = nn.Linear(512, reduce_dim)
        # number of layers in the encoder
        self.depth = len(extract_layers)
        #
        self.reduce_blocks = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(self.depth)])
        # MHA blocks used to process the activations
        self.mha_blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim,
                                                              nhead=mha_heads) for _ in range(self.depth)])  
        self.trans_conv = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=8, stride=8)
        )
        
    def forward(self, encoder_out):

        batch_size = encoder_out[0].shape[0]
        
        text_cond, image_cond, activations = encoder_out
        combined_cond = torch.mul(text_cond, image_cond)
        
        a = None

        for i, (skip, mha, reduce) in enumerate(zip(activations, self.mha_blocks, self.reduce_blocks)):
            if a is None:
                a = reduce(skip)
            else:
                a = a + reduce(skip)
            if (self.cond_layer==None or i==(self.cond_layer-1)):
                a = self.film_mul(combined_cond)*a + self.film_add(combined_cond)
            a = mha(a)

        # Ignoring the CLS token
        a = a[:, 1:, :]  
        # Changing a -> (batch size, features, tokens)
        a = a.permute(0, 2, 1)
        size = int(math.sqrt(a.shape[2]))
        a = a.view(batch_size, a.shape[1], size, size)
        a = self.trans_conv(a)
            
        return a