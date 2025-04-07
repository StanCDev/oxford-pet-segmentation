import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoProcessor, CLIPTextModel, CLIPVisionModelWithProjection
import torch
import torch.nn as nn
import math

# Using CLIP Model 'clip-vit-base-patch32' from Hugging face
model_id = "openai/clip-vit-base-patch32"

class CLIP_Encoder(nn.Module):
    def __init__(self, model_id = model_id):
        super(CLIP_Encoder, self).__init__()
        # Load the CLIP model (probably do not need this for segmentation???)
        self.model = CLIPModel.from_pretrained(model_id)
        # Freeze the parameters of the model as we are not training the encoder
        for p in self.model.parameters():
            p.requires_grad = False
        # preprocessing the images (resizing, normalizing, etc.)
        self.preprocess = AutoProcessor.from_pretrained(model_id)
        # Load the tokenizer so the text prompt can be tokenized
        self.tokenizer = AutoTokenizer.from_pretrained(model_id) 
        # Loads the text encoder for the prompt
        self.text_encoder = CLIPTextModel.from_pretrained(model_id)

        # Freeze the parameters of the text model as we are not training the encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        
        # Load the image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, output_hidden_states = True)
        
        # Freeze the parameters of the text model as we are not training the encoder
        for p in self.image_encoder.parameters():
            p.requires_grad = False
    
    # Function to tokenize and then encode the text prompts
    def textForward(self, prompt):
        tokenized = self.tokenizer([prompt], padding=True, return_tensors='pt')
        prompt_embedding = self.text_encoder(**tokenized)

        return prompt_embedding

    # Function to preprocess and then encode the image features
    def visualForward(self, image):
        preprocessed = self.preprocess(images=image, return_tensors='pt')
        image_embedding = self.image_encoder(**preprocessed)

        return image_embedding

        
    def forward(self, image, prompt, layers = [8, 9, 10, 11]):
        # runs text and image through the frozen encoder to get text and image embeddings
        text_op = self.textForward(prompt)
        image_op_temp = self.visualForward(image)
        text_encoding = text_op[1]
        image_encoding = image_op_temp[0]
        # extracts the hidden states from the image encoder 
        mid_layers = []
        for i in range(len(layers)):
            mid_layers.append(image_op_temp['hidden_states'][layers[i]])
        
        return text_encoding, image_encoding, mid_layers

class CLIP_Decoder(nn.Module):
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
        super(CLIP_Decoder, self).__init__()
        
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
            ###CLIP 2
            # nn.ReLU(),
            # nn.ConvTranspose2d(reduce_dim // 2, reduce_dim // 4, kernel_size=4, stride=4),
            # nn.ReLU(),
            # nn.ConvTranspose2d(reduce_dim // 4, 3, kernel_size=4, stride=2, padding=1),
            ###CLIP 1
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim // 2, 3, kernel_size=8, stride=8),
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
    

class Clip(nn.Module):
    """
    Includes encoder and decoder methods for an autoencoder.
    """
    def __init__(self, model_id : str = model_id):
        super(Clip, self).__init__()
        self.encoder = CLIP_Encoder(model_id)
        self.decoder = CLIP_Decoder()

    def forward(self, image, prompt, layers = [8, 9, 10, 11]):
        encoder_out = self.encoder(image,prompt,layers)
        decoder_out = self.decoder(encoder_out)
        return decoder_out