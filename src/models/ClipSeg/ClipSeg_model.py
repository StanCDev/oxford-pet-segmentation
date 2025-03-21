#Importing Clip model
#! conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
#! pip install ftfy regex tqdm
#! pip install git+https://github.com/openai/CLIP.git

import torch
import math
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import CLIPModel, AutoTokenizer, AutoProcessor, CLIPTextModel, CLIPVisionModelWithProjection

from torch.utils.data import Dataset, DataLoader
import time
import torch.optim.lr_scheduler as lr_scheduler

# Using CLIP Model 'clip-vit-base-patch32' from Hugging face
model_id = "openai/clip-vit-base-patch32"

class Encoder(nn.Module):
    def __init__(self, model_id = model_id):
        super(Encoder, self).__init__()

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
     
    def textForward(self, prompt):
        '''
        Function to tokenize and then encode the text prompts
    Args:
    -----
        prompt (string): text prompt for encoding.
    
    Returns:
    --------
        Embedded text prompt.
    '''
        tokenized = self.tokenizer([prompt], padding=True, return_tensors='pt')
        prompt_embedding = self.text_encoder(**tokenized)
        return prompt_embedding

    def visualForward(self, image):
        '''
        Function to preprocess and then encode the image features
        Args:
        -----
        image (PIL image???): image to be encoded.
    
        Returns:
        --------
        Embedded embedded image.
        '''        
        preprocessed = self.preprocess(images=image, return_tensors='pt')
        image_embedding = self.image_encoder(**preprocessed)

        return image_embedding

        
    def forward(self, image, prompt, layers = [8, 9, 10, 11]):
        '''
        Runs text and image through the frozen encoder to get text and image embeddings and extracts image features from hidden layers specified
        Args:
        -----
        image (PIL image???): image to be encoded.
        prompt (string): text prompt for encoding.
        layers (list): list of hidden layers to extract features
    
        Returns:
        --------
        text_encoding (tensor): text embedding
        image_encoding (tensor): image embedding
        mid_layers (list): hidden layers from image encoder
        '''            
        text_op = self.textForward(prompt)
        image_op_temp = self.visualForward(image)

        # Extract text encoding 
        text_encoding = text_op[1]

        # Extract image encoding in the final layer
        image_encoding = image_op_temp[0]

        # Extracts the hidden states from the image encoder at the specified layers.  image_op_temp['hidden_states'] contains all of the hidden states.
        mid_layers = []
        for i in range(len(layers)):
            mid_layers.append(image_op_temp['hidden_states'][layers[i]])
        
        return text_encoding, image_encoding, mid_layers
    
class Decoder(nn.Module):
    
    def __init__(self, reduce_dim=128, cond_layer = None,
                 extract_layers=[8, 9, 10, 11], mha_heads=4):
        super(Decoder, self).__init__()
        ''' Attributes:
        reduce_dim: Reduces the dimnsionality from 768 in the encoder to reduce_dim in the decoder.
        cond_layer: The layer to apply FiLM conditioning to. If None, it is applied to all layers.
        extract_layers: The layers to extract features from the encoder.
        mha_heads: The number of heads for the Multi Head Attention (MHA) blocks.
        '''        
        self.cond_layer = cond_layer
        # FiLM layers used to apply feature-wise transformations to the activations from the encoder (mul is for scaling and add is used for shifting)
        self.film_mul = nn.Linear(512, reduce_dim) 
        self.film_add = nn.Linear(512, reduce_dim)
        # number of layers in the encoder
        self.depth = len(extract_layers)
        #
        self.reduce_blocks = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(self.depth)])
        # Multi Head Attention (MHA) blocks used to process the activations
        self.mha_blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim,
                                                              nhead=mha_heads) for _ in range(self.depth)])  
        # Transpose convolutional layers to upsample the image
        self.trans_conv = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=8, stride=8)
        )
        
    def forward(self, encoder_out):
        '''
        Function to preprocess and then encode the image features
        Args:
        -----
        encoder_out (tuple): output from the encoder containing text embedding, image embeddings and feature maps from the specified encoder layers.
    
        Returns:
        --------
        Reconstructed segmented image.
        ''' 
        batch_size = encoder_out[0].shape[0]
        
        text_cond, image_cond, activations = encoder_out
        # Combine the text and image embeddings to produce a single conditioning tensor
        combined_cond = torch.mul(text_cond, image_cond)
        
        a = None
        # Loop through the activations and apply FiLM layers and MHA blocks
        for i, (skip, mha, reduce) in enumerate(zip(activations, self.mha_blocks, self.reduce_blocks)):
            if a is None:
                a = reduce(skip)
            else:
                a = a + reduce(skip)
            if (self.cond_layer==None or i==(self.cond_layer-1)):
                a = self.film_mul(combined_cond)*a + self.film_add(combined_cond)
            a = mha(a)

        # Ignoring the clasification (CLS) token
        a = a[:, 1:, :]  
        # Changing a -> (batch size, features, tokens)
        a = a.permute(0, 2, 1)
        size = int(math.sqrt(a.shape[2]))
        a = a.view(batch_size, a.shape[1], size, size)
        a = self.trans_conv(a)
            
        return a
    


# Is this needed???
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
if device == 'cuda':
    torch.cuda.empty_cache()

#Creating custom dataloader class

class Dog_Cat_Dataset(Dataset):
    def __init__(self, split = None):
        super(Dog_Cat_Dataset, self).__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        if split == 'train':

            self.input_images = []
            self.outputs = []
         #   self.task = unique task id for each image???
            # loop over images 
            # for i in range():
            #   temp_ip = Image.open(input image)
            #   temp_op = Image.open(mask)
            #   temp_ip = temp_ip.resize((352, 352))
            #   temp_op = temp_op.resize((224, 224))
            #   self.input_images.append(np.array(temp_ip, dtype=np.float32))
            #   self.outputs.append(np.array(temp_op, dtype = np.float32)/255)

        #elif split == 'val':
            #as above for validation data

        #elif split == 'test':
            #as above for test data
   
                
                
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
    #   return self.phrases[i] for item i,  self.input_images[i], self.outputs[i], self.tasks[i]   
    #   return (prompt, image, mask, task/id)

# Loading the train dataset by creating an instance of the Dog_Cat_Dataset class

tic = time.time()
data = Dog_Cat_Dataset('train')
train = DataLoader(data, batch_size=1, shuffle=True)
toc = time.time()
print(f"train set loaded. Time taken = {toc - tic} sec")

# Loading the validation and test dataset
# Code as above for validation data and test data

#from encoder_model import Encoder 
#from decoder_model import Decoder
#Only use if split out encoder and decoder into separate files
encoder1 = Encoder()
num_e = sum(e.numel() for e in encoder1.parameters() if e.requires_grad)
print("No of trainable parameters in the encoder -", num_e)

decoder1 = Decoder()
num_d = sum(d.numel() for d in decoder1.parameters() if d.requires_grad)
print("No of trainable parameters in the decoder -", num_d)

# Training the model
# Hyper-parameters
criterion = nn.BCEWithLogitsLoss() # ??? need to change this as it is for binary classification
optimizer = torch.optim.SGD(decoder1.parameters(), lr=1e-4)
# learning rate reduces 
gamma = 0.9
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

# Training Loop
n_total_steps = len(train)
loss_history = []
num_epochs = 10

for epoch in range(num_epochs):
    for i, (phrase, ip_image, op_image, _) in enumerate(train):
        # check if the image is RGB
        if ip_image[0].dim() != 3:
            continue
        
        # Use CLIP to encode with frozen weights
        encodings = encoder1(transforms.ToPILImage()(ip_image[0].permute(2, 0, 1)).convert("RGB"), phrase[0])
            
        # Uses custom decoder
        output = decoder1(encodings)
        
        loss = criterion(output[0][0], op_image[0])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Printing the loss at regular intervals
        if (i+1) % 2000 == 0:
            print(f"loss at epoch {epoch+1}/{num_epochs}, iteration {i+1}/{n_total_steps}: {loss.item()}")
            loss_history.append(loss.item())

    # Step the scheduler after each epoch
    scheduler.step()

# Saving the decoder weights
torch.save(decoder1.state_dict(), "decoder1_BCELoss.pth")


#Evaluating the model on Val set
# Calculating the accuracies based on pixel-by-pixel, iou and dice scores on the val set
sigmoid = nn.Sigmoid()
with torch.no_grad():
    accuracy_pixel_by_pixel = []
    accuracy_iou = []
    accuracy_dice_score = []
    
    n_total_steps = len(val)
    
    for i, (pharase, ip_image, op_image, _) in enumerate(val):

        encodings = encoder1(transforms.ToPILImage()(ip_image[0].permute(2, 0, 1)).convert("RGB"), phrase[0])
        output = decoder1(encodings)
        accuracy_pixel_by_pixel.append(torch.sum((output[0][0]>0) == op_image[0])/(224*224))

        output = sigmoid(output)
        
        intersection = torch.sum((output[0][0]>0)*op_image[0]) 
        union = torch.sum(output[0][0] > 0)+torch.sum(op_image[0]) - intersection
        accuracy_iou.append(intersection/union)
        
        numerator = 2*torch.sum((output[0][0]>0)*op_image[0]) + 1e-6
        denominator = torch.sum(op_image[0]**2) + torch.sum((output[0][0]>0)**2) + 1e-6
        accuracy_dice_score.append(numerator/denominator)
        
    print(f"accuracy pixel-by-pixel is - {100*sum(accuracy_pixel_by_pixel)/len(accuracy_pixel_by_pixel)}%")
    print(f"accuracy by iou is - {100*sum(accuracy_iou)/len(accuracy_iou)}%")
    print(f"accuracy by dice-scores is - {100*sum(accuracy_dice_score)/len(accuracy_dice_score)}%")

# Calculating the accuracies based on pixel-by-pixel, iou and dice scores on the test set
sigmoid = nn.Sigmoid()
with torch.no_grad():
    accuracy_pixel_by_pixel = []
    accuracy_iou = []
    accuracy_dice_score = []
    
    n_total_steps = len(test)
    
    for i, (pharase, ip_image, op_image, _) in enumerate(test):
        encodings = encoder1(transforms.ToPILImage()(ip_image[0].permute(2, 0, 1)).convert("RGB"), phrase[0])
        output = decoder1(encodings)
        accuracy_pixel_by_pixel.append(torch.sum((output[0][0]>0) == op_image[0])/(224*224))

        output = sigmoid(output)
        
        intersection = torch.sum((output[0][0]>0)*op_image[0]) 
        union = torch.sum(output[0][0] > 0)+torch.sum(op_image[0]) - intersection
        accuracy_iou.append(intersection/union)
        
        numerator = 2*torch.sum((output[0][0]>0)*op_image[0]) + 1e-6
        denominator = torch.sum(op_image[0]**2) + torch.sum((output[0][0]>0)**2) + 1e-6
        accuracy_dice_score.append(numerator/denominator)
        
    print(f"accuracy by pixel-by-pixel is - {100*sum(accuracy_pixel_by_pixel)/len(accuracy_pixel_by_pixel) :.6f}%")
    print(f"accuracy by iou is - {100*sum(accuracy_iou)/len(accuracy_iou) :.6f}%")
    print(f"accuracy by dice-scores is - {100*sum(accuracy_dice_score)/len(accuracy_dice_score) :.6f}%")