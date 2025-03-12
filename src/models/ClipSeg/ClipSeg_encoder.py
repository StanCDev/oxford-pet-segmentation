import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer, AutoProcessor, CLIPTextModel, CLIPVisionModelWithProjection

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
    
