import torch
import numpy as np
import os
import random
from PIL import Image

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)

CAT = (1,0,0)
DOG = (0,1,0)
BACKGROUND = (0,0,1)

# Generaly utilies
##################
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def color_to_mask(r: int, g: int, b: int):
    color_map = {
        WHITE: BACKGROUND,
        BLACK: BACKGROUND,
        GREEN: DOG,
        RED: CAT,
    }
    if not((r,g,b) in color_map):
        raise ValueError(f"mask color is not WHITE/BLACK/RED/GREEN but ({r},{g},{b})")
    else:
        return color_map[(r,g,b)]

def label_to_one_hot(y : Image):
    """
    Convert a PIL image label of size (w,h,3) to a one hot representation of the mask (w,h,3) as a numpy array.
    (1,0,0) = cat, (0,1,0) = dog, (0,0,1) = background.
    Note that in the original image: red = cat, green = dog, black = background, white = border
    """
    data = np.asarray(y)
    _, _, ch = data.shape
    if (ch != 3):
        raise ValueError(f"Label image expects 3 channels, obtained = {ch}")
    
    vectorized = np.vectorize(color_to_mask)
    return vectorized(data[:,:,0], data[:,:,1], data[:,:,2])



# Metrics
#########
def IoU(y, y_pred):
    return 0