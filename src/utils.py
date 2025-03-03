import torch
import numpy as np
import os
import random
from PIL import Image

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (128,0,0) ## as shown in the input data
GREEN = (0,128,0)

CAT = np.array([1,0,0])
DOG = np.array([0,1,0])
BACKGROUND = np.array([0,0,1])

# Generaly utilies
##################
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


COLOR_MAP = {
    WHITE: BACKGROUND,
    BLACK: BACKGROUND,
    GREEN: DOG,
    RED: CAT,
}

def label_to_one_hot(y: Image):
    """
    Convert a PIL image label (w,h,3) to a one-hot mask (w,h,3) as a numpy array.
    (1,0,0) = cat, (0,1,0) = dog, (0,0,1) = background.
    """

    # Ensure input image has 3 channels
    data = np.asarray(y,dtype=np.uint8)
    w, h, ch = data.shape
    if (ch != 3):
        raise ValueError(f"Label image expects 3 channels, obtained = {ch}")

    # Create an empty output array
    out = np.zeros(data.shape)

    # Generate a mask for each class using NumPy indexing
    for color, label in COLOR_MAP.items():
        mask = np.all(data == np.array(color), axis=-1)  # Find pixels matching the color
        out[mask] = label  # Assign corresponding one-hot value

    # Check for any unmapped pixels
    if not np.all(np.any(out, axis=-1)):  # If any pixel is still (0,0,0), it's an error
        raise ValueError("Label image contains unknown colors.")

    return out


# Metrics
#########
def IoU(y, y_pred):
    return 0