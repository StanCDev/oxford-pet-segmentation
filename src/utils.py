import torch
import numpy as np
import os
import random
from PIL import Image

from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt

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
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


COLOR_MAP = {
    WHITE: BACKGROUND,
    BLACK: BACKGROUND,
    GREEN: DOG,
    RED: CAT,
}

COLOR_MAP_INV = {
    (1,0,0): CAT,
    (0,1,0): DOG,
    (0,0,1): BACKGROUND
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

def one_hot_to_label(y: Image):
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
    for label, color in COLOR_MAP_INV.items():
        mask = np.all(data == np.array(label), axis=-1)  # Find pixels matching the color
        out[mask] = color  # Assign corresponding one-hot value

    # Check for any unmapped pixels
    if not np.all(np.any(out, axis=-1)):  # If any pixel is still (0,0,0), it's an error
        raise ValueError("Label image contains unknown colors.")

    return out

def color_mask(y: torch.Tensor):
    H, W = y.shape
    # Create an empty RGB image (H, W, 3)
    rgb_image = torch.zeros((H, W, 3), dtype=torch.uint8)

    # Define class-to-color mapping
    rgb_image[y == 0] = torch.tensor([128, 0, 0], dtype=torch.uint8)    # Black (Class 0)
    rgb_image[y == 1] = torch.tensor([0, 128, 0], dtype=torch.uint8)  # Red (Class 1)
    rgb_image[y == 2] = torch.tensor([0, 0, 0], dtype=torch.uint8)  # Green (Class 2)
    rgb_image = to_pil_image(rgb_image.permute(2,0,1)).convert("RGB")
    return rgb_image


# Metrics
#########
def IoU(y : np.array , y_pred : np.array):
    """
    y, y_pred have shape (N, ch, w, h)
    """
    N1, ch1, w1, h1 = y.shape
    N2, ch2, w2, h2 = y_pred.shape
    assert N1 == N2 and ch1 == ch2 and w1 == w2 and h1 == h2, f"dimensions of y and y_pred are different: ({y.shape}) vs. ({y_pred.shape})"
    assert ch1 == 3, "must have 3 channels"
    
    iou_per_class = []
    
    for c in range(3):
        intersection = np.logical_and(y[:, c, :, :], y_pred[:, c, :, :]).sum()
        union = np.logical_or(y[:, c, :, :], y_pred[:, c, :, :]).sum()
        
        iou = intersection / (union + 1e-7)  # Avoid division by zero
        iou_per_class.append(iou)
    
    iou_per_class = np.array(iou_per_class)
    mean_iou = np.mean(iou_per_class)
    return mean_iou


def dice(y : np.array , y_pred : np.array):
    """
    y, y_pred have shape (N, ch, w, h)
    """
    N1, ch1, w1, h1 = y.shape
    N2, ch2, w2, h2 = y_pred.shape
    assert N1 == N2 and ch1 == ch2 and w1 == w2 and h1 == h2, f"dimensions of y and y_pred are different: ({y.shape}) vs. ({y_pred.shape})"
    assert ch1 == 3, "must have 3 channels"
    
    dice_per_class = []
    eps=1e-7
    
    for c in range(3):
        y_c = y[:, c, :, :]
        y_pred_c = y_pred[:, c, :, :]
        intersection = np.logical_and(y_c, y_pred_c).sum()
        total = y_c.sum() + y_pred_c.sum()
        dice = (2.0 * intersection) / (total + eps)
        dice_per_class.append(dice)
    
    dice_per_class = np.array(dice_per_class)
    mean_dice = np.mean(dice_per_class)
    return mean_dice

def accuracy(y : np.array , y_pred : np.array):
    """
    Calculates batch accuracy of two np.arrays
    y, y_pred have shape (N, w, h)
    """
    assert len(y.shape) == len(y_pred.shape), f"arrays must have same shape, y has shape {y.shape}, and y_pred has shape {y_pred.shape}"
    if len(y.shape) == 3:
        N1, w1, h1 = y.shape
        N2, w2, h2 = y_pred.shape
        assert N1 == N2 and w1 == w2 and h1 == h2, f"dimensions of y and y_pred are different: ({y.shape}) vs. ({y_pred.shape})"
        return np.sum(y == y_pred) / (w1 * h1 * N1)
    elif len(y.shape) == 4:
        N1, w1, ch1, h1 = y.shape
        N2, w2, ch2, h2 = y_pred.shape
        assert N1 == N2 and w1 == w2 and h1 == h2 and ch1 == ch2, f"dimensions of y and y_pred are different: ({y.shape}) vs. ({y_pred.shape})"
        return np.sum(y == y_pred) / (w1 * h1 * N1 * ch1)
    else:
        raise ValueError("Not 3 or 4 dimensional arrays")


# Plotting
#########

def plot_metric(x : np.array, y: np.array, e: np.array, y_val: np.array, e_val: np.array, xlabel : str, ylabel: str, show_val : bool = False):
    plt.plot(x, y, marker='o', linestyle='-', color='b', label="training mean")
    plt.fill_between(x, y - e, y + e, color='b', alpha=0.2, label="training standard deviation")

    if show_val:
        plt.plot(x, y_val, marker='s', linestyle='-', color='r', label="Validation mean") 
        plt.fill_between(x, y_val - e_val, y_val + e_val, color='r', alpha=0.2, label="Validation standard deviation")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

def plot_training_metrics(trainer, show_val : bool = False):
    x = np.array([i for i in range(len(trainer.loss_mu))])
    y = np.array(trainer.loss_mu)
    e = np.array(trainer.loss_sigma)

    plot_metric(x,y,e, None, None, "Iteration number", "average loss per epoch, with stdev",show_val=False)

    x = [i for i in range(len(trainer.IoU_mu))]
    y = np.array(trainer.IoU_mu)
    y_val = np.array(trainer.val_IoU_mu)
    e = np.array(trainer.IoU_sigma)
    e_val = np.array(trainer.val_IoU_sigma)

    plot_metric(x,y,e,y_val,e_val,"Iteration number", "average IoU per epoch, with stdev",show_val)

    x = [i for i in range(len(trainer.acc_mu))]
    y = np.array(trainer.acc_mu)
    y_val = np.array(trainer.val_acc_mu)
    e = np.array(trainer.acc_sigma)
    e_val = np.array(trainer.val_acc_sigma)

    plot_metric(x,y,e,y_val,e_val,"Iteration number", "average acc per epoch, with stdev",show_val)