import torch
import numpy as np
import os
import random
from PIL import Image
from pathlib import Path

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
    torch.backends.cudnn.deterministic = True


COLOR_MAP = {
    WHITE: BACKGROUND,
    BLACK: BACKGROUND,
    GREEN: DOG,
    RED: CAT,
}

COLOR_MAP_INV = dict([((one_hot[0], one_hot[1], one_hot[2]), color) for color, one_hot in COLOR_MAP.items()])

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

def accuracy(y : np.array , y_pred : np.array):
    """
    Calculates batch accuracy of two np.arrays
    y, y_pred have shape (N, w, h)
    """
    N1, w1, h1 = y.shape
    N2, w2, h2 = y_pred.shape
    assert N1 == N2 and w1 == w2 and h1 == h2, f"dimensions of y and y_pred are different: ({y.shape}) vs. ({y_pred.shape})"
    return np.sum(y == y_pred) / (w1 * h1 * N1)


# Plotting
#########

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss_iter(loss: list):
    Y = np.array(moving_average(loss,20))
    X = np.array([i for i in range(Y.size)])
    plt.plot(X, Y)
    plt.xlabel("Iteration number")
    plt.ylabel("loss")
    plt.show()




def plot_loss_iou_temp(trainer):
    x = np.array([i for i in range(len(trainer.loss_mu))])
    y = np.array(trainer.loss_mu)
    e = np.array(trainer.loss_sigma)

    plt.plot(x, y, marker='o', linestyle='-', color='b', label="mean")
    plt.fill_between(x, y - e, y + e, color='b', alpha=0.2, label="Standard deviation")
    plt.xlabel("Iteration number")
    plt.ylabel("average loss per epoch, with stdev")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

    x = [i for i in range(len(trainer.IoU_mu))]
    y = trainer.IoU_mu
    e = trainer.IoU_sigma

    plt.plot(x, y, marker='o', linestyle='-', color='b', label="mean")
    plt.fill_between(x, y - e, y + e, color='b', alpha=0.2, label="Standard deviation")
    plt.xlabel("Iteration number")
    plt.ylabel("average IoU per epoch, with stdev")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()

    x = [i for i in range(len(trainer.acc_mu))]
    y = trainer.acc_mu
    e = trainer.acc_sigma

    plt.plot(x, y, marker='o', linestyle='-', color='b', label="mean")
    plt.fill_between(x, y - e, y + e, color='b', alpha=0.2, label="Standard deviation")
    plt.xlabel("Iteration number")
    plt.ylabel("average acc per epoch, with stdev")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()