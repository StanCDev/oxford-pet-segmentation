import numpy as np
from scipy.signal import convolve2d
from skimage.util import random_noise
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from models.dataset import SegmentationDataset
from main import load_model
from utils import dice, IoU
from torch.utils.data import random_split

from PIL import Image

def gaussian_noise(img : Image, sigma : int):
    img = np.array(img)
    Gaussian_noise = np.random.normal(loc=0, scale=sigma, size=img.shape)
    perturb_image = img + Gaussian_noise
    perturb_image = np.clip(perturb_image, 0, 255)
    perturb_image = perturb_image.astype(np.uint8)
    return perturb_image

def gaussian_blur(img : Image, nbr_times : int):
    assert nbr_times >= 0 and nbr_times == int(nbr_times), "Must be an integer greater than or equal to 0"
    blur_filter = (1/16) * np.array([[1,2,1],[2,4,2],[1,2,1]])
    input_img = np.array(img)
    perturb_image = np.zeros(input_img.shape, dtype=np.uint8)
    for _ in range(nbr_times):
        # Apply to each channel of the image
        for chn in range(3):
            perturb_image[..., chn ] = convolve2d(input_img[...,chn],blur_filter, mode="same").astype(np.uint8)
        input_img = perturb_image
    return input_img

def salt_and_pepper(img : Image, amount : float):
    assert amount >= 0 and amount <= 1, "amount must be in 0,1 range"
    input_img = np.array(img) 
    perturb_image = random_noise(input_img, mode='s&p', clip=True, amount=amount)
    perturb_image = (255*perturb_image).astype(np.uint8)
    return perturb_image

def contrast_change(img : Image, factor : float):
    input_img = np.array(img)         
    perturb_image = ((input_img).astype(np.int16) * factor).clip(0, 255)  
    perturb_image = perturb_image.astype(np.uint8)
    return perturb_image

def brightness_change(img : Image, factor : int):
    input_img = np.array(img).astype(np.uint8) 
    perturb_img = ((input_img).astype(np.int16) + factor).clip(0, 255)        
    perturb_image = perturb_img.astype(np.uint8)
    perturb_image = (np.clip(perturb_image, 0, 255)).astype(np.uint8)
    return perturb_image

def occlusion(img : Image, edge_len : int):
    # Occlusion in random location by setting a square region to zero
    perturb_image = np.array(img).astype(np.uint8)
    height, width = perturb_image.shape[:2]
    # Get random coordinates for the square region
    x = np.random.randint(0, height)
    y = np.random.randint(0, width)
    w = edge_len
    h = edge_len
    perturb_image[x:x+w, y:y+h] = 0
    perturb_image = np.clip(perturb_image, 0, 255).astype(np.uint8)
    return perturb_image

perturbation_values = {
    "gaussian_noise" : list(range(0,19,2)),
    "gaussian_blur" : list(range(0,10)),
    "contrast_increase" : [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
    "contrast_decrease" : [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
    "brightness_increase" : list(5 * i for i in range(0,10)),
    "brightness_decrease" : list(-5 * i for i in range(0,10)),
    "occlusion" : list(5 * i for i in range(0,10)),
    "salt_and_pepper" : list(0.02 * i for i in range(0,10)),
}

perturbation_functions = {
    "gaussian_noise" : gaussian_noise,
    "gaussian_blur" : gaussian_blur,
    "contrast_increase" : contrast_change,
    "contrast_decrease" : contrast_change,
    "brightness_increase" : brightness_change,
    "brightness_decrease" : brightness_change,
    "occlusion" : occlusion,
    "salt_and_pepper" : salt_and_pepper,
}

def display(perturbation_results : dict[str, np.array]) -> None:
    
    for perturb, y in perturbation_results.items():
        x = perturbation_values[perturb]
        x = np.array(x)

        plt.plot(x, y, label=perturb, marker='o')
        plt.title(f"{perturb} perturbation vs. Dice Score")
        plt.xlabel(f"{perturb} parameter")
        plt.ylabel("Average Dice Score")
        plt.grid(True)
        plt.show()


def main(args):
    model = load_model(args)

    x_y = SegmentationDataset(
        Path(args.train), 
        Path(args.label),
        Path(args.json),
        nn_type=args.nn_type,
        )
    
    train_size = int(0.2 * len(x_y))
    val_size = len(x_y) - train_size

    x_y, _ = random_split(x_y, [train_size, val_size])
    
    dataloader = DataLoader(x_y, batch_size=1, shuffle=True)

    perturbation_results = {perturb: np.zeros(len(values)) for perturb, values in perturbation_values.items()}

    with torch.no_grad():
        #1. Iterate over every image

        for i, batch in enumerate(dataloader):
            # start = time.time()
            if args.nn_type == "CLIP":
                (prompt, x, y, _) = batch
                assert len(prompt) == 1 and len(x) == 1 and len(y) == 1, "Must have batch size of 1"
                prompt = prompt[0]
            else:
                x, y = batch

            if args.nn_type == "CLIP":
                torch_to_PIL = transforms.ToPILImage()
                img = torch_to_PIL(x[0]).convert("RGB")
            else:
                raise ValueError("Support for other models than CLIP not implemented yet")

            # Iterate over every perturbation mode
            for perturbation, values in perturbation_values.items():
                func = perturbation_functions.get(perturbation)

                if func is None:
                    raise ValueError("Function is None! indexed non existing function in perturbation_functions")
                
                for j, value in enumerate(values):
                    assert type(value) == int or type(value) == float, f"Value must be a float or an int but is a {type(value)}"
                    perturbed_img : np.array = func(img, value)
                    #Convert image back to Image.RGB
                    perturbed_img = Image.fromarray(perturbed_img).convert("RGB")
                    #get output mask
                    logits = model.forward(img, prompt)
                    #compare output mask with ground truth
                    y_pred_classes = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # (N, W, H)
                    y_pred_one_hot = F.one_hot(y_pred_classes, num_classes=3).permute(0, 3, 1, 2)  # (N, 3, W, H)

                    dice_score = dice(y_pred_one_hot.numpy(), y.numpy())

                    perturbation_results[perturbation][j] += dice_score
            
            print('\rCalculated robustness for sample {}/{}'.
                format(i + 1, len(dataloader)), end='')

    for perturb in perturbation_results.keys():
        ###Scaling by number of data points to get an average
        perturbation_results[perturb] = perturbation_results[perturb] / float(len(dataloader))

    display(perturbation_results=perturbation_results)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=None, type=str, help="path to directory with train images")
    parser.add_argument('--label', default=None, type=str, help="path to directory with label images")
    parser.add_argument('--json', default=None, type=str, help="path json file to overwrite and save mapping to")
    parser.add_argument('--load',default=None, type=str, help="path to .pth file with model parameters")
    parser.add_argument('--nn_type',default="CLIP", type=str, help="type of Neural Net being loaded")
    parser.add_argument('--nn_batch_size', default = 1, type=int)
    
    args = parser.parse_args()
    if not(args.train) or not(args.label):
        raise ValueError("Must input train and label path if resize/augment of images wanted or json map")
    
    main(args)
    pass