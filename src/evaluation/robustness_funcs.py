import numpy as np
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.signal import convolve2d, correlate2d
from skimage.util import random_noise
from PIL import Image
from pathlib import Path
import argparse


param_values = {
    "gaussian_noise" : [range(0,19,2)],
    "gaussian_blur" : [range(0,10)],
    "contrast_increase" : [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25],
    "contrast_decrease" : [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10],
    "brightness_increase" : [5 * i for i in range(0,10)],
    "brightness_decrease" : [-5 * i for i in range(0,10)],
    "occlusion" : [5 * i for i in range(0,10)],
    "salt_and_pepper" : [0.02 * i for i in range(0,10)],
}

def gaussian_noise(img, sigma : int):
    img = np.array(img)
    Gaussian_noise = np.random.normal(0, sigma, img.shape)
    perturb_image = img + Gaussian_noise
    perturb_image = np.clip(perturb_image, 0, 255)
    perturb_image = perturb_image.astype(np.uint8)
    return perturb_image

def gaussian_blue(img, nbr_times : int):
    assert nbr_times > 0 and nbr_times == int(nbr_times), "Must be an integer greater than 0"
    blur_filter = (1/16) * np.array([[1,2,1],[2,4,2],[1,2,1]])
    input_img = np.array(img)
    perturb_image = np.zeros(input_img.shape, dtype=np.uint8)
    for _ in range(nbr_times):
        # Apply to each channel of the image
        for chn in range(3):
            perturb_image[..., chn ] = convolve2d(input_img[...,chn],blur_filter, mode="same").astype(np.uint8)
        input_img = perturb_image
    return input_img

def salt_and_pepper(img, amount : float):
    assert amount >= 0 and amount <= 1, "amount must be in 0,1 range"
    input_img = np.array(img) 
    perturb_image = random_noise(input_img, mode='s&p', clip=True, amount=amount)
    perturb_image = (255*perturb_image).astype(np.uint8)
    return perturb_image

def contrast_change(img, factor : float):
    input_img = np.array(img)         
    perturb_image = ((input_img).astype(np.int16) * factor).clip(0, 255)  
    perturb_image = perturb_image.astype(np.uint8)
    return perturb_image

def brightness_change(img, factor : int):
    input_img = np.array(img).astype(np.uint8) 
    perturb_img = ((input_img).astype(np.int16) + factor).clip(0, 255)        
    perturb_image = perturb_img.astype(np.uint8)
    perturb_image = (np.clip(perturb_image, 0, 255)).astype(np.uint8)

def occlusion(img, edge_len : int):
    # Occlusion in random location by setting a square region to zero
    perturb_image = np.array(img).astype(np.uint8)
    height, width = perturb_image.shape[:2]
    print(height, width)
    # Get random coordinates for the square region
    x = np.random.randint(0, height)
    y = np.random.randint(0, width)
    w = edge_len
    h = edge_len
    perturb_image[x:x+w, y:y+h] = 0
    perturb_image = np.clip(perturb_image, 0, 255).astype(np.uint8)
    return perturb_image


def main(args):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=None, type=str, help="path to directory with train images")
    parser.add_argument('--label', default=None, type=str, help="path to directory with label images")
    parser.add_argument('--json', default=None, type=str, help="path json file to overwrite and save mapping to")
    parser.add_argument('--model_file',default=None, type=str, help="path to .pth file with model parameters")
    parser.add_argument('--nn_type',default=None, type=str, help="type of Neural Net being loaded")
    
    args = parser.parse_args()
    if not(args.train) or not(args.label):
        raise ValueError("Must input train and label path if resize/augment of images wanted or json map")
    
    main(args)
    pass