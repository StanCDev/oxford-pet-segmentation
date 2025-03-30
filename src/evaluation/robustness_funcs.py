import numpy as np
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.signal import convolve2d, correlate2d
from skimage.util import random_noise
from PIL import Image
from pathlib import Path

#from utils import read_img_mono, display_img, save_img

def perturbate_image(img, perturb_fn : str, perturb_params: float, dim : tuple[int,int]) -> np.ndarray:
    """
    Runs defined robustness test on image.

    Args:
    -----
        img : image to apply 
        perturb_fn (str): perturbation function to apply to image.
        perturb_params (float): perturbation parameters to apply to image.
        dim (tuple[int,int]): image dimension with default 242x242.

    Returns:
    --------
        (np.ndarrray) perturbed image
    """
    if perturb_fn == "gaussian":
        mean = 0
        var = perturb_params
        img = np.array(img)
        Gaussian_noise = np.random.normal(mean, var, img.shape)
        perturb_image = img + Gaussian_noise
        perturb_image = np.clip(perturb_image, 0, 255)
        perturb_image = perturb_image.astype(np.uint8)
    
    elif perturb_fn == "gaussian_blur":
        # Gaussian blur
        blur_filter = (1/16) * np.array([[1,2,1],[2,4,2],[1,2,1]])
        input_img = np.array(img)
        perturb_image = np.zeros(input_img.shape, dtype=np.uint8)
        for i in range(round(perturb_params)):
            # Apply to each channel of the image
            for chn in range(3):
                perturb_image[..., chn ] = convolve2d(input_img[...,chn],blur_filter, mode="same").astype(np.uint8)
            input_img = perturb_image
    
    elif perturb_fn == "salt_and_pepper":
        # Salt and pepper noise
        scale = perturb_params
        input_img = np.array(img) 
        perturb_image = random_noise(input_img, mode='s&p', clip=True, amount=scale)
        perturb_image = (255*perturb_image).astype(np.uint8)
    
    elif perturb_fn == 'contrast_change':
        # Increase contrast by scaling pixel values
        input_img = np.array(img)         
        perturb_image = ((input_img).astype(np.int16) * perturb_params).clip(0, 255)  
      #  perturb_image = np.clip(perturb_image, 0, 255)
        perturb_image = perturb_image.astype(np.uint8)

    elif perturb_fn == 'brightness_change':
        # Increase contrast by scaling pixel values
        input_img = np.array(img).astype(np.uint8) 
        perturb_img = ((input_img).astype(np.int16) + perturb_params).clip(0, 255)        
        perturb_image = perturb_img.astype(np.uint8)
        perturb_image = (np.clip(perturb_image, 0, 255)).astype(np.uint8)

    elif perturb_fn == 'occlusion':
        # Occlusion in random location by setting a square region to zero
        
        perturb_image = np.array(img).astype(np.uint8)
        height, width = perturb_image.shape[:2]
        print(height, width)
        # Get random coordinates for the square region
        x = np.random.randint(0, height)
        y = np.random.randint(0, width)
        w = perturb_params
        h = perturb_params
        perturb_image[x:x+w, y:y+h] = 0
        perturb_image = np.clip(perturb_image, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown perturbation function: {perturb_fn}")


    return perturb_image

def perturb_directory(
        src_dir : Path,  
        dest_dir: Path,
        dim : tuple[int,int], 
        perturb_fn, 
        perturb_params,
        print_progress : bool = True
        ) -> None:
    """
    Perturb all images in a directory to specified dimensions

    Args:
    -----
        src_dir (Pathlib.Path): path where source images are located.

        dest_dir (Pathlib.Path): path where perturbed images are saved.

        dim (tuple[int,int]): image dimension to perturb.

        perturb_fn (str): perturbation function to apply to image.

        perturb_params (float): perturbation parameters to apply to image.

        print_progress (bool): obtain a progress bar showing number of resized images out of total count, default=True.

    Returns:
    --------
        None
    """
    accepted_file_types = {".jpg", ".jpeg", ".png"}
    count = 0
    dir_files = src_dir.iterdir()

    nbr_images = len(list(filter(lambda image_path: (image_path.suffix.lower() in accepted_file_types) , dir_files)))
    print(f"Found {nbr_images} images in {src_dir}")
    for image_path in src_dir.iterdir():
        if image_path.suffix.lower() in accepted_file_types:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                count += 1
                out_image = perturbate_image(img, perturb_fn, perturb_params, dim)
                print(f"Perturbing {image_path.name}...")
                #out_image.save(dest_dir / f"{Path(image_path).name}")
                out_image = Image.fromarray(out_image)
                out_image.save(dest_dir / f"{Path(image_path).name}")
                if print_progress:
                    print(f"Perturbed {count} images / {nbr_images}")
    return

#src_dir = Path("../../test/res_out/")
#dest_dir = Path("../../test/res_perturb/")
#perturb_directory(src_dir, dest_dir, (256, 256), 'gaussian', 50)
