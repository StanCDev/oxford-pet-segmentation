"""
This module provides functions to augment images in a common directory using multiple augmentation options
"""
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
import numpy as np

def augment_image(
    img : np.ndarray,
    img_lab : np.ndarray,
    col_jit : float, 
    rand_rot : float, 
    rand_hflip : float,
    el_trans_a : float,
    el_trans_s : float
    ) -> np.ndarray: 
    """
    augment a numpy array of type uint8 with color jitter, random rotation and random horizontal flip

    Args:
    -----
        img : image to augment
        img_lab : label image to augment

        col_jit (float): color jitter probability factor
        rand_rot (float): random rotation probability factor
        rand_hflip (float): random horizontal flip probability factor
        el_trans_a and el_trans_s are the alpha and sigma values for the elastic transform
    Returns:
    --------
        (np.ndarrray) augmented image and (np.ndarray) label image
    """
    transform = v2.Compose(
        [
            v2.RandomRotation(degrees=rand_rot),
            v2.RandomHorizontalFlip(p=rand_hflip)
        ]
        )
    transform_1 = v2.Compose(
        [
            v2.ColorJitter(brightness=col_jit, contrast=col_jit, saturation=col_jit, hue=col_jit), 
            v2.ElasticTransform(alpha = el_trans_a, sigma = el_trans_s)
        ]
        )    
    img = transform_1(img) # separated out color jitter from other transforms
    return transform(img, img_lab) 

def aug_directory(
        src_dir : Path, 
        dest_dir: Path, 
        src_dir_lab : Path,  
        dest_dir_lab: Path,  
        col_jit : float,
        rand_rot : float,
        rand_hflip : float,
        el_trans_a : float,
        el_trans_s : float,
        print_progress : bool = True
        ) -> None:
    """
    Augments all images in a directory with defined probablity factors and apply the same augmentation to label images with the same name

    Args:
    -----
        src_dir (Pathlib.Path): path where source images are located.
        dest_dir (Pathlib.Path): path where resized images are saved.
        src_dir_lab (Pathlib.Path): path where source label images are located.
        dest_dir_lab (Pathlib.Path): path where resized label images are saved.

        col_jit (float): color jitter factor
        rand_rot (float): random rotation factor (degrees)
        rand_hflip (float): random horizontal flip factor
        el_trans_a and el_trans_s are the alpha and sigma values for the elastic transform
        print_progress (bool): obtain a progress bar showing number of resized images out of total count, default=True.

    Returns:
    --------
        None
    """
    accepted_file_types = {".jpg", ".jpeg", ".png"}
    count = 0
    dir_files = src_dir.iterdir()
    dir_files_lab = src_dir_lab.iterdir()  
    nbr_images = len(list(filter(lambda image_path: (image_path.suffix.lower() in accepted_file_types) , dir_files)))
    nbr_labels = len(list(filter(lambda image_path: (image_path.suffix.lower() in accepted_file_types) , dir_files_lab))) 

    label_info=list()
    for image_path in src_dir.iterdir():
        if image_path.suffix.lower() in accepted_file_types:
            with Image.open(image_path) as img, Image.open(src_dir_lab / f"{Path(image_path).stem}.png") as img_lab: # RT added to open label image (***hard wired .png extension at the moment***)
#                img = img.convert("RGB")  
#                img_lab = img_lab.convert("L") 
                count += 1
                out_image, out_lab = augment_image(img=img, img_lab = img_lab, col_jit=col_jit, rand_rot=rand_rot, rand_hflip=rand_hflip, el_trans_a=el_trans_a, el_trans_s=el_trans_s)   
                out_image.save(dest_dir / f"{Path(image_path).name}")
                out_lab.save(dest_dir_lab / f"{Path(image_path).stem}.png") 
                if print_progress:
                    print(f"Augmented {count} images / {nbr_images}")
    return