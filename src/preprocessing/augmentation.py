"""
This module provides functions to augment images in a common directory using multiple augmentation options
"""
from pathlib import Path
from torchvision.transforms import v2, InterpolationMode
from PIL import Image
import numpy as np

def augment_image(
    img : np.ndarray,
    img_lab : np.ndarray,
    col_jit : float, 
    rand_rot : float, 
    rand_hflip : float,
    el_trans_a : float,
    el_trans_s : float,
    el_trans_prob : float,
    col_jit_prob : float
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
        el_trans_prob : float is the probability of applying the elastic transform
        col_jit_prob : float is the probability of applying the color jitter
    Returns:
    --------
        (np.ndarrray) augmented image and (np.ndarray) label image
    """
    #generates transformation that applies elastic transforms to image and labels with defined probability factor
    random_apply_transform = v2.RandomApply([
                        v2.ElasticTransform(alpha = el_trans_a, sigma = el_trans_s, interpolation=InterpolationMode.NEAREST)
        ], p= el_trans_prob)
    
    #generates transformation to apply  random rotation and horizontal flip to all images, and random elastic transform defined above
    transform = v2.Compose([v2.RandomRotation(degrees=rand_rot,interpolation=InterpolationMode.NEAREST),
            v2.RandomHorizontalFlip(p=rand_hflip), random_apply_transform])

    #Random apply color jitter to image only with 50% probability
    random_apply_transform_1 = v2.RandomApply([
            v2.ColorJitter(brightness=col_jit, contrast=col_jit, saturation=col_jit, hue=col_jit)
        ], p=col_jit_prob)
    transform_1 = v2.Compose(
        [random_apply_transform_1]
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
        el_trans_prob : float,
        col_jit_prob : float,   
        print_progress : bool = True,
        in_place : bool = False
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
        el_trans_prob : float is the probability of applying the elastic transform
        col_jit_prob : float is the probability of applying the color jitter
        print_progress (bool): obtain a progress bar showing number of resized images out of total count, default=True.
        in_place (bool): augment images and overwrite source directory. Dest directory ignored

    Returns:
    --------
        None
    """
    accepted_file_types = {".jpg", ".jpeg", ".png"}
    count = 0
    dir_files = src_dir.iterdir()
    dir_files_lab = src_dir_lab.iterdir()  
    nbr_images = len(list(filter(lambda image_path: (image_path.suffix.lower() in accepted_file_types) , dir_files)))

    for image_path in src_dir.iterdir():
        if image_path.suffix.lower() in accepted_file_types:
            with Image.open(image_path) as img:
                label_path = src_dir_lab / f"{Path(image_path).stem}.png"
                with Image.open(label_path) as img_lab: # RT added to open label image (***hard wired .png extension at the moment***)
                    img = img.convert("RGB")  
                    img_lab = img_lab.convert("RGB") 
                    count += 1
                    assert img.size == img_lab.size, f"Image and label must be of the same size. They are not! Images in question: {image_path} , {label_path}"
                    out_image, out_lab = augment_image(img=img, img_lab = img_lab, col_jit=col_jit, rand_rot=rand_rot, rand_hflip=rand_hflip, el_trans_a=el_trans_a, el_trans_s=el_trans_s, el_trans_prob=el_trans_prob, col_jit_prob=col_jit_prob)
                    if in_place:
                        train_ext = image_path.suffix.lower()
                        out_image.save(image_path.parent / (image_path.stem + "_aug" + train_ext))

                        label_ext = label_path.suffix.lower()
                        out_lab.save(label_path.parent / (label_path.stem + "_aug" + label_ext)) 
                    else:
                        out_image.save(dest_dir / f"{Path(image_path).name}")
                        out_lab.save(dest_dir_lab / f"{Path(image_path).stem}.png") 
                    if print_progress:
                        print(f"Augmented {count} images / {nbr_images}")
    return