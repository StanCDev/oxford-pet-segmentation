"""
This module provides functions to resize images in a common directory to a uniform dimension
"""
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
import numpy as np

def resize_image(img, dim : tuple[int,int]) -> np.ndarray:
    """
    Resize a numpy array of type uint8 to specified dimensions

    Args:
    -----
        img : image to resize

        dim (tuple[int,int]): image dimension to resize to.

    Returns:
    --------
        (np.ndarrray) resized image
    """
    ## data = resize(img, dim, anti_aliasing=True) * 256
    ### Note that this is the same type as output array
    ###return data.astype(np.uint8)
    transform = v2.Resize(dim, antialias=True)
    return transform(img)

def resize_directory(
        src_dir : Path, 
        dest_dir: Path, 
        dim : tuple[int,int], 
        print_progress : bool = True
        ) -> None:
    """
    Resize all images in a directory to specified dimensions

    Args:
    -----
        src_dir (Pathlib.Path): path where source images are located.

        dest_dir (Pathlib.Path): path where resized images are saved.

        dim (tuple[int,int]): image dimension to resize to.

        print_progress (bool): obtain a progress bar showing number of resized images out of total count, default=True.

    Returns:
    --------
        None
    """
    accepted_file_types = {".jpg", ".jpeg", ".png"}
    count = 0
    dir_files = src_dir.iterdir()

    nbr_images = len(list(filter(lambda image_path: (image_path.suffix.lower() in accepted_file_types) , dir_files)))

    for image_path in src_dir.iterdir():
        if image_path.suffix.lower() in accepted_file_types:
            with Image.open(image_path) as img:
                count += 1
                out_image = resize_image(img=img, dim=dim)
                out_image.save(dest_dir / f"{Path(image_path).name}")
                if print_progress:
                    print(f"Resized {count} images / {nbr_images}")
    return