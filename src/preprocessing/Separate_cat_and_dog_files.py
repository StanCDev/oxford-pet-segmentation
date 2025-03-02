'''Separate cat and dog files'''

from pathlib import Path
from PIL import Image
import numpy as np

def separate_cat_dog(src_dir: Path, 
                     src_dir_lab: Path,
                      target_dir_cat: Path, 
                      target_dir_lab_cat: Path, 
                      target_dir_dog: Path, 
                      target_dir_lab_dog: Path) -> None:
    """ 
    Separate cat and dog files from the source directory and save them in the target directory
    Args:
    -----
        src_dir (Path): source directory for image files
        src_dir_lab (Path): source directory for label files
        target_dir_cat (Path): target directory for image files
        target_dir_lab_cat (Path): target directory for label files
        target_dir_dog (Path): target directory for image files
        target_dir_lab_dog (Path): target directory for label files  
    Returns:
    --------
        None
    """


    count = 0
    dir_files = src_dir.iterdir()
    dir_files_lab = src_dir_lab.iterdir()  
 
    label_info=list()
    for label_path in src_dir_lab.iterdir():
        with Image.open(label_path) as img_lab, Image.open(src_dir / f"{Path(label_path).stem}.jpg") as img: # RT added to open label image (***hard wired .png extension at the moment***)
            unique_in_labels, count = np.unique(img_lab, return_counts=True)
            if unique_in_labels[1] == 1:
                img_lab.save(target_dir_lab_cat / f"{Path(label_path).name}")
                print(target_dir_lab_cat / f"{Path(label_path).name}")
                img.save(target_dir_cat / f"{Path(label_path).name}")
                print(target_dir_cat / f"{Path(label_path).name}")
            elif unique_in_labels[1] == 2:
                img_lab.save(target_dir_lab_dog / f"{Path(label_path).name}")
                img.save(target_dir_dog / f"{Path(label_path).name}")
    return


separate_cat_dog(Path('res_out'), Path('res_out\\label'), Path('res_out\\cat'), Path('res_out\\cat\\label'), Path('res_out\\dog'), Path('res_out/dog/label'))
