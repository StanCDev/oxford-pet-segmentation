import pytest
import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing.augmentation_update import aug_directory, augment_image


def test_aug_directory():
    src_dir = Path("res_out/")
    dest_dir = Path("res_aug/")
# dest_dir_lab and src_dir_lab added for labels
    src_dir_lab = Path("res_out/label")
    dest_dir_lab = Path("res_aug/label")
# dest_dir_lab and src_dir_lab included for labels
    aug_directory(src_dir, dest_dir, src_dir_lab, dest_dir_lab, col_jit = 0.5, rand_rot = 45, rand_hflip = 0.5, el_trans_a = 50, el_trans_s =5) # Test with % probability for colorjitter, % for random rotation and % for random horizontal flip
    return
