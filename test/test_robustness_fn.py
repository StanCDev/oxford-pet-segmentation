import pytest
import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.robustness_funcs import perturb_directory,  perturbate_image

def test_perturb_directory():
    src_dir = Path("res_out/")
    dest_dir = Path("res_perturb/")
    perturb_directory(src_dir, dest_dir, (256, 256), 'occlusion', 45)
    return
