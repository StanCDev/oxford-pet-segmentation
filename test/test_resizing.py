import pytest
import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing.resizing import resize_directory, resize_image

def test_resize_directory():
    src_dir = Path("res/")
    dest_dir = Path("res_out/")
    resize_directory(src_dir, dest_dir, (256,256))
    return