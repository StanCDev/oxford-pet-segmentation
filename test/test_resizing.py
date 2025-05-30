import pytest
import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing.resizing import resize_directory

def test_resize_directory():
    src_dir = Path("res_out/")
    resize_directory(src_dir, (256,256))
    return

def test_label_resize_directory():
    src_l_dir = Path("res_out/label")
    resize_directory(src_l_dir, (256,256))
    return