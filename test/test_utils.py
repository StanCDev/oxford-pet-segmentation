import pytest
import sys
from pathlib import Path

import numpy as np

from PIL import Image

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import label_to_one_hot, COLOR_MAP, RED, GREEN, WHITE, CAT, DOG, BACKGROUND, BLACK


def test_on_image_basic():
    y = np.zeros((5,5,3),dtype=np.uint8)
    y[0,0,:] = np.array(RED) #red
    y[1,1,:] = np.array(GREEN) #green
    y[3,3,:] = np.array(WHITE) #white
    out = label_to_one_hot(Image.fromarray(y))
    assert np.all(out[0,0,:] == CAT), "cat here"
    assert np.all(out[1,1,:] == DOG), "dog here"
    assert np.all(out[3,3,:] == BACKGROUND), "backgound here"

def test_on_images():
    labels = Path("res_out/label/")

    for image_path in labels.iterdir():
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image = Image.open(image_path).convert("RGB")
            arr = np.array(image)
            w,h,ch = arr.shape
            result = label_to_one_hot(image)

            acceptable_colors = {RED,GREEN,WHITE,BLACK}
            for i in range(w):
                for j in range(h):
                    mask_color = arr[i,j,:]
                    assert mask_color.size == 3
                    r = mask_color[0]
                    g = mask_color[1]
                    b = mask_color[2]
                    assert (r,g,b) in acceptable_colors, f"(r,g,b) = ({r},{g},{b}) not in acceptable values"
                    mask_value = result[i,j,:]
                    assert mask_value.size == 3
                    assert np.all(COLOR_MAP.get((r,g,b)) == mask_value), f"bad mask value. (r,g,b) = ({r},{g},{b}). Expected mask = {COLOR_MAP.get(r,g,b)} got {mask_value}"


# def test_IoU():
#     labels = Path("res_out/label/")
#     for image_path in labels.iterdir():
#         if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
#             image = Image.open(image_path).convert("RGB")
#             temp = np.array(image)
#             arr = np.zeros((1))