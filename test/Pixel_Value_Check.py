

from pathlib import Path
from PIL import Image
import numpy as np

path = Path("res_aug/label/")
label_info_in = np.array([])
label_info_out = np.array([])
for label_path in path.iterdir():
    with Image.open(label_path) as img_lab: 
        unique_in_labels, count = np.unique(img_lab, return_counts=True)
        print(unique_in_labels, count, sum(count))
