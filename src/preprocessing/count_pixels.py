import argparse

from pathlib import Path
from PIL import Image
import numpy as np

RED = np.array([128, 0, 0])
GREEN = np.array([0, 128, 0])
WHITE = np.array([255, 255, 255])
BLACK = np.array([0, 0, 0])

def main(args):
    path = Path(args.path)

    accepted_file_types = {".jpg", ".jpeg", ".png"}

    count = {"RED" : 0, "GREEN" : 0, "WHITE" : 0, "BLACK" : 0}
    mapping = {"RED" : RED, "GREEN" : GREEN, "WHITE" : WHITE, "BLACK" : BLACK}

    dir_files = path.iterdir()
    for image_path in dir_files:
        if image_path.suffix.lower() in accepted_file_types:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                data = np.array(img)

                assert len(data.shape) == 3, "Images must have 3 channels"

                for color, color_array in mapping.items():
                    total = np.sum(np.all(data == color_array, axis=-1))
                    count[color] += total
    ### Print results
    print(f"Red = {count["RED"]}, Green = {count["GREEN"]}, White = {count["WHITE"]}, Black = {count["BLACK"]}")
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None,
                        help="Path to label dataset")
    args = parser.parse_args()
    main(args)