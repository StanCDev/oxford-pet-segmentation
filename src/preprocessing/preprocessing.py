"""
Main preprocessing script that resizes + augments data 
"""
import argparse

from preprocessing.resizing import resize_directory
from pathlib import Pathlib



def main() -> None:
    ###1. Resize all images in the same directory

    ###2. Augment all resized images to output directory

    ###3. Create json mapping
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_path', default=None, type=str, help="path json file to overwrite and save mapping to")
    parser.add_argument('--data_path', default=None, type=str, help="path to your dataset")
    args = parser.parse_args()
    main(Path(args.json_path), Path(args.data_path))