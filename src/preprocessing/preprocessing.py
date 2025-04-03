"""
Main preprocessing script that resizes + augments data 
"""
import argparse
from pathlib import Path
import json

from resizing import resize_directory
from augmentation import aug_directory
from make_json import make_json_file

from torchvision.transforms import InterpolationMode

# train_dim = (256,256)
# label_dim = (256,256)
train_dim = (352,352)
label_dim = (224,224)


def main(args) -> None:
    ###1. Find set differences of images
    train = Path(args.train)
    label = Path(args.label)

    train_list = train.iterdir()
    label_list = label.iterdir()

    train_set = set([path.stem for path in train_list])
    label_set = set([path.stem for path in label_list])

    diff = train_set.difference(label_set)
    print(f"Directories differ in {len(diff)} file(s) being: {diff}")

    ###2. Resize all images
    if args.resize:
        resize_directory(train, train_dim, True)
        resize_directory(label, label_dim, True,interpolation_mode=InterpolationMode.NEAREST)

    ###3. Augment all resized images
    if args.augment:
        aug_directory(
            train, 
            None, 
            label, 
            None, 
            col_jit = 0.5, 
            rand_rot = 45, 
            rand_hflip = 0.5, 
            el_trans_a = 50, 
            el_trans_s =5, 
            el_trans_prob= 0.1, 
            col_jit_prob = 0.5, 
            in_place=True
        ) 

        train_list = train.iterdir()
        label_list = label.iterdir()
    ###4. Create json mapping
    if args.json_file:
        make_json_file(train,label,Path(args.json))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=None, type=str, help="path to directory with train images")
    parser.add_argument('--label', default=None, type=str, help="path to directory with label images")
    parser.add_argument('--json', default=None, type=str, help="path json file to overwrite and save mapping to")
    parser.add_argument('--resize', action="store_true",default=False)
    parser.add_argument('--augment', action="store_true",default=False)
    parser.add_argument('--json_file', action="store_true",default=False)
    args = parser.parse_args()
    if not(args.train) or not(args.label):
        raise ValueError("Must input train and label path if resize/augment of images wanted or json map")
    
    main(args)