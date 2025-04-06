"""
Main preprocessing script that resizes + augments data, generates a json mapping, and generates prompt dataset
"""
import argparse
from pathlib import Path

from resizing import resize_directory
from augmentation import aug_directory
from make_json import make_json_file
from image_centre import mask_centre_directory

from torchvision.transforms import InterpolationMode



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

    train_dim = (args.train_dim, args.train_dim)
    label_dim = (args.label_dim, args.label_dim)
    
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

    if args.prompt_image:
        mask_centre_directory(train, label, True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', default=None, type=str, help="path to directory with train images")
    parser.add_argument('--label', default=None, type=str, help="path to directory with label images")
    parser.add_argument('--json', default=None, type=str, help="path json file to overwrite and save mapping to")
    parser.add_argument('--resize', action="store_true",default=False)
    parser.add_argument('--augment', action="store_true",default=False)
    parser.add_argument('--json_file', action="store_true",default=False)
    parser.add_argument('--prompt_image', action="store_true",default=False)

    parser.add_argument('--train_dim', default=352, type=int, help="training image dimension")
    parser.add_argument('--label_dim', default=224, type=int, help="label image dimension")
    
    args = parser.parse_args()
    if not(args.train) or not(args.label):
        raise ValueError("Must input train and label path if resize/augment of images wanted or json map")
    
    main(args)