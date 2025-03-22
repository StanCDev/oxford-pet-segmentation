from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from PIL import Image

from pathlib import Path

import json

from utils import label_to_one_hot

import re

DOGS = set(["american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle", "boxer", "chihuahua", "english_cocker_spaniel", "english_setter", "german_shorthaired", "great_pyrenees","havanese", "japanese_chin", "keeshond", "leonberger", "miniature_pinscher", "newfoundland", "pomeranian", "pug", "saint_bernard", "samoyed", "scottish_terrier", "shiba_inu", "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier"])
CATS = set(["abyssinian", "bengal", "birman", "bombay", "british_shorthair", "egyptian_mau", "maine_coon", "persian", "ragdoll", "russian_blue", "siamese", "sphynx"])

DOG_PROMPT = "an image of a dog"
CAT_PROMPT = "an image of a cat"

class SegmentationDataset(Dataset):
    """
    Create a train/test segmentation dataset
    """
    def __init__(self, image_path : Path, image_label_path : Path, json_mapping_path : Path, nn_type : str, train : bool = True, transform=ToTensor()):
        """
        """
        self.image_path = image_path
        self.image_label_path = image_label_path
        self.json_mapping_path = json_mapping_path
        self.train = train
        self.transform = transform
        self.accepted_exts = {".jpg", ".jpeg", ".png"}
        self.nn_type = nn_type

        with open(json_mapping_path, "r") as json_file:
            self.index_mapping = json.loads(json_file.read())
            self.index_mapping = {int(k): v for k, v in self.index_mapping.items()}

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        extension_train = ".jpg"
        extension_label = ".png"

        img_name : str = self.index_mapping.get(idx)
        if img_name is None:
            raise ValueError(f"Index not found in json file at {self.json_mapping_path}")
        # Load image and mask
        with Image.open(self.image_path / f"{img_name}{extension_train}") as image:
            ################################## Autoencoder ##################################
            if self.nn_type == "autoencoder":
                image = image.convert("RGB")
                if self.transform:
                        image = self.transform(image).float()
                return image, image
            ################################## OTHER ##################################
            if self.train:
                with Image.open(self.image_label_path / f"{img_name}{extension_label}") as label:
                    # Apply transformations
                    label = label.convert("RGB")
                    image = image.convert("RGB")
                    label = label_to_one_hot(label)
                    if self.transform:
                        image = self.transform(image).float()
                        label = self.transform(label).float()
                    ################################## CLIP ##################################
                    if self.nn_type == "CLIP":
                        animal_name = None
                        prompt = None

                        #0. convert img_name to lowercase
                        img_name = img_name.lower()
                        #1. get animal from img_name
                        pattern = re.compile("([a-z]+_)+", re.IGNORECASE)
                        result = pattern.match(img_name)
                        match result:
                            case None:
                                raise ValueError(f"Image name doesn't match the regular expression '([a-z]+_)+'. image name = {img_name}")
                            case _:
                                animal_name = result.group()
                                animal_name = animal_name[:-1]
                        
                        #2. check if animal is in cat or dog
                        if animal_name in CATS:
                            prompt = CAT_PROMPT
                        elif animal_name in DOGS:
                            prompt = DOG_PROMPT
                        else:
                            raise ValueError(f"Animal name is neither in the list of dogs or cats! animal name = {animal_name}")
                        
                        return prompt, image, label, idx
                    ################################## other ##################################
                    else:
                        return image, label
            else:
                image = self.transform(image).float()
                return image