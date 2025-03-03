from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from PIL import Image

from pathlib import Path

import json

from utils import label_to_one_hot

class SegmentationDataset(Dataset):
    """
    Create a train/test segmentation dataset
    """
    def __init__(self, image_path : Path, image_label_path : Path, json_mapping_path : Path, train : bool = True, transform=ToTensor()):
        """
        """
        self.image_path = image_path
        self.image_label_path = image_label_path
        self.json_mapping_path = json_mapping_path
        self.train = train
        self.transform = transform
        self.accepted_exts = {".jpg", ".jpeg", ".png"}

        with open(json_mapping_path, "r") as json_file:
            self.index_mapping = json.loads(json_file.read())
            self.index_mapping = {int(k): v for k, v in self.index_mapping.items()}

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        ### CHANGE THIS ASAP ### 
        extension_train = ".jpg"
        extension_label = ".png"
        ###
        img_name = self.index_mapping.get(idx)
        if img_name is None:
            raise ValueError(f"Index not found in json file at {self.json_mapping_path}")
        # Load image and mask
        with Image.open(self.image_path / f"{img_name}{extension_train}") as image:
            if self.train:
                with Image.open(self.image_label_path / f"{img_name}{extension_label}") as label:
                    # Apply transformations
                    label = label.convert("RGB")
                    image = image.convert("RGB")
                    label = label_to_one_hot(label)
                    if self.transform:
                        image = self.transform(image).float()
                        label = self.transform(label).float()
                    return image, label
            else:
                image = self.transform(image).float()
                return image