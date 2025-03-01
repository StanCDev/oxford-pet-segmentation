from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from PIL import Image
from pathlib import Path

from ..utils import color_to_mask

class SegmentationDataset(Dataset):
    """
    Create a train/test segmentation dataset
    """
    def __init__(self, image_path : Path, image_label_path : Path, train : bool = True, transform=ToTensor):
        """
        """
        self.image_path = image_path
        self.image_label_path = image_label_path
        self.train = train
        self.transform = transform
        self.accepted_file_types = {".jpg", ".jpeg", ".png"}

    def __len__(self):
        dir_files = self.image_path.iterdir()
        nbr_images = len(list(filter(lambda image_path: (image_path.suffix.lower() in self.accepted_file_types) , dir_files)))

        if self.train:
            dir_files_label = self.image_label_path.iterdir()
            nbr_images_label = len(list(filter(lambda image_path: (image_path.suffix.lower() in self.accepted_file_types) , dir_files_label)))

            if nbr_images != nbr_images_label:
                raise ValueError("Different number of images and labels!")

        return nbr_images

    def __getitem__(self, idx):
        # ADD FILE EXTENSION
        extension = "...???"
        # Load image and mask
        with Image.open(self.image_path / f"{idx}" + extension) as image:
            if self.train:
                with Image.open(self.image_label_path / f"{idx}" + extension) as label:
                    # Apply transformations
                    label = color_to_mask(label)
                    if self.transform:
                        image = self.transform(image)
                        label = self.transform(label)
                    return image, label
            else:
                image = self.transform(image)
                return image