from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

class SegmentationDataset(Dataset):
    def __init__(self, image_path : Path, image_label_path : Path, transform=None):
        self.image_path = image_path
        self.image_label_path = image_label_path
        self.transform = transform
        self.accepted_file_types = {".jpg", ".jpeg", ".png"}

    def __len__(self):
        
        dir_files = self.image_path.iterdir()
        nbr_images = len(list(filter(lambda image_path: (image_path.suffix.lower() in self.accepted_file_types) , dir_files)))

        dir_files_label = self.image_label_path.iterdir()
        nbr_images_label = len(list(filter(lambda image_path: (image_path.suffix.lower() in self.accepted_file_types) , dir_files_label)))

        if nbr_images != nbr_images_label:
            raise ValueError("Different number of images and labels!")

        return nbr_images

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_path + f"{idx}")
        mask = Image.open(self.mask_paths[idx] + f"{idx}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask