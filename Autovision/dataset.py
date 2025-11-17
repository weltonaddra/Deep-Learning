import os
import torch
import random
import numpy as np
import torchvision
from torchvision import datasets, transforms
from PIL import Image, ImageOps
from config import Config


# Resizes Input images to match the expected input size while keeping the aspect ratio
def letterbox_to_square(img: Image.Image, target=224) -> Image:
    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    pad_w, pad_h = target - new_w, target - new_h
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2

    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=Config.PAD_FILL_RGB)


class ChestXRayDataset:
    # Handles transforms, dataset creation, and dataloaders.

    def __init__(self, data_dir=Config.DATA_DIR, img_size=Config.IMG_SIZE, subset_fraction=1.0, seed=42) -> None:
        self.data_dir = data_dir  # Path to dataset directory
        self.img_size = img_size  # The size the image should be after being resized
        self.transforms = self._build_transforms()  
        self.subset_fraction = subset_fraction   # size of the dataset to use (for quick testing)
        self.seed = seed  # seed for producing consistent results

        # Loads the original training datasets so it can get the class names 
        orig_train_ds = datasets.ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transforms['train'])
        self.class_names = orig_train_ds.classes

        # Calls functions to build the datasets, data loaders and gets the current device type 
        self.datasets = self._build_datasets()
        self.dataloaders = self._build_dataloaders()
        self.dataset_sizes = {x: len(self.datasets[x]) for x in ['train', 'val', 'test', 'selected']}
        self.device = self._get_device()

        print(f"Classes: {self.class_names}")
        print(f"Train images: {self.dataset_sizes['train']}, Val images: {self.dataset_sizes['val']}, Test images: {self.dataset_sizes['test']}")
        


    def _build_datasets(self) -> dict:
        # Splits the dataset into 4 splits: train, val, test, selected (This is for t)
        splits = ['train', 'val', 'test','selected']
        datasets_dict = {}

        for split in splits:
            # Sets the image folder for the split 
            ds = datasets.ImageFolder(os.path.join(self.data_dir, split), transform=self.transforms[split])

            # If a smaller fraction of the dataset is picked, it sets a random seed and splits the dataset accordingly 
            if self.subset_fraction < 1.0:
                random.seed(self.seed)
                np.random.seed(self.seed)
                indices = np.random.choice(len(ds), int(len(ds) * self.subset_fraction), replace=False)
                ds = torch.utils.data.Subset(ds, indices)

            datasets_dict[split] = ds
            
        return datasets_dict

    def _build_transforms(self) -> dict:
        letterbox_224 = transforms.Lambda(lambda im: letterbox_to_square(im, target=self.img_size))
        return {
            'train': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                letterbox_224,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5, fill=Config.PAD_FILL_RGB),
                transforms.ToTensor(),
                transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
            ]),
            'test': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                letterbox_224,
                transforms.ToTensor(),
                transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
            ]),
            'selected': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                letterbox_224,
                transforms.ToTensor(),
                transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
            ]),
            'val': transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                letterbox_224,
                transforms.ToTensor(),
                transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
            ])
        }

    def _build_dataloaders(self) -> dict:
        return {
            'train': torch.utils.data.DataLoader(
                self.datasets['train'], batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS),
            'test': torch.utils.data.DataLoader(
                self.datasets['test'], batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS),
            'selected': torch.utils.data.DataLoader(
                self.datasets['selected'], batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS),           
            'val': torch.utils.data.DataLoader(
                self.datasets['val'], batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
        }

    def _get_device(self) -> str:
        try:
            return torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        except AttributeError:
            return "cuda" if torch.cuda.is_available() else "cpu"
