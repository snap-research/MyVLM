import numpy as np
import torch
import torchvision
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import List, NamedTuple, Tuple
from myvlm.common import VALID_IMAGE_EXTENSIONS


class Paths(NamedTuple):
    train_positive_paths: List[Path]
    train_negative_paths: List[Path]
    val_positive_paths: List[Path]
    val_negative_paths: List[Path]


def get_train_image_transforms() -> torchvision.transforms.Compose:
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomRotation(degrees=30),
    ])
    return train_transform


def load_and_split_image_paths(positive_image_root: Path,
                               negative_image_root: Path,
                               n_positive_samples: int = 4,
                               train_percent_for_negatives: float = 0.5) -> Paths:
    # First, lets get the positive images
    positive_image_paths = [p for p in positive_image_root.glob("*") if p.suffix in VALID_IMAGE_EXTENSIONS]
    train_positive_paths = np.random.choice(positive_image_paths,
                                            size=min(n_positive_samples, len(positive_image_paths)),
                                            replace=False).tolist()
    val_positive_paths = [p for p in positive_image_paths
                          if p not in train_positive_paths and p.suffix in VALID_IMAGE_EXTENSIONS]
    # Now we'll split the negative images
    negative_image_paths = list(negative_image_root.glob("*"))
    np.random.shuffle(negative_image_paths)
    train_negative_paths = negative_image_paths[:int(len(negative_image_paths) * train_percent_for_negatives)]
    val_negative_paths = negative_image_paths[int(len(negative_image_paths) * train_percent_for_negatives):]
    print(f"Number of positive train images: {len(train_positive_paths)}")
    print(f"Number of negative train images: {len(train_negative_paths)}")
    print(f"Number of positive val images: {len(val_positive_paths)}")
    print(f"Number of negative val images: {len(val_negative_paths)}")

    # Let's make sure that there is no leakage
    train_paths = set(train_positive_paths + train_negative_paths)
    val_paths = set(val_positive_paths + val_negative_paths)
    if len(train_paths.intersection(val_paths)) > 0:
        raise Exception("Oops! There is leakage between train and val!")

    return Paths(train_positive_paths, train_negative_paths, val_positive_paths, val_negative_paths)


class ClassificationDataset(Dataset):

    def __init__(self, positive_paths: List[Path], negative_paths: List[Path], processor, transforms=None):
        self.samples = []
        for path in positive_paths:
            self.samples.append({"image": path, "label": 1})
        for path in negative_paths:
            self.samples.append({"image": path, "label": 0})
        self.processor = processor
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        image = Image.open(item['image']).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        if self.processor is not None:
            image = self.processor(image)
        return image, item['label']
