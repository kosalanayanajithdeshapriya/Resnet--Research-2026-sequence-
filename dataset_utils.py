"""SplitFolderDataset lives in its own module (not a notebook cell) because Windows
DataLoader workers (num_workers>0) spawn subprocesses that re-import __main__, and a
class defined inline in a notebook cell isn't importable there -> AttributeError.
"""
import os

from PIL import Image
from torch.utils.data import Dataset


class SplitFolderDataset(Dataset):
    """Reads dataset/<class>/<split>/*.jpg -- i.e. class-first, split-nested-inside-class
    layout -- as opposed to torchvision's ImageFolder which expects split-first/class-inside.
    Classes are sorted alphabetically to match ImageFolder's convention.
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, root, split, transform, classes=None):
        self.transform = transform
        self.classes = classes or sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        self.samples = []
        for label_idx, cls in enumerate(self.classes):
            split_dir = os.path.join(root, cls, split)
            for fname in sorted(os.listdir(split_dir)):
                if fname.lower().endswith(self.IMG_EXTS):
                    self.samples.append((os.path.join(split_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class PooledDataset(Dataset):
    """Wraps an explicit list of (path, label) samples with a given transform.
    Used for cross-validation folds, where each fold needs the same pooled
    samples re-sliced with a train or eval transform.
    """

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
