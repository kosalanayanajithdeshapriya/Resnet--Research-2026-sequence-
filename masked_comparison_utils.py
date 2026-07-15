"""Loader for "tomato final dataset/" and "tomato final dataset_masked/", which are laid
out split-first/class-inside (root/<split>/<class>/*.jpg) -- the opposite nesting order
from dataset_utils.SplitFolderDataset (root/<class>/<split>/*.jpg), which reads the older
"dataset/" folder. Kept as its own module (not added to dataset_utils.py, which the
original resnet50.ipynb depends on and must not change) but mirrors the same interface
and conventions (alphabetical class order, same extension filter, plain (path, label)
samples list) so training/eval code is otherwise identical between the two notebooks.

Also lives outside the notebook so Windows DataLoader workers (num_workers>0) can import it.
"""
import os

from PIL import Image
from torch.utils.data import Dataset


class SplitFirstFolderDataset(Dataset):
    """Reads root/<split>/<class>/*.jpg."""

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, root, split, transform, classes=None):
        self.transform = transform
        split_root = os.path.join(root, split)
        self.classes = classes or sorted(
            d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))
        )
        self.samples = []
        for label_idx, cls in enumerate(self.classes):
            class_dir = os.path.join(split_root, cls)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(self.IMG_EXTS):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
