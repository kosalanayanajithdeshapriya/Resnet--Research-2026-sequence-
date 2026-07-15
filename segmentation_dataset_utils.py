"""Binary plant/background segmentation dataset built from the same COCO polygon
annotations used for the classification masking (tomato final dataset/<split>/_annotations.coco.json).
All polygons on an image (regardless of growth-stage category) are unioned into one
binary mask: 1 = plant/leaf material, 0 = background. Images with no annotation are
skipped (same 3 images identified during the original COCO validation).

Uses torchvision.transforms.v2 with tv_tensors so the image and its mask receive the
IDENTICAL random crop/flip/rotation each call -- a plain torchvision.transforms.Compose
applied separately to image and mask would pick different random parameters for each
and misalign them.
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import tv_tensors


def build_binary_mask(height, width, annotations):
    """Same rasterization approach as scripts/generate_masked_dataset.py's polygon_mask,
    duplicated here (rather than imported) so this module has no dependency on that
    script and can't be affected by changes to it.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    rle_mask = None

    for ann in annotations:
        seg = ann["segmentation"]
        if isinstance(seg, dict):
            from pycocotools import mask as mask_utils
            decoded = mask_utils.decode(seg)
            if rle_mask is None:
                rle_mask = np.zeros((height, width), dtype=np.uint8)
            rle_mask = np.maximum(rle_mask, decoded.astype(np.uint8))
        else:
            for part in seg:
                if len(part) < 6:
                    continue
                pairs = list(zip(part[0::2], part[1::2]))
                draw.polygon(pairs, fill=1)

    mask_arr = np.array(mask, dtype=np.uint8)
    if rle_mask is not None:
        mask_arr = np.maximum(mask_arr, rle_mask)
    return mask_arr  # HxW, values in {0, 1}


class CocoPlantSegmentationDataset(Dataset):
    IMG_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, root, split, joint_transform, classes=None):
        """root: e.g. 'tomato final dataset'. split: 'train'/'valid'/'test'.
        joint_transform: a torchvision.transforms.v2 pipeline applied to
        (tv_tensors.Image, tv_tensors.Mask) together.
        """
        self.joint_transform = joint_transform
        split_root = os.path.join(root, split)
        self.classes = classes or sorted(
            d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))
        )

        json_path = os.path.join(root, split, "_annotations.coco.json")
        with open(json_path, encoding="utf-8") as f:
            coco = json.load(f)
        fname_to_image = {im["file_name"]: im for im in coco["images"]}
        anns_by_image_id = {}
        for ann in coco["annotations"]:
            anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

        self.samples = []  # (path, image_id, height, width)
        for cls in self.classes:
            class_dir = os.path.join(split_root, cls)
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(self.IMG_EXTS):
                    continue
                image_entry = fname_to_image.get(fname)
                if image_entry is None or not anns_by_image_id.get(image_entry["id"]):
                    continue  # no annotation -- skip, consistent with generate_masked_dataset.py
                self.samples.append((
                    os.path.join(class_dir, fname),
                    image_entry["id"],
                    image_entry["height"],
                    image_entry["width"],
                ))

        self._anns_by_image_id = anns_by_image_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, image_id, height, width = self.samples[idx]
        img = Image.open(path).convert("RGB")
        mask_arr = build_binary_mask(height, width, self._anns_by_image_id[image_id])

        img_t = tv_tensors.Image(img)
        mask_t = tv_tensors.Mask(torch.from_numpy(mask_arr).unsqueeze(0))  # 1xHxW

        img_t, mask_t = self.joint_transform(img_t, mask_t)
        return img_t, mask_t.squeeze(0).long()
