"""Generates a background-masked mirror of "tomato final dataset/" for testing
whether the ResNet50 classifier relies on background rather than the plant.

For every image with a matching COCO polygon annotation in the split's
_annotations.coco.json, all annotated regions (regardless of which of the 4
growth-stage categories they belong to -- every annotation on a given image
already shares that image's class, confirmed during validation) are rasterized
into one binary mask; pixels outside every polygon are set to black, pixels
inside are left untouched. Images with no matching annotation are skipped and
logged, never copied unmasked.

Usage: python scripts/generate_masked_dataset.py
"""
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "tomato final dataset")
DST_ROOT = os.path.join(PROJECT_ROOT, "tomato final dataset_masked")
REPORT_PATH = os.path.join(PROJECT_ROOT, "scripts", "masking_report.csv")
EXAMPLES_PATH = os.path.join(PROJECT_ROOT, "scripts", "masking_examples.png")

SPLITS = ["train", "valid", "test"]
CLASSES = ["developing", "flowering", "fruiting", "seeding"]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def polygon_mask(height, width, annotations):
    """Rasterizes every annotation's segmentation for one image into a single
    binary mask (255 = plant/keep, 0 = background). Handles both the normal
    COCO polygon format and the one COCO RLE-encoded segmentation found during
    validation.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    rle_mask = None

    for ann in annotations:
        seg = ann["segmentation"]
        if isinstance(seg, dict):
            # COCO RLE format: {"counts": ..., "size": [h, w]}
            from pycocotools import mask as mask_utils
            decoded = mask_utils.decode(seg)  # HxW array of 0/1
            if rle_mask is None:
                rle_mask = np.zeros((height, width), dtype=np.uint8)
            rle_mask = np.maximum(rle_mask, decoded.astype(np.uint8) * 255)
        else:
            for part in seg:
                if len(part) < 6:
                    continue  # degenerate polygon, <3 points
                pairs = list(zip(part[0::2], part[1::2]))
                draw.polygon(pairs, fill=255)

    mask_arr = np.array(mask)
    if rle_mask is not None:
        mask_arr = np.maximum(mask_arr, rle_mask)
    return mask_arr


def apply_mask(img, mask_arr):
    img_arr = np.array(img.convert("RGB"))
    out = img_arr.copy()
    out[mask_arr == 0] = 0
    return Image.fromarray(out)


def main():
    rows = []
    # One example per class first, then a couple of extra (split, cls) combos
    # for split diversity, once every class already has an example.
    example_targets = [(None, c) for c in CLASSES] + [("test", "fruiting"), ("valid", "seeding")]
    example_candidates = {t: None for t in example_targets}

    for split in SPLITS:
        json_path = os.path.join(SRC_ROOT, split, "_annotations.coco.json")
        with open(json_path, encoding="utf-8") as f:
            coco = json.load(f)

        fname_to_image = {im["file_name"]: im for im in coco["images"]}
        anns_by_image_id = {}
        for ann in coco["annotations"]:
            anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

        for cls in CLASSES:
            src_dir = os.path.join(SRC_ROOT, split, cls)
            dst_dir = os.path.join(DST_ROOT, split, cls)
            os.makedirs(dst_dir, exist_ok=True)

            filenames = sorted(
                f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)
            )

            for fname in filenames:
                image_entry = fname_to_image.get(fname)
                annotations = anns_by_image_id.get(image_entry["id"], []) if image_entry else []

                had_annotation = bool(annotations)
                masked = False

                if had_annotation:
                    src_path = os.path.join(src_dir, fname)
                    img = Image.open(src_path).convert("RGB")
                    width, height = img.size
                    mask_arr = polygon_mask(height, width, annotations)
                    masked_img = apply_mask(img, mask_arr)
                    dst_path = os.path.join(dst_dir, fname)
                    masked_img.save(dst_path, quality=95)
                    masked = True

                    for split_key, cls_key in example_targets:
                        if cls_key != cls:
                            continue
                        if split_key is not None and split_key != split:
                            continue
                        if example_candidates[(split_key, cls_key)] is None:
                            example_candidates[(split_key, cls_key)] = (src_path, dst_path, split, cls)
                            break
                else:
                    print(f"[skip] no annotation: {split}/{cls}/{fname}")

                rows.append({
                    "filename": fname,
                    "split": split,
                    "growth_stage": cls,
                    "had_annotation": had_annotation,
                    "masked": masked,
                })

            print(f"{split}/{cls}: {len(filenames)} images processed")

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "split", "growth_stage", "had_annotation", "masked"])
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_masked = sum(r["masked"] for r in rows)
    n_skipped = n_total - n_masked
    print(f"\nDone. {n_masked}/{n_total} images masked, {n_skipped} skipped (no annotation).")
    print(f"Report saved to {REPORT_PATH}")

    save_examples(example_candidates)


def save_examples(example_candidates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [v for v in example_candidates.values() if v is not None]
    if not pairs:
        print("No examples available to plot.")
        return

    fig, axes = plt.subplots(2, len(pairs), figsize=(4 * len(pairs), 8))
    if len(pairs) == 1:
        axes = axes.reshape(2, 1)

    for col, (src_path, dst_path, split, cls) in enumerate(pairs):
        orig = Image.open(src_path).convert("RGB")
        masked = Image.open(dst_path).convert("RGB")
        axes[0, col].imshow(orig)
        axes[0, col].set_title(f"{cls} ({split})\noriginal")
        axes[0, col].axis("off")
        axes[1, col].imshow(masked)
        axes[1, col].set_title("masked")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(EXAMPLES_PATH, dpi=120)
    print(f"Example comparison figure saved to {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
