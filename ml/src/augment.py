"""Offline augmentation of the TRAIN split only (YOLO boxes stay in sync)."""
import argparse
from pathlib import Path

import albumentations as A
import cv2
import yaml


def build(cfg):
    return A.Compose(
        [
            A.HorizontalFlip(p=cfg["hflip"]),
            A.RandomBrightnessContrast(p=cfg["brightness_contrast"]),
            A.HueSaturationValue(p=cfg["hsv"]),
            A.GaussNoise(p=cfg["noise"]),
            A.Affine(scale=(0.85, 1.15), translate_percent=(-0.05, 0.05),
                     rotate=(-10, 10), p=cfg["affine"]),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_ids"], min_visibility=0.3),
    )


def read_labels(path):
    boxes, classes = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
        x, y = min(max(x, 0), 1), min(max(y, 0), 1)
        w, h = min(max(w, 1e-4), 1), min(max(h, 1e-4), 1)
        boxes.append([x, y, w, h])
        classes.append(c)
    return boxes, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/datasets/v1")
    ap.add_argument("--config", default="ml/configs/augment.yaml")
    ap.add_argument("--copies", type=int, default=2)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    aug = build(cfg)
    root = Path(a.dataset)
    img_dir, lab_dir = root / "images/train", root / "labels/train"
    made = 0
    for img_path in sorted(img_dir.glob("*")):
        if "_aug" in img_path.stem:       # don't augment augmented files on re-runs
            continue
        lab_path = lab_dir / f"{img_path.stem}.txt"
        image = cv2.imread(str(img_path))
        if image is None or not lab_path.exists():
            continue
        boxes, classes = read_labels(lab_path)
        for i in range(a.copies):
            out = aug(image=image, bboxes=boxes, class_ids=classes)
            if boxes and not out["bboxes"]:
                continue                  # every box was cropped away
            stem = f"{img_path.stem}_aug{i}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), out["image"])
            (lab_dir / f"{stem}.txt").write_text("\n".join(
                f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                for c, (x, y, w, h) in zip(out["class_ids"], out["bboxes"])))
            made += 1
    print(f"wrote {made} augmented images to {img_dir}")


if __name__ == "__main__":
    main()
