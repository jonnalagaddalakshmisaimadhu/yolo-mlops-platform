"""Build a YOLO dataset (train/val/test + data.yaml) from CVAT's YOLO export.

Expected input:
  data/annotations/images/*.jpg
  data/annotations/labels/*.txt     (class x_center y_center w h, normalised)
  data/annotations/classes.txt      (one class name per line)
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/annotations/images")
    ap.add_argument("--labels", default="data/annotations/labels")
    ap.add_argument("--classes", default="data/annotations/classes.txt")
    ap.add_argument("--out", default="data/datasets/v1")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    labels_dir = Path(a.labels)
    pairs = [(p, labels_dir / f"{p.stem}.txt")
             for p in sorted(Path(a.images).iterdir()) if p.suffix.lower() in EXTS]
    pairs = [(i, l) for i, l in pairs if l.exists()]  # keep only labelled images
    if len(pairs) < 3:
        raise SystemExit(f"Need at least 3 labelled images, found {len(pairs)}")

    random.Random(a.seed).shuffle(pairs)
    n = len(pairs)
    n_test = max(1, round(n * a.test))
    n_val = max(1, round(n * a.val))
    splits = {"test": pairs[:n_test],
              "val": pairs[n_test:n_test + n_val],
              "train": pairs[n_test + n_val:]}

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for name, items in splits.items():
        (out / "images" / name).mkdir(parents=True)
        (out / "labels" / name).mkdir(parents=True)
        for img, lab in items:
            shutil.copy2(img, out / "images" / name / img.name)
            shutil.copy2(lab, out / "labels" / name / lab.name)

    names = [l.strip() for l in Path(a.classes).read_text().splitlines() if l.strip()]
    # No absolute `path:` on purpose, so the dataset stays portable (DVC/Git friendly).
    (out / "data.yaml").write_text(yaml.safe_dump({
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": dict(enumerate(names)),
    }, sort_keys=False))
    print({k: len(v) for k, v in splits.items()}, "->", out)


if __name__ == "__main__":
    main()
