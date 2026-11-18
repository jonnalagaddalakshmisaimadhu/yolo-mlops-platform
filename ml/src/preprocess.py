"""Resize oversized images and drop unreadable ones: data/raw -> data/processed."""
import argparse
from pathlib import Path
import cv2

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw")
    ap.add_argument("--dst", default="data/processed")
    ap.add_argument("--max-size", type=int, default=1280, help="longest side in px")
    a = ap.parse_args()

    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for p in sorted(Path(a.src).rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            bad += 1
            print(f"skipped unreadable: {p}")
            continue
        h, w = img.shape[:2]
        scale = a.max_size / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / f"{p.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ok += 1
    print(f"processed {ok} images, skipped {bad}")


if __name__ == "__main__":
    main()
