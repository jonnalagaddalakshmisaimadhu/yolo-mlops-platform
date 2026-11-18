"""Download and extract a curated Hard Hat / Helmet dataset in YOLO format.

Sources: Hugging Face (keremberke/hard-hat-detection)
Outputs:
  data/annotations/images/*.jpg
  data/annotations/labels/*.txt
  data/annotations/classes.txt
"""
import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data/test.zip"
ANNOTATIONS_DIR = Path("data/annotations")
IMAGES_DIR = ANNOTATIONS_DIR / "images"
LABELS_DIR = ANNOTATIONS_DIR / "labels"
CLASSES_FILE = ANNOTATIONS_DIR / "classes.txt"


def download_and_setup(limit: int = 500):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Hard Hat dataset (~114 MB) from {DATA_URL}...")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req) as resp:
        zip_bytes = resp.read()
    
    print(f"Downloaded {len(zip_bytes) / (1024 * 1024):.1f} MB. Extracting and converting to YOLO format...")
    z = zipfile.ZipFile(io.BytesIO(zip_bytes))
    
    coco_data = json.loads(z.read("_annotations.coco.json").decode("utf-8"))
    
    # Map category id to name & index
    categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    class_names = [categories[k] for k in sorted(categories.keys())]
    
    CLASSES_FILE.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"Saved classes to {CLASSES_FILE}: {class_names}")

    # Build image map: id -> (filename, width, height)
    image_info = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco_data["images"]}
    
    # Group annotations by image_id
    ann_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann)

    extracted_count = 0
    # Process up to `limit` images for efficient training
    target_img_ids = list(image_info.keys())[:limit]
    
    for img_id in target_img_ids:
        file_name, width, height = image_info[img_id]
        if file_name not in z.namelist():
            continue
        
        # Save image
        img_path = IMAGES_DIR / file_name
        with open(img_path, "wb") as f:
            f.write(z.read(file_name))
        
        # Convert bounding boxes to YOLO format: <class_idx> <x_center> <y_center> <w> <h> (normalized)
        label_lines = []
        for ann in ann_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            x_min, y_min, w, h = ann["bbox"]
            if w <= 0 or h <= 0 or width <= 0 or height <= 0:
                continue
            x_center = (x_min + w / 2.0) / width
            y_center = (y_min + h / 2.0) / height
            norm_w = w / width
            norm_h = h / height
            
            # Clip between 0 and 1
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            label_lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        label_file = LABELS_DIR / f"{Path(file_name).stem}.txt"
        label_file.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        extracted_count += 1

    print(f"Successfully processed {extracted_count} labeled images into {IMAGES_DIR} and {LABELS_DIR}!")


if __name__ == "__main__":
    download_and_setup(limit=500)
