"""Evaluate on the test split: precision, recall, mAP50, mAP50-95, per-class mAP."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/weights/best.pt")
    ap.add_argument("--data", default="data/datasets/v1/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=0)
    a = ap.parse_args()

    m = YOLO(a.weights).val(data=str(Path(a.data).resolve()), imgsz=a.imgsz,
                            split=a.split, device=a.device)
    report = {
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "map50": float(m.box.map50), "map50_95": float(m.box.map),
        "per_class_map50_95": {m.names[int(c)]: float(v)
                               for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])},
    }
    Path("models/eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
