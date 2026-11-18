"""Export trained weights (ONNX by default) into models/exported/."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/weights/best.pt")
    ap.add_argument("--format", default="onnx", help="onnx | engine (TensorRT) | openvino ...")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    out = YOLO(a.weights).export(format=a.format, imgsz=a.imgsz)
    dest = Path("models/exported")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, dest / Path(out).name)
    print("exported ->", dest / Path(out).name)


if __name__ == "__main__":
    main()
