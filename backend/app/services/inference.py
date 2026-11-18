import io
import os
import time
from pathlib import Path

from PIL import Image

MODEL_PATH = os.getenv("MODEL_PATH", "/models/weights/best.pt")
CONF = float(os.getenv("CONF_THRESHOLD", "0.25"))


class Detector:
    """Lazy-loads the model so the API starts (and /health works) before training is done."""

    def __init__(self):
        self.model = None

    @property
    def ready(self) -> bool:
        return Path(MODEL_PATH).exists()

    @property
    def name(self) -> str:
        return Path(MODEL_PATH).name

    def _load(self):
        if self.model is None:
            from ultralytics import YOLO  # imported here: heavy
            self.model = YOLO(MODEL_PATH)
        return self.model

    def predict(self, data: bytes, conf: float = CONF) -> dict:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        t0 = time.perf_counter()
        result = self._load().predict(img, conf=conf, verbose=False)[0]
        latency = (time.perf_counter() - t0) * 1000
        dets = [
            {"label": result.names[int(c)], "confidence": round(float(s), 4),
             "box": [round(float(v), 1) for v in xyxy]}  # x1, y1, x2, y2 in pixels
            for xyxy, s, c in zip(result.boxes.xyxy.tolist(),
                                  result.boxes.conf.tolist(), result.boxes.cls.tolist())
        ]
        return {"width": img.width, "height": img.height,
                "latency_ms": round(latency, 1), "detections": dets}


detector = Detector()
