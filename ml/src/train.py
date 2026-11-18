"""Train YOLO and log params, metrics and weights to MLflow."""
import argparse
import os
import shutil
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ml/configs/train.yaml")
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())

    settings.update({"mlflow": False})  # we log explicitly, avoid duplicate runs
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg["name"]):
        mlflow.log_params({k: v for k, v in cfg.items() if k != "experiment"})
        mlflow.log_artifact(cfg["data"], "dataset")

        model = YOLO(cfg["model"])
        res = model.train(
            data=str(Path(cfg["data"]).resolve()), epochs=cfg["epochs"], imgsz=cfg["imgsz"],
            batch=cfg["batch"], device=cfg["device"],
            project=str(Path(cfg["project"]).resolve()), name=cfg["name"], exist_ok=True,
        )
        mlflow.log_metrics({
            "precision": float(res.box.mp), "recall": float(res.box.mr),
            "map50": float(res.box.map50), "map50_95": float(res.box.map),
        })
        best = Path(res.save_dir) / "weights" / "best.pt"
        mlflow.log_artifact(str(best), "weights")
        Path("models/weights").mkdir(parents=True, exist_ok=True)
        shutil.copy(best, "models/weights/best.pt")
        print("saved models/weights/best.pt")


if __name__ == "__main__":
    main()
