# CV Platform: base project

> **Production-grade MLOps platform to train YOLO models, track with MLflow, and serve with FastAPI & React.**

Annotate → train YOLO → track in MLflow → serve with FastAPI → use from React.

## Flow Diagram

```mermaid
flowchart TD
    %% Phase 1
    subgraph P1["Phase 1: Setup & Services"]
        A["Start Infrastructure<br/>(make up)"] --> B["Services Running:<br/>Postgres, MinIO, MLflow, API, Frontend"]
        C["Setup Python ML Env<br/>(make setup)"]
    end

    %% Phase 2
    subgraph P2["Phase 2: Data Preparation & Annotation"]
        D["Add Raw Images<br/>(data/raw/)"] --> E["Preprocess & Resize<br/>(make preprocess)"]
        E --> F["Annotate Objects<br/>(CVAT / Roboflow - YOLO Format)"]
        F --> G["Place Annotations<br/>(data/annotations/images, labels, classes.txt)"]
        G --> H["Train/Val/Test Split<br/>(make split -> data/datasets/v1/)"]
        H --> I["Data Augmentation<br/>(make augment)"]
    end

    %% Phase 3
    subgraph P3["Phase 3: Training & Experiment Tracking"]
        I --> J["Train YOLO Model<br/>(make train)"]
        J <--> K["MLflow & MinIO<br/>(Track Loss, mAP, Artifacts)"]
        J --> L["Trained Model Saved<br/>(models/weights/best.pt)"]
    end

    %% Phase 4
    subgraph P4["Phase 4: Evaluation & Model Export"]
        L --> M["Evaluate Metrics<br/>(make eval -> models/eval.json)"]
        L --> N["Export to ONNX<br/>(make export - Optional for Speed)"]
    end

    %% Phase 5
    subgraph P5["Phase 5: Serving & Live UI Testing"]
        L & N --> O["FastAPI Backend<br/>(:8000 /predict)"]
        O <--> P["React Frontend UI<br/>(:3000 Detector)"]
        P --> Q["Upload Image / Webcam<br/>See Live Bounding Boxes!"]
    end

    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
```

## Project Implementation Flow: Workplace Safety & Helmet Detection

```mermaid
flowchart TD
    %% Step 1
    subgraph S1["Step 1: Data Acquisition"]
        A["Download Safety Helmet Dataset<br/>(Roboflow / Kaggle Open Data)"] --> B["Store in data/raw/<br/>(~200-500 Images)"]
    end

    %% Step 2
    subgraph S2["Step 2: Data Processing & Labels"]
        B --> C["Preprocess & Resize<br/>python ml/src/preprocess.py"]
        C --> D["Annotations in YOLO Format<br/>classes.txt, images/, labels/"]
    end

    %% Step 3
    subgraph S3["Step 3: Dataset Splitting & Augmentation"]
        D --> E["Split Dataset (Train: 70%, Val: 20%, Test: 10%)<br/>make split -> data/datasets/v1/"]
        E --> F["Data Augmentation (Flips, Brightness, Crop)<br/>make augment"]
    end

    %% Step 4
    subgraph S4["Step 4: Model Training & Experiment Tracking"]
        F --> G["YOLOv8 Training<br/>make train (ml/configs/train.yaml)"]
        G <--> H["MLflow Dashboard (:5000)<br/>Track Loss, Precision, Recall, mAP50"]
        G --> I["Saved Weights<br/>models/weights/best.pt"]
    end

    %% Step 5
    subgraph S5["Step 5: Evaluation & Optimization"]
        I --> J["Evaluate Model on Test Set<br/>make eval -> models/eval.json"]
        I --> K["Optional: Export to ONNX<br/>make export"]
    end

    %% Step 6
    subgraph S6["Step 6: Deployment & Live Testing"]
        I --> L["FastAPI Inference Server<br/>http://localhost:8000/predict"]
        L <--> M["React Web UI<br/>http://localhost:3000"]
        M --> N["Upload Worker Photos or Test Webcam<br/>Live Bounding Boxes Appear!"]
    end

    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> S6
```

## Services (docker compose)

| Service | URL |
|---|---|
| Frontend (React) | http://localhost:3000 |
| API docs (FastAPI) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| MinIO console | http://localhost:9001 (login from `.env`) |
| PostgreSQL | localhost:5432 |

CVAT is run separately from its own official compose setup so upgrades don't touch this project.

## Quick start

```bash
git init && dvc init                  # once
make up                               # postgres, minio, mlflow, api, frontend
make setup                            # python env for ml/ (install CUDA PyTorch first)
source .venv/bin/activate
```

## Workflow

1. Put images in `data/raw/` then `make preprocess` → `data/processed/`
2. Annotate `data/processed/` in CVAT and export in **YOLO** format
3. Copy the export so you have:
   ```
   data/annotations/images/*.jpg
   data/annotations/labels/*.txt
   data/annotations/classes.txt      # one class name per line, in class-id order
   ```
4. `make split` → `data/datasets/v1/` (train/val/test + `data.yaml`)
5. `make augment` → adds augmented copies to the **train** split only
6. `dvc add data/datasets/v1` → version the dataset, commit the `.dvc` file
7. `make train` → logs to MLflow, saves `models/weights/best.pt`
8. `make eval` → test-split mAP and per-class mAP in `models/eval.json`
9. `docker compose restart api` → the UI at :3000 now runs your model

Optional: `make export` writes ONNX to `models/exported/`
(set `MODEL_PATH=/models/exported/best.onnx` in `.env` to serve it).

## Notes

- Train from the repo root; paths in the configs are relative to it.
- No GPU? Set `device: cpu` in `ml/configs/train.yaml`.
- Ultralytics YOLO is AGPL-3.0: fine for learning and portfolios, check the license before commercial use.
- Next steps: SAM 2 auto-annotation script, DVC remote on MinIO, a model registry in MLflow, auth on the API.
