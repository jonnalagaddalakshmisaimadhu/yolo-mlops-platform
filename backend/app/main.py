import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import Prediction
from .services.inference import detector

MAX_BYTES = 15 * 1024 * 1024


@asynccontextmanager
async def lifespan(_):
    for attempt in range(10):  # wait for Postgres on cold start
        try:
            Base.metadata.create_all(engine)
            break
        except Exception:
            if attempt == 9:
                raise
            time.sleep(2)
    yield


app = FastAPI(title="CV Platform API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector.ready, "model": detector.name}


@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not detector.ready:
        raise HTTPException(503, "No model found. Train one (make train) and restart the api.")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Upload an image file (JPG, PNG, WebP).")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 15 MB.")
    try:
        out = detector.predict(data)
    except OSError:
        raise HTTPException(400, "That file could not be read as an image.")

    db.add(Prediction(filename=file.filename or "upload", model_name=detector.name,
                      num_detections=len(out["detections"]),
                      detections=out["detections"], latency_ms=out["latency_ms"]))
    db.commit()
    return out


@app.get("/predictions")
def predictions(limit: int = 20, db: Session = Depends(get_db)):
    rows = db.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(min(limit, 100))).all()
    return [{"id": r.id, "filename": r.filename, "model": r.model_name,
             "num_detections": r.num_detections, "latency_ms": r.latency_ms,
             "created_at": r.created_at.isoformat()} for r in rows]
