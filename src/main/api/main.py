from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from anpr.analytics.dedup import find_duplicates
from anpr.analytics.monitoring import LatencyTracker, js_divergence
from anpr.api.schemas import DriftResponse, LatencyResponse, OccupancyResponse, PredictResponse
from anpr.config import get_settings
from anpr.inference.pipeline import ANPRInferenceEngine

app = FastAPI(title="ANPR API", version="1.0.0")
settings = get_settings()


_engine: ANPRInferenceEngine | None = None
latency_tracker = LatencyTracker()
prediction_history: list[str] = []


def get_engine() -> ANPRInferenceEngine:
    global _engine
    if _engine is None:
        if not Path(settings.model_path).exists():
            raise RuntimeError(
                f"Model checkpoint not found at {settings.model_path}. Train first or set ANPR_MODEL_PATH."
            )
        _engine = ANPRInferenceEngine(
            model_path=settings.model_path,
            charset=settings.charset,
            image_size=settings.image_size,
            max_label_length=settings.max_label_length,
            device=settings.device,
        )
    return _engine


def _decode_image(upload: UploadFile) -> np.ndarray:
    payload = upload.file.read()
    arr = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    return image


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)) -> PredictResponse:
    engine = get_engine()
    image = _decode_image(file)
    start = time.perf_counter()
    pred = engine.process_frame(image)
    latency_tracker.add(time.perf_counter() - start)
    prediction_history.append(pred["plate_text"])
    return PredictResponse(**pred)


@app.post("/predict/batch")
def predict_batch(files: list[UploadFile] = File(...)) -> dict[str, object]:
    engine = get_engine()
    outputs = []
    plate_texts = []

    for file in files:
        image = _decode_image(file)
        start = time.perf_counter()
        pred = engine.process_frame(image)
        latency_tracker.add(time.perf_counter() - start)
        outputs.append(pred)
        plate_texts.append(pred["plate_text"])
        prediction_history.append(pred["plate_text"])

    duplicates = find_duplicates(plate_texts)
    return {"predictions": outputs, "duplicates": duplicates}


@app.get("/analytics/occupancy", response_model=OccupancyResponse)
def occupancy() -> OccupancyResponse:
    engine = get_engine()
    return OccupancyResponse(occupancy=engine.zone_counts)


@app.get("/analytics/latency", response_model=LatencyResponse)
def latency() -> LatencyResponse:
    return LatencyResponse(**latency_tracker.summary())


@app.get("/analytics/drift", response_model=DriftResponse)
def drift() -> DriftResponse:
    if len(prediction_history) < 20:
        return DriftResponse(js_divergence=0.0)

    mid = len(prediction_history) // 2
    reference = prediction_history[:mid]
    current = prediction_history[mid:]
    score = js_divergence(reference, current)
    return DriftResponse(js_divergence=score)
