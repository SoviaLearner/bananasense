import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bananasense.config import (  # noqa: E402
    CLASS_NAMES,
    CONFUSION_MATRIX_PATH,
    HISTORY_PATH,
    LABELS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    TEST_DIR,
    TRAIN_DIR,
)
from src.bananasense.data import count_images  # noqa: E402
from src.bananasense.predict import predict_image  # noqa: E402


app = FastAPI(title="BananaSense API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def get_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.get("/api/metrics")
def metrics():
    return {
        "labels": read_json(LABELS_PATH, CLASS_NAMES),
        "metrics": read_json(METRICS_PATH, {}),
        "history": read_json(HISTORY_PATH, {}),
        "confusion_matrix": read_json(CONFUSION_MATRIX_PATH, []),
        "train_counts": count_images(TRAIN_DIR),
        "test_counts": count_images(TEST_DIR),
        "model_ready": MODEL_PATH.exists(),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum tersedia. Jalankan training terlebih dahulu.")
    image_bytes = await file.read()
    try:
        best, results = predict_image(image_bytes, model, read_json(LABELS_PATH, CLASS_NAMES))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Gambar tidak dapat diproses: {error}") from error
    return {"filename": file.filename, "prediction": best, "probabilities": results}
