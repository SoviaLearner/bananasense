import json
import os
import sys
from functools import lru_cache
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
HF_REPO_ID = os.getenv("HF_REPO_ID", "")
HF_TOKEN = os.getenv("HF_TOKEN") or None
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "bananasense.keras")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3001")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact(local_path: Path, filename: str):
    if local_path.exists():
        return local_path
    if not HF_REPO_ID:
        return None
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="model",
            token=HF_TOKEN,
        )
    )


@lru_cache(maxsize=1)
def get_model():
    model_path = resolve_artifact(MODEL_PATH, HF_MODEL_FILENAME)
    if model_path is None:
        return None
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def read_huggingface_json(local_path: Path, filename: str, default):
    artifact_path = resolve_artifact(local_path, filename)
    return read_json(artifact_path, default) if artifact_path else default


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists() or bool(HF_REPO_ID), "model_source": "huggingface" if HF_REPO_ID and not MODEL_PATH.exists() else "local"}


@app.get("/api/metrics")
def metrics():
    return {
        "labels": read_huggingface_json(LABELS_PATH, "class_names.json", CLASS_NAMES),
        "metrics": read_huggingface_json(METRICS_PATH, "metrics.json", {}),
        "history": read_huggingface_json(HISTORY_PATH, "training_history.json", {}),
        "confusion_matrix": read_huggingface_json(CONFUSION_MATRIX_PATH, "confusion_matrix.json", []),
        "train_counts": count_images(TRAIN_DIR),
        "test_counts": count_images(TEST_DIR),
        "model_ready": MODEL_PATH.exists() or bool(HF_REPO_ID),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum tersedia. Jalankan training terlebih dahulu.")
    image_bytes = await file.read()
    try:
        labels = read_huggingface_json(LABELS_PATH, "class_names.json", CLASS_NAMES)
        best, results = predict_image(image_bytes, model, labels)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Gambar tidak dapat diproses: {error}") from error
    return {"filename": file.filename, "prediction": best, "probabilities": results}
