import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .config import CONFUSION_MATRIX_PATH, METRICS_PATH


def evaluate_model(model, test_ds, class_names, metrics_path: Path = METRICS_PATH):
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "classification_report": report,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CONFUSION_MATRIX_PATH.write_text(
        json.dumps(confusion_matrix(true_labels, predicted_labels).tolist(), indent=2),
        encoding="utf-8",
    )
    return metrics
