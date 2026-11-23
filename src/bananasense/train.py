import argparse
import json
from pathlib import Path

import tensorflow as tf

from .callbacks import GradientDescentTracker
from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    EPOCHS,
    HISTORY_PATH,
    LABELS_PATH,
    MODEL_PATH,
    MODEL_DIR,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
)
from .data import load_datasets
from .evaluate import evaluate_model
from .model import build_model


def train_model(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    output_path: Path = MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    tf.keras.utils.set_random_seed(SEED)
    train_ds, validation_ds, test_ds = load_datasets(train_dir, test_dir, batch_size=batch_size)
    model = build_model(num_classes=len(CLASS_NAMES))
    tracker = GradientDescentTracker(next(iter(train_ds.take(1))), HISTORY_PATH)
    callbacks = [
        tracker,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=callbacks)
    history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
    history_data["gradient_norm"] = tracker.gradient_norms
    history_data["learning_rate"] = tracker.learning_rates
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history_data, indent=2), encoding="utf-8")
    LABELS_PATH.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    metrics = evaluate_model(model, test_ds, CLASS_NAMES)
    val_accuracy = history_data.get("val_accuracy", [])
    val_loss = history_data.get("val_loss", [])
    best_epoch = int(max(range(len(val_accuracy)), key=val_accuracy.__getitem__) + 1) if val_accuracy else 0
    metrics.update(
        {
            "best_epoch": best_epoch,
            "best_val_accuracy": float(val_accuracy[best_epoch - 1]) if best_epoch else 0.0,
            "best_val_loss": float(val_loss[best_epoch - 1]) if best_epoch and val_loss else 0.0,
            "epochs_completed": len(history_data.get("loss", [])),
        }
    )
    from .config import METRICS_PATH

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model tersimpan: {output_path}")
    print(f"Test accuracy: {metrics['test_accuracy']:.2%}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Training CNN BananaSense")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    train_model(args.train_dir, args.test_dir, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
