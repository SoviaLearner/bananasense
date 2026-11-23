from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "bananasense.keras"
HISTORY_PATH = MODEL_DIR / "training_history.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
LABELS_PATH = MODEL_DIR / "class_names.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 20
SEED = 42
CLASS_NAMES = ["mentah", "setengah-matang", "matang", "terlalu-matang"]
