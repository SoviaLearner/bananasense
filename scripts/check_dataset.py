import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bananasense.config import CLASS_NAMES, TEST_DIR, TRAIN_DIR  # noqa: E402
from src.bananasense.data import count_images  # noqa: E402


def main():
    train_counts = count_images(TRAIN_DIR)
    test_counts = count_images(TEST_DIR)
    print("BananaSense dataset check")
    print("=" * 28)
    for class_name in CLASS_NAMES:
        print(f"{class_name:18} train={train_counts[class_name]:4} test={test_counts[class_name]:4}")
    missing = [name for name in CLASS_NAMES if train_counts[name] == 0 or test_counts[name] == 0]
    if missing:
        raise SystemExit(f"Kelas kosong pada train/test: {', '.join(missing)}")
    print("Dataset siap digunakan.")


if __name__ == "__main__":
    main()
