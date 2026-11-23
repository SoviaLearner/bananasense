from pathlib import Path

from src.bananasense.config import CLASS_NAMES
from src.bananasense.data import count_images


def test_expected_class_names():
    assert CLASS_NAMES == ["mentah", "setengah-matang", "matang", "terlalu-matang"]


def test_count_images_returns_all_classes(tmp_path: Path):
    for class_name in CLASS_NAMES:
        (tmp_path / class_name).mkdir()
    counts = count_images(tmp_path)
    assert list(counts) == CLASS_NAMES
    assert all(value == 0 for value in counts.values())
