from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .config import BATCH_SIZE, CLASS_NAMES, IMAGE_SIZE, SEED


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(directory: Path):
    paths = []
    labels = []
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_directory = directory / class_name
        class_paths = sorted(
            path for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths.extend(str(path) for path in class_paths)
        labels.extend([label_index] * len(class_paths))
    if not paths:
        raise ValueError(f"Tidak ada gambar yang ditemukan di {directory}")
    return np.array(paths), np.array(labels, dtype="int32")


def make_dataset(paths, labels, batch_size, image_size, shuffle):
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        return image, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(min(len(paths), 1000), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_datasets(
    train_dir: Path,
    test_dir: Path,
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
):
    train_paths, train_labels = collect_images(train_dir)
    test_paths, test_labels = collect_images(test_dir)
    train_paths, validation_paths, train_labels, validation_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.2,
        random_state=SEED,
        stratify=train_labels,
    )
    return (
        make_dataset(train_paths, train_labels, batch_size, image_size, shuffle=True),
        make_dataset(validation_paths, validation_labels, batch_size, image_size, shuffle=False),
        make_dataset(test_paths, test_labels, batch_size, image_size, shuffle=False),
    )


def count_images(directory: Path) -> dict[str, int]:
    return {
        class_name: sum(
            1
            for file_path in (directory / class_name).rglob("*")
            if file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for class_name in CLASS_NAMES
    }
