from pathlib import Path

from .config import BATCH_SIZE, CLASS_NAMES, IMAGE_SIZE, SEED

def load_datasets(
    train_dir: Path,
    test_dir: Path,
    batch_size: int = BATCH_SIZE,
    image_size: tuple[int, int] = IMAGE_SIZE,
):
    import tensorflow as tf

    autotune = tf.data.AUTOTUNE
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return (
        train_ds.prefetch(autotune),
        validation_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
    )


def count_images(directory: Path) -> dict[str, int]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {
        class_name: sum(
            1
            for file_path in (directory / class_name).rglob("*")
            if file_path.suffix.lower() in extensions
        )
        for class_name in CLASS_NAMES
    }
