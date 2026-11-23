from pathlib import Path

import tensorflow as tf

from .model import calculate_gradient_norm


class GradientDescentTracker(tf.keras.callbacks.Callback):
    """Mencatat besar gradien setiap epoch untuk visualisasi dashboard."""

    def __init__(self, sample_batch, output_path: Path):
        super().__init__()
        self.sample_images, self.sample_labels = sample_batch
        self.output_path = output_path
        self.gradient_norms: list[float] = []

    def on_epoch_end(self, epoch, logs=None):
        gradient_norm = calculate_gradient_norm(
            self.model, self.sample_images, self.sample_labels
        )
        self.gradient_norms.append(gradient_norm)
        print(f" - gradient_norm: {gradient_norm:.6f}")

