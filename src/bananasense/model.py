import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    learning_rate: float = 0.001,
) -> keras.Model:
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )
    inputs = keras.Input(shape=input_shape, name="image")
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255, name="normalization")(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="probabilities")(x)
    model = keras.Model(inputs, outputs, name="bananasense_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def calculate_gradient_norm(model: keras.Model, images, labels) -> float:
    loss_function = keras.losses.SparseCategoricalCrossentropy()
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    valid_gradients = [gradient for gradient in gradients if gradient is not None]
    if not valid_gradients:
        return 0.0
    return float(tf.linalg.global_norm(valid_gradients).numpy())
