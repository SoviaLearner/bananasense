from io import BytesIO

import numpy as np
from PIL import Image

from .config import IMAGE_SIZE


def predict_image(image_source, model, class_names):
    if hasattr(image_source, "read"):
        image_source = image_source.read()
    if isinstance(image_source, (bytes, bytearray)):
        image = Image.open(BytesIO(image_source))
    else:
        image = Image.open(image_source)
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype="float32")
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    results = [
        {"label": class_names[index], "probability": float(probabilities[index])}
        for index in order
    ]
    return results[0], results
