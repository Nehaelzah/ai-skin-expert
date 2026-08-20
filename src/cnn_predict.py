"""Run a local prediction with a trained CNN image-classification model.

Educational use only. This script must not be used for clinical diagnosis or
to make treatment decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an image class with a local CNN model.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--class-map",
        type=Path,
        help="Optional JSON class-index mapping. Defaults to the model's .classes.json file.",
    )
    parser.add_argument("--image-size", type=int, default=150)
    return parser.parse_args()


def load_class_labels(path: Path) -> list[str]:
    class_map = json.loads(path.read_text(encoding="utf-8"))
    return [label for label, _ in sorted(class_map.items(), key=lambda item: item[1])]


def main() -> None:
    args = parse_args()
    if not args.model.is_file() or not args.image.is_file():
        raise ValueError("--model and --image must both point to existing files.")
    class_map_path = args.class_map or args.model.with_suffix(".classes.json")
    if not class_map_path.is_file():
        raise ValueError("A class-map JSON file is required to interpret predictions.")

    model = load_model(args.model)
    labels = load_class_labels(class_map_path)
    loaded_image = image.load_img(args.image, target_size=(args.image_size, args.image_size))
    image_array = image.img_to_array(loaded_image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    probabilities = model.predict(image_array, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    print(json.dumps({"predicted_class": labels[class_index], "confidence": float(probabilities[class_index])}))


if __name__ == "__main__":
    main()
