"""Train the CNN component of the AI Skin Expert portfolio project.

This script requires separate training and validation directories. It is for
educational image-classification experimentation only, not clinical use.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image CNN.")
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_directories(train_dir: Path, validation_dir: Path) -> None:
    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise ValueError("Both --train-dir and --validation-dir must be existing directories.")
    if train_dir.resolve() == validation_dir.resolve():
        raise ValueError("Training and validation directories must be different.")


def build_model(image_size: int, class_count: int) -> Sequential:
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    args = parse_args()
    validate_directories(args.train_dir, args.validation_dir)
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise ValueError("Epochs, batch size, and image size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    generator_options = {
        "target_size": (args.image_size, args.image_size),
        "batch_size": args.batch_size,
        "class_mode": "categorical",
    }
    train_generator = train_datagen.flow_from_directory(args.train_dir, shuffle=True, **generator_options)
    validation_generator = validation_datagen.flow_from_directory(
        args.validation_dir, shuffle=False, **generator_options
    )
    if train_generator.class_indices != validation_generator.class_indices:
        raise ValueError("Training and validation class-folder mappings must match.")

    model = build_model(args.image_size, len(train_generator.class_indices))
    model.fit(train_generator, epochs=args.epochs, validation_data=validation_generator)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_model)
    class_map_path = args.output_model.with_suffix(".classes.json")
    class_map_path.write_text(json.dumps(train_generator.class_indices, indent=2), encoding="utf-8")
    evaluation = model.evaluate(validation_generator, verbose=0)
    print(f"Validation accuracy: {evaluation[1]:.4f}")
    print(f"Saved model to: {args.output_model}")
    print(f"Saved class mapping to: {class_map_path}")


if __name__ == "__main__":
    main()
