from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
]


def _validate_split(data_dir: Path, split: str) -> Path:
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {split_dir}. Expected train/ and test/."
        )

    present = {path.name for path in split_dir.iterdir() if path.is_dir()}
    missing = set(CLASS_NAMES) - present
    if missing:
        raise FileNotFoundError(
            f"Missing class folders in {split_dir}: {sorted(missing)}"
        )
    return split_dir


def _augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def _prepare(
    dataset: tf.data.Dataset,
    training: bool,
    use_augmentation: bool,
) -> tf.data.Dataset:
    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)

    if training and use_augmentation:
        augment = _augmentation()

        def preprocess(images, labels):
            images = tf.cast(images, tf.float32)
            images = augment(images, training=True)
            return rescale(images), labels
    else:
        def preprocess(images, labels):
            images = tf.cast(images, tf.float32)
            return rescale(images), labels

    return dataset.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def load_datasets(
    data_dir: str | Path,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 128,
    validation_split: float = 0.15,
    seed: int = 42,
    use_augmentation: bool = True,
):
    data_dir = Path(data_dir)
    train_dir = _validate_split(data_dir, "train")
    test_dir = _validate_split(data_dir, "test")

    common_train = dict(
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )

    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        subset="training",
        shuffle=True,
        **common_train,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        subset="validation",
        shuffle=False,
        **common_train,
    )

    test_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    train_ds = _prepare(train_raw, training=True, use_augmentation=use_augmentation)
    val_ds = _prepare(val_raw, training=False, use_augmentation=False)
    test_ds = _prepare(test_raw, training=False, use_augmentation=False)

    return train_ds, val_ds, test_ds


def compute_training_class_weights(
    data_dir: str | Path,
) -> Dict[int, float]:
    train_dir = _validate_split(Path(data_dir), "train")
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = train_dir / class_name
        count = sum(
            1 for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        labels.extend([class_index] * count)

    if not labels:
        raise ValueError(f"No training images found under {train_dir}")

    labels_array = np.asarray(labels)
    classes = np.unique(labels_array)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_array,
    )
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}
