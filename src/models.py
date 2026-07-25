from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow.keras import Model, layers


NUM_CLASSES = 4


def _conv(
    x: tf.Tensor,
    filters: int,
    batch_norm_after: bool,
    name: str,
) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        activation=None if batch_norm_after else "relu",
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(x)

    if batch_norm_after:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
        x = layers.Activation("relu", name=f"{name}_relu")(x)

    return x


def _classification_head(
    x: tf.Tensor,
    dense_units: int = 256,
    dense_dropout: float = 0.50,
) -> tf.Tensor:
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_256")(x)
    x = layers.Dropout(dense_dropout, name="dense_dropout")(x)
    return layers.Dense(NUM_CLASSES, activation="softmax", name="prediction")(x)


def build_architecture_1(
    input_shape=(224, 224, 3),
    filters: Sequence[int] = (32, 64),
    block_dropout: float = 0.25,
    dense_dropout: float = 0.50,
) -> Model:
    """(Conv2D x2 - BatchNorm - MaxPool - Dropout) x2."""
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block, block_filters in enumerate(filters, start=1):
        x = _conv(x, block_filters, False, f"b{block}_c1")
        x = _conv(x, block_filters, False, f"b{block}_c2")
        x = layers.BatchNormalization(name=f"b{block}_bn")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"b{block}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"b{block}_dropout")(x)

    outputs = _classification_head(x, dense_dropout=dense_dropout)
    return Model(inputs, outputs, name="alzheimer_cnn_architecture_1")


def build_architecture_2(
    input_shape=(224, 224, 3),
    filters: Sequence[int] = (32, 64),
    block_dropout: float = 0.25,
    dense_dropout: float = 0.50,
) -> Model:
    """((Conv2D - BatchNorm) x2 - MaxPool - Dropout) x2."""
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block, block_filters in enumerate(filters, start=1):
        x = _conv(x, block_filters, True, f"b{block}_c1")
        x = _conv(x, block_filters, True, f"b{block}_c2")
        x = layers.MaxPooling2D(pool_size=2, name=f"b{block}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"b{block}_dropout")(x)

    outputs = _classification_head(x, dense_dropout=dense_dropout)
    return Model(inputs, outputs, name="alzheimer_cnn_architecture_2")


def build_architecture_3(
    input_shape=(224, 224, 3),
    filters: Sequence[int] = (32, 64, 128),
    block_dropout: float = 0.25,
    dense_dropout: float = 0.50,
) -> Model:
    """
    (Conv2D x3 - BatchNorm - MaxPool - Dropout) x2
    - Conv2D x2 - BatchNorm - MaxPool - Dropout.
    """
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block, block_filters in enumerate(filters[:2], start=1):
        for conv_index in range(1, 4):
            x = _conv(x, block_filters, False, f"b{block}_c{conv_index}")
        x = layers.BatchNormalization(name=f"b{block}_bn")(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"b{block}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"b{block}_dropout")(x)

    x = _conv(x, filters[2], False, "b3_c1")
    x = _conv(x, filters[2], False, "b3_c2")
    x = layers.BatchNormalization(name="b3_bn")(x)
    x = layers.MaxPooling2D(pool_size=2, name="b3_pool")(x)
    x = layers.Dropout(block_dropout, name="b3_dropout")(x)

    outputs = _classification_head(x, dense_dropout=dense_dropout)
    return Model(inputs, outputs, name="alzheimer_cnn_architecture_3")


def build_model(
    architecture: int,
    input_shape=(224, 224, 3),
    learning_rate: float = 1e-3,
) -> Model:
    builders = {
        1: build_architecture_1,
        2: build_architecture_2,
        3: build_architecture_3,
    }

    if architecture not in builders:
        raise ValueError("architecture must be one of: 1, 2, 3")

    model = builders[architecture](input_shape=input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
