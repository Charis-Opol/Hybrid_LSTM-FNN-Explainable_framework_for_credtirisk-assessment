"""Feed-forward static encoder for borrower-level features."""

from __future__ import annotations

import tensorflow as tf


class StaticEncoder:
    """Static borrower feature encoder returning a 64-dimensional embedding."""

    def __init__(
        self,
        number_of_features: int,
        name: str = "static_encoder",
    ) -> None:
        self.number_of_features = number_of_features
        self.name = name

    def build(self) -> tf.keras.Model:
        """Build the static encoder model.

        Architecture:
            Dense(64, relu, l2) -> BatchNormalization -> Dropout(0.4)
            -> Dense(64, relu, l2)

        Returns:
            Keras model whose output shape is ``(number_of_borrowers, 64)``.
        """

        inputs = tf.keras.Input(shape=(self.number_of_features,), name="static_input")
        x = tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            name="static_embedding",
        )(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)


def build_static_encoder(number_of_features: int) -> tf.keras.Model:
    """Build a static encoder model."""

    return StaticEncoder(number_of_features=number_of_features).build()

