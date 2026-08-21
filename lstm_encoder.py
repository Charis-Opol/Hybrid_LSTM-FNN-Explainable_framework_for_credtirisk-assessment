"""LSTM temporal encoder for borrower transaction sequences."""

from __future__ import annotations

import tensorflow as tf


class TemporalEncoder:
    """Temporal sequence encoder that returns a 64-dimensional embedding."""

    def __init__(
        self,
        sequence_length: int,
        number_of_features: int,
        mask_value: float = 0.0,
        cell_type: str = "gru",
        name: str = "temporal_encoder",
    ) -> None:
        self.sequence_length = sequence_length
        self.number_of_features = number_of_features
        self.mask_value = mask_value
        if cell_type not in ("gru", "lstm"):
            raise ValueError(f"cell_type must be 'gru' or 'lstm', got {cell_type!r}")
        self.cell_type = cell_type
        self.name = name

    def build(self) -> tf.keras.Model:
        """Build the recurrent encoder model with attention.

        Architecture:
            Masking -> GRU|LSTM(64, return_sequences=True) -> MultiHeadAttention
            -> GlobalAveragePooling1D -> Dense(64)

        Returns:
            Keras model whose output shape is ``(number_of_borrowers, 64)``.
        """

        inputs = tf.keras.Input(
            shape=(self.sequence_length, self.number_of_features),
            name="temporal_input",
        )
        x = tf.keras.layers.Masking(mask_value=self.mask_value)(inputs)
        recurrent_layer = (
            tf.keras.layers.GRU if self.cell_type == "gru" else tf.keras.layers.LSTM
        )
        x = recurrent_layer(
            64,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2,
        )(x)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=2,
            key_dim=32,
            name="temporal_attention",
        )(x, x)
        attention = tf.keras.layers.Add()([x, attention])
        attention = tf.keras.layers.LayerNormalization()(attention)
        attention = tf.keras.layers.GlobalAveragePooling1D()(attention)
        outputs = tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-3),
            name="temporal_embedding",
        )(attention)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)


def build_temporal_encoder(
    sequence_length: int,
    number_of_features: int,
    mask_value: float = 0.0,
    cell_type: str = "gru",
) -> tf.keras.Model:
    """Build a temporal encoder model."""

    return TemporalEncoder(
        sequence_length=sequence_length,
        number_of_features=number_of_features,
        mask_value=mask_value,
        cell_type=cell_type,
    ).build()

