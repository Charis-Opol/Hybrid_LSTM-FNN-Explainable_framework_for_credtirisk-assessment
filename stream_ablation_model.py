"""Single-stream ablations of the hybrid model.

Builds the same classification head used by build_hybrid_model
(Dense(64) -> BN -> Dropout(0.4) -> Dense(32) -> BN -> Dropout(0.3) ->
sigmoid) on top of just one branch -- temporal (GRU or LSTM +
attention) or static (FNN) -- instead of the concatenated embedding.
Holding the head fixed isolates how much each stream contributes to
the full hybrid's classification performance.
"""

from __future__ import annotations

from typing import Literal

import tensorflow as tf

from fnn_encoder import StaticEncoder
from hybrid_model import F1Score
from lstm_encoder import TemporalEncoder

Stream = Literal["temporal", "static"]


def _classification_head(x: tf.Tensor) -> tf.Tensor:
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(x)


def build_stream_ablation_model(
    stream: Stream,
    sequence_length: int | None = None,
    temporal_features: int | None = None,
    static_features: int | None = None,
    learning_rate: float = 0.001,
    cell_type: str = "gru",
) -> tf.keras.Model:
    """Build a single-stream classifier with the hybrid model's head.

    Args:
        stream: "temporal" for GRU/LSTM + attention only, "static" for
            the FNN branch only.
        sequence_length, temporal_features: required when stream="temporal".
        static_features: required when stream="static".
        learning_rate: Adam optimizer learning rate.
        cell_type: Recurrent cell for the temporal branch, "gru" or "lstm"
            (ignored when stream="static").

    Returns:
        Compiled Keras model with AUC, precision, recall, accuracy, and F1.
    """

    if stream == "temporal":
        encoder = TemporalEncoder(
            sequence_length=sequence_length,
            number_of_features=temporal_features,
            cell_type=cell_type,
        ).build()
        model_name = f"temporal_only_{cell_type}"
    elif stream == "static":
        encoder = StaticEncoder(number_of_features=static_features).build()
        model_name = "static_only_fnn"
    else:
        raise ValueError(f"stream must be 'temporal' or 'static', got {stream!r}")

    output = _classification_head(encoder.output)
    model = tf.keras.Model(inputs=encoder.input, outputs=output, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            F1Score(name="f1"),
        ],
    )
    return model
