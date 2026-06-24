"""Hybrid LSTM-FNN credit risk model."""

from __future__ import annotations

import tensorflow as tf

from fnn_encoder import StaticEncoder
from lstm_encoder import TemporalEncoder


@tf.keras.utils.register_keras_serializable(package="credit_risk")
class F1Score(tf.keras.metrics.Metric):
    """Binary F1 metric computed from precision and recall counts."""

    def __init__(self, name: str = "f1", threshold: float = 0.5, **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.greater_equal(y_pred, self.threshold)

        true_positive = tf.cast(tf.logical_and(y_true, y_pred), self.dtype)
        false_positive = tf.cast(tf.logical_and(~y_true, y_pred), self.dtype)
        false_negative = tf.cast(tf.logical_and(y_true, ~y_pred), self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            true_positive *= sample_weight
            false_positive *= sample_weight
            false_negative *= sample_weight

        self.true_positives.assign_add(tf.reduce_sum(true_positive))
        self.false_positives.assign_add(tf.reduce_sum(false_positive))
        self.false_negatives.assign_add(tf.reduce_sum(false_negative))

    def result(self) -> tf.Tensor:
        precision = self.true_positives / (
            self.true_positives + self.false_positives + tf.keras.backend.epsilon()
        )
        recall = self.true_positives / (
            self.true_positives + self.false_negatives + tf.keras.backend.epsilon()
        )
        return 2 * precision * recall / (
            precision + recall + tf.keras.backend.epsilon()
        )

    def reset_state(self) -> None:
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

    def get_config(self) -> dict[str, float | str]:
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config


def build_hybrid_model(
    sequence_length: int,
    temporal_features: int,
    static_features: int,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build and compile the hybrid credit risk classifier.

    Args:
        sequence_length: Number of monthly time steps in temporal input.
        temporal_features: Number of features per monthly time step.
        static_features: Number of static borrower features.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras model with AUC, precision, recall, accuracy, and F1.
    """

    temporal_encoder = TemporalEncoder(
        sequence_length=sequence_length,
        number_of_features=temporal_features,
    ).build()
    static_encoder = StaticEncoder(number_of_features=static_features).build()

    temporal_input = temporal_encoder.input
    static_input = static_encoder.input
    temporal_embedding = temporal_encoder(temporal_input)
    static_embedding = static_encoder(static_input)

    x = tf.keras.layers.Concatenate(name="hybrid_embedding")(
        [temporal_embedding, static_embedding]
    )
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
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(x)

    model = tf.keras.Model(
        inputs=[temporal_input, static_input],
        outputs=output,
        name="hybrid_lstm_fnn_credit_risk_model",
    )
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

