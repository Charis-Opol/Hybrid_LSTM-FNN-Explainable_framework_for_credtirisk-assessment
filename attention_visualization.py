"""Attention-weight and embedding extraction for the hybrid GRU-attention
model and the Transformer-encoder hybrid model.

Keras's ``MultiHeadAttention`` computes attention weights internally but
only returns them if the layer is called with ``return_attention_scores=
True``. Neither ``lstm_encoder.py`` nor ``transformer_model.py`` requests
them, so a trained model's attention weights are not otherwise retrievable.

Each builder function here mirrors its production counterpart's layer
stack exactly (same layers, same order, same names) so that a trained
model's weights transfer via ``set_weights()`` -- the instrumented model
adds extra *outputs* tapped from existing layers, not extra parameters.
"""

from __future__ import annotations

import tensorflow as tf

from config import RANDOM_SEED
from hybrid_model import F1Score


def build_hybrid_model_with_attention(
    sequence_length: int,
    temporal_features: int,
    static_features: int,
) -> tf.keras.Model:
    """Same architecture as ``build_hybrid_model``, plus the temporal
    self-attention weights and the pre-output fused embedding as extra
    outputs: ``[risk_score, temporal_attention_weights, hybrid_embedding]``.
    """
    temporal_input = tf.keras.Input(shape=(sequence_length, temporal_features), name="temporal_input")
    x = tf.keras.layers.Masking(mask_value=0.0)(temporal_input)
    x = tf.keras.layers.GRU(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)(x)
    attention_output, attention_scores = tf.keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=32, name="temporal_attention",
    )(x, x, return_attention_scores=True)
    attention = tf.keras.layers.Add()([x, attention_output])
    attention = tf.keras.layers.LayerNormalization()(attention)
    attention = tf.keras.layers.GlobalAveragePooling1D()(attention)
    temporal_embedding = tf.keras.layers.Dense(
        64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3), name="temporal_embedding",
    )(attention)

    static_input = tf.keras.Input(shape=(static_features,), name="static_input")
    s = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(static_input)
    s = tf.keras.layers.BatchNormalization()(s)
    s = tf.keras.layers.Dropout(0.4)(s)
    static_embedding = tf.keras.layers.Dense(
        64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3), name="static_embedding",
    )(s)

    fused = tf.keras.layers.Concatenate(name="hybrid_embedding")([temporal_embedding, static_embedding])
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(fused)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    risk_score = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(x)

    return tf.keras.Model(
        inputs=[temporal_input, static_input],
        outputs=[risk_score, attention_scores, fused],
        name="hybrid_lstm_fnn_with_attention",
    )


def build_transformer_hybrid_model_with_attention(
    sequence_length: int,
    temporal_features: int,
    static_features: int,
    model_dim: int = 64,
    num_heads: int = 4,
    key_dim: int = 16,
    feed_forward_dim: int = 128,
    num_encoder_layers: int = 2,
    static_dense_units: int = 64,
    fusion_dense_units: int = 64,
    second_dense_units: int = 32,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-3,
) -> tf.keras.Model:
    """Same architecture as ``build_transformer_hybrid_model``, plus each
    encoder block's self-attention weights and the pre-output fused
    embedding as extra outputs:
    ``[risk_score, [attention_weights_per_layer...], hybrid_embedding]``.
    """
    l2 = tf.keras.regularizers.l2(l2_reg)

    temporal_input = tf.keras.Input(shape=(sequence_length, temporal_features), name="temporal_input")
    padding_mask = tf.keras.layers.Lambda(
        lambda t, tf_backend: tf_backend.reduce_any(tf_backend.not_equal(t, 0.0), axis=-1),
        arguments={"tf_backend": tf}, output_shape=(sequence_length,), name="compute_padding_mask",
    )(temporal_input)

    x = tf.keras.layers.Dense(model_dim, kernel_regularizer=l2, name="temporal_projection")(temporal_input)
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    position_embeddings = tf.keras.layers.Embedding(
        input_dim=sequence_length, output_dim=model_dim, name="positional_embedding",
    )(positions)
    x = x + position_embeddings

    attention_mask = tf.keras.layers.Lambda(
        lambda m: m[:, None, :], output_shape=(1, sequence_length), name="expand_attention_mask",
    )(padding_mask)

    attention_weights_per_layer = []
    for layer_index in range(num_encoder_layers):
        name = f"encoder_block_{layer_index + 1}"
        attention_output, attention_scores = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name=f"{name}_self_attention",
        )(x, x, attention_mask=attention_mask, return_attention_scores=True)
        attention_weights_per_layer.append(attention_scores)
        attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_attention_norm")(x)

        feed_forward = tf.keras.layers.Dense(feed_forward_dim, activation="relu", kernel_regularizer=l2)(x)
        feed_forward = tf.keras.layers.Dense(model_dim, kernel_regularizer=l2)(feed_forward)
        feed_forward = tf.keras.layers.Dropout(dropout_rate)(feed_forward)
        x = tf.keras.layers.Add()([x, feed_forward])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_feedforward_norm")(x)

    pooled = tf.keras.layers.Lambda(
        lambda inputs, tf_backend: (
            tf_backend.reduce_sum(inputs[0] * tf_backend.cast(inputs[1], tf_backend.float32)[..., tf_backend.newaxis], axis=1)
            / tf_backend.maximum(tf_backend.reduce_sum(tf_backend.cast(inputs[1], tf_backend.float32)[..., tf_backend.newaxis], axis=1), 1e-6)
        ),
        arguments={"tf_backend": tf}, output_shape=(model_dim,), name="masked_global_average_pool",
    )([x, padding_mask])

    temporal_embedding = tf.keras.layers.Dense(
        model_dim, activation="relu", kernel_regularizer=l2, name="temporal_embedding",
    )(pooled)

    static_input = tf.keras.Input(shape=(static_features,), name="static_input")
    s = tf.keras.layers.Dense(static_dense_units, activation="relu", kernel_regularizer=l2)(static_input)
    s = tf.keras.layers.BatchNormalization()(s)
    s = tf.keras.layers.Dropout(dropout_rate)(s)
    static_embedding = tf.keras.layers.Dense(
        static_dense_units, activation="relu", kernel_regularizer=l2, name="static_embedding",
    )(s)

    fused = tf.keras.layers.Concatenate(name="hybrid_embedding")([temporal_embedding, static_embedding])
    f = tf.keras.layers.Dense(fusion_dense_units, activation="relu", kernel_regularizer=l2)(fused)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(dropout_rate)(f)
    f = tf.keras.layers.Dense(second_dense_units, activation="relu", kernel_regularizer=l2)(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(max(dropout_rate - 0.1, 0.1))(f)
    risk_score = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(f)

    return tf.keras.Model(
        inputs=[temporal_input, static_input],
        outputs=[risk_score, *attention_weights_per_layer, fused],
        name="transformer_encoder_hybrid_with_attention",
    )


def _collect_weighted_layers_by_name(model: tf.keras.Model) -> dict[str, tf.keras.layers.Layer]:
    """Recursively map layer name -> layer, descending into nested
    sub-models (e.g. hybrid_model.py wraps TemporalEncoder/StaticEncoder
    as their own Model instances used as layers).
    """
    layers_by_name: dict[str, tf.keras.layers.Layer] = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layers_by_name.update(_collect_weighted_layers_by_name(layer))
        elif layer.get_weights():
            layers_by_name[layer.name] = layer
    return layers_by_name


def load_weights_into_attention_model(source_model: tf.keras.Model, attention_model: tf.keras.Model) -> None:
    """Transfer weights from a trained production model into its
    attention-exposing twin, matched by layer name (not position) --
    the production models nest TemporalEncoder/StaticEncoder as their own
    sub-models, so a flat positional ``get_weights()``/``set_weights()``
    can silently line up same-shaped layers in the wrong order.
    """
    source_layers = _collect_weighted_layers_by_name(source_model)
    target_layers = _collect_weighted_layers_by_name(attention_model)

    missing = set(target_layers) - set(source_layers)
    if missing:
        raise ValueError(f"No matching source layer(s) for: {sorted(missing)}")

    for name, target_layer in target_layers.items():
        target_layer.set_weights(source_layers[name].get_weights())
