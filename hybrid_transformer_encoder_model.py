"""Hybrid model variant: Transformer encoder replaces the GRU entirely.

Standard Transformer encoder block (Vaswani et al., 2017) applied to
the monthly transaction sequence: learned positional embeddings +
multi-head self-attention + feed-forward sublayer, each with residual
connections and layer normalization, no recurrence at all. The static
branch and fusion (concatenation) are kept identical to hybrid_model.py
so this ablation isolates the effect of recurrence vs. pure attention.

Ablation counterpart to hybrid_model.py (GRU + self-attention + concat)
and hybrid_cross_attention_model.py (GRU + self-attention + cross-attention
fusion).
"""

from __future__ import annotations

import tensorflow as tf

from hybrid_model import F1Score


def transformer_encoder_block(
    x: tf.Tensor,
    num_heads: int,
    key_dim: int,
    feed_forward_dim: int,
    dropout_rate: float,
    l2_reg: float,
    name: str,
) -> tf.Tensor:
    """One Transformer encoder block: self-attention + feed-forward, each
    with a residual connection and layer normalization."""

    l2 = tf.keras.regularizers.l2(l2_reg)
    model_dim = x.shape[-1]

    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"{name}_self_attention"
    )(x, x)
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    x = tf.keras.layers.Add()([x, attention_output])
    x = tf.keras.layers.LayerNormalization(name=f"{name}_attention_norm")(x)

    feed_forward = tf.keras.layers.Dense(
        feed_forward_dim, activation="relu", kernel_regularizer=l2
    )(x)
    feed_forward = tf.keras.layers.Dense(model_dim, kernel_regularizer=l2)(feed_forward)
    feed_forward = tf.keras.layers.Dropout(dropout_rate)(feed_forward)
    x = tf.keras.layers.Add()([x, feed_forward])
    x = tf.keras.layers.LayerNormalization(name=f"{name}_feedforward_norm")(x)

    return x


def build_transformer_hybrid_model(
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
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build the Transformer-encoder hybrid model.

    Args:
        sequence_length: Number of monthly time steps in the temporal input.
        temporal_features: Number of features per monthly time step.
        static_features: Number of static borrower features.
        model_dim: Dimension the raw temporal features are projected to
            before positional embeddings and the encoder stack.
        num_heads: Attention heads per encoder block.
        key_dim: Key dimension per attention head.
        feed_forward_dim: Hidden size of each encoder block's feed-forward
            sublayer.
        num_encoder_layers: Number of stacked Transformer encoder blocks.
        static_dense_units: Units in the static (FNN) branch's Dense layers.
        fusion_dense_units: Units in the first post-fusion Dense layer.
        second_dense_units: Units in the second post-fusion Dense layer.
        dropout_rate: Dropout applied throughout.
        l2_reg: L2 regularization strength applied to Dense layers.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras model with AUC, precision, recall, accuracy, and F1.
    """
    l2 = tf.keras.regularizers.l2(l2_reg)

    # --- Temporal branch: project to model_dim, add positional embeddings,
    # then stack Transformer encoder blocks (no recurrence). ---
    temporal_input = tf.keras.Input(
        shape=(sequence_length, temporal_features), name="temporal_input"
    )
    x = tf.keras.layers.Masking(mask_value=0.0)(temporal_input)
    x = tf.keras.layers.Dense(model_dim, kernel_regularizer=l2, name="temporal_projection")(x)

    positions = tf.range(start=0, limit=sequence_length, delta=1)
    position_embeddings = tf.keras.layers.Embedding(
        input_dim=sequence_length, output_dim=model_dim, name="positional_embedding"
    )(positions)
    x = tf.keras.layers.Add(name="add_positional_embedding")([x, position_embeddings])

    for layer_index in range(num_encoder_layers):
        x = transformer_encoder_block(
            x,
            num_heads=num_heads,
            key_dim=key_dim,
            feed_forward_dim=feed_forward_dim,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            name=f"encoder_block_{layer_index + 1}",
        )

    pooled = tf.keras.layers.GlobalAveragePooling1D()(x)
    temporal_embedding = tf.keras.layers.Dense(
        model_dim, activation="relu", kernel_regularizer=l2, name="temporal_embedding"
    )(pooled)

    # --- Static branch: identical to hybrid_model.py's StaticEncoder ---
    static_input = tf.keras.Input(shape=(static_features,), name="static_input")
    s = tf.keras.layers.Dense(
        static_dense_units, activation="relu", kernel_regularizer=l2
    )(static_input)
    s = tf.keras.layers.BatchNormalization()(s)
    s = tf.keras.layers.Dropout(dropout_rate)(s)
    static_embedding = tf.keras.layers.Dense(
        static_dense_units, activation="relu", kernel_regularizer=l2, name="static_embedding"
    )(s)

    # --- Fusion: concatenation, same as the original hybrid model ---
    fused = tf.keras.layers.Concatenate(name="hybrid_embedding")(
        [temporal_embedding, static_embedding]
    )
    f = tf.keras.layers.Dense(
        fusion_dense_units, activation="relu", kernel_regularizer=l2
    )(fused)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(dropout_rate)(f)
    f = tf.keras.layers.Dense(
        second_dense_units, activation="relu", kernel_regularizer=l2
    )(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(max(dropout_rate - 0.1, 0.1))(f)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(f)

    model = tf.keras.Model(
        inputs=[temporal_input, static_input],
        outputs=output,
        name="transformer_encoder_hybrid_model",
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
