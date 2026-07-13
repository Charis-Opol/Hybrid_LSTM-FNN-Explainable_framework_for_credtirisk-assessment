"""Hybrid model variant: cross-attention fusion of static and temporal branches.

Instead of pooling the temporal sequence to a single embedding and
concatenating it with the static embedding (as in hybrid_model.py),
this variant keeps the full per-month temporal sequence and lets the
static embedding act as a query that attends over it. This allows the
model to learn, per borrower, which months matter most given that
borrower's static profile -- rather than applying the same fixed
fusion to every borrower.

Ablation counterpart to hybrid_model.py (GRU + self-attention + concat)
and hybrid_transformer_encoder_model.py (Transformer encoder + concat).
"""

from __future__ import annotations

import tensorflow as tf

from hybrid_model import F1Score


def build_cross_attention_hybrid_model(
    sequence_length: int,
    temporal_features: int,
    static_features: int,
    gru_units: int = 64,
    self_attention_heads: int = 2,
    self_attention_key_dim: int = 32,
    cross_attention_heads: int = 2,
    cross_attention_key_dim: int = 32,
    static_dense_units: int = 64,
    fusion_dense_units: int = 64,
    second_dense_units: int = 32,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-3,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Build the cross-attention fusion hybrid model.

    Args:
        sequence_length: Number of monthly time steps in the temporal input.
        temporal_features: Number of features per monthly time step.
        static_features: Number of static borrower features.
        gru_units: GRU hidden units in the temporal branch.
        self_attention_heads: Heads for the temporal self-attention (as in
            the original hybrid model).
        self_attention_key_dim: Key dimension for temporal self-attention.
        cross_attention_heads: Heads for the static-queries-temporal
            cross-attention fusion.
        cross_attention_key_dim: Key dimension for the cross-attention fusion.
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

    # Explicit padding mask: True where a time step has any nonzero feature.
    # Computed directly from the raw input and passed explicitly into both
    # MultiHeadAttention calls below, rather than relying on Keras's
    # automatic mask propagation -- MultiHeadAttention's call signature
    # uses "attention_mask", not the generic "mask" argument Keras looks
    # for during automatic propagation, so an implicit mask silently gets
    # ignored (or, in the Transformer-encoder variant, corrupted) rather
    # than actually applied.
    temporal_input = tf.keras.Input(
        shape=(sequence_length, temporal_features), name="temporal_input"
    )
    padding_mask = tf.keras.layers.Lambda(
        lambda t: tf.reduce_any(tf.not_equal(t, 0.0), axis=-1),
        output_shape=(sequence_length,),
        name="compute_padding_mask",
    )(temporal_input)  # shape: (batch, sequence_length), boolean

    # --- Temporal branch: Masking -> GRU -> self-attention (kept as a full
    # sequence output, not pooled, so it can be queried by the static branch) ---
    x = tf.keras.layers.Masking(mask_value=0.0)(temporal_input)
    x = tf.keras.layers.GRU(
        gru_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.2
    )(x)

    self_attention_mask = tf.keras.layers.Lambda(
        lambda m: m[:, tf.newaxis, :],
        output_shape=(1, sequence_length),
        name="expand_self_attention_mask",
    )(padding_mask)

    self_attended = tf.keras.layers.MultiHeadAttention(
        num_heads=self_attention_heads,
        key_dim=self_attention_key_dim,
        name="temporal_self_attention",
    )(x, x, attention_mask=self_attention_mask)
    x = tf.keras.layers.Add()([x, self_attended])
    temporal_sequence = tf.keras.layers.LayerNormalization()(x)
    # temporal_sequence shape: (batch, sequence_length, gru_units)

    # --- Static branch: Dense -> BatchNorm -> Dropout -> Dense ---
    static_input = tf.keras.Input(shape=(static_features,), name="static_input")
    s = tf.keras.layers.Dense(
        static_dense_units, activation="relu", kernel_regularizer=l2
    )(static_input)
    s = tf.keras.layers.BatchNormalization()(s)
    s = tf.keras.layers.Dropout(dropout_rate)(s)
    static_embedding = tf.keras.layers.Dense(
        static_dense_units, activation="relu", kernel_regularizer=l2,
        name="static_embedding",
    )(s)
    # static_embedding shape: (batch, static_dense_units)

    # --- Cross-attention fusion: static embedding (query) attends over the
    # temporal sequence (key/value). Output length matches the query
    # length, i.e. one vector per borrower. ---
    query = tf.keras.layers.Reshape((1, static_dense_units))(static_embedding)
    cross_attention_mask = tf.keras.layers.Lambda(
        lambda m: m[:, tf.newaxis, :],
        output_shape=(1, sequence_length),
        name="expand_cross_attention_mask",
    )(padding_mask)
    cross_attended = tf.keras.layers.MultiHeadAttention(
        num_heads=cross_attention_heads,
        key_dim=cross_attention_key_dim,
        name="static_queries_temporal_cross_attention",
    )(query=query, value=temporal_sequence, key=temporal_sequence, attention_mask=cross_attention_mask)
    cross_attended = tf.keras.layers.Reshape((static_dense_units,))(cross_attended)

    fused = tf.keras.layers.Add(name="residual_fusion")([static_embedding, cross_attended])
    fused = tf.keras.layers.LayerNormalization()(fused)

    # --- Classification head ---
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
        name="cross_attention_hybrid_model",
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
