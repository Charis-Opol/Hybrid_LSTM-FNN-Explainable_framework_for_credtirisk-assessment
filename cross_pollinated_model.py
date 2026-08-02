"""Cross-pollinated model: pairs each branch with the architecture that
ablation showed actually depends on that stream.

The interpretability study (see models/.../interpretability_report/) found
that the hybrid GRU-Attention model's predictions barely move when its
temporal branch is ablated (correlation 0.985 with the real predictions --
i.e. it relies almost entirely on the static FNN branch), while the
Transformer-Encoder Hybrid's predictions barely move when its *static*
branch is ablated (0.955 -- it relies on the temporal self-attention
branch). Neither model's "unused" branch is doing much work.

This architecture takes the branch each one actually uses: the
Transformer-encoder's temporal self-attention stack (not the GRU), fused
with the hybrid model's static FNN encoder (not the transformer's static
Dense stack) -- same fusion head shape as both source architectures, so
this is a fair like-for-like addition to the model comparison.
"""

from __future__ import annotations

import tensorflow as tf

from fnn_encoder import StaticEncoder
from hybrid_model import F1Score
from lstm_encoder import TemporalEncoder
from attention_visualization import ExpandMaskLayer, MaskedGlobalAveragePooling1D, PaddingMaskLayer


def build_cross_pollinated_model(
    sequence_length: int,
    temporal_features: int,
    static_features: int,
    model_dim: int = 64,
    num_heads: int = 4,
    key_dim: int = 16,
    feed_forward_dim: int = 128,
    num_encoder_layers: int = 2,
    fusion_dense_units: int = 64,
    second_dense_units: int = 32,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-3,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Transformer temporal encoder + hybrid static (FNN) encoder.

    Args mirror build_transformer_hybrid_model / build_hybrid_model so the
    two source architectures' hyperparameters carry over unchanged.
    """
    l2 = tf.keras.regularizers.l2(l2_reg)

    # --- Temporal branch: transformer's self-attention encoder stack ---
    temporal_input = tf.keras.Input(shape=(sequence_length, temporal_features), name="temporal_input")
    padding_mask = PaddingMaskLayer(name="compute_padding_mask")(temporal_input)

    x = tf.keras.layers.Dense(model_dim, kernel_regularizer=l2, name="temporal_projection")(temporal_input)
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    position_embeddings = tf.keras.layers.Embedding(
        input_dim=sequence_length, output_dim=model_dim, name="positional_embedding",
    )(positions)
    x = x + position_embeddings

    attention_mask = ExpandMaskLayer(name="expand_attention_mask")(padding_mask)

    for layer_index in range(num_encoder_layers):
        name = f"encoder_block_{layer_index + 1}"
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, name=f"{name}_self_attention",
        )(x, x, attention_mask=attention_mask)
        attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_attention_norm")(x)

        feed_forward = tf.keras.layers.Dense(feed_forward_dim, activation="relu", kernel_regularizer=l2)(x)
        feed_forward = tf.keras.layers.Dense(model_dim, kernel_regularizer=l2)(feed_forward)
        feed_forward = tf.keras.layers.Dropout(dropout_rate)(feed_forward)
        x = tf.keras.layers.Add()([x, feed_forward])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_feedforward_norm")(x)

    pooled = MaskedGlobalAveragePooling1D(name="masked_global_average_pool")([x, padding_mask])
    temporal_embedding = tf.keras.layers.Dense(
        model_dim, activation="relu", kernel_regularizer=l2, name="temporal_embedding",
    )(pooled)

    # --- Static branch: hybrid model's FNN encoder ---
    static_encoder = StaticEncoder(number_of_features=static_features).build()
    static_input = static_encoder.input
    static_embedding = static_encoder(static_input)

    # --- Fusion head (same shape as both source architectures) ---
    fused = tf.keras.layers.Concatenate(name="hybrid_embedding")([temporal_embedding, static_embedding])
    f = tf.keras.layers.Dense(fusion_dense_units, activation="relu", kernel_regularizer=l2)(fused)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(dropout_rate)(f)
    f = tf.keras.layers.Dense(second_dense_units, activation="relu", kernel_regularizer=l2)(f)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(max(dropout_rate - 0.1, 0.1))(f)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(f)

    model = tf.keras.Model(
        inputs=[temporal_input, static_input],
        outputs=output,
        name="cross_pollinated_model",
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


def build_cross_pollinated_model_reverse(
    sequence_length: int,
    temporal_features: int,
    static_features: int,
    static_dense_units: int = 64,
    fusion_dense_units: int = 64,
    second_dense_units: int = 32,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-3,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """The other pairing: hybrid's GRU-Attention temporal encoder + the
    transformer's plain Dense static branch.

    Built to test whether the first cross-pollinated model's finding
    (temporal-branch dominance persists even after swapping which static
    encoder it's paired with) generalizes: if GRU-based temporal *still*
    loses the branch-importance competition here -- now paired with a
    weaker static branch than either original architecture had -- that's
    stronger evidence this is a GRU-vs-self-attention gradient-competition
    effect, not a property of any specific dataset stream.
    """
    l2 = tf.keras.regularizers.l2(l2_reg)

    temporal_encoder = TemporalEncoder(
        sequence_length=sequence_length,
        number_of_features=temporal_features,
    ).build()
    temporal_input = temporal_encoder.input
    temporal_embedding = temporal_encoder(temporal_input)

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
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(f)

    model = tf.keras.Model(
        inputs=[temporal_input, static_input],
        outputs=output,
        name="cross_pollinated_model_reverse",
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
