"""Integrated Gradients (Sundararajan et al., 2017) implemented directly
against tf.keras models.

This is Captum's flagship attribution method, but Captum itself is
PyTorch-only and can't attach to a tf.keras model. The algorithm is short
enough (interpolate along a straight-line path from a baseline to the real
input, average the gradient of the output w.r.t. the input along that
path, scale by (input - baseline)) to reimplement natively, which avoids
standing up a second ML framework and a parallel, separately-trained model
just to get one attribution method.

Reference baseline: the dataset's per-feature mean (an "average borrower"),
not zero -- static features are raw-scaled (age, income, ...) where zero
is not a meaningful "absence" value the way it is for the padded temporal
sequence. Using the same mean-baseline convention for both inputs keeps
the comparison between temporal and static attributions consistent.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def _interpolate(baseline: np.ndarray, target: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """Build ``len(alphas)`` points on the straight-line path from
    ``baseline`` to ``target`` for a single example.
    """
    alphas_reshaped = alphas.reshape((-1,) + (1,) * baseline.ndim)
    return baseline[None, ...] + alphas_reshaped * (target - baseline)[None, ...]


def integrated_gradients_single(
    model: tf.keras.Model,
    temporal_input: np.ndarray,
    static_input: np.ndarray,
    baseline_temporal: np.ndarray,
    baseline_static: np.ndarray,
    m_steps: int = 50,
    output_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Integrated Gradients for one borrower.

    Args:
        model: Multi-input Keras model; ``model([temporal, static])``'s
            output at ``output_index`` is the scalar being attributed
            (the risk_score head -- attention/embedding outputs, if any,
            are ignored).
        temporal_input: Shape (sequence_length, temporal_features).
        static_input: Shape (static_features,).
        baseline_temporal, baseline_static: Same shapes as the inputs above.
        m_steps: Number of interpolation steps (trapezoidal rule).
        output_index: Which model output is the risk score.

    Returns:
        (temporal_attributions, static_attributions, convergence_delta).
        ``convergence_delta`` should be close to zero: it's
        ``sum(attributions) - (F(x) - F(baseline))``, the standard IG
        sanity check that the implementation is correct.
    """
    alphas = np.linspace(0.0, 1.0, m_steps + 1)

    interpolated_temporal = _interpolate(baseline_temporal, temporal_input, alphas).astype(np.float32)
    interpolated_static = _interpolate(baseline_static, static_input, alphas).astype(np.float32)

    temporal_tensor = tf.convert_to_tensor(interpolated_temporal)
    static_tensor = tf.convert_to_tensor(interpolated_static)

    with tf.GradientTape() as tape:
        tape.watch(temporal_tensor)
        tape.watch(static_tensor)
        outputs = model([temporal_tensor, static_tensor], training=False)
        target = outputs[output_index] if isinstance(outputs, (list, tuple)) else outputs
        target = tf.reshape(target, [-1])

    gradients_temporal, gradients_static = tape.gradient(target, [temporal_tensor, static_tensor])

    # Trapezoidal rule: average interior points fully, endpoints at half weight.
    weights = np.ones(m_steps + 1)
    weights[0] = weights[-1] = 0.5
    weights = weights / m_steps

    average_gradient_temporal = np.tensordot(weights, gradients_temporal.numpy(), axes=(0, 0))
    average_gradient_static = np.tensordot(weights, gradients_static.numpy(), axes=(0, 0))

    temporal_attributions = (temporal_input - baseline_temporal) * average_gradient_temporal
    static_attributions = (static_input - baseline_static) * average_gradient_static

    predictions = model([tf.convert_to_tensor(np.stack([baseline_temporal, temporal_input]).astype(np.float32)),
                          tf.convert_to_tensor(np.stack([baseline_static, static_input]).astype(np.float32))], training=False)
    prediction_values = predictions[output_index] if isinstance(predictions, (list, tuple)) else predictions
    prediction_values = np.asarray(prediction_values).reshape(-1)
    convergence_delta = float(
        temporal_attributions.sum() + static_attributions.sum() - (prediction_values[1] - prediction_values[0])
    )

    return temporal_attributions, static_attributions, convergence_delta


def integrated_gradients_batch(
    model: tf.keras.Model,
    X_temporal: np.ndarray,
    X_static: np.ndarray,
    baseline_temporal: np.ndarray,
    baseline_static: np.ndarray,
    m_steps: int = 50,
    output_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrated Gradients for every row in ``X_temporal``/``X_static``,
    one borrower at a time (each borrower needs its own ``m_steps``-point
    interpolated path, so this isn't batched across borrowers).
    """
    n = len(X_temporal)
    temporal_attributions = np.zeros_like(X_temporal, dtype=np.float32)
    static_attributions = np.zeros_like(X_static, dtype=np.float32)
    convergence_deltas = np.zeros(n, dtype=np.float32)

    for i in range(n):
        temporal_attr, static_attr, delta = integrated_gradients_single(
            model, X_temporal[i], X_static[i], baseline_temporal, baseline_static,
            m_steps=m_steps, output_index=output_index,
        )
        temporal_attributions[i] = temporal_attr
        static_attributions[i] = static_attr
        convergence_deltas[i] = delta

    return temporal_attributions, static_attributions, convergence_deltas
