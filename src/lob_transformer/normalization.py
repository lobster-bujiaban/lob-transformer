from __future__ import annotations

import numpy as np


class LayerNorm:
    """Normalize each token across its features, then apply learned scale and bias."""

    def __init__(self, dimensions: int, epsilon: float = 1e-5) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.weight = np.ones(dimensions)
        self.bias = np.zeros(dimensions)
        self.epsilon = epsilon

    def forward(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors)
        if values.ndim != 2 or values.shape[1] != self.weight.size:
            raise ValueError("vectors must have shape [sequence_length, dimensions]")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("vectors must contain floating-point values")
        mean = values.mean(axis=-1, keepdims=True)
        variance = values.var(axis=-1, keepdims=True)
        normalized = (values - mean) / np.sqrt(variance + self.epsilon)
        return normalized * self.weight + self.bias

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        return self.forward(vectors)
