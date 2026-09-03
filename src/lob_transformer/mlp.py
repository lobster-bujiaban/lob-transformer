from __future__ import annotations

import numpy as np


class MLP:
    """Token-wise feed-forward network: expand features, ReLU, project back."""

    def __init__(
        self,
        dimensions: int,
        *,
        expansion: int = 4,
        seed: int = 7,
        rng: np.random.Generator | None = None,
    ) -> None:
        if dimensions <= 0 or expansion <= 0:
            raise ValueError("dimensions and expansion must be positive")
        generator = rng if rng is not None else np.random.default_rng(seed)
        hidden_size = dimensions * expansion
        self.up_weight = generator.normal(0, dimensions**-0.5, (dimensions, hidden_size))
        self.down_weight = generator.normal(0, hidden_size**-0.5, (hidden_size, dimensions))

    def forward(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors)
        if values.ndim != 2 or values.shape[1] != self.up_weight.shape[0]:
            raise ValueError("vectors must have shape [sequence_length, dimensions]")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("vectors must contain floating-point values")
        hidden = np.maximum(0, values @ self.up_weight)
        return hidden @ self.down_weight

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        return self.forward(vectors)
