from __future__ import annotations

import numpy as np


class Embedding:
    """Map token IDs to dense vectors through a learnable lookup table."""

    def __init__(
        self,
        vocab_size: int,
        dimensions: int,
        *,
        seed: int = 7,
        rng: np.random.Generator | None = None,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")

        generator = rng if rng is not None else np.random.default_rng(seed)
        scale = dimensions**-0.5
        self.weight = generator.normal(0, scale, (vocab_size, dimensions))

    @property
    def vocab_size(self) -> int:
        return self.weight.shape[0]

    @property
    def dimensions(self) -> int:
        return self.weight.shape[1]

    def forward(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        """Look up one embedding vector for each token ID."""
        ids = np.asarray(token_ids)
        if ids.ndim != 1:
            raise ValueError("token IDs must be one-dimensional")
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("token IDs must be integers")
        if ids.size and (np.any(ids < 0) or np.any(ids >= self.vocab_size)):
            raise ValueError("token ID is outside the vocabulary")
        return self.weight[ids.astype(np.int64, copy=False)]

    def __call__(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        return self.forward(token_ids)
