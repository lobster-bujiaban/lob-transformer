from __future__ import annotations

import numpy as np

from .attention import CausalSelfAttention
from .mlp import MLP
from .normalization import LayerNorm


class TransformerBlock:
    """Pre-Norm decoder block with two residual connections."""

    def __init__(
        self,
        dimensions: int,
        heads: int,
        *,
        seed: int = 7,
        rng: np.random.Generator | None = None,
    ) -> None:
        generator = rng if rng is not None else np.random.default_rng(seed)
        self.attention_norm = LayerNorm(dimensions)
        self.attention = CausalSelfAttention(dimensions, heads, rng=generator)
        self.mlp_norm = LayerNorm(dimensions)
        self.mlp = MLP(dimensions, rng=generator)

    def forward(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors)
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))

    def forward_cached(self, vectors: np.ndarray, cache=None):
        x = np.asarray(vectors)
        attended, cache = self.attention.forward_cached(self.attention_norm(x), cache)
        x = x + attended
        return x + self.mlp(self.mlp_norm(x)), cache

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        return self.forward(vectors)
