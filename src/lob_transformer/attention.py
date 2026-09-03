from __future__ import annotations

import numpy as np

from .rope import RotaryPositionEmbedding


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


class CausalSelfAttention:
    """Multi-head self-attention that cannot attend to future tokens."""

    def __init__(
        self,
        dimensions: int,
        heads: int,
        *,
        seed: int = 7,
        rng: np.random.Generator | None = None,
    ) -> None:
        if dimensions <= 0 or heads <= 0:
            raise ValueError("dimensions and heads must be positive")
        if dimensions % heads:
            raise ValueError("dimensions must be divisible by heads")
        self.head_size = dimensions // heads
        if self.head_size % 2:
            raise ValueError("attention head dimensions must be even for RoPE")

        self.dimensions = dimensions
        self.heads = heads
        self.rope = RotaryPositionEmbedding(self.head_size)
        generator = rng if rng is not None else np.random.default_rng(seed)
        scale = dimensions**-0.5
        self.query_weight = generator.normal(0, scale, (dimensions, dimensions))
        self.key_weight = generator.normal(0, scale, (dimensions, dimensions))
        self.value_weight = generator.normal(0, scale, (dimensions, dimensions))
        self.output_weight = generator.normal(0, scale, (dimensions, dimensions))

    def forward(
        self,
        vectors: np.ndarray,
        *,
        return_weights: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        values = np.asarray(vectors)
        if values.ndim != 2 or values.shape[1] != self.dimensions:
            raise ValueError("vectors must have shape [sequence_length, dimensions]")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("vectors must contain floating-point values")
        if values.shape[0] == 0:
            raise ValueError("sequence must not be empty")

        length = values.shape[0]
        query = self._split_heads(values @ self.query_weight)
        key = self._split_heads(values @ self.key_weight)
        value = self._split_heads(values @ self.value_weight)
        query = self.rope(query)
        key = self.rope(key)

        scores = query @ key.transpose(0, 2, 1) / np.sqrt(self.head_size)
        future_mask = np.triu(np.ones((length, length), dtype=bool), k=1)
        scores = np.where(future_mask, -np.inf, scores)
        attention_weights = softmax(scores, axis=-1)

        attended = attention_weights @ value
        merged = attended.transpose(1, 0, 2).reshape(length, self.dimensions)
        output = merged @ self.output_weight
        if return_weights:
            return output, attention_weights
        return output

    def _split_heads(self, vectors: np.ndarray) -> np.ndarray:
        length = vectors.shape[0]
        return vectors.reshape(length, self.heads, self.head_size).transpose(1, 0, 2)

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        output = self.forward(vectors)
        assert isinstance(output, np.ndarray)
        return output
