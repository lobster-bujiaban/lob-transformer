from __future__ import annotations

import numpy as np


class RotaryPositionEmbedding:
    """Inject token positions by rotating pairs of vector dimensions."""

    def __init__(self, dimensions: int, base: float = 10_000.0) -> None:
        if dimensions <= 0 or dimensions % 2:
            raise ValueError("RoPE dimensions must be a positive even number")
        if base <= 0:
            raise ValueError("RoPE base must be positive")
        self.dimensions = dimensions
        self.base = base
        self.inverse_frequencies = 1.0 / (
            base ** (np.arange(0, dimensions, 2, dtype=np.float64) / dimensions)
        )

    def forward(
        self,
        vectors: np.ndarray,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply RoPE along the penultimate (sequence) axis."""
        values = np.asarray(vectors)
        if values.ndim < 2 or values.shape[-1] != self.dimensions:
            raise ValueError("vectors must end with [sequence_length, dimensions]")
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError("vectors must contain floating-point values")

        sequence_length = values.shape[-2]
        if positions is None:
            position_values = np.arange(sequence_length, dtype=np.float64)
        else:
            position_values = np.asarray(positions)
            if position_values.shape != (sequence_length,):
                raise ValueError("positions must match the sequence length")
            if not np.issubdtype(position_values.dtype, np.number):
                raise TypeError("positions must contain numbers")

        angles = position_values[:, None] * self.inverse_frequencies[None, :]
        shape = (1,) * (values.ndim - 2) + angles.shape
        cosine = np.cos(angles).reshape(shape)
        sine = np.sin(angles).reshape(shape)
        even = values[..., 0::2]
        odd = values[..., 1::2]

        rotated = np.empty_like(values)
        rotated[..., 0::2] = even * cosine - odd * sine
        rotated[..., 1::2] = even * sine + odd * cosine
        return rotated

    def __call__(
        self,
        vectors: np.ndarray,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.forward(vectors, positions)
