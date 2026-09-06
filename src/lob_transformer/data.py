"""Sample contiguous next-token windows without padding or dropping corpus tails."""
from __future__ import annotations

import numpy as np


class TextWindows:
    def __init__(self, token_ids, context_length: int, seed: int = 7):
        self.ids = np.asarray(token_ids)
        if self.ids.ndim != 1 or self.ids.size < 2:
            raise ValueError("corpus must contain at least two tokens")
        if not np.issubdtype(self.ids.dtype, np.integer):
            raise TypeError("corpus token IDs must be integers")
        if type(context_length) is not int or context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        self.length = min(context_length, self.ids.size - 1)
        self.window_count = self.ids.size - self.length
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size: int):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        starts = self.rng.integers(0, self.window_count, size=batch_size)
        indices = starts[:, None] + np.arange(self.length + 1)
        windows = self.ids[indices]
        return windows[:, :-1], windows[:, 1:]


def split_corpus(token_ids, validation_fraction=0.1):
    """Split contiguous tokens before window sampling; neither side shares positions."""
    ids = np.asarray(token_ids)
    if not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if ids.ndim != 1 or ids.size < 4:
        raise ValueError("validated corpus training requires at least four tokens")
    count = max(2, int(ids.size * validation_fraction))
    if ids.size - count < 2:
        raise ValueError("split must leave at least two tokens in each set")
    return ids[:-count], ids[-count:]
