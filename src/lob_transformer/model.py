from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .block import TransformerBlock
from .embedding import Embedding
from .normalization import LayerNorm


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    dimensions: int = 64
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.context_length, self.dimensions, self.heads, self.layers) <= 0:
            raise ValueError("vocab_size, context_length, dimensions, heads and layers must be positive")
        if self.dimensions % self.heads:
            raise ValueError("dimensions must be divisible by heads")
        if (self.dimensions // self.heads) % 2:
            raise ValueError("attention head dimensions must be even for RoPE")


class TinyGPT:
    """Decoder-only Transformer forward pass with causal attention."""

    def __init__(self, config: ModelConfig, seed: int = 7) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        self.token_embedding = Embedding(config.vocab_size, config.dimensions, rng=rng)
        self.layers = [
            TransformerBlock(config.dimensions, config.heads, rng=rng)
            for _ in range(config.layers)
        ]
        self.final_norm = LayerNorm(config.dimensions)
        self.lm_head = self.token_embedding.weight.T.copy()

    def forward(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids)
        if ids.ndim != 1 or not 0 < ids.size <= self.config.context_length:
            raise ValueError("token sequence must be non-empty, one-dimensional and fit the context length")
        x = self.token_embedding(ids)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x) @ self.lm_head

    def _forward_cached(self, token_ids, caches=None):
        """Incremental forward; cache ownership stays with the generation request."""
        ids = np.asarray(token_ids)
        offset = 0 if caches is None else caches[0][0].shape[1]
        if ids.ndim != 1 or not 0 < len(ids) <= self.config.context_length - offset:
            raise ValueError("cached tokens must fit the context length")
        x = self.token_embedding(ids)
        updated = []
        for index, layer in enumerate(self.layers):
            x, cache = layer.forward_cached(x, None if caches is None else caches[index])
            updated.append(cache)
        return self.final_norm(x) @ self.lm_head, updated

    def generate(self, token_ids: list[int], count: int, *, use_cache: bool = False) -> list[int]:
        if type(count) is not int or count < 0:
            raise ValueError("count must be a non-negative integer")
        if type(use_cache) is not bool:
            raise ValueError("use_cache must be a boolean")
        result = list(token_ids)
        if not result:
            raise ValueError("prompt must not be empty")
        # Validate even for zero-token requests. Long prompts retain the original sliding window.
        self.token_embedding(np.asarray(result))
        caches = None
        for _ in range(count):
            if not use_cache:
                logits = self.forward(result[-self.config.context_length:])
            elif caches is None or caches[0][0].shape[1] >= self.config.context_length:
                # Dropping an old token changes retained tokens' deeper-layer states.
                # Rebuild with positions starting at zero to match uncached sliding-window inference.
                logits, caches = self._forward_cached(result[-self.config.context_length:])
            else:
                logits, caches = self._forward_cached(result[-1:], caches)
            result.append(int(np.argmax(logits[-1])))
        return result
