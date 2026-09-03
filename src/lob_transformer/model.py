from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedding import Embedding
from .rope import RotaryPositionEmbedding


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=axis, keepdims=True)


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    dimensions: int = 64
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        if self.dimensions % self.heads:
            raise ValueError("dimensions must be divisible by heads")
        if (self.dimensions // self.heads) % 2:
            raise ValueError("attention head dimensions must be even for RoPE")


class TinyGPT:
    """Decoder-only Transformer forward pass with causal attention."""

    def __init__(self, config: ModelConfig, seed: int = 7) -> None:
        self.config = config
        rng = np.random.default_rng(seed)
        scale = config.dimensions ** -0.5
        self.token_embedding = Embedding(config.vocab_size, config.dimensions, rng=rng)
        self.rope = RotaryPositionEmbedding(config.dimensions // config.heads)
        self.layers = []
        for _ in range(config.layers):
            self.layers.append({
                "q": rng.normal(0, scale, (config.dimensions, config.dimensions)),
                "k": rng.normal(0, scale, (config.dimensions, config.dimensions)),
                "v": rng.normal(0, scale, (config.dimensions, config.dimensions)),
                "o": rng.normal(0, scale, (config.dimensions, config.dimensions)),
                "up": rng.normal(0, scale, (config.dimensions, config.dimensions * 4)),
                "down": rng.normal(0, scale, (config.dimensions * 4, config.dimensions)),
            })
        self.lm_head = self.token_embedding.weight.T.copy()

    def _attention(self, x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
        length, dimensions = x.shape
        head_size = dimensions // self.config.heads
        q = (x @ layer["q"]).reshape(length, self.config.heads, head_size).transpose(1, 0, 2)
        k = (x @ layer["k"]).reshape(length, self.config.heads, head_size).transpose(1, 0, 2)
        v = (x @ layer["v"]).reshape(length, self.config.heads, head_size).transpose(1, 0, 2)
        q = self.rope(q)
        k = self.rope(k)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_size)
        scores = np.where(np.triu(np.ones((length, length), dtype=bool), 1), -1e9, scores)
        weights = softmax(scores, axis=-1)
        attended = (weights @ v).transpose(1, 0, 2).reshape(length, dimensions)
        return attended @ layer["o"]

    def forward(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) > self.config.context_length:
            raise ValueError("token sequence must be one-dimensional and fit the context length")
        x = self.token_embedding(ids)
        for layer in self.layers:
            x = x + self._attention(x, layer)
            hidden = np.maximum(0, x @ layer["up"])
            x = x + hidden @ layer["down"]
        return x @ self.lm_head

    def generate(self, token_ids: list[int], count: int) -> list[int]:
        result = list(token_ids)
        for _ in range(count):
            logits = self.forward(result[-self.config.context_length:])
            result.append(int(np.argmax(logits[-1])))
        return result
