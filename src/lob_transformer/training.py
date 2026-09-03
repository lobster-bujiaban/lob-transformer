"""Explicit NumPy backpropagation and full-sequence SGD for the tiny decoder."""
from __future__ import annotations

import numpy as np

from .attention import CausalSelfAttention, softmax
from .model import TinyGPT
from .normalization import LayerNorm
from .data import TextWindows


def _sgd_step(gradients, learning_rate):
    if not all(np.isfinite(g).all() for _, g in gradients):
        raise ValueError("non-finite gradients; reduce learning_rate")
    norm = np.sqrt(sum(float(np.sum(g * g)) for _, g in gradients))
    factor = 1 / max(1.0, norm)
    for parameter, gradient in gradients:
        parameter -= learning_rate * factor * gradient


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean next-token loss and its gradient, using stable log-sum-exp."""
    logits, targets = np.asarray(logits), np.asarray(targets)
    if logits.ndim != 2 or 0 in logits.shape or targets.shape != (logits.shape[0],):
        raise ValueError("expected non-empty logits [tokens, vocab] and targets [tokens]")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("targets must be integers")
    if np.any(targets < 0) or np.any(targets >= logits.shape[1]):
        raise ValueError("target outside vocabulary")
    if not np.isfinite(logits).all():
        raise ValueError("logits must be finite")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    total = exp.sum(axis=-1, keepdims=True)
    rows = np.arange(len(targets))
    loss = np.mean(np.log(total[:, 0]) - shifted[rows, targets])
    gradient = exp / total
    gradient[rows, targets] -= 1
    return float(loss), gradient / len(targets)


def _norm_forward(norm: LayerNorm, x: np.ndarray):
    centered = x - x.mean(axis=-1, keepdims=True)
    inverse_std = 1 / np.sqrt(x.var(axis=-1, keepdims=True) + norm.epsilon)
    normalized = centered * inverse_std

    def backward(dy):
        scaled = dy * norm.weight
        dx = inverse_std * (
            scaled - scaled.mean(axis=-1, keepdims=True)
            - normalized * (scaled * normalized).mean(axis=-1, keepdims=True)
        )
        return dx, [(norm.weight, (dy * normalized).sum(axis=0)),
                    (norm.bias, dy.sum(axis=0))]

    return normalized * norm.weight + norm.bias, backward


def _attention_forward(attention: CausalSelfAttention, x: np.ndarray):
    length, dimensions = x.shape

    def split(v):
        return v.reshape(length, attention.heads, attention.head_size).transpose(1, 0, 2)

    def merge(v):
        return v.transpose(1, 0, 2).reshape(length, dimensions)

    q = attention.rope(split(x @ attention.query_weight))
    k = attention.rope(split(x @ attention.key_weight))
    v = split(x @ attention.value_weight)
    scale = attention.head_size**-0.5
    scores = (q @ k.transpose(0, 2, 1)) * scale
    mask = np.triu(np.ones((length, length), dtype=bool), 1)
    weights = softmax(np.where(mask, -np.inf, scores))
    merged = merge(weights @ v)

    def backward(dy):
        dmerged = split(dy @ attention.output_weight.T)
        dw = dmerged @ v.transpose(0, 2, 1)
        dv = weights.transpose(0, 2, 1) @ dmerged
        ds = weights * (dw - (dw * weights).sum(axis=-1, keepdims=True))
        # The transpose of a rotation is the same rotation at negative positions.
        positions = -np.arange(length)
        dq = merge(attention.rope((ds @ k) * scale, positions))
        dk = merge(attention.rope((ds.transpose(0, 2, 1) @ q) * scale, positions))
        dv = merge(dv)
        dx = (dq @ attention.query_weight.T + dk @ attention.key_weight.T
              + dv @ attention.value_weight.T)
        return dx, [
            (attention.query_weight, x.T @ dq),
            (attention.key_weight, x.T @ dk),
            (attention.value_weight, x.T @ dv),
            (attention.output_weight, merged.T @ dy),
        ]

    return merged @ attention.output_weight, backward


def loss_and_gradients(model: TinyGPT, token_ids, targets):
    """Return loss and (parameter, gradient) pairs for every trainable array.

    The training forward mirrors inference but retains local backward closures.
    No weights are changed here; repeated token IDs accumulate embedding gradients.
    """
    ids = np.asarray(token_ids)
    if ids.ndim != 1 or not 0 < ids.size <= model.config.context_length:
        raise ValueError("input must be non-empty and fit the context length")
    x = model.token_embedding(ids)
    tape = []
    for block in model.layers:
        normalized, norm1_backward = _norm_forward(block.attention_norm, x)
        attended, attention_backward = _attention_forward(block.attention, normalized)
        x = x + attended
        normalized, norm2_backward = _norm_forward(block.mlp_norm, x)
        preactivation = normalized @ block.mlp.up_weight
        hidden = np.maximum(0, preactivation)
        x = x + hidden @ block.mlp.down_weight
        tape.append((block, normalized, preactivation, hidden,
                     norm1_backward, attention_backward, norm2_backward))
    x, final_backward = _norm_forward(model.final_norm, x)
    loss, dlogits = cross_entropy(x @ model.lm_head, targets)
    gradients = [(model.lm_head, x.T @ dlogits)]
    dx, local = final_backward(dlogits @ model.lm_head.T)
    gradients.extend(local)
    for block, normalized, preactivation, hidden, norm1, attention, norm2 in reversed(tape):
        dh = (dx @ block.mlp.down_weight.T) * (preactivation > 0)
        gradients.extend([(block.mlp.down_weight, hidden.T @ dx),
                          (block.mlp.up_weight, normalized.T @ dh)])
        residual, local = norm2(dh @ block.mlp.up_weight.T)
        gradients.extend(local)
        dx = dx + residual
        residual, local = attention(dx)
        gradients.extend(local)
        residual, local = norm1(residual)
        gradients.extend(local)
        dx = dx + residual
    embedding_gradient = np.zeros_like(model.token_embedding.weight)
    np.add.at(embedding_gradient, ids, dx)
    gradients.append((model.token_embedding.weight, embedding_gradient))
    return loss, gradients


def train(model: TinyGPT, token_ids, *, steps: int = 200, learning_rate: float = 0.05):
    """Fit one sequence with SGD; return initial loss and loss after each update."""
    ids = np.asarray(token_ids)
    if ids.ndim != 1 or not 2 <= ids.size <= model.config.context_length + 1:
        raise ValueError("training text must have 2 to context_length + 1 tokens")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    inputs, targets = ids[:-1], ids[1:]
    loss, gradients = loss_and_gradients(model, inputs, targets)
    history = [loss]
    for _ in range(steps):
        _sgd_step(gradients, learning_rate)
        loss, gradients = loss_and_gradients(model, inputs, targets)
        history.append(loss)
    return history


def batch_loss_and_gradients(model: TinyGPT, inputs, targets):
    """Accumulate per-window gradients, averaging before clipping/updating."""
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    if inputs.ndim != 2 or 0 in inputs.shape or targets.shape != inputs.shape:
        raise ValueError("inputs and targets must have matching non-empty [batch, tokens] shapes")
    accumulated = None
    total = 0.0
    for x, y in zip(inputs, targets):
        loss, gradients = loss_and_gradients(model, x, y)
        total += loss
        if accumulated is None:
            accumulated = [(parameter, gradient.copy()) for parameter, gradient in gradients]
        else:
            for (parameter, summed), (other, gradient) in zip(accumulated, gradients):
                assert parameter is other
                summed += gradient
    for _, gradient in accumulated:
        gradient /= len(inputs)
    return total / len(inputs), accumulated


def train_corpus(model: TinyGPT, token_ids, *, steps=200, learning_rate=0.05,
                 batch_size=4, seed=7, progress=None):
    """Random-window SGD; report a fixed training sample, not validation loss."""
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    windows = TextWindows(token_ids, model.config.context_length, seed)
    if np.any(windows.ids < 0) or np.any(windows.ids >= model.config.vocab_size):
        raise ValueError("corpus token ID is outside the vocabulary")
    # Separate RNG leaves training-window sampling independent of evaluation.
    evaluation = TextWindows(token_ids, model.config.context_length, seed + 1).sample(batch_size)

    def evaluate():
        return float(np.mean([cross_entropy(model.forward(x), y)[0]
                              for x, y in zip(*evaluation)]))

    history = [(0, evaluate())]
    if progress:
        progress(*history[-1])
    for step in range(1, steps + 1):
        inputs, targets = windows.sample(batch_size)
        _, gradients = batch_loss_and_gradients(model, inputs, targets)
        _sgd_step(gradients, learning_rate)
        if step % 10 == 0 or step == steps:
            history.append((step, evaluate()))
            if progress:
                progress(*history[-1])
    return history
