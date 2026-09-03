"""Versioned, pickle-free checkpoints containing weights, config and vocabulary."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .model import ModelConfig, TinyGPT
from .tokenizer import CharacterTokenizer


def _parameters(model: TinyGPT) -> dict[str, np.ndarray]:
    parameters = {"embedding": model.token_embedding.weight, "lm_head": model.lm_head,
                  "final_norm.weight": model.final_norm.weight,
                  "final_norm.bias": model.final_norm.bias}
    for index, block in enumerate(model.layers):
        for component in ("attention_norm", "mlp_norm", "attention", "mlp"):
            module = getattr(block, component)
            names = {"attention": ("query_weight", "key_weight", "value_weight", "output_weight"),
                     "mlp": ("up_weight", "down_weight")}.get(component, ("weight", "bias"))
            for name in names:
                parameters[f"layers.{index}.{component}.{name}"] = getattr(module, name)
    return parameters


def save_checkpoint(path: str | Path, model: TinyGPT, tokenizer: CharacterTokenizer) -> None:
    """Atomically replace a checkpoint, preserving the exact requested filename."""
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("tokenizer vocabulary size does not match model")
    parameters = _parameters(model)
    if not all(np.isfinite(value).all() for value in parameters.values()):
        raise ValueError("cannot save non-finite model weights")
    metadata = json.dumps({"version": 1, "config": asdict(model.config),
                           "vocabulary": tokenizer.itos}, ensure_ascii=False)
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as file:
            temporary = file.name
            np.savez_compressed(file, metadata=np.array(metadata), **parameters)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path) -> tuple[TinyGPT, CharacterTokenizer]:
    """Restore a model and its original token IDs; reject incompatible archives."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata["version"] != 1:
                raise ValueError("unsupported checkpoint version")
            config = metadata["config"]
            if not isinstance(config, dict) or any(type(v) is not int or v <= 0 for v in config.values()):
                raise ValueError("invalid model config")
            vocabulary = metadata["vocabulary"]
            if (not isinstance(vocabulary, list) or not vocabulary
                    or vocabulary[0] != CharacterTokenizer.UNK_TOKEN
                    or any(not isinstance(c, str) or len(c) != 1 for c in vocabulary[1:])
                    or len(set(vocabulary)) != len(vocabulary)):
                raise ValueError("invalid vocabulary")
            tokenizer = CharacterTokenizer()
            tokenizer.itos = vocabulary
            tokenizer.stoi = {token: index for index, token in enumerate(vocabulary)}
            model_config = ModelConfig(**config)
            if model_config.vocab_size != tokenizer.vocab_size:
                raise ValueError("vocabulary size does not match model")
            model = TinyGPT(model_config)
            parameters = _parameters(model)
            if set(archive.files) != {"metadata", *parameters}:
                raise ValueError("checkpoint parameter names do not match model")
            for name, parameter in parameters.items():
                value = archive[name]
                if (value.shape != parameter.shape or value.dtype != parameter.dtype
                        or not np.isfinite(value).all()):
                    raise ValueError(f"invalid parameter: {name}")
                parameter[...] = value
            return model, tokenizer
    except (KeyError, TypeError, AttributeError, ValueError, EOFError, BadZipFile) as error:
        raise ValueError(f"invalid checkpoint: {error}") from error
