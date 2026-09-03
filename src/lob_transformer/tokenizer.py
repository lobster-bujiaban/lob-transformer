from __future__ import annotations


class CharacterTokenizer:
    """Deterministic UTF-8 character tokenizer for the learning baseline."""

    def __init__(self, text: str = "") -> None:
        alphabet = sorted(set(text))
        self.itos = ["<unk>"] + alphabet
        self.stoi = {char: index for index, char in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(char, 0) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[index] if 0 <= index < len(self.itos) else "<unk>" for index in token_ids)
