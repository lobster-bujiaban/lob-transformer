from __future__ import annotations

from collections.abc import Iterable


class CharacterTokenizer:
    """A deterministic character-level tokenizer.

    The vocabulary is built from the distinct Unicode characters in ``text``.
    Index 0 is reserved for characters that were not present during fitting.
    """

    UNK_TOKEN = "<unk>"
    UNK_ID = 0

    def __init__(self, text: str = "") -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        alphabet = sorted(set(text))
        self.itos = [self.UNK_TOKEN, *alphabet]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs, mapping unseen characters to ``UNK_ID``."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return [self.stoi.get(char, self.UNK_ID) for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        """Convert token IDs back to text; invalid IDs become ``<unk>``."""
        characters: list[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("token IDs must be integers")
            if 0 <= token_id < self.vocab_size:
                characters.append(self.itos[token_id])
            else:
                characters.append(self.UNK_TOKEN)
        return "".join(characters)
