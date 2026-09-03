from __future__ import annotations

import argparse

from .model import ModelConfig, TinyGPT
from .tokenizer import CharacterTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(prog="lob-transformer", description="A from-scratch tiny Transformer")
    sub = parser.add_subparsers(dest="command")
    tokenize = sub.add_parser("tokenize", help="encode and decode text with the character tokenizer")
    tokenize.add_argument("--text", required=True)
    forward = sub.add_parser("forward", help="run a Transformer forward pass")
    forward.add_argument("--text", required=True)
    generate = sub.add_parser("generate", help="generate tokens from a prompt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    source_text = args.prompt if args.command == "generate" else args.text
    tokenizer = CharacterTokenizer(source_text)
    prompt_ids = tokenizer.encode(source_text)
    if not prompt_ids:
        parser.error("text/prompt must not be empty")
    if args.command == "tokenize":
        print({
            "text": source_text,
            "token_ids": prompt_ids,
            "decoded": tokenizer.decode(prompt_ids),
            "vocab_size": tokenizer.vocab_size,
        })
        return
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = TinyGPT(config)
    if args.command == "forward":
        logits = model.forward(prompt_ids)
        print({"tokens": len(prompt_ids), "vocab_size": config.vocab_size, "logits_shape": list(logits.shape)})
    else:
        print(tokenizer.decode(model.generate(prompt_ids, args.tokens)))
