from __future__ import annotations

import argparse

from .model import ModelConfig, TinyGPT
from .tokenizer import CharacterTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(prog="lob-transformer", description="A from-scratch tiny Transformer")
    sub = parser.add_subparsers(dest="command")
    forward = sub.add_parser("forward", help="run a Transformer forward pass")
    forward.add_argument("--text", required=True)
    generate = sub.add_parser("generate", help="generate tokens from a prompt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    tokenizer = CharacterTokenizer(args.text if args.command == "forward" else args.prompt)
    config = ModelConfig(vocab_size=len(tokenizer.itos))
    model = TinyGPT(config)
    prompt_ids = tokenizer.encode(args.text if args.command == "forward" else args.prompt)
    if not prompt_ids:
        parser.error("text/prompt must not be empty")
    if args.command == "forward":
        logits = model.forward(prompt_ids)
        print({"tokens": len(prompt_ids), "vocab_size": config.vocab_size, "logits_shape": list(logits.shape)})
    else:
        print(tokenizer.decode(model.generate(prompt_ids, args.tokens)))
