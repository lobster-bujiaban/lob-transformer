from __future__ import annotations

import argparse
from pathlib import Path

from .attention import CausalSelfAttention
from .checkpoint import load_checkpoint, save_checkpoint
from .embedding import Embedding
from .model import ModelConfig, TinyGPT
from .rope import RotaryPositionEmbedding
from .tokenizer import CharacterTokenizer
from .training import train, train_corpus
from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="lob-transformer", description="A from-scratch tiny Transformer")
    sub = parser.add_subparsers(dest="command")
    tokenize = sub.add_parser("tokenize", help="encode and decode text with the character tokenizer")
    tokenize.add_argument("--text", required=True)
    embedding = sub.add_parser("embedding", help="look up embedding vectors for text")
    embedding.add_argument("--text", required=True)
    embedding.add_argument("--dimensions", type=int, default=8)
    rope = sub.add_parser("rope", help="apply rotary position embeddings to text vectors")
    rope.add_argument("--text", required=True)
    rope.add_argument("--dimensions", type=int, default=8)
    attention = sub.add_parser("attention", help="run causal multi-head self-attention")
    attention.add_argument("--text", required=True)
    attention.add_argument("--dimensions", type=int, default=8)
    attention.add_argument("--heads", type=int, default=2)
    forward = sub.add_parser("forward", help="run a Transformer forward pass")
    forward.add_argument("--text", required=True)
    generate = sub.add_parser("generate", help="generate tokens from a prompt")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--tokens", type=int, default=16)
    generate.add_argument("--checkpoint", help="load trained weights and vocabulary")
    training = sub.add_parser("train", help="fit a short text using NumPy backpropagation and SGD")
    source = training.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", help="UTF-8 corpus file; enables random-window mini-batch training")
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--context-length", type=int, default=128)
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--steps", type=int, default=200)
    training.add_argument("--learning-rate", type=float, default=0.05)
    training.add_argument("--dimensions", type=int, default=32)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--layers", type=int, default=2)
    training.add_argument("--tokens", type=int, default=16)
    training.add_argument("--save", help="save weights and vocabulary to this checkpoint file")
    serving = sub.add_parser("serve", help="serve a checkpoint over a local HTTP JSON API")
    serving.add_argument("--checkpoint", required=True)
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "serve":
        try:
            serve(args.checkpoint, args.host, args.port)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return
    source_text = args.prompt if args.command == "generate" else args.text
    if args.command == "train" and args.file:
        try:
            source_text = Path(args.file).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            parser.error(str(error))
    if args.command == "generate" and args.checkpoint:
        try:
            if not source_text or args.tokens < 0:
                raise ValueError("prompt must not be empty and tokens must be non-negative")
            model, tokenizer = load_checkpoint(args.checkpoint)
            ids = tokenizer.encode(source_text)
            if tokenizer.UNK_ID in ids:
                raise ValueError("prompt contains characters absent from the saved vocabulary")
            print(tokenizer.decode(model.generate(ids, args.tokens)))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return
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
    if args.command == "embedding":
        try:
            vectors = Embedding(tokenizer.vocab_size, args.dimensions)(prompt_ids)
        except ValueError as error:
            parser.error(str(error))
        print({
            "token_ids": prompt_ids,
            "embedding_shape": list(vectors.shape),
            "vectors": vectors.round(4).tolist(),
        })
        return
    if args.command == "rope":
        try:
            vectors = Embedding(tokenizer.vocab_size, args.dimensions)(prompt_ids)
            rotated = RotaryPositionEmbedding(args.dimensions)(vectors)
        except ValueError as error:
            parser.error(str(error))
        print({
            "input_shape": list(vectors.shape),
            "output_shape": list(rotated.shape),
            "vectors": rotated.round(4).tolist(),
        })
        return
    if args.command == "attention":
        try:
            vectors = Embedding(tokenizer.vocab_size, args.dimensions)(prompt_ids)
            output, weights = CausalSelfAttention(args.dimensions, args.heads).forward(
                vectors,
                return_weights=True,
            )
        except ValueError as error:
            parser.error(str(error))
        print({
            "input_shape": list(vectors.shape),
            "output_shape": list(output.shape),
            "weights_shape": list(weights.shape),
            "head_0_weights": weights[0].round(4).tolist(),
        })
        return
    if args.command == "train":
        try:
            if args.tokens < 0:
                raise ValueError("tokens must be non-negative")
            config = ModelConfig(vocab_size=tokenizer.vocab_size, dimensions=args.dimensions,
                                 heads=args.heads, layers=args.layers, context_length=args.context_length)
            model = TinyGPT(config, seed=args.seed)
            if args.file:
                print({"corpus_tokens": len(prompt_ids), "vocab_size": tokenizer.vocab_size,
                       "window_length": min(config.context_length, len(prompt_ids) - 1),
                       "batch_size": args.batch_size}, flush=True)
                samples = train_corpus(
                    model, prompt_ids, steps=args.steps, learning_rate=args.learning_rate,
                    batch_size=args.batch_size, seed=args.seed,
                    progress=lambda step, loss: print(
                        {"step": step, "train_sample_loss": round(loss, 6)}, flush=True))
                history = [loss for _, loss in samples]
            else:
                history = train(model, prompt_ids, steps=args.steps, learning_rate=args.learning_rate)
            if args.save:
                save_checkpoint(args.save, model, tokenizer)
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
        print({"steps": args.steps, "initial_loss": round(history[0], 6),
               "final_loss": round(history[-1], 6)})
        print(tokenizer.decode(model.generate(prompt_ids[:1], args.tokens)))
        return
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = TinyGPT(config)
    if args.command == "forward":
        logits = model.forward(prompt_ids)
        print({"tokens": len(prompt_ids), "vocab_size": config.vocab_size, "logits_shape": list(logits.shape)})
    else:
        print(tokenizer.decode(model.generate(prompt_ids, args.tokens)))
