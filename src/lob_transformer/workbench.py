"""One background training job, separate from the active inference model."""
import math
import threading
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .checkpoint import load_checkpoint, save_checkpoint
from .model import ModelConfig, TinyGPT
from .tokenizer import CharacterTokenizer
from .training import train_corpus


class Workbench:
    def __init__(self, checkpoint):
        self.model, self.tokenizer = load_checkpoint(checkpoint)
        self.name = Path(checkpoint).name
        self.directory = Path(checkpoint).resolve().parent / "web-checkpoints"
        self.lock = threading.Lock()
        self.job = {"status": "idle"}
        self.result = None

    def snapshot(self):
        with self.lock:
            return {**self.job, "history": list(self.job.get("history", []))}

    def info(self):
        return {"status": "ok", "model": self.name, **asdict(self.model.config),
                "vocabulary": self.tokenizer.itos[1:]}

    def start(self, request):
        allowed = {"text", "steps", "batch_size", "context_length", "dimensions", "heads", "layers", "learning_rate"}
        if set(request) - allowed:
            raise ValueError("训练参数包含不支持的字段")
        text = request.get("text")
        if not isinstance(text, str) or not 2 <= len(text) <= 100000:
            raise ValueError("训练语料需为 2～100000 个字符")
        tokenizer = CharacterTokenizer(text)
        if tokenizer.vocab_size > 512:
            raise ValueError("网页训练最多支持 511 种不同字符；更大语料请使用 CLI")
        defaults = {"steps": 200, "batch_size": 4, "context_length": 32,
                    "dimensions": 32, "heads": 4, "layers": 2}
        limits = {"steps": 2000, "batch_size": 16, "context_length": 128,
                  "dimensions": 64, "heads": 8, "layers": 2}
        values = {}
        for key, default in defaults.items():
            value = request.get(key, default)
            if type(value) is not int or not 1 <= value <= limits[key]:
                raise ValueError(f"{key} 必须为 1～{limits[key]} 的整数")
            values[key] = value
        rate = request.get("learning_rate", 0.05)
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 1:
            raise ValueError("学习率必须大于 0 且不超过 1")
        config = ModelConfig(vocab_size=tokenizer.vocab_size,
                             **{key: values[key] for key in ("context_length", "dimensions", "heads", "layers")})
        with self.lock:
            if self.job["status"] == "running":
                raise RuntimeError("已有训练任务正在运行")
            job_id = uuid4().hex[:12]
            self.job = {"id": job_id, "status": "running", "step": 0,
                        "steps": values["steps"], "history": [], "elapsed": 0,
                        "corpus_tokens": len(text), "vocab_size": tokenizer.vocab_size}
            self.result = None

        def run():
            started = time.monotonic()
            try:
                model = TinyGPT(config)

                def progress(step, loss):
                    with self.lock:
                        self.job.update(step=step, loss=loss, elapsed=round(time.monotonic() - started, 2))
                        self.job["history"].append({"step": step, "loss": loss})

                train_corpus(model, tokenizer.encode(text), steps=values["steps"],
                             batch_size=values["batch_size"], learning_rate=rate, progress=progress)
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f"train-{job_id}.npz"
                save_checkpoint(path, model, tokenizer)
                with self.lock:
                    self.result = (model, tokenizer, path.name)
                    self.job.update(status="completed", checkpoint=str(path),
                                    elapsed=round(time.monotonic() - started, 2))
            except Exception as error:
                with self.lock:
                    self.job.update(status="failed", error=str(error))

        threading.Thread(target=run, daemon=True, name=f"train-{job_id}").start()
        return self.snapshot()

    def activate(self, job_id):
        with self.lock:
            if self.job.get("id") != job_id or self.job["status"] != "completed" or self.result is None:
                raise ValueError("指定训练任务尚未完成，无法启用")
            self.model, self.tokenizer, self.name = self.result
            self.job["activated"] = True
        return self.info()
