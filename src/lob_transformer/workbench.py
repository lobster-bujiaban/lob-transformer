"""One background training job, separate from the active inference model."""
import math
import base64
import hashlib
import io
import json
import tempfile
import zipfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .checkpoint import load_checkpoint, save_checkpoint
from .model import ModelConfig, TinyGPT
from .tokenizer import CharacterTokenizer
from .training import train_corpus
from .data import split_corpus
from .time_task import dataset, validate_splits, fit, evaluate, TrainingCancelled


class Workbench:
    def __init__(self, checkpoint):
        self.model, self.tokenizer = load_checkpoint(checkpoint)
        self.path = Path(checkpoint).resolve()
        self.task = self.task_for(self.path)
        self.name = Path(checkpoint).name
        self.directory = Path(checkpoint).resolve().parent / "web-checkpoints"
        self.lock = threading.Lock()
        self.job = {"status": "idle"}
        self.result = None
        self.cancel = threading.Event()

    def snapshot(self):
        with self.lock:
            return {**self.job, "history": list(self.job.get("history", []))}

    def info(self):
        return {"status": "ok", "model": self.name, "task": self.task, **asdict(self.model.config),
                "vocabulary": self.tokenizer.itos[1:]}

    def start(self, request):
        if request.get("task") == "time":
            return self.start_time(request)
        request = dict(request)
        if request.pop("task", "text") != "text":
            raise ValueError("不支持的训练模式")
        allowed = {"text", "steps", "batch_size", "context_length", "dimensions", "heads", "layers", "learning_rate", "validation_fraction"}
        if set(request) - allowed:
            raise ValueError("训练参数包含不支持的字段")
        text = request.get("text")
        if not isinstance(text, str) or not 4 <= len(text) <= 1000000:
            raise ValueError("训练语料需为 4～1000000 个字符")
        fraction = request.get("validation_fraction", 0.1)
        if type(fraction) not in (int, float):
            raise ValueError("验证集比例必须为数字")
        train_ids, val_ids = split_corpus(range(len(text)), fraction)
        tokenizer = CharacterTokenizer(text)
        if tokenizer.vocab_size > 8192:
            raise ValueError("网页训练最多支持 8191 种不同字符；请减少语料中的字符种类")
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
            self.cancel.clear()
            job_id = uuid4().hex[:12]
            self.job = {"id": job_id, "task": "text", "status": "running", "step": 0,
                        "steps": values["steps"], "history": [], "elapsed": 0,
                        "train_tokens": len(train_ids), "val_tokens": len(val_ids),
                        "corpus_tokens": len(text), "vocab_size": tokenizer.vocab_size}
            self.result = None

        def run():
            started = time.monotonic()
            try:
                model = TinyGPT(config)

                prompt = tokenizer.encode(text[:min(8, config.context_length)])
                before = tokenizer.decode(model.generate(prompt, 16))

                def progress(**row):
                    if self.cancel.is_set():
                        raise TrainingCancelled("训练已停止")
                    with self.lock:
                        self.job.update(**row, loss=row["train_loss"], elapsed=round(time.monotonic() - started, 2))
                        self.job["history"].append({**row, "loss": row["train_loss"]})

                train_corpus(model, tokenizer.encode(text), steps=values["steps"],
                             batch_size=values["batch_size"], learning_rate=rate, progress=progress,
                             validation_fraction=fraction)
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f"train-{job_id}.npz"
                save_checkpoint(path, model, tokenizer)
                with self.lock:
                    self.result = (model, tokenizer, path, "text")
                    self.job.update(status="completed", checkpoint=str(path),
                                    comparison={"prompt": tokenizer.decode(prompt), "before": before,
                                                "after": tokenizer.decode(model.generate(prompt, 16))},
                                    elapsed=round(time.monotonic() - started, 2))
            except TrainingCancelled as error:
                with self.lock:
                    self.job.update(status="cancelled", error=str(error))
            except Exception as error:
                with self.lock:
                    self.job.update(status="failed", error=str(error))

        threading.Thread(target=run, daemon=True, name=f"train-{job_id}").start()
        return self.snapshot()

    def activate(self, job_id):
        with self.lock:
            if self.job.get("id") != job_id or self.job["status"] != "completed" or self.result is None:
                raise ValueError("指定训练任务尚未完成，无法启用")
            self.model, self.tokenizer, self.path, self.task = self.result
            self.name = self.path.name
            self.job["activated"] = True
        return self.info()


    @staticmethod
    def task_for(path):
        meta = path.with_suffix('.meta.json')
        if meta.exists():
            try:
                return 'time' if json.loads(meta.read_text()).get('task') == 'time' else 'text'
            except (OSError, ValueError):
                return 'text'
        if path.name == 'model.npz' and (path.parent / 'report.json').exists():
            return 'time'
        return 'text'

    def model_paths(self):
        root = Path(__file__).resolve().parents[2]
        paths = {self.path, root / 'model.npz', root / 'data/time-task/model.npz'}
        if self.directory.exists():
            paths.update(self.directory.glob('train-*.npz'))
            paths.update(self.directory.glob('upload-*.npz'))
            paths.update(p for p in self.directory.glob('time-*/model.npz')
                         if (p.parent / 'report.json').is_file())
        return {hashlib.sha256(str(p).encode()).hexdigest()[:16]: p
                for p in sorted(paths) if p.is_file()}

    def models(self):
        return [{"id": key, "name": path.parent.name + '/' + path.name,
                 "task": self.task_for(path), "active": path == self.path,
                 "has_report": (path.parent / 'report.json').exists()}
                for key, path in self.model_paths().items()]

    def select(self, model_id):
        path = self.model_paths().get(model_id)
        if path is None:
            raise ValueError('模型不存在，请刷新列表')
        model, tokenizer = load_checkpoint(path)
        with self.lock:
            self.model, self.tokenizer, self.path = model, tokenizer, path
            self.task, self.name = self.task_for(path), path.name
            self.job['activated'] = False
        return self.info()

    def artifact(self, model_id, kind):
        path = self.model_paths().get(model_id)
        if path is None:
            raise ValueError('模型不存在')
        if kind == 'model':
            return path
        if kind not in ('report', 'train', 'validation', 'test') or self.task_for(path) != 'time':
            raise ValueError('不支持的下载类型')
        artifact = path.parent / (kind + ('.json' if kind == 'report' else '.jsonl'))
        if not artifact.is_file():
            raise ValueError('这个模型没有保存对应文件')
        return artifact

    def upload(self, request):
        if request.get('task') not in ('text', 'time'):
            raise ValueError('请选择模型类型')
        try:
            raw = base64.b64decode(request.get('data', ''), validate=True)
            if not raw or len(raw) > 8 * 1024 * 1024:
                raise ValueError('模型文件须小于 8 MiB')
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                if sum(info.file_size for info in archive.infolist()) > 64 * 1024 * 1024:
                    raise ValueError('模型解压后不能超过 64 MiB')
        except (TypeError, zipfile.BadZipFile) as error:
            raise ValueError('请选择有效的 NPZ 模型文件') from error
        self.directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix='.npz', dir=self.directory) as file:
            file.write(raw)
            file.flush()
            model, tokenizer = load_checkpoint(file.name)
        if request['task'] == 'time':
            needed = set(''.join(r['input'] + '=' + r['output'] for r in dataset()['train']))
            if not needed <= set(tokenizer.stoi) or model.config.context_length < 20:
                raise ValueError('该模型词表或上下文不满足时间转换要求')
        path = self.directory / ('upload-' + uuid4().hex[:12] + '.npz')
        save_checkpoint(path, model, tokenizer)
        path.with_suffix('.meta.json').write_text(json.dumps({'task': request['task']}))
        return {'models': self.models()}

    def stop(self):
        with self.lock:
            if self.job['status'] == 'running':
                self.cancel.set()
                self.job['stopping'] = True
        return self.snapshot()

    def start_time(self, request):
        if set(request) - {'task', 'steps', 'splits'}:
            raise ValueError('时间训练参数包含不支持的字段')
        steps = request.get('steps', 4000)
        if type(steps) is not int or not 1 <= steps <= 10000:
            raise ValueError('训练步数需为 1～10000 的整数')
        splits = validate_splits(request.get('splits', dataset()))
        with self.lock:
            if self.job['status'] == 'running':
                raise RuntimeError('已有训练任务正在运行')
            self.cancel.clear()
            job_id = uuid4().hex[:12]
            self.job = dict(id=job_id, task='time', status='running', step=0,
                            steps=steps, history=[], elapsed=0,
                            counts={key:len(rows) for key,rows in splits.items()})
            self.result = None
        def run():
            started = time.monotonic()
            try:
                def progress(**row):
                    with self.lock:
                        self.job.update(**row, elapsed=round(time.monotonic()-started, 2))
                        self.job['history'].append(row)
                directory = self.directory / ('time-' + job_id)
                model, tokenizer, report = fit(directory, steps, splits=splits,
                                               progress=progress, cancelled=self.cancel.is_set)
                path = directory / 'model.npz'
                with self.lock:
                    self.result = (model, tokenizer, path, 'time')
                    self.job.update(status='completed', report=report, checkpoint=str(path),
                                    elapsed=round(time.monotonic()-started, 2))
            except TrainingCancelled as error:
                with self.lock:
                    self.job.update(status='cancelled', error=str(error))
            except Exception as error:
                with self.lock:
                    self.job.update(status='failed', error=str(error))
        threading.Thread(target=run, daemon=True, name='time-' + job_id).start()
        return self.snapshot()
