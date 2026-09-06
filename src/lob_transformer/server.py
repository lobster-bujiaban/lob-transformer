"""Small local JSON inference server using only the Python standard library."""
from __future__ import annotations

import json
from importlib.resources import files
from http.server import BaseHTTPRequestHandler, HTTPServer

from .workbench import Workbench
from .time_task import dataset, examples, evaluate
from .inference import timed_generate, benchmark
import re


def create_server(checkpoint: str, host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    """Load once and serve serially to bound simultaneous inference work."""
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    workbench = Workbench(checkpoint)

    class Handler(BaseHTTPRequestHandler):
        def download(self, body, filename, content_type="application/octet-stream"):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def respond(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = files("lob_transformer").joinpath("static/index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/app.js":
                body = files("lob_transformer").joinpath("static/app.js").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/models":
                self.respond(200, workbench.models())
            elif self.path == "/time-data":
                self.respond(200, dataset())
            elif self.path.startswith("/artifacts/"):
                try:
                    _, _, model_id, kind = self.path.split('/')
                    path = workbench.artifact(model_id, kind)
                    self.download(path.read_bytes(), path.name)
                except (ValueError, OSError) as error:
                    self.respond(404, {"error": str(error)})
            elif self.path == "/health":
                self.respond(200, workbench.info())
            elif self.path == "/training":
                self.respond(200, workbench.snapshot())
            else:
                self.respond(404, {"error": "route not found"})

        def do_POST(self):
            if self.path not in ("/generate", "/train", "/activate", "/stop", "/select-model", "/upload-model", "/convert-time", "/evaluate", "/benchmark"):
                self.respond(404, {"error": "route not found"})
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.respond(403, {"error": "不允许跨站请求"})
                return
            if self.headers.get_content_type() != "application/json":
                self.respond(415, {"error": "Content-Type must be application/json"})
                return
            if self.headers.get("Transfer-Encoding"):
                self.respond(400, {"error": "Transfer-Encoding is not supported"})
                return
            lengths = self.headers.get_all("Content-Length", [])
            if not lengths:
                self.respond(411, {"error": "Content-Length is required"})
                return
            if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                self.respond(400, {"error": "invalid Content-Length"})
                return
            length = int(lengths[0])
            limit = 16 * 1024 * 1024 if self.path in ("/train", "/upload-model") else 16384
            if length > limit:
                self.respond(413, {"error": f"request body exceeds {limit} bytes"})
                return
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete request body")
                request = json.loads(body.decode("utf-8"))
                if not isinstance(request, dict):
                    raise ValueError("request must be a JSON object")
                if self.path == "/stop":
                    self.respond(200, workbench.stop())
                    return
                if self.path == "/select-model":
                    self.respond(200, workbench.select(request.get("id")))
                    return
                if self.path == "/upload-model":
                    self.respond(200, workbench.upload(request))
                    return
                if self.path == "/evaluate":
                    model_id = request.get("id")
                    path = workbench.artifact(model_id, "model")
                    test_path = workbench.artifact(model_id, "test")
                    from .checkpoint import load_checkpoint
                    model, tokenizer = load_checkpoint(path)
                    rows = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
                    result = evaluate(model, tokenizer, rows)
                    self.respond(200, {"test": result, "passed": result["accuracy"] >= .95})
                    return
                if self.path == "/train":
                    self.respond(202, workbench.start(request))
                    return
                if self.path == "/activate":
                    self.respond(200, workbench.activate(request.get("id")))
                    return
                model, tokenizer = workbench.model, workbench.tokenizer
                if set(request) - {"prompt", "tokens", "use_cache"}:
                    raise ValueError("only prompt, tokens and use_cache are supported")
                prompt = request.get("prompt")
                tokens = request.get("tokens", 16)
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError("prompt must be a non-empty string")
                if type(tokens) is not int or not 0 <= tokens <= 256:
                    raise ValueError("tokens must be an integer between 0 and 256")
                use_cache = request.get("use_cache", True)
                if type(use_cache) is not bool:
                    raise ValueError("use_cache must be a boolean")
                time_task = self.path == "/convert-time" or (self.path == "/benchmark" and workbench.task == "time")
                if time_task:
                    if workbench.task != "time":
                        raise ValueError("请先启用时间转换模型")
                    if prompt not in {row['input'] for row in examples()}:
                        raise ValueError("不支持的时间表达，请使用页面列出的中文时间格式")
                    prompt += "="
                    tokens = 5
                ids = tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    raise ValueError(f"提示词超过 {model.config.context_length} 个字符的上下文限制")
                if tokenizer.UNK_ID in ids:
                    unknown = "".join(dict.fromkeys(c for c in prompt if c not in tokenizer.stoi))
                    raise ValueError(f"当前词表不包含这些字符：{unknown[:40]}。请先用包含这些字符的语料训练。")
                if self.path == "/benchmark":
                    self.respond(200, benchmark(model, tokenizer, ids, tokens))
                    return
                result, metrics = timed_generate(model, ids, tokens, use_cache)
                completion = tokenizer.decode(result[len(ids):])
                if time_task and not re.fullmatch(r'(?:[01][0-9]|2[0-3]):[0-5][0-9]', completion):
                    raise ValueError(f"模型未生成有效时间：{completion!r}")
            except RuntimeError as error:
                self.respond(409, {"error": str(error)})
                return
            except TimeoutError:
                self.respond(408, {"error": "request body timed out"})
                return
            except (ValueError, UnicodeError, RecursionError, OSError, TypeError) as error:
                self.respond(400, {"error": str(error)})
                return
            self.respond(200, {"text": tokenizer.decode(result),
                               "completion": completion, "metrics": metrics,
                               "prompt_tokens": len(ids), "generated_tokens": tokens})

    return HTTPServer((host, port), Handler)


def serve(checkpoint: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    with create_server(checkpoint, host, port) as server:
        address, bound_port = server.server_address[:2]
        print(f"Serving on http://{address}:{bound_port} (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
