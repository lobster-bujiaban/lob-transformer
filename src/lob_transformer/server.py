"""Small local JSON inference server using only the Python standard library."""
from __future__ import annotations

import json
from importlib.resources import files
from http.server import BaseHTTPRequestHandler, HTTPServer

from .workbench import Workbench


def create_server(checkpoint: str, host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    """Load once and serve serially to bound simultaneous inference work."""
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    workbench = Workbench(checkpoint)

    class Handler(BaseHTTPRequestHandler):
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
            elif self.path == "/health":
                self.respond(200, workbench.info())
            elif self.path == "/training":
                self.respond(200, workbench.snapshot())
            else:
                self.respond(404, {"error": "route not found"})

        def do_POST(self):
            if self.path not in ("/generate", "/train", "/activate"):
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
            limit = 1048576 if self.path == "/train" else 16384
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
                if self.path == "/train":
                    self.respond(202, workbench.start(request))
                    return
                if self.path == "/activate":
                    self.respond(200, workbench.activate(request.get("id")))
                    return
                model, tokenizer = workbench.model, workbench.tokenizer
                if set(request) - {"prompt", "tokens"}:
                    raise ValueError("only prompt and tokens are supported")
                prompt = request.get("prompt")
                tokens = request.get("tokens", 16)
                if not isinstance(prompt, str) or not prompt:
                    raise ValueError("prompt must be a non-empty string")
                if type(tokens) is not int or not 0 <= tokens <= 256:
                    raise ValueError("tokens must be an integer between 0 and 256")
                ids = tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    raise ValueError(f"提示词超过 {model.config.context_length} 个字符的上下文限制")
                if tokenizer.UNK_ID in ids:
                    unknown = "".join(dict.fromkeys(c for c in prompt if c not in tokenizer.stoi))
                    raise ValueError(f"当前词表不包含这些字符：{unknown[:40]}。请先用包含这些字符的语料训练。")
            except RuntimeError as error:
                self.respond(409, {"error": str(error)})
                return
            except TimeoutError:
                self.respond(408, {"error": "request body timed out"})
                return
            except (ValueError, UnicodeError, RecursionError) as error:
                self.respond(400, {"error": str(error)})
                return
            result = model.generate(ids, tokens)
            self.respond(200, {"text": tokenizer.decode(result),
                               "completion": tokenizer.decode(result[len(ids):]),
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
