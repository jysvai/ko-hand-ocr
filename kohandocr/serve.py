"""판독기를 **LLM 서버와 같은 말**로 띄운다 — OpenAI 호환 + Ollama 호환.

    ko-hand-ocr-serve                         # 허깅페이스에서 단일 모델을 받아 띄운다
    ko-hand-ocr-serve --ensemble              # 앙상블(모델 3개). 더 정확하고 7배 느리다
    ko-hand-ocr-serve --model ./ko-hand-ocr-single --port 8765

왜 이렇게 하나. 이 모델은 LM Studio·Ollama·vLLM 에 **모델로는** 못 올린다. 셋 다
글을 이어 쓰는 LLM 을 돌리는 엔진이고(LM Studio·Ollama 는 llama.cpp 의 GGUF, vLLM 은
자기 모델 목록), 이 모델은 그림 인코더 + 교차 주의 디코더(TrOCR)라 그 목록에 없다.
대신 **그 엔진들이 밖으로 내놓는 말(API)** 을 똑같이 한다. 그러면 Open WebUI 같은
화면, `openai`·`ollama` 클라이언트, vLLM 을 부르던 코드가 주소만 바꿔서 부른다.

    GET  /v1/models          POST /v1/chat/completions             OpenAI (vLLM·LM Studio 서버와 같은 말)
    GET  /api/tags           POST /api/chat   POST /api/generate   Ollama
    GET  /api/version        POST /api/show
    POST /read               그림 바이트를 그대로 받아 줄 목록을 JSON 으로 (curl 용)

그림은 메시지에 **base64 로 실어 보낸 것만** 받는다. 주소(`http://…`)를 주면 받으러
가지 않고 거절한다 — 이 판독기는 아무 데도 나가지 않는다. 글로 물은 것은 쓰지 않는다.
그림을 줄로 잘라 읽고 줄마다 한 줄씩 돌려준다.

기본은 `127.0.0.1` 에만 연다. 비밀번호가 없으니 `--host 0.0.0.0` 은 믿는 망에서만.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from . import __version__

REPO = "localdeel/ko-hand-ocr"
# LM Studio 1234, Ollama 11434, vLLM 8000 과 안 겹치게.
PORT = 8765
# 한 요청의 몸통 상한. 사진 몇 장을 base64 로 실어도 이 안에 든다.
BODY_MAX = 64 * 1024 * 1024
HINT = "그림을 같이 보내 주세요. 이 모델은 손글씨 사진을 읽어 줄마다 한 줄씩 글로 돌려줍니다."


class Unreadable(ValueError):
    """요청에서 그림을 꺼낼 수 없다. 400 으로 돌려준다."""


def decode_image(text: str | None) -> bytes:
    """`data:image/png;base64,...` 또는 맨 base64 -> 바이트."""
    if not isinstance(text, str) or not text:
        raise Unreadable("그림이 비어 있다")
    if text.startswith(("http://", "https://")):
        raise Unreadable("그림 주소는 받으러 가지 않는다. base64 로 실어 보내라 "
                         "(data:image/jpeg;base64,...)")
    if text.startswith("data:"):
        head, _, text = text.partition(",")
        if ";base64" not in head:
            raise Unreadable("data: 주소는 base64 여야 한다")
    try:
        return base64.b64decode(text)
    except (binascii.Error, ValueError) as err:
        raise Unreadable("base64 를 풀 수 없다: %s" % err) from None


def openai_images(messages: list | None) -> list[bytes]:
    """OpenAI 꼴 메시지에서 **마지막 사용자 말**의 그림만 꺼낸다.

    채팅 화면은 앞서 주고받은 말을 통째로 다시 보낸다. 앞 그림까지 읽으면 새 그림을
    보낼 때마다 옛 그림의 답이 앞에 붙는다.
    """
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        found = []
        for part in message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url")
                found.append(decode_image(url.get("url") if isinstance(url, dict) else url))
        return found
    return []


def ollama_images(messages: list | None) -> list[bytes]:
    """Ollama 꼴 메시지(`images` 에 base64 목록)에서 마지막 사용자 말의 그림만."""
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return [decode_image(one) for one in message.get("images") or []]
    return []


def fetch(model: str, ensemble: bool) -> Path:
    """`--model` 이 폴더면 그대로, 아니면 허깅페이스 저장소 이름으로 보고 받는다.

    앙상블은 저장소의 `ensemble/` 에 있다. 단일 모델만 쓸 때 그 450MB 를 받지 않는다.
    """
    spot = Path(model).expanduser()
    if spot.is_dir():
        return spot / "ensemble" if ensemble and (spot / "ensemble").is_dir() else spot
    from huggingface_hub import snapshot_download

    if ensemble:
        return Path(snapshot_download(model, allow_patterns=["ensemble/*"])) / "ensemble"
    return Path(snapshot_download(model, ignore_patterns=["ensemble/*", "assets/*"]))


class Engine:
    """판독기 하나를 올려 두고 **한 번에 한 요청씩** 읽는다.

    torch 가 이미 CPU 코어를 다 쓴다. 요청 둘을 겹쳐 읽으면 둘 다 느려질 뿐이다.
    """

    def __init__(self, folder: Path, name: str = "", device: str = "cpu") -> None:
        from .reader import Reader

        self.reader = Reader(folder, device=device)
        self.name = name or ("ko-hand-ocr-ensemble" if self.reader.also else "ko-hand-ocr")
        self.size = sum(p.stat().st_size for p in Path(folder).rglob("*.safetensors"))
        # `ollama list` 는 지문 앞 12자를 잘라 보여 준다. 비어 있으면 거기서 죽는다.
        digest = hashlib.sha256()
        with open(Path(folder) / "model.safetensors", "rb") as fp:
            for piece in iter(lambda: fp.read(1 << 20), b""):
                digest.update(piece)
        self.digest = digest.hexdigest()
        self.modified = _when((Path(folder) / "model.safetensors").stat().st_mtime)
        self.lock = threading.Lock()

    def lines(self, data: bytes) -> list[str]:
        """사진 한 장 -> 줄 목록. 줄을 못 찾으면 그림 전체를 한 줄로 읽는다.

        이미 한 줄만 잘라 보낸 그림에서 줄 자르기가 아무것도 못 찾는 일이 있다.
        """
        from PIL import Image, UnidentifiedImageError

        try:
            Image.open(BytesIO(data)).verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as err:
            raise Unreadable("그림으로 열 수 없다 (jpg·png·webp 같은 그림 파일만 읽는다): %s"
                             % type(err).__name__) from None
        with self.lock:
            found = self.reader.read_photo(data)
            if not found:
                found = self.reader.read([Image.open(BytesIO(data)).convert("L")])
        return [one for one in found if one]

    def text(self, images: list[bytes]) -> str:
        """그림 여럿 -> 대답 글. 그림 사이는 빈 줄로 가른다."""
        if not images:
            return HINT
        return "\n\n".join("\n".join(self.lines(one)) for one in images)


class Server(ThreadingHTTPServer):
    engine: Engine

    def handle_error(self, request, client_address) -> None:
        # 손님이 먼저 끊은 것은 잘못이 아니다. Windows 는 그때마다 10054 를 길게 찍는다.
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def _when(stamp: float | None = None) -> str:
    moment = datetime.fromtimestamp(stamp, timezone.utc) if stamp else datetime.now(timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ko-hand-ocr/" + __version__

    server: Server

    @property
    def engine(self) -> Engine:
        return self.server.engine

    # ── 보내기 ───────────────────────────────────────────────────
    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, value) -> None:
        self._send(code, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _stream(self, kind: str, pieces: list[bytes]) -> None:
        """흘려 보내기. 길이를 미리 안 적고 연결을 닫아 끝을 알린다."""
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for piece in pieces:
            self.wfile.write(piece)
            self.wfile.flush()
        self.close_connection = True

    def _body(self) -> bytes:
        size = int(self.headers.get("Content-Length") or 0)
        if size > BODY_MAX:
            raise Unreadable("요청이 너무 크다 (%dMB 까지)" % (BODY_MAX // 1048576))
        return self.rfile.read(size) if size else b""

    # ── 받기 ─────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802 — http.server 이 정한 이름
        path = self.path.split("?")[0].rstrip("/")
        name = self.engine.name
        if path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "created": 0, "owned_by": "ko-hand-ocr"}]})
        elif path == "/api/tags":
            self._json(200, {"models": [{
                "name": name, "model": name, "modified_at": self.engine.modified,
                "size": self.engine.size, "digest": self.engine.digest,
                "details": {"format": "safetensors", "family": "trocr",
                            "parameter_size": "41M", "quantization_level": "F32"}}]})
        elif path == "/api/version":
            self._json(200, {"version": __version__})
        elif path == "":
            self._send(200, b"ko-hand-ocr is running", "text/plain; charset=utf-8")
        else:
            self._json(404, {"error": "없는 자리: %s" % self.path})

    def do_HEAD(self) -> None:  # noqa: N802
        # Ollama 클라이언트는 `HEAD /` 로 서버가 살아 있나 먼저 본다.
        self.send_response(200 if self.path.split("?")[0].rstrip("/") == "" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        openai = path in ("/v1/chat/completions", "/chat/completions")
        try:
            body = self._body()
            if path == "/read":
                self._json(200, {"model": self.engine.name, "lines": self.engine.lines(body)})
                return
            ask = json.loads(body or b"{}")
            if openai:
                self._openai(ask)
            elif path == "/api/chat":
                self._ollama(ask, ollama_images(ask.get("messages")), chat=True)
            elif path == "/api/generate":
                images = [decode_image(one) for one in ask.get("images") or []]
                self._ollama(ask, images, chat=False)
            elif path == "/api/show":
                self._json(200, {
                    "modelfile": "", "parameters": "", "template": "",
                    "details": {"format": "safetensors", "family": "trocr",
                                "parameter_size": "41M", "quantization_level": "F32"},
                    "model_info": {}, "capabilities": ["completion", "vision"]})
            else:
                self._json(404, {"error": "없는 자리: %s" % self.path})
        except (Unreadable, json.JSONDecodeError, AttributeError, TypeError) as err:
            message = str(err) if isinstance(err, Unreadable) else "요청 꼴이 틀렸다: %s" % err
            if openai:
                self._json(400, {"error": {"message": message, "type": "invalid_request_error"}})
            else:
                self._json(400, {"error": message})

    def _openai(self, ask: dict) -> None:
        text = self.engine.text(openai_images(ask.get("messages")))
        head = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()),
                "model": self.engine.name}
        if not ask.get("stream"):
            self._json(200, {**head, "object": "chat.completion", "choices": [{
                "index": 0, "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
            return
        chunk = {**head, "object": "chat.completion.chunk"}
        pieces = [
            {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text},
                                   "finish_reason": None}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        self._stream("text/event-stream; charset=utf-8",
                     [("data: %s\n\n" % json.dumps(p, ensure_ascii=False)).encode("utf-8")
                      for p in pieces] + [b"data: [DONE]\n\n"])

    def _ollama(self, ask: dict, images: list[bytes], chat: bool) -> None:
        start = time.perf_counter_ns()
        # 빈 요청은 Ollama 에서 '모델을 올려 두라' 는 뜻이다. 읽을 것이 없으니 빈 답.
        empty = not images and not (ask.get("messages") if chat else ask.get("prompt"))
        text = "" if empty else self.engine.text(images)
        head = {"model": self.engine.name, "created_at": _when()}

        def say(words: str) -> dict:
            return {"message": {"role": "assistant", "content": words}} if chat else {"response": words}

        last = {**head, **say(""), "done": True, "done_reason": "load" if empty else "stop",
                "total_duration": time.perf_counter_ns() - start, "load_duration": 0,
                "prompt_eval_count": 0, "eval_count": 0, "eval_duration": 0}
        # Ollama 는 `stream` 을 안 주면 흘려 보낸다.
        if ask.get("stream", True) is False:
            self._json(200, {**last, **say(text)})
            return
        rows = ([{**head, **say(text), "done": False}] if text else []) + [last]
        self._stream("application/x-ndjson",
                     [(json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in rows])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ko-hand-ocr-serve",
                                 description="손글씨 판독기를 OpenAI·Ollama 호환 서버로 띄운다")
    ap.add_argument("--model", default=REPO,
                    help="모델 폴더, 또는 허깅페이스 저장소 (기본: %(default)s)")
    ap.add_argument("--ensemble", action="store_true",
                    help="앙상블(모델 3개)로 읽는다. 더 정확하고 7배 느리다")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--device", default="cpu", help="cpu 또는 cuda")
    ap.add_argument("--name", default="", help="밖에 알릴 모델 이름")
    args = ap.parse_args(argv)

    folder = fetch(args.model, args.ensemble)
    server = Server((args.host, args.port), Handler)
    server.engine = Engine(folder, args.name, args.device)
    base = "http://%s:%d" % (args.host, args.port)
    print("ko-hand-ocr %s — 모델 %s (%s)" % (__version__, server.engine.name, folder))
    print("  OpenAI 호환  %s/v1" % base)
    print("  Ollama 호환  %s" % base)
    print("  그대로       curl --data-binary @scan.jpg %s/read" % base, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
