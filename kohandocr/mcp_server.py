"""손글씨 판독을 MCP **도구**로 내놓는다 — LM Studio 같은 채팅 앱의 모델이 부른다.

    pip install "ko-hand-ocr[mcp]"
    ko-hand-ocr-mcp                     # stdio. 손님 앱이 띄운다. 손으로 띄울 일은 없다

LM Studio(0.3.17 부터)의 `mcp.json`:

    {"mcpServers": {"ko-hand-ocr": {"command": "ko-hand-ocr-mcp"}}}

그 뒤 채팅 모델(Qwen3, Gemma 3 …)에게 "C:\\scan.jpg 읽어 줘" 라고 하면 모델이
`read_handwriting` 에 그 경로를 넘기고, 읽은 줄을 받아 대답에 쓴다. 정리·요약·서식
채우기는 채팅 모델이 하고, 글자를 알아보는 것만 이쪽이 한다.

**채팅 창에 붙인 그림은 못 받는다.** 그 그림은 채팅 모델에게 가지 도구에게 오지
않는다. 그래서 파일 경로를 받는다.

모델은 **처음 부를 때** 올린다. 손님 앱은 서버가 뜨자마자 도구 목록을 묻는데, 그때
모델을 올리고 있으면(처음 한 번은 허깅페이스에서 156MB 를 받는다) 기다리다 끊는다.

stdout 은 MCP 가 말을 주고받는 줄이다. 모델을 올리며 찍히는 글이 거기 섞이면 손님이
말을 못 알아듣는다. 그래서 올리고 읽는 동안 `print` 를 stderr 로 돌린다.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
from pathlib import Path

from .serve import BODY_MAX, REPO, Engine, Unreadable, fetch

# 예상한 실패는 `ToolError` 로 낸다. SDK 2 는 다른 예외를 '고장' 으로 보고 까닭을 감춘
# 채 "Error executing tool" 만 돌려준다 — 채팅 모델이 파일이 없다는 것도 모른다.
try:                                                         # MCP SDK 2
    from mcp.server.mcpserver import MCPServer as _Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:
    try:                                                     # MCP SDK 1
        from mcp.server.fastmcp import FastMCP as _Server
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError:
        raise SystemExit('MCP 꾸러미가 없다: pip install "ko-hand-ocr[mcp]"') from None

server = _Server("ko-hand-ocr")
_options = {"model": REPO, "ensemble": False, "device": "cpu"}
_engine: Engine | None = None
_lock = threading.Lock()


def _load() -> Engine:
    global _engine
    with _lock:
        if _engine is None:
            with contextlib.redirect_stdout(sys.stderr):
                _engine = Engine(fetch(_options["model"], _options["ensemble"]),
                                 device=_options["device"])
    return _engine


@server.tool()
def read_handwriting(path: str) -> str:
    """Read handwritten Korean/English text from an image file on this computer.

    Give the full path of a photo or scan (jpg, png, webp). Returns the text, one line per
    handwritten line, top to bottom. 손글씨 사진 파일을 읽어 줄마다 한 줄씩 글로 돌려준다.
    """
    spot = Path(path.strip().strip("\"'")).expanduser()
    if not spot.is_file():
        raise ToolError("파일이 없다: %s" % spot)
    if spot.stat().st_size > BODY_MAX:
        raise ToolError("파일이 너무 크다 (%dMB 까지)" % (BODY_MAX // 1048576))
    engine = _load()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            found = engine.lines(spot.read_bytes())
    except Unreadable as err:
        raise ToolError(str(err)) from None
    return "\n".join(found) if found else "(읽을 글씨를 찾지 못했다)"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ko-hand-ocr-mcp",
                                 description="손글씨 판독을 MCP 도구로 내놓는다 (stdio)")
    ap.add_argument("--model", default=REPO,
                    help="모델 폴더, 또는 허깅페이스 저장소 (기본: %(default)s)")
    ap.add_argument("--ensemble", action="store_true",
                    help="앙상블(모델 3개)로 읽는다. 더 정확하고 7배 느리다")
    ap.add_argument("--device", default="cpu", help="cpu 또는 cuda")
    args = ap.parse_args(argv)
    _options.update(model=args.model, ensemble=args.ensemble, device=args.device)
    server.run()


if __name__ == "__main__":
    main()
