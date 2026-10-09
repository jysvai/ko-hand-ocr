"""견줌 상대 하나를 **제 환경에서** 띄워 두고 줄 그림을 읽힌다. `tools/vs.py` 가 부른다.

    <그 엔진의 python> tools/rival_worker.py easyocr  [--device cuda]
    <그 엔진의 python> tools/rival_worker.py paddle
    python             tools/rival_worker.py tesseract --tesseract <tesseract.exe>
    <vlm 의 python>    tools/rival_worker.py vlm --repo Qwen/Qwen3-VL-2B-Instruct --prompt "..."

장치는 `load` 를 부를 때 준다(아래).

왜 따로 띄우나. EasyOCR 은 torchvision·OpenCV 를, PaddleOCR 은 paddle 이라는 다른
프레임워크를 끌고 온다. 학습 venv 에 깔면 torch 판이 바뀔 수 있고(앱 venv 의 CPU
torch 로 학습이 2장/초로 돈 적이 있다), 그러면 견주려다 학습을 망친다. 그래서
엔진마다 `~/.cache/ko-hand-ocr/rivals/<엔진>` 에 따로 깔고, 여기서는 표준
라이브러리와 PIL·numpy 와 그 엔진만 쓴다.

주고받는 것은 줄마다 JSON 한 줄이다(stdin -> stdout).

    {"op": "load", "device": "cuda"}           -> {"seconds", "params", "disk_mb", "what"}
    {"op": "read", "paths": [png, ...]}        -> {"texts": [...], "seconds", "peak_mb"}
    {"op": "to",   "device": "cpu"}            -> {"seconds"}
    {"op": "quit"}

**시간은 이쪽에서 잰다.** 그림을 파일에서 읽어 메모리에 올린 **뒤부터** 엔진이 답을
내놓을 때까지다. 파이프로 주고받는 시간과 PNG 를 쓰고 푸는 시간은 우리가 붙인 것이라
상대의 빠르기에 넣으면 안 된다. 다만 Tesseract 는 명령줄 프로그램이라 그림을 제가
파일에서 읽는다 — 그 시간은 그 엔진을 쓰는 사람도 똑같이 치르므로 넣는다.

엔진이 찍는 로그는 전부 stderr 로 보낸다. stdout 에 한 글자라도 섞이면 답이 깨진다.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

# 답을 내보낼 길은 따로 쥐고, 원래 stdout(파이썬과 C 둘 다)은 stderr 로 돌린다.
_CHAN = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
os.dup2(2, 1)
sys.stdout = sys.stderr

from PIL import Image                                    # noqa: E402


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def _size_mb(paths) -> float:
    return sum(Path(p).stat().st_size for p in paths if Path(p).is_file()) / 2**20


# ── 엔진들 ────────────────────────────────────────────────────────────

class EasyOCR:
    """`easyocr.Reader(['ko', 'en'])`. 한국어 인식기(korean_g2)는 영문도 같이 든다.

    검출기는 끈다(`detector=False`). 줄은 우리 자르기가 이미 잘라 주었으므로
    `recognize` 에 그림 전체를 한 칸으로 넘긴다 — README 의 사용법 그대로다.
    """

    what = "easyocr korean_g2 (ko+en), recognize() 에 줄 그림 전체를 한 칸으로"

    def load(self, device: str) -> dict:
        import easyocr
        import torch

        self.torch = torch
        self.device = device
        start = time.perf_counter()
        self.reader = easyocr.Reader(["ko", "en"], gpu=device.startswith("cuda"),
                                     detector=False, verbose=False)
        took = time.perf_counter() - start
        net = self.reader.recognizer
        params = sum(p.numel() for p in net.parameters())
        store = Path(self.reader.model_storage_directory)
        disk = _size_mb([store / "korean_g2.pth"])
        return {"seconds": took, "params": params, "disk_mb": disk, "what": self.what,
                "version": easyocr.__version__}

    def prepare(self, paths):
        import numpy as np
        return [np.array(Image.open(p).convert("RGB")) for p in paths]

    def read(self, arrays) -> list[str]:
        out = []
        for arr in arrays:
            got = self.reader.recognize(arr, detail=0, paragraph=False)
            out.append(_nfc(" ".join(got)))
        return out

    def peak(self):
        if self.device.startswith("cuda"):
            return self.torch.cuda.max_memory_allocated() / 2**20
        return None

    def reset_peak(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()

    def held(self):
        if self.device.startswith("cuda"):
            return self.torch.cuda.memory_allocated() / 2**20
        return 0.0


class Paddle:
    """PaddleOCR 3.x 의 `TextRecognition(model_name="korean_PP-OCRv5_mobile_rec")`.

    줄 하나를 읽는 인식기만 쓴다(검출은 우리 자르기). 모델 카드의 사용법 그대로이고,
    전처리(높이 48 로 맞추기)는 그쪽이 제 것으로 한다.
    """

    model = "korean_PP-OCRv5_mobile_rec"
    what = "paddleocr TextRecognition(korean_PP-OCRv5_mobile_rec)"

    def load(self, device: str) -> dict:
        import paddle
        import paddleocr
        from paddleocr import TextRecognition

        self.device = device
        start = time.perf_counter()
        self.net = TextRecognition(model_name=self.model,
                                   device="gpu" if device.startswith("cuda") else "cpu")
        took = time.perf_counter() - start
        home = Path(os.environ.get("PADDLE_PDX_CACHE_HOME",
                                   Path.home() / ".paddlex")) / "official_models" / self.model
        files = [p for p in home.glob("*") if p.is_file()]
        weights = [p for p in files if p.suffix == ".pdiparams"]
        params = self._count(home) or None
        return {"seconds": took, "params": params, "disk_mb": _size_mb(files),
                "weights_mb": _size_mb(weights), "what": self.what,
                "version": "paddleocr %s / paddle %s" % (paddleocr.__version__,
                                                          paddle.__version__)}

    @staticmethod
    def _count(home: Path) -> int:
        """추론 프로그램을 열어 매개변수 모양을 센다. 못 세면 0 — 짐작한 값을 적지 않는다.

        `paddle.load` 로 가중치 파일을 열면 첫 텐서 하나(모양 [16])만 나온다. 파일
        크기를 4 로 나누는 셈은 맞아 보여도 짐작이다. 프로그램을 읽으면 234개를
        하나씩 셀 수 있다.
        """
        import numpy as np
        import paddle
        try:
            paddle.enable_static()
            exe = paddle.static.Executor(paddle.CPUPlace())
            prog, _, _ = paddle.static.load_inference_model(str(home / "inference"), exe)
            return int(sum(np.prod(v.shape) for v in prog.global_block().all_parameters()))
        except Exception as why:                         # noqa: BLE001
            print("낱값 수를 못 셌다: %s" % why, file=sys.stderr)
            return 0
        finally:
            paddle.disable_static()

    def prepare(self, paths):
        import numpy as np
        # paddlex 는 넘겨받은 배열을 OpenCV 처럼 BGR 로 본다.
        return [np.ascontiguousarray(np.array(Image.open(p).convert("RGB"))[:, :, ::-1])
                for p in paths]

    def read(self, arrays) -> list[str]:
        got = list(self.net.predict(input=arrays, batch_size=len(arrays)))
        return [_nfc(one["rec_text"]) for one in got]

    def peak(self):
        if self.device.startswith("cuda"):
            import paddle
            return paddle.device.cuda.max_memory_allocated() / 2**20
        return None

    def reset_peak(self):
        if self.device.startswith("cuda"):
            import paddle
            paddle.device.cuda.reset_max_memory_allocated()

    def held(self):
        if self.device.startswith("cuda"):
            import paddle
            return paddle.device.cuda.memory_allocated() / 2**20
        return 0.0


class Tesseract:
    """Tesseract 5 LSTM, `-l kor+eng --psm 7`(한 줄). conda-forge 판을 따로 깔았다.

    줄마다 프로그램을 새로 띄우면 띄우는 값만 줄당 0.1초가 넘는다. 그건 그 엔진의
    빠르기가 아니라 부르는 방식의 값이라, 그림 목록을 한 번에 넘긴다(`tesseract
    목록.txt stdout`). 쪽 사이는 폼 피드(\\f)로 갈린다.
    """

    langs = "kor+eng"

    def __init__(self, exe: str) -> None:
        self.exe = exe
        self.what = "tesseract %s, --oem 1 --psm 7 -l %s" % ("?", self.langs)

    def load(self, device: str) -> dict:
        start = time.perf_counter()
        version = subprocess.run([self.exe, "--version"], capture_output=True,
                                 text=True).stdout.splitlines()[0].strip()
        took = time.perf_counter() - start
        data = Path(os.environ.get("TESSDATA_PREFIX", Path(self.exe).parent.parent.parent
                                   / "share" / "tessdata"))
        self.data = data
        disk = _size_mb([data / ("%s.traineddata" % one) for one in self.langs.split("+")])
        self.what = "%s, --oem 1 --psm 7 -l %s" % (version, self.langs)
        return {"seconds": took, "params": None, "disk_mb": disk, "what": self.what,
                "version": version}

    def prepare(self, paths):
        return list(paths)

    def read(self, paths) -> list[str]:
        with tempfile.TemporaryDirectory() as tmp:
            book = Path(tmp) / "list.txt"
            book.write_text("\n".join(str(Path(p).resolve()) for p in paths) + "\n",
                            encoding="utf-8")
            env = dict(os.environ, TESSDATA_PREFIX=str(self.data), OMP_THREAD_LIMIT="1")
            run = subprocess.run([self.exe, str(book), "stdout", "--oem", "1",
                                  "--psm", "7", "-l", self.langs],
                                 capture_output=True, env=env)
        pages = run.stdout.decode("utf-8", "replace").split("\f")
        texts = [_nfc(" ".join(page.split())) for page in pages]
        texts = (texts + [""] * len(paths))[:len(paths)]
        return texts

    def peak(self):
        return None

    def reset_peak(self):
        pass

    def held(self):
        return 0.0


class Vlm:
    """그림과 글을 같이 보는 큰 모델(VLM). 모델 카드의 사용법 그대로 부른다.

    - `AutoModelForImageTextToText` + 대화 틀(`apply_chat_template`). 원격 코드는 안 쓴다.
    - 정밀도는 **그 모델이 나온 정밀도**다(`dtype="auto"`, transformers 5 의 기본값 —
      config 에 적힌 bfloat16). GPU 에서도 CPU 에서도 같다. 모델 카드대로 쓰는 사람이
      실제로 겪는 값이 그것이고, 2B 를 float32 로 올리면 8GB 카드에 안 들어간다.
      처음에는 CPU 만 float32 로 올렸는데, 그건 우리가 붙인 설정이라 뺐다.
    - 뽑기는 **욕심(greedy)** 이다. 생성 설정에 표본 뽑기가 켜져 있는 모델이 있는데,
      그러면 같은 그림을 두 번 읽혀도 답이 달라서 잰 값을 다시 못 만든다.
    - 길이 상한은 ko-trocr 와 같은 64 토큰이고, 닿은 줄 수를 센다.
    """

    tokens = 64

    def __init__(self, repo: str, prompt: str) -> None:
        self.repo, self.prompt = repo, prompt
        self.what = "%s, 물음 %r, 욕심 뽑기, 최대 %d토큰" % (repo, prompt, self.tokens)
        self.capped = 0

    def load(self, device: str) -> dict:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.torch, self.device = torch, device
        start = time.perf_counter()
        self.proc = AutoProcessor.from_pretrained(self.repo)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.repo, dtype="auto").to(device).eval()
        dtype = self.model.dtype
        # 멈추는 표. 대화 틀이 턴 끝에 `<|im_end|>` 를 쓰면 거기서도 멈춘다(2026-10-08).
        # Qwen3.5 는 generation_config 가 없어서 eos 가 `<|endoftext|>` 뿐이다 — 그대로 두면
        # 대답 뒤에 다음 턴을 지어내며 상한까지 채우고, 그 꼬리가 답에 붙어 **그쪽이 손해**다.
        # Qwen3-VL 은 이미 그 표에서 멈추고(그대로), PaddleOCR-VL 은 그 표를 안 쓴다(그대로).
        eos = self.model.generation_config.eos_token_id
        self.stop = list(eos) if isinstance(eos, list) else ([eos] if eos is not None else [])
        tok = self.proc.tokenizer
        frame = getattr(self.proc, "chat_template", None) or tok.chat_template or ""
        if "<|im_end|>" in frame and "<|im_end|>" in tok.get_vocab():
            end = tok.convert_tokens_to_ids("<|im_end|>")
            if end not in self.stop:
                self.stop.append(end)
        took = time.perf_counter() - start
        from huggingface_hub import snapshot_download
        # 아직 안 올린 판(내 폴더)도 같은 자로 잴 수 있게.
        spot = Path(self.repo) if Path(self.repo).is_dir() else Path(snapshot_download(self.repo))
        disk = sum(p.resolve().stat().st_size for p in spot.glob("*.safetensors")) / 2**20
        return {"seconds": took,
                "params": sum(p.numel() for p in self.model.parameters()),
                "disk_mb": disk, "what": self.what,
                "version": "transformers %s, %s" % (transformers.__version__,
                                                    str(dtype).replace("torch.", ""))}

    def prepare(self, paths):
        return [Image.open(p).convert("RGB") for p in paths]

    def read(self, images) -> list[str]:
        torch = self.torch
        talk = [[{"role": "user", "content": [{"type": "image", "image": im},
                                              {"type": "text", "text": self.prompt}]}]
                for im in images]
        batch = self.proc.apply_chat_template(
            talk, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**batch, max_new_tokens=self.tokens,
                                      do_sample=False, eos_token_id=self.stop or None)
        made = ids[:, batch["input_ids"].shape[1]:]
        eos = set(self.stop)
        for row in made.tolist():
            if len(row) >= self.tokens and not eos & set(row):
                self.capped += 1
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        texts = self.proc.batch_decode(made, skip_special_tokens=True)
        # 여러 줄로 내놓으면(한 줄 그림인데도) 한 줄로 잇는다.
        return [_nfc(" ".join(t.split())) for t in texts]

    def peak(self):
        if self.device.startswith("cuda"):
            return self.torch.cuda.max_memory_allocated() / 2**20
        return None

    def reset_peak(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()

    def held(self):
        if self.device.startswith("cuda"):
            return self.torch.cuda.memory_allocated() / 2**20
        return 0.0


# ── 주고받기 ──────────────────────────────────────────────────────────

def say(obj: dict) -> None:
    _CHAN.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _CHAN.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["easyocr", "paddle", "tesseract", "vlm"])
    ap.add_argument("--tesseract", default="")
    ap.add_argument("--repo", default="", help="vlm: Hugging Face 모델 이름")
    ap.add_argument("--prompt", default="", help="vlm: 그림과 같이 넘길 물음")
    args = ap.parse_args()

    def build():
        if args.engine == "easyocr":
            return EasyOCR()
        if args.engine == "paddle":
            return Paddle()
        if args.engine == "vlm":
            return Vlm(args.repo, args.prompt)
        return Tesseract(args.tesseract)

    engine = None
    for line in sys.stdin:
        if not line.strip():
            continue
        ask = json.loads(line)
        try:
            if ask["op"] == "load":
                engine = build()
                say(engine.load(ask.get("device", "cpu")))
            elif ask["op"] == "to":
                start = time.perf_counter()
                engine = build()
                engine.load(ask["device"])
                say({"seconds": time.perf_counter() - start})
            elif ask["op"] == "reset_peak":
                engine.reset_peak()
                say({})
            elif ask["op"] == "held":
                say({"mb": engine.held()})
            elif ask["op"] == "read":
                ready = engine.prepare(ask["paths"])
                start = time.perf_counter()
                texts = engine.read(ready)
                took = time.perf_counter() - start
                say({"texts": texts, "seconds": took, "peak_mb": engine.peak(),
                     "capped": getattr(engine, "capped", 0)})
            elif ask["op"] == "quit":
                say({})
                break
        except Exception as why:                         # noqa: BLE001
            import traceback
            traceback.print_exc()
            say({"error": "%s: %s" % (type(why).__name__, why)})


if __name__ == "__main__":
    main()
