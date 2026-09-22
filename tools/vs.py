"""공개된 다른 한국어 OCR 과 **같은 자로** 견준다.

    python tools/vs.py                          # 전부
    python tools/vs.py --fonts 120 --wide 30    # 시험지 줄 수를 줄여 빨리
    python tools/vs.py --skip-cpu               # CPU 재기를 건너뛴다(오래 걸린다)

견주는 쪽은 `ddobokki/ko-trocr` 다. 한국어 손글씨를 읽는 공개 모델이 사실상
그것뿐이라 README 가 처음부터 그것을 견줌 대상으로 적어 두었는데, 정작 **우리가
직접 재 본 적이 없었다.** 남의 카드에 적힌 숫자를 우리 숫자 옆에 놓는 것은
견준 것이 아니다. 자가 다르기 때문이다.

## 무엇을 같게 두었나

견줌이 성립하려면 **모델만 다르고 나머지는 같아야** 한다.

- 같은 그림. 사진은 우리 줄 자르기(`kohandocr.page`)로 자른 **같은 칸**을
  둘 다에게 준다. ko-trocr 에는 줄 자르기가 없으므로 우리 것을 빌려준 셈이다.
  글씨체 시험지는 `synth.render` 가 그린 **날것**을 준다 — 우리 쪽 전처리
  (`cell.fit`, 64x640 letterbox)를 걸어서 주면 그쪽만 두 번 찌그러진다.
- 같은 자. 자모 닮음·CER·줄 통째 일치를 **한 함수**로 재고, 둘 다 NFC 로
  고른 뒤 띄어쓰기를 지우고 견준다(`eval_photos.closeness_tight` 와 같은 뜻).
- 같은 정밀도. 둘 다 float32 다. ko-trocr 카드에는 float16 이 적혀 있는데,
  정확도를 그쪽에 맞춰 깎을 이유가 없다. 대신 **빠르기는 float16 도 같이 잰다**
  (그쪽이 유리한 자리를 감추면 안 된다).
- 같은 GPU, 같은 순간. 학습이 도는 중에 재면 다른 것을 재게 되므로
  `bench.busy()` 로 막는다.

## 무엇이 같지 않은가 — 이것도 같이 적는다

- **시험 글꼴을 우리만 뺐다.** 여섯 벌·스물네 벌은 *우리* 학습에서 뺀 것이고,
  ko-trocr 가 AI Hub 에서 무엇을 봤는지는 우리가 모른다. 그쪽에 유리하게
  겹쳐 있을 수도 있다. **그러니 이 숫자는 그쪽에 불리한 쪽으로 기울어 있지
  않다.** 사진 열한 장은 어느 쪽도 본 적이 없다.
- **만든 목적이 다르다.** ko-trocr 는 AI Hub 「다양한 형태의 한글 문자 OCR」과
  「공공행정문서 OCR」로 배웠다. 손글씨만 노린 모델이 아니다. 손글씨에서 지는
  것이 그 모델이 나쁘다는 뜻이 아니라 **다른 일을 하라고 만든 것**이라는 뜻이다.
- **길이 상한.** ko-trocr 의 `generation_config` 는 `max_length: 16` 이다.
  글자 단위 토크나이저라 열여섯 글자에서 잘린다. 그대로 재면 긴 줄을 통째로
  버리는 셈이라 `max_new_tokens=64` 로 풀어 주고, **몇 줄이 상한에 닿았는지**
  같이 센다.
- **우리는 판을 셋 쓴다.** 앙상블은 36M x 3 이고 흔들기까지 세면 판독을 아홉
  번 돈다. 그래서 「판 하나(36M)」도 같이 적는다. 크기로 견주려면 그쪽을 본다.
  **숫자를 여기 적어 두면 낡는다** — 이름표에 붙는 크기는 잰 값에서 뽑는다.

성적표는 `runs/VS.json` 에 남는다. README 의 그림과 표는 손으로 옮겨
적은 것이 아니라 **전부 그 파일에서 나온 것**이다.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT if (ROOT / "data" / "eval").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "tools"))

from transformers import logging as _hf_logging          # noqa: E402

_hf_logging.set_verbosity_error()

import bench as B                                        # noqa: E402
import eval_photos as E                                  # noqa: E402
import holdout as H                                      # noqa: E402
from kohandocr import page as photo                      # noqa: E402
from kohandocr import reader as R                        # noqa: E402
from kohandocr import synth                              # noqa: E402

BOOK = ROOT / "runs" / "ENSEMBLE.json"
CARD = ROOT / "runs" / "VS.json"

RIVAL = "ddobokki/ko-trocr"
# ko-trocr 가 실제로 쓰는 상한은 16 이다. 열여섯 글자에서 자른 값을 정확도로
# 적으면 그 모델을 재는 것이 아니라 그 설정을 재는 것이 된다. 풀어 주고, 대신
# 닿은 줄 수를 센다.
RIVAL_TOKENS = 64

# 묶음을 몇 줄씩 넣을지는 **재서 정한다.** 예전에는 양쪽 다 16줄로 박아
# 두었는데, 그러면 ko-trocr 쪽만 3.8배 비싸게 돌린다 — 그 모델은 인코더가
# 384x384(패치 576개)라 1줄 묶음에서 이미 카드를 다 쓰고, 16줄에서는 줄당
# 0.086초가 0.331초가 된다. 답은 묶음 크기와 무관하니 정확도는 안 바뀌지만,
# **480벌을 재는 데 몇 시간이 더 든다.** 그래서 빠르기를 먼저 재고 그 결과에서
# 가장 싼 묶음을 골라 판독에 쓴다.
CHUNK = 16                       # 아직 안 쟀을 때 기댈 자리


# ── 자 ────────────────────────────────────────────────────────────────

def _tight(text: str) -> str:
    """띄어쓰기를 지운 NFC 꼴. **두 모델의 답을 같은 꼴로 만들어 놓고 잰다.**

    ko-trocr 는 홑자모를 그대로 흘리는 일이 있다(실측: 'ᄉ' 가 답에 섞였다).
    NFC 로 고르지 않으면 자모 하나가 음절 하나처럼 세어져 CER 이 부풀거나
    깎인다. 어느 쪽으로 틀리는지 예측이 안 되므로 반드시 고른다.
    """
    return "".join(unicodedata.normalize("NFC", text).split())


def _edits(got: str, want: str) -> int:
    """레벤슈타인 거리. 줄이 짧으니 표를 그대로 채운다."""
    if not want:
        return len(got)
    row = list(range(len(want) + 1))
    for i, a in enumerate(got, 1):
        new = [i]
        for j, b in enumerate(want, 1):
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1]


def marks(got: str, want: str) -> dict:
    """한 줄의 성적. 세 가지를 같이 낸다 — 고치는 자리가 다르기 때문이다.

    `jamo`  자모 닮음. 이 저장소가 내내 써 온 자라 예전 숫자와 이어진다.
            '팀'을 '탐'으로 읽으면 자모 셋 중 하나만 틀려 0.67 이다.
    `cer`   글자 오류율. 남들이 OCR 을 잴 때 쓰는 자다. 같은 '팀->탐'이
            1.0(글자 하나 통째로 틀림)이다. **두 자는 다른 말을 한다.**
            밖에 내보낼 때 어느 자로 잰 값인지 안 적으면 견줄 수 없다.
    `exact` 줄을 통째로 맞혔나. 서식에서 값을 뽑아 쓰려면 결국 이것이다.
    """
    g, w = _tight(got), _tight(want)
    return {"jamo": E.closeness_tight(got, want),
            "edits": _edits(g, w), "chars": len(w),
            "exact": 1 if g == w and w else 0}


def roll(rows: list[dict]) -> dict:
    """줄 성적을 묶는다. **CER 은 줄마다 평균 내지 않는다** — 그러면 한 글자
    짜리 줄과 서른 글자 줄이 같은 무게가 된다. 글자를 다 모아 놓고 나눈다."""
    chars = sum(r["chars"] for r in rows) or 1
    return {"lines": len(rows),
            "chars": sum(r["chars"] for r in rows),
            "jamo": 100 * sum(r["jamo"] for r in rows) / max(len(rows), 1),
            "cer": 100 * sum(r["edits"] for r in rows) / chars,
            "exact": 100 * sum(r["exact"] for r in rows) / max(len(rows), 1)}


# ── 줄을 갈래로 나눈다 ────────────────────────────────────────────────
#
# 합계만 보면 **어디서 지고 어디서 이기는지** 안 보인다. 실제로 ko-trocr 는
# 짧은 한글 낱말은 잘 읽는데 라틴 약어에서 무너진다('Codex 서약' -> 'Codox시왁').
# 그 둘을 한 숫자로 뭉개면 고칠 자리도, 쓸 자리도 알 수 없다.

def kind(text: str) -> str:
    """줄 하나가 어느 갈래인가. **겹치지 않게** 앞에서부터 하나만 고른다."""
    t = unicodedata.normalize("NFC", text)
    if any("a" <= c.lower() <= "z" for c in t):
        return "라틴 섞임"
    if any(c.isdigit() for c in t):
        return "숫자 섞임"
    if ":" in t:
        return "서식 라벨"
    if len(_tight(t)) <= 6:
        return "한글 낱말·이름"
    return "한글 문장"


def span(text: str) -> str:
    """길이 갈래. ko-trocr 는 정사각형(384x384)으로 눌러 넣으므로 **긴 줄일수록
    불리하고**, 우리는 64x640 이라 그 반대다. 길이로 갈라야 그것이 보인다."""
    n = len(_tight(text))
    if n <= 5:
        return "1~5자"
    if n <= 10:
        return "6~10자"
    if n <= 20:
        return "11~20자"
    return "21자 이상"


def cut_by(rows: list[tuple[str, dict]], fn) -> dict:
    out: dict[str, list[dict]] = {}
    for text, mark in rows:
        out.setdefault(fn(text), []).append(mark)
    return {key: roll(vals) for key, vals in sorted(out.items())}


# ── 견주는 두 쪽 ──────────────────────────────────────────────────────

class Ours:
    """우리 판. `tools/bench.py` 와 **같은 길**로 올리고 같은 길로 읽는다.

    앱이 지나는 길(`Reader.read_trust`)이어야 한다. 벤치 전용 지름길로 재면
    그 숫자는 앱의 숫자가 아니다.
    """

    def __init__(self, name: str, members: list[str]) -> None:
        self.name, self.members = name, members
        self.reader = None

    def load(self, device: str) -> float:
        start = time.perf_counter()
        self.reader = B.build(self.members, device)
        return time.perf_counter() - start

    def to(self, device: str) -> None:
        self.reader.model.to(device)
        for _, extra in self.reader.also:
            extra.to(device)
        self.reader.device = device

    def read(self, images: list[Image.Image]) -> list[str]:
        got = self.reader.read_trust(images, beams=R.BEAMS)
        return [text for text, _ in got]

    def trust(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        return self.reader.read_trust(images, beams=R.BEAMS)

    def params(self) -> int:
        n = sum(p.numel() for p in self.reader.model.parameters())
        return n + sum(sum(p.numel() for p in m.parameters())
                       for _, m in self.reader.also)

    def disk(self) -> float:
        """받아서 깔아야 하는 크기(MB). 파라미터 수와 따로 적는다 — 앙상블은
        파라미터가 셋이면 디스크도 셋이지만, 같은 크기여도 저장 정밀도가
        다르면 어긋난다."""
        return sum(one.stat().st_size for m in self.members
                   for one in Path(m).glob("*") if one.is_file()) / 2**20

    def free(self) -> None:
        self.reader = None
        torch.cuda.empty_cache()


class Rival:
    """`ddobokki/ko-trocr`. 그쪽 카드가 시키는 대로 `TrOCRProcessor` 로 먹인다.

    우리 전처리를 걸지 않는다. 남의 모델에 우리 눈금을 씌워 놓고 못 읽는다고
    하면 그것은 견준 것이 아니다.
    """

    def __init__(self, repo: str = RIVAL, half: bool = False,
                 beams: int = R.BEAMS) -> None:
        self.name = repo + (" (fp16)" if half else "")
        self.repo, self.half, self.beams = repo, half, beams
        self.capped = 0            # 길이 상한에 닿은 줄 수
        self.model = self.proc = None

    def load(self, device: str) -> float:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        start = time.perf_counter()
        self.proc = TrOCRProcessor.from_pretrained(self.repo)
        kind = {"dtype": torch.float16} if self.half else {"dtype": torch.float32}
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.repo, **kind).to(device).eval()
        self.device = device
        return time.perf_counter() - start

    def to(self, device: str) -> None:
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def read(self, images: list[Image.Image]) -> list[str]:
        if not images:
            return []
        rgb = [one.convert("RGB") for one in images]
        px = self.proc(images=rgb, return_tensors="pt").pixel_values
        px = px.to(self.device, dtype=self.model.dtype)
        ids = self.model.generate(px, num_beams=self.beams,
                                  max_new_tokens=RIVAL_TOKENS,
                                  early_stopping=self.beams > 1)
        # 상한에 닿았나. **줄마다** 센다 — 묶음 하나가 닿았다고 그 묶음 전부를
        # 세면 숫자가 묶음 크기만큼 부풀고, 그러면 '잘려서 진 것'과 '못 읽어서
        # 진 것'을 가를 수가 없다. 끝 토큰(eos)이 안 나온 줄이 잘린 줄이다.
        eos = self.model.generation_config.eos_token_id
        for row in ids.tolist():
            made = row[1:]                       # 시작 토큰은 뺀다
            if len(made) >= RIVAL_TOKENS and eos not in made:
                self.capped += 1
        return [unicodedata.normalize("NFC", t).strip()
                for t in self.proc.batch_decode(ids, skip_special_tokens=True)]

    def params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def disk(self) -> float:
        """HF 캐시에 받아 둔 가중치 크기(MB). 심볼릭 링크 너머의 실제 파일을 잰다."""
        spot = Path(self.model.config.name_or_path or "")
        if not spot.is_dir():
            from huggingface_hub import snapshot_download
            spot = Path(snapshot_download(self.repo))
        return sum(one.resolve().stat().st_size
                   for one in spot.glob("*.safetensors")) / 2**20

    def free(self) -> None:
        self.model = self.proc = None
        torch.cuda.empty_cache()


# ── 시험지 ────────────────────────────────────────────────────────────

def sheet(count: int, fonts_dir: str, only: str, part: str) -> list[tuple[str, Image.Image]]:
    """`tools/holdout.sheet` 과 **같은 씨앗·같은 그림**인데 `fit` 을 안 건다.

    holdout 은 64x640 으로 맞춰 둔 것을 돌려준다. 그 letterbox 를 ko-trocr 에
    주면 384x384 로 또 눌리면서 글씨가 왼쪽 위 구석에 몰린다 — 그 모델을 재는
    것이 아니라 우리 전처리를 재는 꼴이다. 날것을 주면 우리 `Reader` 는 어차피
    안에서 `fit` 하므로 holdout 과 똑같은 것을 보고, 그쪽은 제 눈금으로 본다.
    """
    import random

    fonts = synth.Fonts(fonts_dir, part=part, only=only)
    rng = random.Random(H.SEED)
    out = []
    from kohandocr import corpus
    for text in corpus.lines(count, seed=H.SEED):
        if H._has_bare_jamo(text):
            continue
        page = synth.render(text, fonts, rng)
        if page is not None:
            out.append((text, page))
    return out


def every_font(fonts_dir: str) -> list[tuple[str, str, str]]:
    """글꼴 폴더에 있는 **전부**. (파일이름, 보기 좋은 이름, 어느 무리) 로 준다.

    무리가 셋이다. 이 구분을 안 적고 합계만 내면 **거짓말이 된다.**

        학습     450벌. *우리* 학습에 쓴 것이다. 여기 점수가 높은 것은
                 '일반화'가 아니라 '배운 것을 외웠나'이고, ko-trocr 에게는
                 그냥 처음 보는 글씨체다. **같은 자리가 아니다.**
        고르기    6벌. 학습에서 뺐지만 조합을 **고르는 데** 쓴다. 그래서
                 이쪽도 완전히 깨끗하지는 않다.
        두루      24벌. 학습에서도 빼고 고르기에도 안 쓴다. 우리 쪽에서
                 가장 깨끗한 자리다.
    """
    kit = synth.Fonts(fonts_dir, part="all")
    out = []
    for path in kit.files:
        if synth._for_test(path.name):
            herd = "고르기 6벌"
        elif synth._for_wide(path.name):
            herd = "두루 24벌"
        else:
            herd = "학습에 쓴 벌"
        out.append((path.name, H._short(path.name), herd))
    return out


def cells_of(jpg: Path) -> list[Image.Image]:
    png = photo.prepare(jpg.read_bytes())
    return [Image.open(io.BytesIO(c["png"])).convert("L")
            for c in photo.lines(png, max_lines=16)]


def read_all(side, rows: list[tuple[str, Image.Image]]) -> list[tuple[str, dict]]:
    step = getattr(side, "chunk", CHUNK)
    out = []
    for i in range(0, len(rows), step):
        chunk = rows[i:i + step]
        got = side.read([img for _, img in chunk])
        out += [(text, marks(g, text)) for (text, _), g in zip(chunk, got)]
    return out


# ── 재는 자리들 ───────────────────────────────────────────────────────

def on_photos(side, jpgs: list[Path], truth: dict, cache: dict) -> dict:
    """사진. 칸은 **미리 한 번만** 잘라 두고 둘 다에게 같은 것을 준다."""
    per_photo, rows, gauge = {}, [], {}
    for jpg in jpgs:
        cells = cache[jpg.name]
        want = truth[jpg.name]["lines"]
        got = side.read(cells)
        pairs = [marks(g, w) for g, w in E.pair(got, want)]
        mean = 100 * sum(p["jamo"] for p in pairs) / max(len(pairs), 1)
        if truth[jpg.name].get("gauge"):
            gauge[jpg.name] = mean
            continue
        per_photo[jpg.name] = mean
        rows += [(w, m) for (_, w), m in zip(E.pair(got, want), pairs)]
    card = roll([m for _, m in rows])
    card.update(per_photo=per_photo, gauge=gauge,
                low_photo=min(per_photo.values()),
                mean_photo=sum(per_photo.values()) / len(per_photo),
                over_goal=sum(1 for v in per_photo.values() if v >= E.GOAL),
                of_photos=len(per_photo))
    card["by_kind"] = cut_by(rows, kind)
    card["by_span"] = cut_by(rows, span)
    return card


def on_trust(side, jpgs: list[Path], truth: dict, cache: dict) -> dict:
    """**못 읽었으면 못 읽었다고 하나.** 점수와 따로 세는 자리다.

    서식에서 틀린 답은 빈칸보다 나쁘다 — 사람이 검토할 자리를 못 찾기 때문이다
    (`reader.TRUST_EDGE`). 그래서 우리는 믿음값이 문턱 아래면 답을 안 낸다.
    세는 것은 점수가 아니라 셋이다.

        막음  잘못 읽은 줄(닮음 0.9 미만)을 물러서서 안 내보냈다   <- 벌어들인 것
        놓침  잘 읽은 줄을 괜히 물러서서 지웠다                  <- 잃은 것
        샘    잘못 읽은 줄이 그냥 나갔다                        <- 남은 위험

    ko-trocr 에는 이에 해당하는 것이 **없다.** 빔 점수를 문턱으로 쓸 수야 있지만
    그건 우리가 그 모델에 붙여 준 것이지 그 모델이 내놓는 것이 아니다. 그래서
    여기는 견줌 표가 아니라 **한쪽에만 있는 것**으로 적는다.
    """
    blocked = missed = leaked = kept = 0
    for jpg in jpgs:
        if truth[jpg.name].get("gauge"):
            continue
        want = truth[jpg.name]["lines"]
        pairs = side.trust(cache[jpg.name])
        said = [text for text, _ in pairs]
        holds = [0.0 <= t < R.TRUST_EDGE for _, t in pairs]
        for ti, gi in _paired(said, want).items():
            guess = said[gi] if gi is not None else ""
            held = bool(gi is not None and holds[gi])
            good = E.closeness_tight(guess, want[ti]) >= 0.9
            if held and good:
                missed += 1
            elif held:
                blocked += 1
            elif good:
                kept += 1
            else:
                leaked += 1
    return {"blocked": blocked, "missed": missed, "leaked": leaked,
            "kept": kept, "edge": R.TRUST_EDGE}


def _paired(got: list[str], want: list[str]) -> dict[int, int | None]:
    """`eval_photos.pair` 와 **글자 하나까지 같은 짝**인데 자리를 돌려준다.

    물러섰는지는 읽은 값이 아니라 **몇 번째 칸이었나**에 붙어 있다. 그런데
    `pair` 는 글월만 돌려주므로 되짚을 수가 없다. 표시를 글월에 붙여서 넘기는
    수를 썼다가 되물렀다 — 그 표시가 닮음 셈에 끼어들어 **짝이 달라진다.**
    재는 자가 재는 대상을 바꾸면 안 된다. 그래서 같은 셈을 여기서 한 번 더
    하고 자리를 들고 나온다.
    """
    scores = sorted(((E.closeness(guess, truth), gi, ti)
                     for ti, truth in enumerate(want)
                     for gi, guess in enumerate(got)), reverse=True)
    taken: set[int] = set()
    chosen: dict[int, int | None] = {}
    for _, gi, ti in scores:
        if ti in chosen or gi in taken:
            continue
        chosen[ti] = gi
        taken.add(gi)
    return {ti: chosen.get(ti) for ti in range(len(want))}


def on_sheets(side, exams: dict[str, list], tag: str) -> dict:
    per_style, rows = {}, []
    for who, made in exams.items():
        got = read_all(side, made)
        rows += got
        per_style[who] = roll([m for _, m in got])
        print("    %-34s 자모 %5.2f%%  CER %5.2f%%  통째 %5.1f%%"
              % (who, per_style[who]["jamo"], per_style[who]["cer"],
                 per_style[who]["exact"]))
    card = roll([m for _, m in rows])
    card.update(styles={k: v["jamo"] for k, v in per_style.items()},
                per_style=per_style,
                low=min(v["jamo"] for v in per_style.values()),
                of_styles=len(per_style))
    card["by_kind"] = cut_by(rows, kind)
    card["by_span"] = cut_by(rows, span)
    print("    %-34s 자모 %5.2f%%  CER %5.2f%%  통째 %5.1f%%  (%s 합계, 최저 벌 %.2f%%)"
          % ("", card["jamo"], card["cer"], card["exact"], tag, card["low"]))
    return card


def on_every(side, sheets: dict[str, tuple[tuple[str, str], list]], tag: str) -> dict:
    """글꼴 **전부**를 한 벌씩 잰다. 무리(학습/고르기/두루)별로 따로 셈한다.

    벌마다 줄 수가 적어(기본 12줄) **한 벌의 값은 흔들린다.** 그래서 벌 단위
    숫자는 성적표에만 남기고, 밖에 내보내는 것은 무리별 합계와 **분포**다 —
    몇 벌이 어느 구간에 있나. 480벌을 한 줄씩 늘어놓아 봐야 아무도 안 읽는다.
    """
    # **줄을 한데 모아 읽는다.** 벌마다 따로 부르면 묶음이 그 벌의 줄 수에
    # 갇힌다 — 벌당 한 줄이면 480번을 한 줄씩 읽게 되고, 앙상블은 그 자리에서
    # 줄당 0.10초가 0.25초가 된다(2.5배). 답은 묶음과 무관하니 점수는 같다.
    order = [(name, text, img)
             for name, (_, made) in sheets.items() for text, img in made]
    print("    %d벌 %d줄을 %d줄씩 묶어 읽는다" % (len(sheets), len(order),
                                                getattr(side, "chunk", CHUNK)))
    start = time.perf_counter()
    scored = read_all(side, [(text, img) for _, text, img in order])

    per_font, herds = {}, {}
    rows_of: dict[str, list[dict]] = {}
    for (name, _, _), (_, mark) in zip(order, scored):
        rows_of.setdefault(name, []).append(mark)
    for name, (who, _) in sheets.items():
        card = roll(rows_of[name])
        per_font[name] = {"who": who[0], "herd": who[1], **card}
        herds.setdefault(who[1], []).extend(rows_of[name])
    print("    %.0fs 걸렸다" % (time.perf_counter() - start))
    out = {"per_font": per_font,
           "herds": {h: roll(rows) for h, rows in sorted(herds.items())}}
    every = list(per_font.values())
    out["of_fonts"] = len(every)
    out["lines"] = sum(v["lines"] for v in every)
    out["chars"] = sum(v["chars"] for v in every)
    # 합계는 줄을 다 모아 놓고 낸다(벌마다 평균 내면 줄 수가 다른 벌이 뒤섞인다).
    everything = [m for rows in herds.values() for m in rows]
    out.update({k: v for k, v in roll(everything).items()
                if k in ("jamo", "cer", "exact")})
    out["low"] = min(v["jamo"] for v in every)
    out["low_font"] = min(per_font, key=lambda k: per_font[k]["jamo"])
    # 분포. 이 그림 하나가 480줄짜리 표보다 많이 말한다.
    edges = (0, 50, 60, 70, 80, 85, 90, 95, 101)
    bins = {}
    for herd in herds:
        counts = [0] * (len(edges) - 1)
        for v in every:
            if v["herd"] != herd:
                continue
            for i in range(len(edges) - 1):
                if edges[i] <= v["jamo"] < edges[i + 1]:
                    counts[i] += 1
                    break
        bins[herd] = counts
    out["bin_edges"] = list(edges)
    out["bins"] = bins
    for herd, card in out["herds"].items():
        n = sum(1 for v in every if v["herd"] == herd)
        print("    %-14s %3d벌  자모 %5.2f%%  CER %5.2f%%  통째 %5.1f%%  "
              "95넘긴벌 %d"
              % (herd, n, card["jamo"], card["cer"], card["exact"],
                 sum(1 for v in every if v["herd"] == herd and v["jamo"] >= 95)))
    print("    %-14s %3d벌  자모 %5.2f%%  CER %5.2f%%  통째 %5.1f%%  최저 %s %.2f%%"
          % ("전부(" + tag + ")", out["of_fonts"], out["jamo"], out["cer"],
             out["exact"], per_font[out["low_font"]]["who"], out["low"]))
    return out


def speed(side, cells: list[Image.Image], device: str, repeat: int,
          sizes=(1, 4, 8, 16, 32)) -> dict:
    """묶음별 줄당 초. **몸풀기를 한 번 하고 잰다** — 첫 판독은 CUDA 커널을
    깎느라 몇 배 느리다. 그것까지 재면 묶음이 작을수록 손해가 커 보인다.

    **넘친 자리를 골라내야 한다.** 이 카드가 8GB 인데 ko-trocr 는 인코더가
    384x384(패치 576개)라 빔 5로 32줄을 넣으면 카드 밖으로 밀려난다. 그때
    torch 는 터지지 않고 **공유 메모리로 새어 나가며** 32줄에 131초를 쓴다
    (꼭대기 8394MB — 카드가 8151MB 다). 그 값을 그대로 적으면 「ko-trocr 는
    줄당 4.1초」가 되는데, 그건 모델의 빠르기가 아니라 **이 노트북의 VRAM 을
    잰 것**이다.

    **샌 것과 그냥 느린 것을 섞으면 안 된다.** 처음에는 '앞 묶음보다 줄당이
    나빠지면 샌 것'으로 세었는데, 그러면 ko-trocr 의 4줄 묶음(꼭대기 1.7GB,
    카드의 22%)까지 「카드 밖으로 샜다」고 적힌다. 그건 거짓이다 — 그 묶음은
    메모리가 넉넉한 채로 그냥 느렸고, 그게 그 모델의 성질이다. 샌 것은 **꼭대기
    VRAM 으로만** 가른다. 느린 묶음은 `fastest` 가 어차피 안 고른다.

    묶음마다 **칸 전부를 한 바퀴** 돈다. 묶음을 한 칸으로 채워 재면 그 칸의
    길이를 재는 것이지 줄당 초를 재는 것이 아니다.
    """
    cuda = device.startswith("cuda")
    total = (torch.cuda.get_device_properties(0).total_memory / 2**20) if cuda else 0.0
    out, best, last_peak = {}, None, 0.0
    for size in sizes:
        # **묶음을 같은 칸으로 채우면 안 된다.** 칸마다 글자 수가 달라서 읽는
        # 시간이 다르고, 자모를 한 개씩 뽑는 판이라 그 차이가 그대로 시간이다.
        # 예전에는 `cells[i % len(cells)]` 로 채워서 **묶음이 1줄이면 첫 칸만
        # 되풀이해** 읽었다. 그 칸은 가로/세로가 1.5(두어 글자)인데 칸의
        # 가운데값은 4.0 이다. 짧은 줄만 재 놓고 '줄당 초'라고 적은 셈이고,
        # `bench.py` 의 실측과 1.8배 어긋났다.
        #
        # 게다가 **양쪽에 다르게 걸린다** — 가장 싼 자리가 한쪽은 16줄이고
        # 한쪽은 1줄이면, 골고루 읽은 값과 짧은 줄만 읽은 값을 나란히 놓게
        # 된다. 그래서 어떤 묶음이든 **칸 전부를 한 바퀴** 돌고 칸 수로 나눈다.
        # 끝이 모자라면 앞으로 돌아가 채워 묶음 크기는 늘 같게 둔다.
        rounds = [[cells[(at + j) % len(cells)] for j in range(size)]
                  for at in range(0, len(cells), size)]
        seen = size * len(rounds)
        # **카드 밖인 것이 이미 드러난 묶음은 안 잰다.**
        #
        # 묶음을 두 배로 키우면 활성값도 대강 두 배다. 앞 묶음의 꼭대기가
        # 4555MB 였다면 다음은 9GB 라 8151MB 짜리 카드에 안 들어간다. 그걸
        # 굳이 돌리면 torch 가 터지는 대신 시스템 메모리로 새면서 한 번에
        # 2~10분을 쓰고, 그러고 나오는 말은 '안 들어간다' 하나뿐이다.
        #
        # 여기서 적는 것은 **짐작한 속도가 아니라 안 들어간다는 사실**이다.
        # 속도 칸은 비워 둔다 — 안 잰 값을 잰 것처럼 적으면 안 된다.
        if cuda and last_peak and last_peak * 2 > 0.90 * total:
            out[str(size)] = {"seconds": None, "per_line": None,
                              "peak_mb": None, "spilled": True, "slower": False,
                              "why": "앞 묶음 %.0fMB 의 두 배가 카드 %.0fMB 를 넘는다"
                                     % (last_peak, total)}
            print("      %2d줄 묶음  이 카드에 안 들어간다 (앞 묶음이 이미 "
                  "%.0fMB / %.0fMB). 안 잰다." % (size, last_peak, total))
            break
        try:
            side.read(rounds[0][:1])
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            spent = []
            for _ in range(repeat):
                if cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                for one in rounds:
                    side.read(one)
                if cuda:
                    torch.cuda.synchronize()
                spent.append(time.perf_counter() - start)
            took = statistics.median(spent)
            peak = (torch.cuda.max_memory_allocated() / 2**20) if cuda else None
            last_peak = peak or 0.0
            per = took / seen
            spilled = bool(cuda and peak and peak > 0.90 * total)
            slower = bool(best is not None and per > best * 1.05)
            out[str(size)] = {"seconds": took, "per_line": per, "peak_mb": peak,
                              "spilled": spilled, "slower": slower,
                              "repeat": repeat}
            if not spilled:
                best = per if best is None else min(best, per)
            print("      %2d줄 묶음  %7.3fs/%d줄  줄당 %.4fs  (%5.1f줄/초)%s%s"
                  % (size, took, seen, per, 1 / per,
                     "  VRAM %4.0fMB" % peak if cuda else "",
                     "   <- 카드 밖으로 샜다. 안 쓴다" if spilled
                     else ("   (더 작은 묶음이 더 싸다)" if slower else "")))
            if spilled:
                # 한 번 샜으면 더 큰 묶음은 더 샌다. 재 봐야 같은 말을 몇 분
                # 더 걸려 되풀이할 뿐이고, 그동안 카드를 붙잡고 있는다.
                print("      더 큰 묶음은 안 잰다 — 여기가 이 카드의 끝이다")
                break
        except (RuntimeError, torch.OutOfMemoryError) as why:
            if "out of memory" not in str(why).lower():
                raise
            torch.cuda.empty_cache()
            out[str(size)] = {"seconds": None, "per_line": None, "spilled": True}
            print("      %2d줄 묶음  메모리가 모자라다" % size)
            break
    return out


def fastest(rows: dict) -> tuple[str, dict] | tuple[None, None]:
    """쪽 환산에 쓸 묶음. **넘친 자리는 뺀다**(`speed` 머리말)."""
    ok = [(k, v) for k, v in rows.items()
          if v.get("per_line") and not v.get("spilled")]
    return min(ok, key=lambda kv: kv[1]["per_line"]) if ok else (None, None)


def main() -> None:
    ap = argparse.ArgumentParser(description="공개 모델과 같은 자로 견준다")
    ap.add_argument("--fonts", type=int, default=120, help="시험 글씨체 한 벌당 줄")
    ap.add_argument("--wide", type=int, default=30, help="두루 재는 벌당 줄")
    ap.add_argument("--every", type=int, default=12,
                    help="글꼴 **전부**를 잴 때 한 벌당 줄. 0 이면 건너뛴다")
    ap.add_argument("--repeat", type=int, default=3, help="시간을 몇 번 재나")
    ap.add_argument("--page-lines", type=int, default=30, help="한 쪽을 몇 줄로 셈하나")
    ap.add_argument("--skip-cpu", action="store_true", help="CPU 재기를 건너뛴다")
    ap.add_argument("--skip-wide", action="store_true")
    ap.add_argument("--font-dir", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--anyway", action="store_true")
    # 「판 하나」로 잴 판. 안 주면 조합의 첫 식구다. **릴리스로 내보내는 판을
    # 준다** — 0.3.0 은 이것이 없어서 견줌표의 판 하나는 v55-2500 을 쟀는데
    # 내보낸 판 하나는 v62 였다. 표의 숫자가 받는 판의 숫자가 아니었다.
    ap.add_argument("--one", default="", help="판 하나로 잴 판 폴더 (예: runs/v81)")
    args = ap.parse_args()

    others = B.busy()
    if others is None and not args.anyway:
        raise SystemExit("학습이 도는지 물어보지 못했다. 다시 부르거나 --anyway.")
    if others and not args.anyway:
        raise SystemExit("학습이 돌고 있다 (pid %s). 세우고 재라." % ", ".join(others))

    font_dir = str(synth.font_folder(args.font_dir))
    truth = json.loads((APP / "data/eval/labels.json").read_text(encoding="utf-8"))
    jpgs = sorted((APP / "data/eval/samples").glob("hand-*.jpg"))
    team = [m.replace("\\", "/")
            for m in json.loads(BOOK.read_text(encoding="utf-8"))["members"]]

    gear = (torch.cuda.get_device_name(0) if args.device.startswith("cuda")
            else "CPU")
    print("자: %s / torch %s / 사진 %d장" % (gear, torch.__version__, len(jpgs)))

    # 칸과 시험지는 **한 번만** 만들어 둘 다에게 같은 것을 준다.
    print("\n칸 자르기(양쪽 공통) ...")
    start = time.perf_counter()
    cells = {jpg.name: cells_of(jpg) for jpg in jpgs}
    cut_all = time.perf_counter() - start
    graded = [j for j in jpgs if not truth[j.name].get("gauge")]
    cut_per_page = cut_all / len(jpgs)
    print("  %d장 %d칸, 장당 %.3fs (CPU. ko-trocr 에는 줄 자르기가 없어 빌려준다)"
          % (len(jpgs), sum(len(v) for v in cells.values()), cut_per_page))

    print("시험지 그리기 ...")
    exams = {H._short(one): sheet(args.fonts, font_dir, one.split()[-1], "test")
             for one in synth.FONT_TEST}
    wide = ({H._short(one): sheet(args.wide, font_dir, one.split()[-1], "wide")
             for one in synth.FONT_WIDE} if not args.skip_wide else {})
    print("  시험 %d벌 x %d줄, 두루 %d벌 x %d줄"
          % (len(exams), min(len(v) for v in exams.values()), len(wide),
             min((len(v) for v in wide.values()), default=0)))

    # 글꼴 **전부**. 무리를 갈라 두고 한 벌씩 시험지를 만든다. 파일 이름을
    # `only` 로 주면 그 한 벌만 남는다 — 이름이 파일 이름이라 반드시 유일하다.
    every: dict[str, tuple[tuple[str, str], list]] = {}
    if args.every:
        start = time.perf_counter()
        for name, who, herd in every_font(font_dir):
            made = sheet(args.every, font_dir, name, "all")
            if made:
                every[name] = ((who, herd), made)
        print("  글꼴 전부 %d벌 x %d줄 (%.0fs 걸려 그렸다)"
              % (len(every), args.every, time.perf_counter() - start))

    flat = [one for jpg in graded for one in cells[jpg.name]]

    # **이름에 크기를 박지 않는다.** 예전 이름표에 `31M` 이 박혀 있었는데
    # 디코더가 6층이 되면서 36M 이 됐고, 박아 둔 숫자는 안 따라 올라갔다.
    # 크기는 `params` 로 재서 적고, 그리는 쪽이 거기서 이름표를 만든다.
    sides = [
        ("ko-hand-ocr 앙상블", lambda: Ours("ours-team", team)),
        ("ko-hand-ocr 판 하나",
         lambda: Ours("ours-one", [args.one.replace("\\", "/")] if args.one else team[:1])),
        ("ddobokki/ko-trocr", lambda: Rival()),
    ]

    cards = []
    for label, make in sides:
        print("\n" + "=" * 74)
        print(label)
        print("=" * 74)
        side = make()
        load_s = side.load(args.device)
        card = {"name": label, "load_seconds": round(load_s, 2),
                "params": side.params(), "disk_mb": side.disk()}
        print("  판 올리기 %.1fs, 파라미터 %.1fM, 디스크 %.0fMB"
              % (load_s, card["params"] / 1e6, card["disk_mb"]))

        # **빠르기를 먼저 잰다.** 두 가지를 얻는다 — 성적표에 적을 줄/초와,
        # 이 모델이 가장 싸게 도는 묶음 크기. 뒤의 판독은 그 묶음으로 돈다.
        # 묶음은 답을 안 바꾸고 시간만 바꾸므로 정확도는 그대로다.
        print("  빠르기 (GPU)")
        card["gpu"] = speed(side, flat, args.device, args.repeat)
        where, _ = fastest(card["gpu"])
        side.chunk = int(where) if where else CHUNK
        card["read_chunk"] = side.chunk
        print("      -> 판독은 %d줄씩 묶어 돈다 (가장 싼 자리)" % side.chunk)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        if isinstance(side, Rival) and args.device.startswith("cuda"):
            # ko-trocr 카드에 적힌 정밀도는 float16 이다. 정확도는 float32 로
            # 재 두고(깎을 이유가 없다) **빠르기만** 그쪽 설정으로 한 번 더
            # 잰다. 그쪽이 유리한 자리를 안 재고 넘어가면 견줌이 아니다.
            print("  빠르기 (GPU, float16 — ko-trocr 카드에 적힌 정밀도)")
            side.model.half()
            card["gpu_fp16"] = speed(side, flat, args.device, args.repeat)
            side.model.float()
            torch.cuda.empty_cache()

        print("  사진 11장")
        card["photos"] = on_photos(side, jpgs, truth, cells)
        p = card["photos"]
        print("    자모 %5.2f%%  CER %5.2f%%  통째 %5.1f%%  최저 장 %5.2f%%  "
              "목표 넘긴 장 %d/%d"
              % (p["jamo"], p["cer"], p["exact"], p["low_photo"],
                 p["over_goal"], p["of_photos"]))

        print("  안 배운 글씨체 6벌")
        card["test_fonts"] = on_sheets(side, exams, "6벌")
        if wide:
            print("  두루 읽나 — 24벌")
            card["wide_fonts"] = on_sheets(side, wide, "24벌")

        if every:
            print("  글꼴 전부 — %d벌" % len(every))
            card["every_font"] = on_every(side, every, "%d벌" % len(every))

        if isinstance(side, Rival):
            card["capped_lines"] = side.capped
        else:
            print("  못 읽었으면 물러서나 (믿음값 %.2f 아래)" % R.TRUST_EDGE)
            card["trust"] = on_trust(side, jpgs, truth, cells)
            t = card["trust"]
            print("    막음 %d줄 / 놓침 %d줄 / 샘 %d줄 / 그냥 맞음 %d줄"
                  % (t["blocked"], t["missed"], t["leaked"], t["kept"]))

        if not args.skip_cpu:
            print("  빠르기 (CPU — 이 모델이 실제로 도는 자리)")
            side.to("cpu")
            card["cpu"] = speed(side, flat, "cpu", max(1, args.repeat - 1),
                                sizes=(1, 8))
        card["cut_per_page"] = cut_per_page
        cards.append(card)
        side.free()

    # ── 한 자리에 모은다 ──────────────────────────────────────────
    n = args.page_lines
    for c in cards:
        where, fit = fastest(c["gpu"])
        if fit:
            c["page_seconds_gpu"] = c["cut_per_page"] + fit["per_line"] * n
            c["page_basis_gpu"] = "%s줄 묶음 실측" % where
            c["lines_per_second_gpu"] = 1 / fit["per_line"]
        where, half = fastest(c.get("gpu_fp16") or {})
        if half:
            c["lines_per_second_gpu_fp16"] = 1 / half["per_line"]
        where, slow = fastest(c.get("cpu") or {})
        if slow:
            c["page_seconds_cpu"] = c["cut_per_page"] + slow["per_line"] * n
            c["page_basis_cpu"] = "%s줄 묶음 실측" % where
            c["lines_per_second_cpu"] = 1 / slow["per_line"]

    def cell(value, how="%8.2f"):
        return (how % value) if value is not None else "%8s" % "-"

    print("\n" + "=" * 104)
    print("%-28s %7s %7s %7s %8s %9s %9s %10s %10s"
          % ("", "자모%", "CER%", "통째%", "최저벌%", "줄/초GPU", "줄/초CPU",
             "%d줄쪽GPU" % n, "%d줄쪽CPU" % n))
    print("-" * 104)
    for c in cards:
        f = c["test_fonts"]
        print("%-28s %7.2f %7.2f %7.2f %8.2f %s %s %s %s"
              % (c["name"], f["jamo"], f["cer"], f["exact"], f["low"],
                 cell(c.get("lines_per_second_gpu"), "%9.1f"),
                 cell(c.get("lines_per_second_cpu"), "%9.2f"),
                 cell(c.get("page_seconds_gpu"), "%9.2fs"),
                 cell(c.get("page_seconds_cpu"), "%9.2fs")))

    out = {"when": time.strftime("%Y-%m-%d %H:%M"), "gear": gear,
           "torch": torch.__version__, "members": team,
           "one": args.one.replace("\\", "/") if args.one else team[0],
           "lines_per_test_font": args.fonts, "lines_per_wide_font": args.wide,
           "lines_per_every_font": args.every,
           "page_lines": n, "beams": R.BEAMS, "rival_max_new_tokens": RIVAL_TOKENS,
           "sides": cards}
    CARD.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n적었다: %s" % CARD.relative_to(ROOT))


if __name__ == "__main__":
    main()
