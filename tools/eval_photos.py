"""실제 손글씨 사진 7장으로 재 본다.

    python tools/eval_photos.py runs/base

합성 이미지에서 잘 나오는 것은 아무 뜻이 없다. 우리가 맞춰야 하는 것은
사람이 실제로 쓴 글씨다. 그래서 앱과 **똑같은 줄 자르기**(`kohandocr.page`)
를 거쳐 같은 조각을 모델에 넣는다.

두 가지를 같이 잰다.
  - 얼마나 맞았나  (자모 단위 / 줄 통째)
  - 얼마나 빨랐나  (사진 한 장당 초)

둘 다 중요하다. 아무리 잘 읽어도 한 장에 1분이면 못 쓴다.
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import sys
import time
import unicodedata
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
# 시험 자료(data/eval)가 저장소 안에 있으면 그쪽이다. 예전에는 이 저장소가
# 앱 폴더 **안에** 있어서 늘 부모를 봤는데, 따로 떼어 낸 뒤로는 부모가
# 바탕화면이라 아무것도 없다. 두 자리를 다 받아 준다.
APP = ROOT if (ROOT / "data" / "eval").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

from kohandocr import page as photo                    # noqa: E402  앱과 같은 줄 자르기
from transformers import logging as _hf_logging        # noqa: E402
from kohandocr import data, model as builder, synth     # noqa: E402
from kohandocr.reader import letters_gate                # noqa: E402
from kohandocr.vocab import Vocab                       # noqa: E402

# generate() 를 부를 때마다 max_new_tokens/max_length 안내가 줄마다 찍혀
# 정작 봐야 할 표를 덮는다. 경고이지 오류가 아니다.
_hf_logging.set_verbosity_error()


GOAL = 90          # 사진 한 장이 넘어야 하는 값(%). 평균이 아니라 **모든 장**이 넘어야 한다.


def jamo(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def closeness(got: str, want: str) -> float:
    return difflib.SequenceMatcher(None, jamo(got), jamo(want)).ratio()


def tight(text: str) -> str:
    """띄어쓰기를 지운 꼴."""
    return "".join(jamo(text).split())


def closeness_tight(got: str, want: str) -> float:
    """띄어쓰기를 빼고 견준다.

    '성명 : 홍길동' 을 '성명: 홍길동' 으로 읽으면 자모 단위로는 5% 를 잃는데,
    서식에서 값을 뽑아 쓰는 데는 아무 차이가 없다. 두 값을 같이 보면
    **남은 오차가 진짜 글자를 못 읽은 것인지, 띄어쓰기뿐인지** 갈린다.
    고치는 방향이 달라지므로 둘 다 찍는다.
    """
    return difflib.SequenceMatcher(None, tight(got), tight(want)).ratio()


def cells_of(jpg: Path, max_lines: int = 16) -> list[Image.Image]:
    png = photo.prepare(jpg.read_bytes())
    return [Image.open(io.BytesIO(cell["png"])).convert("L")
            for cell in photo.lines(png, max_lines=max_lines)]


def read(model, vocab, cells, device, beams: int) -> list[str]:
    if not cells:
        return []
    batch = torch.stack([data.to_gray(synth.fit(cell)) for cell in cells])
    pixels = data.to_pixels(batch.to(device))
    with torch.no_grad():
        # 앱(`Reader.read`)과 **같은 조건**으로 읽는다. 홀자모를 앱에서만 빼면
        # 여기서 잰 점수가 앱의 점수가 아니게 된다.
        ids = model.generate(pixels, num_beams=beams, max_new_tokens=128,
                             early_stopping=beams > 1,
                             logits_processor=letters_gate(vocab))
    return [vocab.decode(row.tolist()) for row in ids]


def pair(got: list[str], want: list[str]) -> list[tuple[str, str]]:
    """읽은 줄과 정답 줄을 가장 비슷한 것끼리 맞춘다.

    줄 수가 다를 수 있다. 잡음 칸이 더 잡히기도 하고, 정답 하나가 두 칸에
    걸치기도 한다.

    정답을 차례대로 훑으며 그때그때 가장 가까운 것을 집으면 안 된다.
    앞선 정답이 뒤 정답의 짝을 먼저 가져가 버린다(실측: '테스트' 가
    '홍서 말' 을 집어가서, 정작 '홍서말' 에는 '615' 가 남았다).
    **확신이 큰 짝부터** 확정해야 한다.
    """
    scores = sorted(
        ((closeness(guess, truth), gi, ti) for ti, truth in enumerate(want)
         for gi, guess in enumerate(got)),
        reverse=True,
    )
    taken_read: set[int] = set()
    chosen: dict[int, int] = {}
    for _, gi, ti in scores:
        if ti in chosen or gi in taken_read:
            continue
        chosen[ti] = gi
        taken_read.add(gi)
    return [(got[chosen[ti]] if ti in chosen else "", truth)
            for ti, truth in enumerate(want)]


def _reader(model, vocab, device):
    """이미 올려 둔 모델로 판독기를 만든다. 두 번 올리지 않는다."""
    from kohandocr.reader import Reader

    reader = Reader.__new__(Reader)
    reader.model, reader.vocab, reader.device, reader.skip = model, vocab, device, 1
    return reader


def main() -> None:
    ap = argparse.ArgumentParser(description="실제 사진으로 재기")
    ap.add_argument("run", help="모델 폴더 (예: runs/base)")
    ap.add_argument("--photos", default=str(APP / "data/eval/samples"))
    ap.add_argument("--labels", default=str(APP / "data/eval/labels.json"))
    ap.add_argument("--beams", type=int, default=5, help="1 이면 그리디. 앱과 같은 값이 기본")
    ap.add_argument("--roster", action="store_true",
                    help="명단(이름·조직)을 알려 주고 그 안에서 고르게 가둔다 — 앱의 실제 조건")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    truth = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    vocab = Vocab.load(Path(args.run) / "vocab.json")
    model = builder.load(args.run).to(args.device).eval()

    # 명단은 일곱 장 전체의 이름·조직을 모은 것. 한 장을 읽을 때 그 장의 답만
    # 주면 시험이 너무 쉬워진다. 실제로도 회사 명단 전체와 맞춰 본다.
    pool: list[str] = []
    if args.roster:
        for row in truth.values():
            pool += [row[key] for key in ("person", "org") if row.get(key)]
        pool = sorted(set(pool))
        print("명단", len(pool), "개:", ", ".join(pool))

    hits = misses = 0
    scores, loose, spent = [], [], []
    pairs_all: list[tuple[str, str]] = []
    by_photo: dict[str, list[float]] = {}
    for jpg in sorted(Path(args.photos).glob("hand-*.jpg")):
        want = truth.get(jpg.name, {}).get("lines", [])
        start = time.perf_counter()
        cells = cells_of(jpg)
        if pool:
            got = [row["constrained"] if row["agrees"] and row["constrained"] else row["text"]
                   for row in _reader(model, vocab, args.device).both(cells, pool)]
        else:
            got = read(model, vocab, cells, args.device, args.beams)
        took = time.perf_counter() - start
        spent.append(took)
        print(f"--- {jpg.name}  {len(cells)}칸  {took:.2f}초")
        matched = pair(got, want)
        pairs_all += matched
        for guess, want_line in matched:
            score = closeness(guess, want_line)
            scores.append(score)
            loose.append(closeness_tight(guess, want_line))
            by_photo.setdefault(jpg.name, []).append(loose[-1])
            hits += guess == want_line
            misses += guess != want_line
            mark = "맞음" if guess == want_line else f"{score*100:3.0f}%"
            print(f"    {mark}  {want_line!r} -> {guess!r}")
        extra = [g for g in got if g not in [p[0] for p in pair(got, want)]]
        if extra:
            print(f"    (덤으로 읽은 칸: {extra})")

    # 사진 단위로도 본다. **평균이 아니라 가장 나쁜 장**이 목표를 정한다 —
    # 한 장이 무너지면 그 서류는 사람이 다시 봐야 하므로, 평균으로는 그걸 못 본다.
    print()
    print("사진별 (띄어쓰기 뺀 자모 닮음)   목표 %d%%" % GOAL)
    ranked = sorted(((n, 100 * sum(v) / len(v), len(v)) for n, v in by_photo.items()),
                    key=lambda row: row[1])
    for name, mean, rows in ranked:
        print("   %s %-9s %5.1f%%   줄 %d개"
              % ("  " if mean >= GOAL else "->", name.replace(".jpg", ""), mean, rows))
    short = [n for n, mean, _ in ranked if mean < GOAL]
    print("   %d / %d 장이 %d%% 를 넘었다%s"
          % (len(ranked) - len(short), len(ranked), GOAL,
             "" if not short else "   못 넘긴 장: "
             + ", ".join(n.replace(".jpg", "") for n in short)))

    print()
    same = sum(1 for guess, want_line in
               ((g, w) for g, w in pairs_all) if tight(guess) == tight(want_line))
    print("줄 맞춤   %d / %d   (띄어쓰기 빼면 %d)" % (hits, hits + misses, same))
    print("자모 닮음 %.1f%%   (띄어쓰기 빼면 %.1f%%)"
          % (100 * sum(scores) / max(len(scores), 1),
             100 * sum(loose) / max(len(loose), 1)))
    print("한 장당   %.2f초  (가장 느린 장 %.2f초)" % (
        sum(spent) / max(len(spent), 1), max(spent, default=0)))


if __name__ == "__main__":
    main()
