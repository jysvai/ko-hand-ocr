"""판을 **합성 시험지**로 재서 서로 가른다.

    python tools/holdout.py runs/v10a-15000 runs/v16-5000 --lines 600

왜 따로 두는가. 사진 11장 37칸으로는 87% 와 90% 를 못 가른다 — 칸을 다시 뽑아
보면 평균이 ±5%p 흔들린다(`tools/pick.py` 가 그 폭을 찍는다). 그 폭 안에서
손잡이를 돌리면 11장에만 맞는 모델이 된다.

여기서는 **한 번도 학습에 안 쓴 씨앗**으로 합성 줄을 수백 개 만들어 잰다.
표본이 많으니 폭이 좁고(600줄이면 ±1%p 안팎) 판을 실제로 가를 수 있다.

물론 이건 합성 실력이다. 그 값이 실제 실력을 대신할 수 있는 근거는 따로 있다 —
`tools/match.py` 가 재는 아홉 항목이 모두 12% 안에 들어 있다는 것이다. 둘을
같이 봐야 한다: **분포가 맞는지는 match, 판을 가르는 것은 여기, 마지막 확인은
pick.**
"""

from __future__ import annotations

import argparse
import random
import sys
import unicodedata
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import logging as _hf_logging          # noqa: E402
from kohandocr import corpus, data, model as builder, synth  # noqa: E402
from kohandocr.reader import letters_only                # noqa: E402
from kohandocr.vocab import Vocab                        # noqa: E402

_hf_logging.set_verbosity_error()

# 학습에서 쓰지 않는 씨앗. 바꾸면 판끼리 견줄 수 없으니 **고정**이다.
SEED = 90210


def sheet(count: int, fonts_dir: str) -> list[tuple[str, object]]:
    """시험지를 만든다. 씨앗이 고정이라 몇 번을 불러도 같은 그림이 나온다."""
    # **학습에 안 쓴 글꼴로만** 만든다. 여기가 이 자의 핵심이다.
    fonts = synth.Fonts(fonts_dir, part="test")
    rng = random.Random(SEED)
    out = []
    for text in corpus.lines(count, seed=SEED):
        page = synth.render(text, fonts, rng)
        if page is not None:
            out.append((text, synth.fit(page)))
    return out


def closeness(guess: str, truth: str) -> float:
    """자모로 풀어 앞에서부터 맞은 몫. `check` 와 같은 잣대다."""
    a = unicodedata.normalize("NFD", guess)
    b = unicodedata.normalize("NFD", truth)
    return sum(1 for x, y in zip(a, b) if x == y) / max(len(b), 1)


def band(marks: list[float], rounds: int = 400) -> float:
    """95% 폭의 절반(%). 이 안이면 같은 판으로 본다."""
    rng = random.Random(7)
    means = sorted(100 * sum(marks[rng.randrange(len(marks))] for _ in marks) / len(marks)
                   for _ in range(rounds))
    return (means[int(0.975 * (rounds - 1))] - means[int(0.025 * (rounds - 1))]) / 2


def score(run: Path, exam: list, device: str, beams: int, batch: int) -> dict:
    vocab = Vocab.load(run / "vocab.json")
    model = builder.load(str(run)).to(device).eval()
    marks, whole = [], 0
    for at in range(0, len(exam), batch):
        chunk = exam[at:at + batch]
        pixels = data.to_pixels(torch.stack(
            [data.to_gray(cell) for _, cell in chunk]).to(device))
        with torch.no_grad():
            ids = model.generate(pixels, num_beams=beams, max_new_tokens=128,
                                 early_stopping=beams > 1,
                                 prefix_allowed_tokens_fn=letters_only(vocab))
        for (truth, _), row in zip(chunk, ids):
            guess = unicodedata.normalize("NFC", vocab.decode(row.tolist())).strip()
            marks.append(closeness(guess, truth))
            whole += guess == truth
    del model
    torch.cuda.empty_cache()
    return {"letters": 100 * sum(marks) / len(marks),
            "whole": 100 * whole / len(marks), "band": band(marks)}


def main() -> None:
    ap = argparse.ArgumentParser(description="합성 시험지로 판을 가른다")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--lines", type=int, default=600)
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fonts", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.fonts = str(synth.font_folder(args.fonts))

    exam = sheet(args.lines, args.fonts)
    print("시험지 %d줄   글꼴 %d벌 (학습에 안 쓴 것)   씨앗 %d" % (
        len(exam), len(synth.Fonts(args.fonts, part="test")), SEED))
    print()
    print("%-22s %8s %8s %8s" % ("판", "글자", "줄 통째", "흔들림"))
    best = None
    for name in args.runs:
        run = Path(name)
        if not (run / "config.json").exists():
            print("%-22s (없음)" % name)
            continue
        got = score(run, exam, args.device, args.beams, args.batch)
        print("%-22s %7.1f%% %7.1f%% %7s" % (
            name[:22], got["letters"], got["whole"], "+-%.1f%%p" % got["band"]))
        if best is None or got["letters"] > best[0]:
            best = (got["letters"], name, got)
    if best:
        print("\n가장 나은 판: %s   글자 %.1f%% (+-%.1f%%p)"
              % (best[1], best[0], best[2]["band"]))


if __name__ == "__main__":
    main()
