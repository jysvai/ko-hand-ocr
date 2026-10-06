"""학습 글꼴마다 **판 하나가 얼마나 읽나** 잰다. 고리의 `hard` 가 이것을 쓴다.

    python tools/hardfonts.py runs/v73-5000                 # runs/HARD-v73-5000.json
    python tools/hardfonts.py runs/v73-5000 --lines 16 --show 30

왜 따로 있나(2026-09-25). v59~v83 스물다섯 바퀴 동안 가장 낮은 하나(동해독도)가
90.1~90.2% 에 붙어 있었다. 그동안 고리가 바꾼 것은 이어받는 자리와 학습률뿐이고,
학습 글꼴 450벌은 **늘 고르게** 뽑았다. 이미 잘 읽는 벌에 걸음 대부분을 쓴다.
여기서 잰 점수로 `synth.Fonts.lean` 이 못 읽는 벌을 더 자주 뽑는다.

**재는 것은 학습 글꼴뿐이다.** 시험용 여섯 벌과 두루 시험용 스물네 벌은
`part="train"` 이 뺀다. 여기서 시험용을 재어 몰아주면 자를 외우는 것이 된다.

글월은 시험지와 같은 씨앗(`holdout.sheet`)이라 **벌마다 같은 글월**을 읽힌다.
벌끼리 견줄 수 있는 것이 그래서다. 흔들기 없이 원본 한 번만 읽는다 — 차례만
알면 되고, 판 하나를 한 번만 올린다.

잰 판은 바뀌지 않으므로 파일이 있으면 다시 안 잰다(`--again` 으로 다시).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import eval_photos as E                                  # noqa: E402
import holdout as H                                      # noqa: E402
from transformers import logging as _hf_logging          # noqa: E402
from kohandocr import model as builder, synth            # noqa: E402
from kohandocr.vocab import Vocab                        # noqa: E402

_hf_logging.set_verbosity_error()


def where(run: str) -> Path:
    return ROOT / "runs" / ("HARD-%s.json" % Path(run).name)


def measure(run: str, lines: int, fonts_dir: str, device: str) -> dict:
    """글꼴 파일 이름 -> 점수(%). 그림을 다 그린 뒤 판을 **한 번만** 올려 읽는다."""
    names = [f.name for f in synth.Fonts(fonts_dir, part="train").files]
    exams, cells = {}, []
    for name in names:
        made = H.sheet(lines, fonts_dir, only=name, part="train")
        if made:
            exams[name] = (len(cells), made)
            cells += [cell for _, cell in made]
    vocab = Vocab.load(Path(run) / "vocab.json")
    model = builder.load(run).to(device).eval()
    got = []
    for at in range(0, len(cells), 40):
        got += E.read(model, vocab, cells[at:at + 40], device, beams=5)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    out = {}
    for name, (at, made) in exams.items():
        hit = [E.closeness_tight(got[at + i], text) for i, (text, _) in enumerate(made)]
        out[name] = round(100 * sum(hit) / len(hit), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="학습 글꼴마다 판 하나로 잰다")
    ap.add_argument("run", help="잴 판 폴더 (고리는 이어받을 판을 준다)")
    ap.add_argument("--lines", type=int, default=12, help="글꼴마다 줄 수")
    ap.add_argument("--show", type=int, default=15, help="가장 못 읽는 벌을 몇 개 보이나")
    ap.add_argument("--again", action="store_true", help="파일이 있어도 다시 잰다")
    ap.add_argument("--fonts", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    fonts_dir = str(synth.font_folder(args.fonts))

    out = where(args.run)
    if out.exists() and not args.again:
        said = json.loads(out.read_text(encoding="utf-8"))
        print("이미 있다: %s (%d벌, %d줄씩)" % (out.name, len(said["fonts"]), said["lines"]))
        return

    start = time.perf_counter()
    scores = measure(args.run, args.lines, fonts_dir, args.device)
    values = sorted(scores.values())
    out.write_text(json.dumps(
        {"run": args.run, "lines": args.lines, "fonts": scores,
         "when": time.strftime("%Y-%m-%d %H:%M")},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print("%s: 학습 글꼴 %d벌 x %d줄  (%.0f초) -> %s"
          % (Path(args.run).name, len(scores), args.lines,
             time.perf_counter() - start, out.name))
    print("  가장 낮은 벌 %.1f%%   10%% 자리 %.1f%%   가운데 %.1f%%   평균 %.1f%%"
          % (values[0], values[len(values) // 10], statistics.median(values),
             statistics.fmean(values)))
    for name, value in sorted(scores.items(), key=lambda row: row[1])[:args.show]:
        print("    %5.1f%%  %s" % (value, name))


if __name__ == "__main__":
    main()
