"""남겨 둔 판들을 한 번에 재서 **어느 것을 쓸지 고른다.**

    python tools/pick.py runs/v10-15000 runs/v10-30000 runs/v10
    python tools/pick.py runs/v10 --beams 1 3 5 8      # 빔도 같이 쓸어 본다

왜 따로 두는가. 두 번 데었다.

  - v8 은 45,000 걸음까지 간 판(80.1%)이 30,000 걸음 판(81.4%)보다 나빴다.
    **마지막이 최고가 아니다.** 그런데 그때 30,000 판은 이미 덮어써진 뒤였다.
    그래서 `train.py --keep` 으로 중간 판을 남기게 했는데, 남겨 놓고도 하나씩
    손으로 재면 결국 안 재게 된다.
  - 고르는 기준이 **평균이 아니다.** 우리가 넘어야 하는 것은 사진 한 장 한 장이다.
    평균이 높은 판이 가장 나쁜 장에서는 더 낮은 일이 실제로 있다.

평균은 `eval_photos.py` 와 조금 다르게 낸다. 여기서는 **사진 한 장에 한 표**이고
저기서는 줄 하나에 한 표다(hand-07 은 줄이 4개라 저기서 표를 더 갖는다).
고르는 데는 장 단위가 맞다 — 우리가 넘어야 하는 것이 장이기 때문이다.

그래서 여기서는 판마다 (평균, **가장 나쁜 장**, 85% 를 넘긴 장 수, 초) 를 같이
찍고 마지막에 골라 준다. 고르는 순서는 **못 넘긴 장이 적은 것 → 가장 나쁜 장이
높은 것 → 평균이 높은 것** 이다.
"""

from __future__ import annotations

import argparse
import json
import random as _random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "tools"))

import eval_photos as E                                  # noqa: E402  같은 잣대를 쓴다
from transformers import logging as _hf_logging          # noqa: E402
from kohandocr import data, model as builder, synth      # noqa: E402
from kohandocr.reader import letters_only                 # noqa: E402
from kohandocr.vocab import Vocab                        # noqa: E402

# generate() 를 부를 때마다 max_new_tokens/max_length 안내가 줄마다 찍혀
# 정작 봐야 할 표를 덮는다. 경고이지 오류가 아니다.
_hf_logging.set_verbosity_error()


def score(run: Path, cells: dict, truth: dict, beams: int, device: str) -> dict:
    vocab = Vocab.load(run / "vocab.json")
    model = builder.load(str(run)).to(device).eval()
    per: dict[str, float] = {}
    every: list[float] = []
    spent = 0.0
    worst_lines: list[tuple[float, str, str, str]] = []
    for name, images in cells.items():
        want = truth.get(name, {}).get("lines", [])
        batch = torch.stack([data.to_gray(synth.fit(c)) for c in images])
        pixels = data.to_pixels(batch.to(device))
        start = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(pixels, num_beams=beams, max_new_tokens=128,
                                 early_stopping=beams > 1,
                                 prefix_allowed_tokens_fn=letters_only(vocab))
        spent += time.perf_counter() - start
        got = [vocab.decode(row.tolist()) for row in ids]
        marks = []
        for guess, line in E.pair(got, want):
            close = E.closeness_tight(guess, line)
            marks.append(close)
            worst_lines.append((close, name, line, guess))
        per[name] = 100 * sum(marks) / max(len(marks), 1)
        every += marks
    del model
    torch.cuda.empty_cache()
    flat = [v for v in per.values()]
    return {
        "per": per,
        "mean": sum(flat) / max(len(flat), 1),
        "low": min(flat, default=0.0),
        "over": sum(1 for v in flat if v >= E.GOAL),
        "photos": len(flat),
        "spent": spent / max(len(flat), 1),
        "worst_lines": sorted(worst_lines)[:6],
        "wobble": wobble(every),
    }


def wobble(marks: list[float], rounds: int = 400) -> float:
    """이 점수가 **얼마나 흔들리는 값인지** 돌려준다(95% 폭의 절반, %).

    왜 필요한가. 사진 11장 37칸으로는 87% 와 90% 를 못 가른다. 실제로 크게
    데었다 — 사진을 조금 기울여 읽었더니 v14 는 **+0.6도**에서 가장 좋았는데
    v16 은 **-1.2도**에서 가장 좋았다. 같은 사진인데 방향이 반대다. 즉 그 3%p 는
    기울기가 만든 것이 아니라, 아슬아슬한 칸들이 판마다 다르게 떨어진 것이다.
    그런 값을 좇아 손잡이를 돌리면 11장에만 맞는 모델이 된다.

    그래서 판을 견줄 때 이 폭을 같이 찍는다. **폭 안에 들면 같은 판으로 본다.**
    """
    if len(marks) < 2:
        return 0.0
    rng = _random.Random(7)
    means = []
    for _ in range(rounds):
        draw = [marks[rng.randrange(len(marks))] for _ in marks]
        means.append(100 * sum(draw) / len(draw))
    means.sort()
    return (means[int(0.975 * (rounds - 1))] - means[int(0.025 * (rounds - 1))]) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description="남겨 둔 판들을 재서 고른다")
    ap.add_argument("runs", nargs="+", help="모델 폴더들")
    ap.add_argument("--beams", type=int, nargs="+", default=[5])
    ap.add_argument("--photos", default=str(APP / "data/eval/samples"))
    ap.add_argument("--labels", default=str(APP / "data/eval/labels.json"))
    ap.add_argument("--lines", type=int, default=0,
                    help="가장 나쁜 줄을 이만큼 같이 찍는다 (0 이면 안 찍음)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    truth = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    # 칸 자르기는 판마다 같으므로 **한 번만** 한다. 판이 넷이면 넷 다 같은 조각을 본다.
    jpgs = sorted(Path(args.photos).glob("hand-*.jpg"))
    cells = {jpg.name: E.cells_of(jpg) for jpg in jpgs}
    names = [jpg.name.replace(".jpg", "").replace("hand-", "") for jpg in jpgs]

    print("사진 %d장 %d칸   목표: **모든 장** %d%% 이상\n"
          % (len(jpgs), sum(len(v) for v in cells.values()), E.GOAL))
    header = "%-22s %6s %6s %7s %6s  " % ("판", "평균", "최저", "넘긴장", "초/장")
    print(header + "  ".join("%5s" % n for n in names))

    table: list[tuple[tuple, str, dict]] = []
    for name in args.runs:
        run = Path(name)
        if not (run / "config.json").exists():
            print("%-22s (없음)" % name)
            continue
        for beams in args.beams:
            got = score(run, cells, truth, beams, args.device)
            label = name if len(args.beams) == 1 else f"{name} b{beams}"
            # 고르는 기준: 못 넘긴 장이 적은 것 -> 가장 나쁜 장 -> 평균
            key = (got["over"], got["low"], got["mean"])
            table.append((key, label, got))
            print("%-22s %5.1f%% %5.1f%% %4d/%-2d %5.2f  " % (
                label[:22], got["mean"], got["low"], got["over"], got["photos"],
                got["spent"]) + "  ".join(
                "%5.1f" % got["per"].get(f"hand-{n}.jpg", 0.0) for n in names))
            print("%-22s (칸을 다시 뽑아 보면 평균이 +-%.1f%%p 흔들린다 — 이 안이면 같은 판이다)"
                  % ("", got["wobble"]))

    if not table:
        raise SystemExit("잰 판이 없다")
    table.sort(reverse=True)
    key, best, got = table[0]
    short = sorted((v, n) for n, v in got["per"].items() if v < E.GOAL)
    print("\n고른 판: %s   평균 %.1f%%  가장 나쁜 장 %.1f%%  %d/%d 장이 %d%% 넘음"
          % (best, got["mean"], got["low"], got["over"], got["photos"], E.GOAL))
    if short:
        print("  아직 못 넘긴 장: " + ", ".join(
            "%s %.1f%%" % (n.replace(".jpg", "").replace("hand-", ""), v) for v, n in short))
    else:
        print("  모든 장이 목표를 넘었다.")

    if args.lines:
        print("\n가장 나쁜 줄 (%s)" % best)
        for close, name, line, guess in got["worst_lines"][:args.lines]:
            print("  %5.1f%%  %-10s %r -> %r"
                  % (100 * close, name.replace(".jpg", ""), line, guess))


if __name__ == "__main__":
    main()
