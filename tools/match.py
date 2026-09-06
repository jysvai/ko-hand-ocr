"""합성 그림이 실제 사진과 같은 분포인지 한 번에 잰다.

    python tools/match.py              # 과녁(kohandocr/measured.json)과 지금 합성을 견준다
    python tools/match.py --update     # 사진을 새로 재서 과녁을 다시 적는다

왜 따로 두는가. 이 프로젝트에서 정확도를 올린 것은 대부분 **모델을 바꾼 것이
아니라 합성 그림을 실제와 같은 분포로 맞춘 것**이었다. 그런데 그 측정을
매번 처음부터 짜면 시간도 토큰도 많이 들고, 무엇보다 **지난번에 무엇을
맞췄는지 잊어버린다.** 그래서 과녁을 `kohandocr/measured.json` 에 적어 두고 여기서 견준다.

한 항목이라도 크게 어긋나면 학습을 걸기 전에 먼저 고친다. 실측으로 배운 것:
합성이 어긋난 채로 학습하면 걸음 수를 늘려도 실제 사진에서는 안 따라온다.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
# 시험 자료(data/eval)가 저장소 안에 있으면 그쪽이다. 예전에는 이 저장소가
# 앱 폴더 **안에** 있어서 늘 부모를 봤는데, 따로 떼어 낸 뒤로는 부모가
# 바탕화면이라 아무것도 없다. 두 자리를 다 받아 준다.
APP = ROOT if (ROOT / "data" / "eval").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))

from kohandocr import corpus, synth                     # noqa: E402

MEMORY = ROOT / "kohandocr/measured.json"
QUANTS = (0.015, 0.10, 0.25, 0.50, 0.75, 0.90, 0.985)

# 사진 폴더는 개인 자료라 꾸러미에 들어가지 않는다. 없으면 견주기를 건너뛴다.
PHOTOS = APP / "data" / "eval" / "samples"
LABELS = APP / "data" / "eval" / "labels.json"

# 칸이 정답 줄과 다른 차례로 나오는 사진. 옆으로 쓴 줄이 있으면 읽기 순서가 다르다.
# index 로 짝지을 때 이걸 안 보면 측정이 통째로 틀린다(지수 0.16 대 0.60).
CELL_ORDER = {"hand-07.jpg": [0, 2, 1, 3]}


def quantiles(values: list[float]) -> list[float]:
    values = sorted(values)
    count = len(values)
    if not count:
        return [0.0] * len(QUANTS)
    return [values[min(count - 1, int(count * q))] for q in QUANTS]


def cell_facts(cell: Image.Image) -> dict:
    """칸 하나에서 잴 것들. 실제 조각과 합성 조각에 똑같이 쓴다."""
    # 밝기값 세는 일은 낱낱이 훑지 않고 히스토그램으로 한다. 픽셀이 4만 개인데
    # 자리는 256개뿐이라 훨씬 싸고, 분위수도 한 번 훑어 구할 수 있다.
    hist = cell.histogram()
    total = sum(hist)
    inked = sum(hist[:synth.INK_EDGE])
    box = cell.point([255 if v < synth.INK_EDGE else 0 for v in range(256)]).getbbox()
    facts: dict[str, float] = {"ink": inked / max(total, 1)}
    if box:
        facts["density"] = inked / max((box[2] - box[0]) * (box[3] - box[1]), 1)
    if inked >= 40:
        # 잉크로 친 픽셀만 골라서 잰다. 픽셀 전부로 재면 종이가 다수라 묻힌다.
        seen = 0
        for value in range(synth.INK_EDGE):
            seen += hist[value]
            if "dark10" not in facts and seen >= inked * 0.10:
                facts["dark10"] = value
            if seen >= inked * 0.50:
                facts["mid"] = value
                break
    if box:
        facts["blob"] = _blob(cell, box)
    if inked >= 40:
        # 획 가장자리가 얼마나 번졌나. 또렷한 사진은 종이에서 잉크로 한 번에
        # 떨어지고, 흐린 사진은 그 사이 회색이 넓다. 잉크로 친 픽셀 대비
        # **문턱 바로 위 회색띠**의 크기로 잰다(초점, 흔들림, 표백이 다 여기 나온다).
        facts["fringe"] = sum(hist[synth.INK_EDGE:236]) / max(inked, 1)
    return facts


def edges(cell: Image.Image) -> dict:
    """**자르기 전 칸**에서 가장자리 여백을 잰다(글자 높이 대비).

    `cell_facts` 와 달리 `fit` 을 거치기 **전에** 재야 한다. fit 은 64x640 으로
    맞추면서 여백을 바꿔 버리므로, 맞춘 뒤에 재면 이 어긋남이 안 보인다.
    실제로 v16 까지 이 항목이 없어서, 합성이 실제의 **네 배** 여백을 두고 있는
    것을 열여섯 판 동안 아무도 못 봤다. 64px 틀에 맞추고 나면 실제 글자는
    61px 인데 합성 글자는 53px 이었다 — 모델이 보는 글자가 줄곧 15% 작았다.
    """
    mask = cell.point([255 if v < synth.INK_EDGE else 0 for v in range(256)])
    box = mask.getbbox()
    if not box:
        return {}
    tall = max(box[3] - box[1], 1)
    return {"edge_up": (box[1] + cell.height - box[3]) / 2 / tall,
            "edge_side": (box[0] + cell.width - box[2]) / 2 / tall}


def _blob(cell: Image.Image, box) -> float:
    """이어진 잉크 덩이의 폭(가운데값). 글자 높이 대비 비율.

    **글자가 얼마나 붙어 있는가**를 잰다. 사람은 획을 이어 쓰는데 글꼴은 떼어 그려서,
    아무것도 안 하면 합성이 실제보다 잘게 쪼개진다(실측 0.46 대 0.65).
    이건 밝기로도 밀도로도 안 잡히는 눈금이라 따로 둔다.
    """
    tall = box[3] - box[1]
    if tall < 8:
        return 0.0
    band = cell.point([255 if v < synth.INK_EDGE else 0 for v in range(256)]).crop(box)
    runs, seen = [], 0
    for x in range(band.width):
        if band.crop((x, 0, x + 1, band.height)).getbbox() is not None:
            seen += 1
        elif seen:
            runs.append(seen / tall)
            seen = 0
    if seen:
        runs.append(seen / tall)
    if not runs:
        return 0.0
    runs.sort()
    return runs[len(runs) // 2]


def fit_curve(pairs: list[tuple[int, float]]) -> dict:
    """가로세로 = a x 글자수^p. 로그 자리에서 직선을 맞춘다."""
    xs = [math.log(n) for n, _ in pairs]
    ys = [math.log(a) for _, a in pairs]
    count = len(xs)
    if count < 3:
        return {"a": 0.0, "p": 0.0, "spread": [0.0, 0.0], "samples": count}
    mx, my = sum(xs) / count, sum(ys) / count
    spread = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / spread if spread else 0.0
    scale = math.exp(my - slope * mx)
    off = sorted(math.log(a / (scale * n ** slope)) for n, a in pairs)
    return {
        "a": round(scale, 3),
        "p": round(slope, 3),
        "spread": [round(math.exp(off[len(off) // 10]), 2),
                   round(math.exp(off[9 * len(off) // 10]), 2)],
        "samples": count,
    }


def _tidy(facts: dict[str, list], shapes: list, sized: list) -> dict:
    out: dict[str, object] = {
        key: [round(v, 4) if key in ("ink", "density", "edge_up", "edge_side", "fringe") else
              round(v, 2) if key == "blob" else round(v)
              for v in quantiles(values)]
        for key, values in facts.items() if values}
    out["aspect"] = [round(v, 2) for v in quantiles(shapes)]
    out["aspect_curve"] = fit_curve(sized)
    out["cells"] = len(shapes)
    return out


def from_photos() -> dict | None:
    """실제 사진에서 과녁을 잰다. 사진이 없으면 None."""
    if not PHOTOS.exists() or not LABELS.exists():
        return None
    from kohandocr import page as photo                 # 앱과 **같은** 줄 자르기를 쓴다

    truth = json.loads(LABELS.read_text(encoding="utf-8"))
    facts: dict[str, list] = {key: [] for key in NAMES if key != "aspect"}
    shapes: list[float] = []
    sized: list[tuple[int, float]] = []
    for jpg in sorted(PHOTOS.glob("hand-*.jpg")):
        lines = truth.get(jpg.name, {}).get("lines", [])
        cells = list(photo.lines(photo.prepare(jpg.read_bytes()), max_lines=16))
        for cell in cells:
            image = Image.open(io.BytesIO(cell["png"])).convert("L")
            shapes.append(image.width / image.height)
            for key, value in {**cell_facts(synth.fit(image)), **edges(image)}.items():
                facts[key].append(value)
        # 글자 수와 짝지으려면 칸 수가 정답 줄 수와 같아야 한다. 다르면 건너뛴다.
        if len(cells) != len(lines):
            continue
        for index, text in zip(CELL_ORDER.get(jpg.name, range(len(cells))), lines):
            image = Image.open(io.BytesIO(cells[index]["png"]))
            letters = len(text.replace(" ", ""))
            if letters:
                sized.append((letters, image.width / image.height))
    if not shapes:
        return None
    out = _tidy(facts, shapes, sized)
    out["photos"] = len(list(PHOTOS.glob("hand-*.jpg")))
    return out


def from_synth(count: int, seed: int, fonts_dir: str) -> dict:
    fonts = synth.Fonts(fonts_dir)
    rng = random.Random(11)
    facts: dict[str, list] = {key: [] for key in NAMES if key != "aspect"}
    shapes: list[float] = []
    sized: list[tuple[int, float]] = []
    for text in corpus.lines(count, seed=seed):
        page = synth.render(text, fonts, rng)
        if page is None:
            continue
        shapes.append(page.width / page.height)
        letters = len(text.replace(" ", ""))
        if letters:
            sized.append((letters, page.width / page.height))
        for key, value in {**cell_facts(synth.fit(page)), **edges(page)}.items():
            facts[key].append(value)
    return _tidy(facts, shapes, sized)


NAMES = {
    "aspect": ("칸 가로/세로", 1.0),
    "ink": ("칸을 덮은 잉크", 100.0),
    "density": ("글자 속 밀도", 100.0),
    "dark10": ("획 진한 10%", 1.0),
    "mid": ("획 가운데 밝기", 1.0),
    "blob": ("이어진 덩이 폭", 1.0),
    "fringe": ("획 가장자리 번짐", 100.0),
    "edge_up": ("여백 위아래", 100.0),
    "edge_side": ("여백 좌우", 100.0),
}


def gaps(target: dict, got: dict) -> dict[str, float]:
    """항목별 어긋남. 과녁의 폭에 견준 상대값이라 밝기와 비율을 같은 자로 잰다."""
    out = {}
    for key in NAMES:
        want, mine = target.get(key), got.get(key)
        if not want or not mine:
            continue
        span = max(want[-1] - want[0], abs(want[len(want) // 2]) * 0.2, 1e-6)
        out[key] = sum(abs(a - b) for a, b in zip(want, mine)) / len(want) / span
    return out


def report(target: dict, got: dict) -> dict[str, float]:
    print("항목            " + "".join("%7.1f%%" % (100 * q) for q in QUANTS))
    off = gaps(target, got)
    for key, (label, scale) in NAMES.items():
        want, mine = target.get(key), got.get(key)
        if not want or not mine:
            continue
        print("%-14s %s   실제" % (label, "".join("%8.1f" % (scale * v) for v in want)))
        print("%-14s %s   합성   어긋남 %+.0f%%"
              % ("", "".join("%8.1f" % (scale * v) for v in mine), 100 * off[key]))
    a, b = target.get("aspect_curve"), got.get("aspect_curve")
    if a and b and a["samples"] >= 3:
        print("\n가로세로 곡선   실제 %.2f x n^%.3f  (흩어짐 %.2f~%.2f, 표본 %d)"
              % (a["a"], a["p"], a["spread"][0], a["spread"][1], a["samples"]))
        print("                합성 %.2f x n^%.3f  (흩어짐 %.2f~%.2f, 표본 %d)"
              % (b["a"], b["p"], b["spread"][0], b["spread"][1], b["samples"]))
        for letters in (3, 6, 10, 17):
            print("                  %2d자 -> 실제 %.1f   합성 %.1f"
                  % (letters, a["a"] * letters ** a["p"], b["a"] * letters ** b["p"]))
    return off


def main() -> None:
    ap = argparse.ArgumentParser(description="합성이 실제와 같은 분포인지 견준다")
    ap.add_argument("--fonts", default="")
    ap.add_argument("--lines", type=int, default=400)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--update", action="store_true", help="사진을 다시 재서 과녁을 고쳐 적는다")
    args = ap.parse_args()
    args.fonts = str(synth.font_folder(args.fonts))

    memory = json.loads(MEMORY.read_text(encoding="utf-8"))
    if args.update:
        fresh = from_photos()
        if fresh is None:
            raise SystemExit(f"사진이 없다: {PHOTOS}")
        memory["measured"] = fresh
        MEMORY.write_text(json.dumps(memory, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
        print("과녁을 다시 적었다: 사진 %d장 %d칸" % (fresh["photos"], fresh["cells"]))
        return

    target = memory["measured"]
    got = from_synth(args.lines, args.seed, args.fonts)
    print("과녁: 사진 %d장 %d칸   /   지금 합성 %d칸\n"
          % (target.get("photos", 0), target.get("cells", 0), got["cells"]))
    off = report(target, got)
    worst = max(off.values(), default=0.0)
    name = max(off, key=lambda k: off[k]) if off else "-"
    print("\n가장 큰 어긋남 %s %.0f%%   %s" % (
        NAMES.get(name, (name, 0))[0], 100 * worst,
        "맞는다" if worst < 0.25 else "**여기부터 고친다. 학습을 걸기 전에.**"))


if __name__ == "__main__":
    main()
