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


def _has_bare_jamo(text: str) -> bool:
    """홑자모(ㄱ, ㅏ, ㄾ …)가 든 줄인가.

    **이런 줄은 시험지에서 뺀다.** 읽을 때 `letters_only` 를 걸어 홑자모를
    답으로 못 내게 하는데(까닭은 `reader.LETTERS_ONLY` 머리말에 있다), 그러면
    정답이 홑자모인 줄은 **맞힐 길이 아예 없다.** 그걸 채점에 넣고 있었다 —
    600줄 중 27줄(4.5%)이 그랬고 23줄은 통째로 홑자모라 언제나 0% 였다.
    답이 막힌 문제를 틀렸다고 세면 점수만 낮아지는 것이 아니라, 손잡이를
    돌려도 안 움직이는 바닥이 생겨서 **판을 가르는 힘이 줄어든다.**

    말뭉치가 이런 줄을 만드는 까닭은 따로 있다. 어휘의 모든 토큰이 학습에
    한 번은 나와야 하기 때문이다(`test_모든_토큰이_학습에_나온다`). 서식에
    실제로 나오는 줄이 아니므로 시험지에서 빼는 것이 맞다.
    """
    return any(0x3131 <= ord(c) <= 0x318E for c in text)


def sheet(count: int, fonts_dir: str, only: str = "") -> list[tuple[str, object]]:
    """시험지를 만든다. 씨앗이 고정이라 몇 번을 불러도 같은 그림이 나온다.

    `only` 를 주면 이름에 그 글자가 든 글씨체 하나로만 만든다(`--per-font`).
    """
    # **학습에 안 쓴 글꼴로만** 만든다. 여기가 이 자의 핵심이다.
    fonts = synth.Fonts(fonts_dir, part="test", only=only)
    rng = random.Random(SEED)
    out = []
    for text in corpus.lines(count, seed=SEED):
        if _has_bare_jamo(text):
            continue
        page = synth.render(text, fonts, rng)
        if page is not None:
            out.append((text, synth.fit(page)))
    return out


def _short(name: str) -> str:
    """'구글 감자꽃 GamjaFlower-Regular.ttf' -> '감자꽃 (GamjaFlower)'"""
    bits = name.replace(".ttf", "").split()
    return "%s (%s)" % (bits[1], bits[-1].split("-")[0]) if len(bits) >= 3 else name


def _label_font(size: int = 13):
    """이름표를 그릴 글꼴. **한글이 나와야 한다.**

    PIL 의 기본 글꼴은 한글을 못 그려서 전부 두부(□)가 된다. 그림을 만들어
    놓고 정작 무슨 글씨체인지 못 읽는 꼴이 났다. 그래서 이 PC 에 이미 깔린
    글꼴을 찾아 쓴다 — 여기 쓰는 것은 **이름표**일 뿐이고, 시험지 글씨는
    손글씨 글꼴로 따로 그린다.
    """
    from PIL import ImageFont

    for spot in ("C:/Windows/Fonts/malgun.ttf",
                 "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                 "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        try:
            return ImageFont.truetype(spot, size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_sample(path: str, fonts_dir: str, per_font: int = 3,
                run: str = "", device: str = "cpu", beams: int = 5) -> None:
    """시험지가 실제로 어떻게 생겼는지 그림 한 장으로 남긴다.

    숫자만 적어 두면 '83%' 가 무엇에 대한 83% 인지 알 수가 없다. 글씨체마다
    몇 줄씩 붙이고, 판을 주면 **그 판이 뭐라 읽었는지와 닮음까지** 같이 적는다.

    **글꼴 파일이 아니라 그림이다.** 글꼴로 글자를 그리는 것은 OFL 도 나눔손글씨
    안내도 허용한다. 재배포하지 않는 것은 `.ttf` 파일 쪽이다(PROVENANCE.md).
    """
    from PIL import Image, ImageDraw

    rows = []
    for name in synth.FONT_TEST:
        try:
            made = sheet(per_font, fonts_dir, only=name.split()[-1])
        except (RuntimeError, FileNotFoundError):
            continue
        for text, cell in made[:per_font]:
            rows.append([_short(name), text, cell, "", None])
    if not rows:
        raise SystemExit("시험지를 못 만들었다. 글꼴 폴더를 보라.")

    if run:
        spot = Path(run)
        vocab = Vocab.load(spot / "vocab.json")
        model = builder.load(str(spot)).to(device).eval()
        for at in range(0, len(rows), 16):
            chunk = rows[at:at + 16]
            pixels = data.to_pixels(torch.stack(
                [data.to_gray(one[2]) for one in chunk]).to(device))
            with torch.no_grad():
                ids = model.generate(pixels, num_beams=beams, max_new_tokens=128,
                                     early_stopping=beams > 1,
                                     prefix_allowed_tokens_fn=letters_only(vocab))
            for one, got in zip(chunk, ids):
                one[3] = unicodedata.normalize("NFC", vocab.decode(got.tolist())).strip()
                one[4] = closeness(one[3], one[1])
        del model

    face = _label_font()
    pad, label = 10, 300
    wide = label + max(one[2].width for one in rows) + pad * 2
    tall = sum(one[2].height + 34 for one in rows) + pad * 2
    made = Image.new("L", (wide, tall), 255)
    draw = ImageDraw.Draw(made)
    y = pad
    for who, text, cell, guess, close in rows:
        draw.text((pad, y), who, fill=70, font=face)
        draw.text((pad, y + 15), "정답  " + text[:30], fill=120, font=face)
        if guess:
            draw.text((pad, y + 30), "읽음  " + guess[:30], fill=120, font=face)
            draw.text((pad, y + 45), "닮음  %.0f%%" % (100 * close), fill=70, font=face)
        made.paste(cell.convert("L"), (label, y))
        y += cell.height + 34
    made.save(path)


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
    ap.add_argument("--per-font", action="store_true",
                    help="글씨체마다 따로 잰다. 합계는 무너진 글씨체를 가린다")
    ap.add_argument("--sample", default="",
                    help="시험지 그림을 이 파일로 남긴다 (글씨체마다 몇 줄씩)")
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

    if args.per_font:
        # 합계는 **무너진 글씨체를 가린다.** 여섯 벌 평균이 83% 여도 그 안에서
        # 20%p 넘게 벌어질 수 있고, 손잡이를 돌릴 때 봐야 하는 것은 낮은 쪽이다.
        pick = Path(best[1]) if best else Path(args.runs[0])
        print("\n글씨체마다 (판 %s, 글씨체당 %d줄)" % (pick.name, args.lines))
        print("%-30s %8s %8s %8s" % ("글씨체", "글자", "줄 통째", "흔들림"))
        rows = []
        for name in synth.FONT_TEST:
            try:
                one = sheet(args.lines, args.fonts, only=name.split()[-1])
            except (RuntimeError, FileNotFoundError):
                print("%-30s (글꼴이 없다)" % _short(name))
                continue
            got = score(pick, one, args.device, args.beams, args.batch)
            rows.append((got["letters"], name))
            print("%-30s %7.1f%% %7.1f%% %7s"
                  % (_short(name), got["letters"], got["whole"],
                     "+-%.1f%%p" % got["band"]))
        if len(rows) > 1:
            rows.sort()
            print("\n가장 어려운 글씨체: %s (%.1f%%)   가장 쉬운 것: %s (%.1f%%)"
                  % (_short(rows[0][1]), rows[0][0], _short(rows[-1][1]), rows[-1][0]))

    if args.sample:
        save_sample(args.sample, args.fonts,
                    run=best[1] if best else args.runs[0],
                    device=args.device, beams=args.beams)
        print("\n시험지 맛보기: %s" % args.sample)


if __name__ == "__main__":
    main()
