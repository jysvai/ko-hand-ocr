"""자체 벤치마크. **얼마나 맞고 얼마나 빠른가**를 한 자리에서 잰다.

    python tools/bench.py                  # 제품 설정(ENSEMBLE.json)으로
    python tools/bench.py --all            # 제품 / 판 하나 / 그리디 셋을 견준다
    python tools/bench.py --repeat 3       # 시간은 세 번 재서 가운데값
    python tools/bench.py --page-lines 30  # 한 쪽을 몇 줄로 셈할 것인가

`tools/verify.py` 와 무엇이 다른가. verify 는 **맞추는 쪽**만 본다 — 글씨체
시험지를 만들어 조합의 성적표를 낸다. 여기는 **앱이 실제로 지나는 길**을
그대로 지나며 시간을 같이 잰다. 사진 바이트를 받아 줄을 자르고(`page`)
판독기(`reader.Reader`)로 읽는다. 앱과 다른 길로 잰 속도는 앱의 속도가 아니다.

시간은 두 토막으로 나눠 찍는다.

    자르기   `page.prepare` + `page.lines`   — CPU 만 쓴다
    읽기     `Reader.read_trust`             — GPU 를 쓴다

나누는 까닭은 고치는 자리가 다르기 때문이다. 자르기가 느리면 판을 아무리
줄여도 안 빨라지고, 읽기가 느리면 조합을 줄이는 것 말고는 길이 없다.

**쪽당 초는 시험지 밀도에 딸려 있다.** 우리 시험 사진은 한 장에 서너 줄인데
남들이 말하는 「한 쪽」은 서른 줄짜리 서류 한 장이다. 그래서 잰 그대로의
쪽당 초와 함께, 줄당 초로 환산한 값을 같이 적는다. 환산 없이 견주면
우리 쪽이 30배 유리한 숫자가 나온다.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
# 시험 자료(data/eval)가 저장소 안에 있으면 그쪽이다.
APP = ROOT if (ROOT / "data" / "eval").exists() else ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "tools"))

from transformers import logging as _hf_logging          # noqa: E402

_hf_logging.set_verbosity_error()

import eval_photos as E                                  # noqa: E402
from kohandocr import model as builder                   # noqa: E402
from kohandocr import page as photo                      # noqa: E402
from kohandocr import reader as R                        # noqa: E402
from kohandocr.reader import Reader                      # noqa: E402
from kohandocr.vocab import Vocab                        # noqa: E402

BOOK = ROOT / "runs" / "ENSEMBLE.json"
CARD = ROOT / "runs" / "BENCH.json"

# 견줄 세 가지. 이름 -> (판을 몇 벌 쓰나, 빔 너비)
#   판을 하나만 쓰면 `Reader` 가 흔들지 않는 길로 간다(`also` 가 비므로).
#   그래서 「판 하나」는 흔들기까지 같이 빠진 값이다.
MODES: dict[str, tuple[int | None, int]] = {
    "제품(앙상블)": (None, R.BEAMS),
    "판 하나": (1, R.BEAMS),
    "판 하나+그리디": (1, 1),
}


def busy() -> list[str]:
    """지금 GPU 를 쥐고 있는 우리 프로세스. **시간을 재기 전에 반드시 본다.**

    학습이 도는 중에 재면 같은 조합이 1.23초에서 1.92초로 나온다. 숫자가
    틀린 것이 아니라 **다른 것을 잰 것**인데, 성적표에는 똑같이 적힌다.
    한 번 그렇게 덮어쓰면 나중에 어느 쪽이 맞는지 알 길이 없다.
    """
    try:
        got = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'kohandocr.train|loop.py' } | "
             "ForEach-Object { $_.ProcessId }"],
            capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    return [one for one in got.stdout.split() if one.strip()]


def sync(device: str) -> None:
    """GPU 는 일을 미뤄 놓고 먼저 돌아온다. 안 맞추면 읽기 시간이 0 으로 나온다."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def build(members: list[str], device: str) -> Reader:
    """판 목록으로 판독기를 만든다. `also/` 폴더를 만들지 않고 바로 올린다.

    `tools/install.py` 는 앱 자리에 곁들이는 판을 **복사해서** 깐다. 여기서
    그러면 120MB 짜리를 셋 복사해 놓고 한 번 재고 버리게 된다. 올리는 결과는
    같으니 메모리에서 바로 엮는다.
    """
    first = Path(members[0])
    one = Reader.__new__(Reader)
    one.vocab = Vocab.load(first / "vocab.json")
    one.model = builder.load(first).to(device).eval()
    one.device = device
    one.skip = 1
    one._letters_fn = None
    one.also = []
    for extra in members[1:]:
        extra = Path(extra)
        one.also.append((Vocab.load(extra / "vocab.json"),
                         builder.load(extra).to(device).eval()))
    return one


def one_pass(reader: Reader, jpgs: list[Path], beams: int, device: str) -> dict:
    """사진을 한 바퀴 돌며 자르고 읽는다. 시간과 읽은 값을 같이 준다."""
    cut_s, read_s, said, boxes = {}, {}, {}, {}
    for jpg in jpgs:
        raw = jpg.read_bytes()

        sync(device)
        start = time.perf_counter()
        png = photo.prepare(raw)
        cells = [Image.open(io.BytesIO(cell["png"])).convert("L")
                 for cell in photo.lines(png, max_lines=16)]
        cut_s[jpg.name] = time.perf_counter() - start

        sync(device)
        start = time.perf_counter()
        got = reader.read_trust(cells, beams=beams)
        sync(device)
        read_s[jpg.name] = time.perf_counter() - start

        said[jpg.name] = got
        boxes[jpg.name] = cells
    return {"cut": cut_s, "read": read_s, "said": said, "cells": boxes}


def score(said: dict[str, list[tuple[str, float]]], truth: dict) -> dict:
    """읽은 값을 정답과 견준다. `tools/eval_photos.py` 와 같은 자를 쓴다.

    **「재기만 하는 장」은 셈에서 뺀다.** `labels.json` 에 `gauge: true` 로
    적은 장은 손글씨 판독의 목표가 아니라 한계를 보려고 넣어 둔 것이다
    (지금은 hand-12 — 저해상도 인쇄 서식에 친필 서명 둘). 어느 판으로 읽어도
    27% 인데다 줄이 열다섯이라, 섞으면 열한 장 평균 93% 가 통째로 73% 로
    끌려 내려간다. 재기는 재되 따로 적는다.
    """
    per_photo, per_line, per_line_tight, gated = {}, [], [], []
    exact = rows = unsure = 0
    gauge_photo, gauge_line = {}, []
    for name, reads in said.items():
        want = truth.get(name, {}).get("lines", [])
        if not want:
            continue
        only_gauge = bool(truth[name].get("gauge"))
        got = [text for text, _ in reads]
        # 앱이 실제로 내보내는 답. 믿음값이 문턱 아래면 **빈칸**이다
        # (`reader.TRUST_EDGE`). 서식에서 틀린 답은 빈칸보다 나쁘기 때문이다.
        held = [("" if 0 <= trust < R.TRUST_EDGE else text) for text, trust in reads]
        unsure += sum(1 for _, trust in reads
                      if 0 <= trust < R.TRUST_EDGE)
        tight_scores = []
        for guess, line in E.pair(got, want):
            tight_scores.append(E.closeness_tight(guess, line))
            if only_gauge:
                gauge_line.append(tight_scores[-1])
                continue
            per_line.append(E.closeness(guess, line))
            per_line_tight.append(tight_scores[-1])
            exact += E.tight(guess) == E.tight(line)
            rows += 1
        if not only_gauge:
            gated += [E.closeness_tight(guess, line)
                      for guess, line in E.pair(held, want)]
        mean = sum(tight_scores) / len(tight_scores)
        (gauge_photo if only_gauge else per_photo)[name] = mean
    return {"photos": per_photo, "gauge_photos": gauge_photo,
            "lines": per_line, "tight": per_line_tight, "gauge_lines": gauge_line,
            "gated": gated, "exact": exact, "rows": rows, "unsure": unsure}


def run(name: str, members: list[str], beams: int, jpgs: list[Path],
        truth: dict, device: str, repeat: int, wide: bool = False) -> dict:
    print("\n" + "=" * 66)
    print("%s   판 %d벌, 빔 %d" % (name, len(members), beams))
    print("=" * 66)

    start = time.perf_counter()
    reader = build(members, device)
    load_s = time.perf_counter() - start
    print("  판 올리기 %.1f초 (한 번만 낸다. 아래 숫자에는 안 넣는다)" % load_s)

    # 첫 판독은 CUDA 커널을 깎느라 몇 배 느리다. 그것까지 재면 쪽당 초가
    # 사진 수에 딸려 흔들린다. 한 장을 버리는 셈 치고 먼저 돌린다.
    one_pass(reader, jpgs[:1], beams, device)

    laps = [one_pass(reader, jpgs, beams, device) for _ in range(repeat)]
    got = laps[0]
    cut = {n: statistics.median(lap["cut"][n] for lap in laps) for n in got["cut"]}
    read = {n: statistics.median(lap["read"][n] for lap in laps) for n in got["read"]}

    card = score(got["said"], truth)
    graded = list(card["photos"])
    marks = dict(card["photos"], **card["gauge_photos"])

    print("\n  %-12s %5s %8s %8s %8s   %7s" %
          ("사진", "칸", "자르기", "읽기", "합계", "닮음"))
    for jpg in jpgs:
        n = jpg.name
        if n not in marks:
            continue
        tag = "     " if n in graded else "재기만"
        print("  %-12s %5d %7.2fs %7.2fs %7.2fs   %6.1f%% %s"
              % (n.replace(".jpg", ""), len(got["cells"][n]), cut[n], read[n],
                 cut[n] + read[n], 100 * marks[n], tag))

    keep = graded
    cells = sum(len(got["cells"][n]) for n in keep)
    cut_all = sum(cut[n] for n in keep)
    read_all = sum(read[n] for n in keep)
    pages = len(keep)
    chars = sum(len("".join(truth[n]["lines"])) for n in keep)

    out = {
        "name": name,
        "members": [m.replace("\\", "/") for m in members],
        "beams": beams,
        "repeat": repeat,
        "pages": pages,
        "cells": cells,
        "chars": chars,
        "load_seconds": round(load_s, 2),
        "cut_per_page": cut_all / pages,
        "read_per_page": read_all / pages,
        "seconds_per_page": (cut_all + read_all) / pages,
        "seconds_per_line": read_all / cells,
        "chars_per_second": chars / (cut_all + read_all),
        "accuracy": sum(card["tight"]) / len(card["tight"]),
        "accuracy_strict": sum(card["lines"]) / len(card["lines"]),
        "accuracy_by_photo": sum(card["photos"].values()) / len(card["photos"]),
        "accuracy_gated": (sum(card["gated"]) / len(card["gated"])
                           if card["gated"] else None),
        "exact_lines": card["exact"],
        "of_lines": card["rows"],
        "unsure_cells": card["unsure"],
        "per_photo": dict(card["photos"]),
        "gauge_photos": dict(card["gauge_photos"]),
        "worst_photo": min((card["photos"][n], n) for n in keep)[1],
        "worst_score": min(card["photos"].values()),
    }
    print("\n  정확도 %.3f (줄 단위)   %.3f (사진 평균)   줄 통째로 맞은 것 %d/%d"
          % (out["accuracy"], out["accuracy_by_photo"], out["exact_lines"],
             out["of_lines"]))
    print("  가장 나쁜 장 %s %.1f%%   못 읽었다고 물러선 칸 %d개%s"
          % (out["worst_photo"].replace(".jpg", ""), 100 * out["worst_score"],
             out["unsure_cells"],
             ("   -> 그 칸을 빈칸으로 내보내면 %.3f"
              % out["accuracy_gated"]) if out["accuracy_gated"] is not None
             and out["unsure_cells"] else ""))

    if wide:
        print("\n  한 번에 몇 줄씩 넣나 (줄당 비용이 얼마나 내려가나)")
        out["throughput"] = throughput(
            reader, [one for n in keep for one in got["cells"][n]], beams, device,
            repeat)
    return out


def throughput(reader: Reader, cells: list, beams: int, device: str, repeat: int,
               sizes: tuple[int, ...] = (1, 3, 8, 16, 32)) -> list[dict]:
    """**한 번에 몇 줄을 넣느냐**로 줄당 비용이 얼마나 달라지나.

    우리 시험 사진은 한 장에 서너 줄이다. 남들이 말하는 「한 쪽」은 서른 줄이다.
    서너 줄로 잰 줄당 초에 30을 곱해 놓고 「서른 줄짜리 쪽은 이만큼」이라고
    하면 **틀린다** — GPU 는 서너 줄로는 놀고 있어서, 묶음을 키우면 줄당 값이
    내려간다. 그 내려가는 폭을 재 두어야 환산이 짐작이 아니게 된다.

    줄이 모자라면 있는 칸을 돌려 쓴다. 같은 그림이라도 도는 일의 양은 같다.

    **한 번만 재면 못 쓴다.** 한 번씩 재 봤더니 16줄이 32줄보다 줄당 값이
    싸게 나왔는데, 이것은 순서가 뒤집힌 것이 아니라 잰 값이 흔들린 것이었다.
    이 값이 「서른 줄짜리 쪽은 몇 초」의 근거가 되므로 여러 번 재서 가운데를
    쓴다.
    """
    out = []
    for size in sizes:
        batch = [cells[i % len(cells)] for i in range(size)]
        try:
            reader.read_trust(batch[:1], beams=beams)      # 몸풀기
            spent = []
            for _ in range(repeat):
                sync(device)
                start = time.perf_counter()
                reader.read_trust(batch, beams=beams)
                sync(device)
                spent.append(time.perf_counter() - start)
            took = statistics.median(spent)
        except RuntimeError as why:
            # 8GB 라 큰 묶음은 못 든다. 그때 벤치 전체가 죽으면 앞서 잰 값까지
            # 잃는다. 못 든 자리는 빈칸으로 적고 넘어간다. 메모리 말고 다른
            # 까닭이면 조용히 삼키지 않는다 — 잘못된 숫자보다 낫다.
            if "out of memory" not in str(why).lower():
                raise
            torch.cuda.empty_cache()
            print("  %3d줄 묶음   메모리가 모자라다" % size)
            out.append({"batch": size, "seconds": None, "per_line": None})
            continue
        out.append({"batch": size, "seconds": took, "per_line": took / size})
        print("  %3d줄 묶음   %6.2fs   줄당 %.3fs" % (size, took, took / size))
    return out


def page_seconds(card: dict, n: int) -> tuple[float, str]:
    """`n` 줄짜리 서류 한 쪽에 드는 초. 무엇으로 환산했는지도 같이 준다.

    묶음별로 재 뒀으면(`--throughput`) **n 에 가장 가까운 묶음**의 줄당 초를
    쓴다. 안 쟀으면 사진 그대로의 줄당 초를 쓰는데, 그것은 서너 줄짜리 묶음의
    값이라 큰 쪽을 **비싸게** 잡는다. 어느 쪽으로 셈했는지 안 적으면 나중에
    두 숫자를 섞어 놓고 견주게 된다.
    """
    rows = [one for one in (card.get("throughput") or []) if one["per_line"]]
    if rows:
        fit = min(rows, key=lambda one: (abs(one["batch"] - n), -one["batch"]))
        return card["cut_per_page"] + fit["per_line"] * n, "%d줄 묶음 실측" % fit["batch"]
    return (card["cut_per_page"] + card["seconds_per_line"] * n,
            "사진 그대로(서너 줄 묶음)의 줄당 초")


def main() -> None:
    ap = argparse.ArgumentParser(description="맞는 정도와 빠르기를 같이 잰다")
    ap.add_argument("runs", nargs="*", help="판 폴더. 안 주면 ENSEMBLE.json")
    ap.add_argument("--all", action="store_true", help="제품/판하나/그리디 셋을 견준다")
    ap.add_argument("--repeat", type=int, default=3, help="시간을 몇 번 재서 가운데값을 쓸까")
    ap.add_argument("--page-lines", type=int, default=30,
                    help="서류 한 쪽을 몇 줄로 셈할까 (환산용)")
    ap.add_argument("--throughput", action="store_true",
                    help="묶음 크기별 줄당 초까지 잰다. 쪽 환산을 짐작 대신 실측으로")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--anyway", action="store_true",
                    help="학습이 도는 중에도 잰다. 시간은 못 쓰고 정확도만 쓸 때")
    args = ap.parse_args()

    # 학습과 GPU 를 나눠 쓰면 **다른 것을 재게 된다.** 성적표에는 똑같이 적히니
    # 여기서 막는다. 정확도만 필요하면 --anyway 로 넘어갈 수 있고, 그때는
    # 성적표에 '나눠 썼다'고 적어 둔다.
    others = busy()
    if others and not args.anyway:
        raise SystemExit(
            "학습이 돌고 있다 (pid %s). GPU 를 나눠 쓰면 시간이 딴 값이 된다.\n"
            "  세우고 재거나, 정확도만 볼 것이면 --anyway 를 준다." % ", ".join(others))

    team = args.runs
    if not team:
        if not BOOK.exists():
            raise SystemExit("runs/ENSEMBLE.json 이 없다. 판을 직접 주라.")
        team = json.loads(BOOK.read_text(encoding="utf-8"))["members"]
    team = [t.replace("\\", "/") for t in team]

    truth = json.loads((APP / "data/eval/labels.json").read_text(encoding="utf-8"))
    jpgs = sorted((APP / "data/eval/samples").glob("hand-*.jpg"))
    if not jpgs:
        raise SystemExit("시험 사진이 없다: data/eval/samples")

    gear = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "CPU"
    print("자: %s / torch %s / 사진 %d장 / 시간은 %d번 재서 가운데값"
          % (gear, torch.__version__, len(jpgs), args.repeat))

    plans = []
    if args.all:
        for name, (many, beams) in MODES.items():
            plans.append((name, team if many is None else team[:many], beams))
    else:
        plans.append(("제품(앙상블)", team, R.BEAMS))

    cards = [run(name, members, beams, jpgs, truth, args.device, args.repeat,
                 args.throughput)
             for name, members, beams in plans]

    # ── 한 자리에 모아 놓고 본다 ───────────────────────────────────
    n = args.page_lines
    print("\n" + "=" * 88)
    print("%-16s %7s %9s %8s %8s %9s %11s" %
          ("", "정확도", "가장나쁜장", "초/쪽", "초/줄", "글자/초",
           "%d줄 쪽 환산" % n))
    print("-" * 88)
    for c in cards:
        c["page_seconds"], c["page_basis"] = page_seconds(c, n)
        print("%-16s %7.3f %8.3f %7.2fs %7.3fs %8.1f %10.2fs" %
              (c["name"], c["accuracy"], c["worst_score"],
               c["seconds_per_page"], c["seconds_per_line"],
               c["chars_per_second"], c["page_seconds"]))
    print("-" * 88)
    print("정확도 = 자모 닮음(띄어쓰기 뺀 값), 0~1. 사진 %d장 %d줄."
          % (cards[0]["pages"], cards[0]["of_lines"]))
    print("초/쪽 = 우리 시험 사진 그대로(한 장 평균 %.1f줄). 옆 칸은 %d줄짜리"
          " 서류 한 쪽으로 환산한 값이고, 줄당 초는 %s 이다."
          % (cards[0]["cells"] / cards[0]["pages"], n, cards[0]["page_basis"]))

    best = cards[0]
    print("\n%s 로 %d줄짜리 200쪽을 읽으면 %.0f초 (%.1f분)."
          % (best["name"], n, best["page_seconds"] * 200,
             best["page_seconds"] * 200 / 60))

    keep = {
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "gear": gear,
        "torch": torch.__version__,
        # 시간을 믿어도 되는 값인지. 참이면 학습과 GPU 를 나눠 쓰며 잰 것이라
        # **시간 칸은 못 쓴다.** 이 표시가 없으면 나중에 두 벌을 섞어 놓는다.
        "shared_gpu": bool(others),
        "page_lines_assumed": n,
        "cards": cards,
    }
    CARD.write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n성적표: %s" % CARD)


if __name__ == "__main__":
    main()
