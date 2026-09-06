"""믿음 문턱(`reader.TRUST_EDGE`)을 **조합 여럿에 대고** 재 본다.

    python tools/trustsweep.py                 # 조합 넷 x 문턱 열 가지
    python tools/trustsweep.py --lines 30      # 글씨체 시험지도 같이 본다
    python tools/trustsweep.py --teams "runs/v20b,runs/v32" "runs/v18-5000,runs/v20b"

왜 만들었나. 2026-09-05 에 지금 조합에서 재 보니 **멀쩡한 손글씨 다섯 줄이
0.69~0.75 로 문턱(0.80) 아래에 걸렸다.** 규정대로 빈칸으로 내보내면 정확도가
0.936 에서 0.774 로 떨어진다. 그런데 `CLAUDE.md` 에 적힌 근거는 "손글씨는
0.96~1.00 이라 사이가 훤히 비어 있다" 였다. 둘 중 하나가 틀렸다.

**문턱은 제품 설정이라 조합 하나만 보고 못 바꾼다.** 사진이 열한 장뿐이라
조합만 바꿔도 넘긴 장이 7~10 장을 오간다(`CLAUDE.md`). 그 폭에 묻히는 차이를
손잡이 덕이라고 착각한 적이 이미 두 번 있다. 그래서 여기서는 **조합을 넷으로
바꿔 가며 같은 방향이 나오는지**만 본다.

무엇을 세나. 문턱은 정확도를 올리는 장치가 **아니다** — 지우면 그 줄은 0 점이
되니 점수는 언제나 내려간다. 문턱이 하는 일은 **틀린 답을 안 내보내는 것**이고,
서식에서 틀린 답은 빈칸보다 나쁘다(사람이 검토할 자리를 못 찾는다). 그러니
세야 할 것은 점수가 아니라 이 셋이다.

    놓침   잘 읽은 줄을 지웠다        <- 손해. 사람이 다시 쳐야 한다
    막음   잘못 읽은 줄을 지웠다      <- 이득. 빈칸이 보이니 사람이 찾아간다
    샘     잘못 읽은 줄이 그냥 나갔다  <- 문턱이 못 잡은 것

좋은 문턱은 **막음이 살아 있으면서 놓침이 0 에 가까운** 자리다. 조합이 바뀌어도
그 자리가 그대로여야 손잡이고, 조합마다 옮겨 다니면 그것은 손잡이가 아니다.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
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

import bench                                             # noqa: E402
import eval_photos as E                                  # noqa: E402
import holdout as H                                      # noqa: E402
from kohandocr import page as photo                      # noqa: E402
from kohandocr import reader as R                        # noqa: E402
from kohandocr import synth                              # noqa: E402

BOOK = ROOT / "runs" / "ENSEMBLE.json"
CARD = ROOT / "runs" / "TRUSTSWEEP.json"

# 재 볼 문턱. 0.0 은 '문턱을 안 쓴다' 이고 지금 쓰는 값은 0.80 이다.
GRID = (0.0, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

# 이 위면 '잘 읽었다'로 센다. 사진 한 장이 넘어야 하는 값과 같은 자리를 쓴다
# (`eval_photos.GOAL`). 다른 자를 쓰면 여기서 나온 '놓침' 이 저기 성적표의
# '넘긴 장' 과 말이 안 맞는다.
GOOD = E.GOAL / 100.0

# 견줄 조합. 지금 쓰는 조합에 **겹치지 않는 것들**을 섞는다. 셋만 보면 우연히
# 같은 방향이 나올 수 있고, 넷이면 하나가 어긋나는 것도 보인다.
SPARE = (
    ["runs/v19-17500", "runs/v10a-15000", "runs/v32"],
    ["runs/v18-5000", "runs/v20b", "runs/v32"],
    ["runs/v20b", "runs/v10a-15000", "runs/v18-5000"],
)


def whole(team: list[str]) -> bool:
    """판이 다 있고 **둘 이상**인가. 판이 하나면 믿음값을 아예 못 낸다."""
    return len(team) >= 2 and all((ROOT / one / "config.json").exists() for one in team)


def pair_at(got: list[str], want: list[str]) -> dict[int, int]:
    """`eval_photos.pair` 와 **같은 짝짓기**를 하되 자리 번호를 준다.

    믿음값은 칸에 붙어 있는데 `E.pair` 는 글월만 돌려준다. 같은 글월을 읽은
    칸이 둘이면 어느 쪽 믿음값인지 알 수가 없다. 그래서 자리로 받는다.

    본문을 베낀 것이라 저쪽이 바뀌면 조용히 갈라진다. 그래서 부를 때마다
    **글월이 저쪽과 같은지 맞춰 본다**(`pair_texts`). 다르면 거기서 멈춘다.
    """
    scores = sorted(
        ((E.closeness(guess, truth), gi, ti) for ti, truth in enumerate(want)
         for gi, guess in enumerate(got)),
        reverse=True,
    )
    taken: set[int] = set()
    chosen: dict[int, int] = {}
    for _, gi, ti in scores:
        if ti in chosen or gi in taken:
            continue
        chosen[ti] = gi
        taken.add(gi)
    return chosen


def pair_texts(got: list[str], want: list[str]) -> dict[int, int]:
    """자리로 짝지은 뒤, 그 자리에서 나온 글월이 `E.pair` 와 같은지 본다."""
    chosen = pair_at(got, want)
    mine = [got[chosen[ti]] if ti in chosen else "" for ti in range(len(want))]
    theirs = [guess for guess, _ in E.pair(got, want)]
    if mine != theirs:
        raise SystemExit("짝짓기가 eval_photos.pair 와 갈라졌다. pair_at 을 맞춰라.")
    return chosen


def look(reader, jpgs: list[Path], truth: dict) -> dict:
    """사진을 읽고 **줄마다 (믿음값, 닮음)** 을 모은다. 문턱은 아직 안 쓴다."""
    rows, spare = [], []
    for jpg in jpgs:
        want = truth.get(jpg.name, {}).get("lines", [])
        if not want or truth[jpg.name].get("gauge"):
            # 「재기만 하는 장」은 뺀다. hand-12 는 어느 판으로 읽어도 27% 이고
            # 열두 칸이 모두 문턱 아래라, 넣으면 어떤 문턱이든 '막음' 이 열둘씩
            # 붙어 자리 고르기가 그 장 하나에 끌려간다.
            continue
        cells = [Image.open(io.BytesIO(one["png"])).convert("L")
                 for one in photo.lines(photo.prepare(jpg.read_bytes()), max_lines=16)]
        said = reader.read_trust(cells)
        got = [text for text, _ in said]
        chosen = pair_texts(got, want)
        for ti, line in enumerate(want):
            at = chosen.get(ti)
            rows.append({
                "photo": jpg.name,
                "trust": None if at is None else said[at][1],
                "close": E.closeness_tight("" if at is None else got[at], line),
            })
        # 정답 줄에 못 붙은 칸 — 잘못 잘린 조각이다. 이것을 지우는 것은
        # 점수에 안 잡히지만 서식에서는 이득이다(없는 칸이 생기지 않는다).
        spare += [said[gi][1] for gi in range(len(got)) if gi not in set(chosen.values())]
    return {"rows": rows, "spare": spare}


def styles(reader, lines: int, fonts: str) -> list[dict]:
    """글씨체 시험지의 믿음값. **문턱의 근거가 여기서 나왔다.**

    `CLAUDE.md` 의 "손글씨는 0.96~1.00" 은 글씨체 시험지에서 잰 값이다.
    시험지는 글꼴에서 바로 찍어 내므로 사진 다듬기·자르기를 안 거친다. 사진이
    같은 분포일 이유가 없는데, 문턱은 그 값을 보고 정해졌다. 그래서 둘을
    나란히 놓고 본다.
    """
    out = []
    for one in synth.FONT_TEST:
        made = H.sheet(lines, fonts, only=one.split()[-1])
        if not made:
            continue
        said = reader.read_trust([image for _, image in made])
        for (text, _), (guess, trust) in zip(made, said):
            out.append({"font": H._short(one), "trust": trust,
                        "close": E.closeness_tight(guess, text)})
    return out


def sweep(rows: list[dict], spare: list[float]) -> list[dict]:
    """문턱마다 놓침·막음·샘을 센다."""
    out = []
    for edge in GRID:
        miss = block = leak = 0
        score = []
        for row in rows:
            gone = row["trust"] is not None and 0 <= row["trust"] < edge
            good = row["close"] >= GOOD
            score.append(0.0 if gone else row["close"])
            if gone and good:
                miss += 1
            elif gone:
                block += 1
            elif not good:
                leak += 1
        out.append({
            "edge": edge,
            "accuracy": sum(score) / len(score),
            "miss": miss, "block": block, "leak": leak,
            "junk": sum(1 for one in spare if 0 <= one < edge),
            "of_junk": len(spare),
        })
    return out


def spread(values: list[float]) -> str:
    """가장 낮은 값 ~ 가운데값. 분포가 겹치는지 한 줄로 본다."""
    kept = sorted(one for one in values if one >= 0)
    if not kept:
        return "잴 수 없음"
    return "%.2f / %.2f / %.2f (최저·1/4·가운데)" % (
        kept[0], kept[len(kept) // 4], statistics.median(kept))


def main() -> None:
    ap = argparse.ArgumentParser(description="믿음 문턱을 조합 여럿에 대고 재 본다")
    ap.add_argument("--teams", nargs="*", default=None,
                    help="조합을 직접 준다. 쉼표로 판을 잇는다")
    ap.add_argument("--lines", type=int, default=0,
                    help="글씨체 시험지도 벌마다 이만큼 본다 (0 이면 사진만)")
    ap.add_argument("--fonts", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.fonts = str(synth.font_folder(args.fonts))

    if args.teams:
        teams = [[one.strip() for one in row.split(",")] for row in args.teams]
    else:
        teams = []
        if BOOK.exists():
            teams.append([one.replace("\\", "/")
                          for one in json.loads(BOOK.read_text(encoding="utf-8"))["members"]])
        teams += [list(one) for one in SPARE]
    teams = [one for one in teams if whole(one)]
    seen, only = set(), []
    for team in teams:                      # 같은 조합을 두 번 재지 않는다
        key = tuple(sorted(team))
        if key not in seen:
            seen.add(key)
            only.append(team)
    teams = only
    if len(teams) < 3:
        raise SystemExit("조합이 셋도 안 된다. 하나만 보고 문턱을 정하면 안 된다.")

    truth = json.loads((APP / "data/eval/labels.json").read_text(encoding="utf-8"))
    jpgs = sorted((APP / "data/eval/samples").glob("hand-*.jpg"))

    print("지금 문턱 %.2f   '잘 읽었다' 기준 %.2f   조합 %d가지"
          % (R.TRUST_EDGE, GOOD, len(teams)))

    cards = []
    for team in teams:
        name = " + ".join(one.replace("runs/", "") for one in team)
        print("\n" + "=" * 78)
        print(name)
        print("=" * 78)
        reader = bench.build(team, args.device)
        got = look(reader, jpgs, truth)
        table = sweep(got["rows"], got["spare"])

        print("  채점 줄 %d개, 정답에 못 붙은 칸 %d개"
              % (len(got["rows"]), len(got["spare"])))
        print("  사진 줄의 믿음값  %s" % spread([r["trust"] for r in got["rows"]
                                            if r["trust"] is not None]))
        seen_styles = []
        if args.lines:
            seen_styles = styles(reader, args.lines, args.fonts)
            print("  글씨체 줄의 믿음값 %s   (%d줄)"
                  % (spread([r["trust"] for r in seen_styles]), len(seen_styles)))

        print("\n  %6s %8s %7s %7s %7s %9s"
              % ("문턱", "정확도", "놓침", "막음", "샘", "조각지움"))
        for row in table:
            here = " <- 지금" if abs(row["edge"] - R.TRUST_EDGE) < 1e-9 else ""
            print("  %6.2f %8.3f %7d %7d %7d %5d/%-4d%s"
                  % (row["edge"], row["accuracy"], row["miss"], row["block"],
                     row["leak"], row["junk"], row["of_junk"], here))

        cards.append({"team": team, "name": name, "table": table,
                      "rows": got["rows"], "spare": got["spare"],
                      "styles": seen_styles})

    # ── 조합을 건너서 같은 방향인가 ────────────────────────────────
    print("\n" + "=" * 78)
    print("조합 %d가지를 건너서 본다 (놓침은 손해, 막음은 이득)" % len(cards))
    print("=" * 78)
    print("  %6s %26s %26s" % ("문턱", "놓친 줄 (조합별)", "막은 줄 (조합별)"))
    for at, edge in enumerate(GRID):
        miss = [card["table"][at]["miss"] for card in cards]
        block = [card["table"][at]["block"] for card in cards]
        here = " <- 지금" if abs(edge - R.TRUST_EDGE) < 1e-9 else ""
        print("  %6.2f %26s %26s%s"
              % (edge, " ".join("%2d" % one for one in miss),
                 " ".join("%2d" % one for one in block), here))

    # 손해가 하나도 없으면서 이득이 가장 큰 자리. 조합마다 따로 본다.
    print("\n  조합마다 '놓침 0' 을 지키는 가장 높은 문턱과 그때 막는 줄:")
    safe = []
    for card in cards:
        rows = [row for row in card["table"] if row["miss"] == 0]
        top = max(rows, key=lambda row: row["edge"])
        safe.append(top["edge"])
        print("    %-44s %.2f  (막음 %d, 샘 %d)"
              % (card["name"], top["edge"], top["block"], top["leak"]))
    print("\n  조합을 건너서 겹치는 자리: %.2f 아래 (가장 낮은 조합이 %.2f)"
          % (min(safe), min(safe)))

    CARD.write_text(json.dumps({
        "when": time.strftime("%Y-%m-%d %H:%M"),
        "edge_now": R.TRUST_EDGE, "good": GOOD, "grid": list(GRID),
        "style_lines": args.lines,
        "cards": cards,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n성적표: %s" % CARD)


if __name__ == "__main__":
    main()
