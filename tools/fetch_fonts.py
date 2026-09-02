"""한글 **손글씨** 글꼴을 더 모은다. 학습 그림을 그리는 데만 쓴다.

    python tools/fetch_fonts.py --list          # 무엇을 받을지만 보여 준다
    python tools/fetch_fonts.py                 # 받는다

왜 모으는가. 손글씨를 읽는 실력은 **본 손씨의 가짓수**가 거의 다 정한다.
나눔손글씨 109벌은 네이버가 낸 전량이라 더 없다. 그래서 밖에서 더 모은다.

라이선스. OFL(SIL Open Font License)만 받는다. OFL 은 글꼴 파일을 **다시
배포**하는 것을 제한하지, 그 글꼴로 **그림을 그리는 것**은 막지 않는다.
그러니 받은 파일은 이 PC 의 글꼴 폴더에만 두고 **저장소에도 꾸러미에도 넣지
않는다**(나눔손글씨와 같은 규칙이다). 모델 가중치는 글자 모양의 저작물이
아니라 픽셀에서 배운 값이라 라이선스가 옮아붙지 않는다.

받는 곳은 google/fonts 저장소다. 파일이 한자리에 있고 주소가 안 바뀐다.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/google/fonts/main"
API = "https://api.github.com/repos/google/fonts/contents"

# google/fonts 안의 한글 글꼴 가운데 **손으로 쓴 모양**인 것들.
# 붓·펜·연필처럼 획이 이어지거나 삐뚤한 것만 고른다. 네모반듯한 고딕/명조는
# 손글씨가 아니라서 넣으면 오히려 '반듯한 글자'쪽으로 끌린다.
HANDS = [
    ("nanumpenscript", "나눔펜스크립트"),
    ("nanumbrushscript", "나눔브러시스크립트"),
    ("gaegu", "개구"),
    ("gamjaflower", "감자꽃"),
    ("himelody", "하이멜로디"),
    ("poorstory", "푸어스토리"),
    ("singleday", "싱글데이"),
    ("dokdo", "독도"),
    ("eastseadokdo", "동해독도"),
    ("kiranghaerang", "기랑해랑"),
    ("yeonsung", "연성"),
    ("cutefont", "귀여운글꼴"),
    ("hanbit", "한빛"),
    ("sunflower", "해바라기"),
    ("jua", "주아"),
    ("dohyeon", "도현"),
    ("gugi", "구기"),
    ("stylish", "스타일리시"),
    ("songmyung", "송명"),
    ("blackhansans", "검은고딕"),
    ("gowundodum", "고운돋움"),
    ("hahmlet", "함렛"),
    ("nanummyeongjo", "나눔명조"),
    ("nanumgothic", "나눔고딕"),
]

# 손글씨가 아닌 것(고딕·명조·display)은 받되 **쓰지는 않는다.** 왜 받나 하면,
# 실제 서식에는 인쇄된 글자도 같이 찍히기 때문이다 — 다만 그건 딴 일이라
# 지금은 표시만 해 둔다.
NOT_HAND = {"sunflower", "jua", "dohyeon", "gugi", "stylish", "songmyung",
            "blackhansans", "gowundodum", "hahmlet", "nanummyeongjo", "nanumgothic"}


def listing(slug: str) -> list[str]:
    """그 글꼴 폴더에 있는 파일 이름들."""
    for area in ("ofl", "apache", "ufl"):
        try:
            with urllib.request.urlopen(f"{API}/{area}/{slug}", timeout=20) as answer:
                items = json.loads(answer.read().decode("utf-8"))
            return [f"{area}/{slug}/{item['name']}" for item in items
                    if item["name"].lower().endswith((".ttf", ".otf"))]
        except Exception:
            continue
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="한글 손글씨 글꼴을 더 모은다")
    ap.add_argument("--into", default="")
    ap.add_argument("--list", action="store_true", help="받지 않고 목록만")
    ap.add_argument("--all", action="store_true", help="손글씨가 아닌 것까지")
    args = ap.parse_args()
    # 받는 자리도 코드에 안 박는다. `kohandocr.synth.font_folder` 와 같은 차례로
    # 고른다 — 받는 곳과 읽는 곳이 갈라지면 받아 놓고도 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kohandocr.synth import font_folder
    args.into = str(font_folder(args.into))

    into = Path(args.into)
    into.mkdir(parents=True, exist_ok=True)
    got = new = 0
    for slug, korean in HANDS:
        if slug in NOT_HAND and not args.all:
            continue
        paths = listing(slug)
        if not paths:
            print("  %-20s 못 찾음" % korean)
            continue
        for path in paths:
            name = "구글 %s %s" % (korean, path.rsplit("/", 1)[-1])
            target = into / name
            got += 1
            if target.exists():
                continue
            if args.list:
                print("  받을 것: %s" % name)
                continue
            try:
                with urllib.request.urlopen(f"{RAW}/{path}", timeout=60) as answer:
                    body = answer.read()
                target.write_bytes(body)
                new += 1
                print("  받았다: %-46s %5.0f KB" % (name, len(body) / 1024))
            except Exception as bad:
                print("  못 받음: %s (%s)" % (name, bad))
    print("\n찾은 파일 %d개, 새로 받은 것 %d개 -> %s" % (got, new, into))
    print("이 파일들은 **저장소에도 꾸러미에도 넣지 않는다.** 그림 그릴 때만 쓴다.")


if __name__ == "__main__":
    main()
