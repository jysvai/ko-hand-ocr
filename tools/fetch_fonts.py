"""한글 **손글씨** 글꼴을 더 모은다. 학습 그림을 그리는 데만 쓴다.

    python tools/fetch_fonts.py --list          # 무엇을 받을지만 보여 준다
    python tools/fetch_fonts.py                 # 받는다

왜 모으는가. 손글씨를 읽는 실력은 **본 손씨의 가짓수**가 거의 다 정한다.
나눔손글씨 109벌은 네이버가 낸 전량이라 더 없다. 그래서 밖에서 더 모은다.

라이선스. 글꼴로 **그림을 그리는 것**을 허용하는 것만 받는다 — OFL, 공공누리
1유형, 그리고 나눔손글씨처럼 '누구나 무료로(상업적으로도) 쓰되 팔지 말라'는
자체 약관. 이런 약관은 글꼴 파일을 **다시 배포**하는 것을 제한하지 그림을
그리는 것은 막지 않는다. 그러니 받은 파일은 이 PC 의 글꼴 폴더에만 두고
**저장소에도 꾸러미에도 넣지 않는다.** 모델 가중치는 글자 모양의 저작물이
아니라 픽셀에서 배운 값이라 라이선스가 옮아붙지 않는다. 벌마다의 근거는
`fonts_extra.json` 과 PROVENANCE.md 에 있다.

받는 곳은 셋이다: google/fonts 저장소, CLOVA 의 나눔손글씨 명단, 그리고
`fonts_extra.json` 에 적은 주소(어비 공식 사이트, fonts-archive, 눈누 CDN).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/google/fonts/main"
API = "https://api.github.com/repos/google/fonts/contents"

# 나눔손글씨 109벌. clova.ai/handwriting 의 목록 화면이 읽는 명단이고, 벌마다
# 내려받기 주소가 들어 있다. **손으로 받지 않는다** — 예전에는 zip 을 받아
# %TEMP% 에 풀고 거기로 링크를 걸어 두었는데, 임시 폴더가 비워지면서 글꼴이
# 통째로 사라졌다. 고리를 다시 띄우려던 날 알았다. 이제는 이 한 줄로 되살린다.
NANUM = "https://clova.ai/static/font.json"

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

# 구글·나눔 **밖**에서 받는 손글씨 글꼴. 명단은 옆의 `fonts_extra.json` 이고,
# 줄마다 한글 이름·라이선스·**그 라이선스를 어디서 확인했나**·주소가 있다.
#
# 고른 기준(2026-09-17, 후보 386벌 -> 302벌). 나눔손글씨가 이미 선 자리에
# 맞춘다 — 나눔도 OFL 이 아니라 '누구나 무료로 쓰되 팔지 말라'는 자체 약관이다.
#
#   넣는다   OFL / 공공누리 1유형 / 원문에 상업적 이용까지 허용한다고 적힌 자체 약관
#   뺀다     공공누리 3유형·CC-BY-ND (변경 금지)
#            약관에 **장평·기울기 변형 금지**가 적힌 것 — 합성이 획을 깎고 기울인다
#            **형식 변환·재배포 금지**, 지정 사이트 밖 배포 금지가 적힌 것
#            (교보손글씨·ACC 는 공식 TTF 가 손으로만 받아져서 웹폰트 사본뿐이다)
#            반듯한 글꼴(borderline 56벌) — 손글씨가 아니면 반듯한 쪽으로 끌린다
#            시험용 여섯 벌과 같은 디자인(독도체=동해독도, 배민 기랑해랑 등)
#
# 주소가 .zip 이면 안의 글꼴 파일을 다 꺼낸다(굵기가 여럿이면 여럿 다 쓴다).
EXTRA_LIST = Path(__file__).resolve().parent / "fonts_extra.json"
FONT_FILES = (".ttf", ".otf", ".woff", ".woff2")


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


# ── 시험용 글씨체와 **같은 디자인**은 학습에 못 들어가게 ────────────────
#
# 시험용 여섯 벌(`synth.FONT_TEST`)은 이름으로 뺀다. 그런데 같은 디자인이
# **다른 이름으로** 들어오면 이름 거르기를 그냥 지나간다. 나눔펜스크립트가
# 그렇다 — 네이버 글꼴 꾸러미에는 '나눔손글씨 펜'(NanumPen.ttf)으로 들어 있다.
# 그것이 학습에 섞이면 시험지가 **본 글씨체**를 재게 되고, 숫자는 오르는데
# 그 숫자가 아무 뜻이 없어진다. 조용히 오르니 알아챌 길도 없다.
#
# 그래서 이름이 아니라 **그림으로** 견준다. 같은 글월을 그려 크기를 맞추고
# 겹치는 넓이를 잰다(IoU). 실측(2026-09-17, 학습 글꼴 117벌 x 시험용 6벌):
# 서로 다른 글꼴끼리는 가장 닮은 짝이 0.445(개구 Bold ~ 감자꽃), 가운데값
# 0.245 다. 같은 파일끼리는 1.0 이다. 0.7 이면 양쪽에서 넉넉히 떨어져 있다.
TWIN = 0.7
TWIN_PROBE = "가나다 손글씨 확인 부서장 전자금융 흙빛 쫓겨"


def _shape(path: Path):
    """글월을 그려 낱말마다 24px 높이로 맞춘 흑백 조각들. 못 그리면 None."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(path), 64)
    out = []
    for word in TWIN_PROBE.split():
        canvas = Image.new("L", (64 * len(word) + 40, 110), 0)
        ImageDraw.Draw(canvas).text((10, 10), word, font=font, fill=255)
        box = canvas.getbbox()
        if box is None:
            return None
        small = canvas.crop(box).resize((24 * len(word), 24))
        out.append(bytes(1 if v > 100 else 0 for v in small.tobytes()))
    return out


def twin_of(path: Path, into: Path) -> tuple[str, float]:
    """시험용 글꼴 가운데 가장 닮은 것과 그 닮음. 시험용이 없으면 ('', 0)."""
    from kohandocr.synth import FONT_TEST

    try:
        mine = _shape(path)
    except OSError:
        return "", 0.0
    if mine is None:
        return "", 0.0
    best = ("", 0.0)
    for name in FONT_TEST:
        other = into / name
        if not other.exists() or other.name == path.name:
            continue
        theirs = _shape(other)
        inter = union = 0
        for a, b in zip(mine, theirs or []):
            inter += sum(x & y for x, y in zip(a, b))
            union += sum(x | y for x, y in zip(a, b))
        score = inter / max(union, 1)
        if score > best[1]:
            best = (name, score)
    return best


def settle(body: bytes, target: Path) -> bool:
    """받은 글꼴을 자리에 놓는다. 시험용과 같은 디자인이면 `twins/` 로 비킨다.

    **`.part` 로 먼저 쓰고 이름을 바꾼다.** 학습 일꾼은 켤 때 `*.ttf` 를
    훑는데, 받는 도중인 반쪽 파일을 집으면 FreeType 이 못 열고 `stroke.json`
    에 '못 쓴다'로 적힌다. 그 표는 다음부터 다시 안 재므로 **영영 빠진다.**
    """
    from kohandocr.synth import FONT_TEST

    into = target.parent
    part = target.with_name(target.name + ".part")
    part.write_bytes(body)
    if target.name not in FONT_TEST:
        like, score = twin_of(part, into)
        if score >= TWIN:
            aside = into / "twins"
            aside.mkdir(exist_ok=True)
            part.replace(aside / target.name)
            print("  비켜 둠: %-44s 시험용 %s 와 닮음 %.2f"
                  % (target.name, like, score))
            return False
    part.replace(target)
    return True


def fetch(url: str, target: Path, show: bool) -> bool:
    """한 파일을 받는다. 이미 있으면 안 받는다. 받았으면 참."""
    if target.exists() or (target.parent / "twins" / target.name).exists():
        return False
    if show:
        print("  받을 것: %s" % target.name)
        return False
    try:
        body = _get(url)
        if not settle(body, target):
            return False
        print("  받았다: %-46s %5.0f KB" % (target.name, len(body) / 1024))
        return True
    except Exception as bad:
        print("  못 받음: %s (%s)" % (target.name, bad))
        return False


def _get(url: str, timeout: int = 60) -> bytes:
    """주소 하나를 받는다.

    주소에 한글과 빈칸이 들어 있으면(`나눔손글씨 가람연꽃.ttf`) 그대로 넘길 때
    urllib 이 ASCII 로 못 바꿔서 죽는다. jsDelivr 는 `@판` 을, 이미 %-로 싼
    주소는 `%` 를 그대로 둬야 한다. 몇몇 곳은 User-Agent 가 없으면 막는다.
    """
    quoted = urllib.parse.quote(url, safe=":/@%?=&")
    ask = urllib.request.Request(quoted, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(ask, timeout=timeout) as answer:
        return answer.read()


def extra(into: Path, show: bool) -> tuple[int, int]:
    """`fonts_extra.json` 을 받는다. 파일 이름은 `그밖 <한글이름> <원래 파일 이름>`."""
    import io
    import zipfile

    got = new = 0
    for row in json.loads(EXTRA_LIST.read_text(encoding="utf-8")):
        korean, url = row["korean"], row["url"]
        tail = urllib.parse.unquote(url.rsplit("/", 1)[-1].split("?")[0])
        if not tail.lower().endswith(".zip"):
            got += 1
            new += fetch(url, into / ("그밖 %s %s" % (korean, tail)), show)
            continue
        if show:
            print("  받을 것: %s (zip)" % korean)
            continue
        try:
            packed = zipfile.ZipFile(io.BytesIO(_get(url, timeout=120)))
        except Exception as bad:
            print("  못 받음: %s (%s)" % (korean, bad))
            continue
        for member in packed.namelist():
            name = member.rsplit("/", 1)[-1]
            if not name.lower().endswith(FONT_FILES) or name.startswith("._"):
                continue
            target = into / ("그밖 %s %s" % (korean, name))
            got += 1
            if target.exists() or (into / "twins" / target.name).exists():
                continue
            body = packed.read(member)
            if settle(body, target):
                new += 1
                print("  받았다: %-46s %5.0f KB" % (target.name, len(body) / 1024))
    return got, new


def nanum(into: Path, show: bool) -> tuple[int, int]:
    """나눔손글씨 109벌. 파일 이름은 CLOVA 가 붙인 그대로(`나눔손글씨 <이름>.ttf`)."""
    with urllib.request.urlopen(NANUM, timeout=30) as answer:
        rows = json.loads(answer.read().decode("utf-8"))["all"]
    got = new = 0
    for row in rows:
        got += 1
        new += fetch(row["url"], into / ("나눔손글씨 %s.ttf" % row["name"]), show)
    return got, new


def main() -> None:
    ap = argparse.ArgumentParser(description="한글 손글씨 글꼴을 더 모은다")
    ap.add_argument("--into", default="")
    ap.add_argument("--list", action="store_true", help="받지 않고 목록만")
    ap.add_argument("--all", action="store_true", help="손글씨가 아닌 것까지")
    ap.add_argument("--no-nanum", action="store_true", help="나눔손글씨 109벌은 건너뛴다")
    args = ap.parse_args()
    # 받는 자리도 코드에 안 박는다. `kohandocr.synth.font_folder` 와 같은 차례로
    # 고른다 — 받는 곳과 읽는 곳이 갈라지면 받아 놓고도 못 찾는다.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kohandocr.synth import font_folder
    args.into = str(font_folder(args.into))

    into = Path(args.into)
    into.mkdir(parents=True, exist_ok=True)
    got = new = 0
    # 구글 것을 **먼저** 받는다. 시험용 여섯 벌이 여기서 오는데, 뒤에 받는
    # 글꼴마다 그 여섯과 견주어 같은 디자인을 걸러 내기 때문이다(`settle`).
    for slug, korean in HANDS:
        if slug in NOT_HAND and not args.all:
            continue
        paths = listing(slug)
        if not paths:
            print("  %-20s 못 찾음" % korean)
            continue
        for path in paths:
            name = "구글 %s %s" % (korean, path.rsplit("/", 1)[-1])
            got += 1
            new += fetch(f"{RAW}/{path}", into / name, args.list)
    if not args.no_nanum:
        more, fresh = nanum(into, args.list)
        got, new = got + more, new + fresh
    more, fresh = extra(into, args.list)
    got, new = got + more, new + fresh
    print("\n찾은 파일 %d개, 새로 받은 것 %d개 -> %s" % (got, new, into))
    print("이 파일들은 **저장소에도 꾸러미에도 넣지 않는다.** 그림 그릴 때만 쓴다.")


if __name__ == "__main__":
    main()
