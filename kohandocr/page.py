"""사진 한 장에서 **글씨 줄을 잘라 낸다.** PIL 만 있으면 된다.

    from kohandocr.page import prepare, lines
    from kohandocr.reader import Reader

    png = prepare(open("scan.jpg", "rb").read())
    cells = [one["png"] for one in lines(png)]     # 줄마다 잘린 그림

`Reader` 는 **이미 잘린 한 줄**을 받는다. 여러 줄이 든 사진을 그대로 넘기면
읽지 못한다. 그래서 이것이 짝으로 있어야 사진에서 글자까지 갈 수 있고,
`Reader.read_photo` 가 둘을 이어 준다.

자르는 방식은 이렇다. 종이 테두리를 찾아 반듯이 펴고, 그림자를 걷어 낸 뒤,
잉크의 가로·세로 그림자(profile)를 번갈아 훑어 골짜기에서 자른다. 글자
인식이 아니라 **잉크가 있고 없고**만 보므로 어느 나라 글씨에나 통한다.

`max_lines` 로 줄 수를 제한한다. 서식 한 장에 줄이 열여섯을 넘는 일은 드물고,
넘게 잡히면 대개 표 테두리나 얼룩을 줄로 잘못 본 것이다.
"""

from __future__ import annotations

import io

from typing import Any

def upright(image: Any) -> Any:
    """사진을 바로 세운다.

    휴대폰은 센서를 눕힌 채로 찍고 방향을 EXIF 에 적어 둔다. 사진을 돌려
    가며 어느 쪽이 맞는지 알아맞힐 필요가 없다 — 파일이 이미 알고 있다.
    (실제 샘플 두 장 모두 Orientation=6 이었고, 이것만으로 바로 섰다.)
    """
    from PIL import ImageOps

    return ImageOps.exif_transpose(image) or image

def flatten(image: Any, blur: int | None = None) -> Any:
    """그림자와 밝기 기울기를 걷어 낸다.

    종이의 밝기는 천천히 변하고 글자는 급격히 어둡다. 크게 흐린 판(종이의
    밝기)으로 원본을 나누면 종이는 하얗게 펴지고 글자만 남는다.

    전체 대비 보정(autocontrast)만으로는 안 된다. 한쪽이 그늘지면 그늘 전체가
    '검은 글자' 로 취급되어 그 부분 글자가 묻힌다.

    나눗셈은 픽셀을 하나씩 돌지 않고 ImageMath 에 통째로 맡긴다. 파이썬
    반복문으로 돌면 사진 한 장에 수백만 번을 돌아 눈에 띄게 느려진다.
    """
    from PIL import ImageFilter, ImageMath

    # 흐리는 정도는 사진 크기에 따라간다. 고정값을 쓰면 축소본에서는 결까지
    # 뭉개져 배경도 종이처럼 하얘진다(실측: 축소본에서 가죽을 종이로 봤다).
    if blur is None:
        blur = max(8, round(max(image.size) * 0.02))
    gray = image.convert("L")
    # 종이의 밝기만 남긴 판. 글자는 흐려져 사라진다.
    paper = gray.filter(ImageFilter.GaussianBlur(radius=blur))
    return ImageMath.lambda_eval(
        # 종이보다 밝은 자리는 255 로 눌러 하얗게, 0 으로 나누는 일은 막는다.
        lambda a: a["convert"](a["min"](a["ink"] * 255 / a["max"](a["paper"], 1), 255), "L"),
        ink=gray,
        paper=paper,
    )

_PAPER_WHITE = 250

_PAPER_SHARE = 0.75

_PAPER_MIN_AREA = 0.25

def paper(image: Any, probe: int = 600) -> tuple[int, int, int, int] | None:
    """사진에서 종이가 놓인 자리를 찾는다. 못 찾으면 None.

    책상 위에 놓고 찍으면 가죽이나 나무결이 함께 찍힌다. 그림자를 걷어 내면
    종이는 새하얘지지만 배경은 얼룩덜룩한 회색으로 남아서, 글씨를 찾을 때
    그 결을 글씨로 본다(실측: 가죽 무늬가 한 칸으로 잡혀 모델에 들어갔다).

    새하얀 픽셀이 줄의 대부분을 차지하는 구간이 종이다. 세로로 먼저 찾고,
    그 안에서 다시 가로로 찾는다 — 순서를 바꾸거나 한 번에 하면 종이 위아래로
    배경이 꽉 찬 사진에서 아무 열도 기준을 넘지 못한다(실측 hand-04).

    찾는 일은 축소본으로 한다. 어디쯤인지만 알면 되고, 원본으로 하면 느리다.
    """
    from PIL import Image

    scale = probe / max(image.size)
    small = image if scale >= 1 else image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BOX
    )
    white = flatten(small).point(lambda p: 255 if p >= _PAPER_WHITE else 0)
    width, height = white.size

    top, bottom = _widest(list(white.resize((1, height), Image.BOX).get_flattened_data()))
    if top is None:
        return None
    band = white.crop((0, top, width, bottom + 1))
    left, right = _widest(list(band.resize((width, 1), Image.BOX).get_flattened_data()))
    if left is None:
        return None

    back = max(image.size) / max(white.size)
    box = (round(left * back), round(top * back),
           round((right + 1) * back), round((bottom + 1) * back))
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area < image.width * image.height * _PAPER_MIN_AREA:
        return None
    return box

_PAPER_GAP = 0.03          # 프로파일 길이에 대한 비율

def _widest(profile: list[int]) -> tuple[int, int] | tuple[None, None]:
    """기준을 넘는 가장 긴 구간. 짧게 끊긴 자리는 이어서 본다."""
    limit = 255 * _PAPER_SHARE
    gap = max(2, round(len(profile) * _PAPER_GAP))
    best: tuple[int, int] | None = None
    run: tuple[int, int] | None = None
    missed = 0
    for index, value in enumerate(profile):
        if value < limit:
            missed += 1
            if run is not None and missed > gap:
                run = None                    # 배경이 길게 이어진다. 여기서 끝.
            continue
        missed = 0
        run = (run[0], index) if run else (index, index)
        if best is None or run[1] - run[0] > best[1] - best[0]:
            best = run
    return best if best else (None, None)

def prepare(data: bytes, max_side: int = 2200) -> bytes:
    """사진 한 장을 판독기에 넘길 수 있는 모양으로 만든다."""
    from PIL import Image

    image = upright(Image.open(io.BytesIO(data)))
    # 종이만 남긴다. 배경을 버리면 판독에 쓸 화소가 전부 글씨 쪽으로 간다
    # (실측 hand-04: 종이가 사진의 45% 라 글씨 크기가 두 배가 됐다).
    found = paper(image)
    if found:
        image = image.crop(found)
    # 줄이는 것이 먼저다. 4천 픽셀짜리 원본을 그대로 흐리게 만들면 그 한
    # 단계에만 몇 초가 든다. 어차피 판독은 줄인 크기로 한다.
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buffer = io.BytesIO()
    flatten(image).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()

_MIN_LINE = 0.012

_MAX_LINE = 0.30

_MIN_INK = 0.015

_MAX_INK = 0.25

_GAP_Y = 0.004

_GAP_X = 1.2

def lines(png: bytes, max_lines: int = 12, pad: int = 6) -> list[dict[str, Any]]:
    """다듬은 쪽에서 글씨 칸을 찾아 하나씩 잘라 낸다.

    손글씨 모델은 글자 한 줄을 받도록 만들어져 있다. 쪽 전체를 통째로 넣으면
    첫 줄만 읽고 만다. 그런데 어디서 나뉘는지는 판독기가 알려 주지 않는다 —
    Windows OCR 은 손글씨에서 글자를 아예 못 찾기 때문이다.

    그래서 글자를 읽지 않고 잉크가 있는 자리만 센다. 그림자를 걷어 낸 뒤라
    종이는 희고 글씨만 검다.

    **가로로 자르고 세로로 자르기를 번갈아 되풀이한다.** 한 번만 훑으면
    나란히 놓인 것들을 못 가른다. 실측 hand-07 은 왼쪽에 '김구팀', 오른쪽에
    문장 두 줄이 있는데, 가로줄로만 묶으면 셋이 한 덩어리가 된다. 세로로
    한 번 가른 뒤 오른쪽만 다시 가로로 갈라야 문장 두 줄이 나온다.
    """
    from PIL import Image

    page = Image.open(io.BytesIO(png)).convert("L")
    width, height = page.size
    # 잉크가 있다고 볼 어두움. 그림자를 걷어 낸 종이는 240 이상이다.
    ink = page.point(lambda p: 255 if p < 200 else 0)

    gap_y = max(3, round(height * _GAP_Y))
    # 가로로 자를 기준은 이 사진의 글자 높이에서 얻는다. 먼저 가로줄로만
    # 훑어 줄 높이를 어림잡고, 그 값으로 칸 사이 간격을 정한다.
    gap_x = max(8, round(_text_height(ink, (0, 0, width, height), gap_y) * _GAP_X))
    boxes = _split(ink, (0, 0, width, height), gap_x, gap_y)

    # 얼룩과 배경을 **먼저** 걷어 낸다. 아래에서 줄 높이를 어림잡을 때 이것들이
    # 섞이면 기준이 무너진다(실측: 종이 뒷면이 비쳐 생긴 23px 짜리 자국 때문에
    # 기준이 낮아져, 멀쩡한 제목 '감독권' 이 반으로 잘렸다).
    kept = [box for box in boxes if _worth_reading(ink, box, height)]

    # 그 다음, 그래도 두꺼운 덩어리는 골짜기에서 가른다. 손으로 쓰면 윗줄의
    # 내려 그은 획과 아랫줄의 올려 그은 획이 닿아서, 두 줄 사이에 빈 행이
    # **한 줄도** 없는 일이 생긴다(실측 hand-07: 최소 잉크량이 0 이 아니라 1).
    found: list[dict[str, Any]] = []
    for box in _unstack(ink, kept):
        padded = (max(0, box[0] - pad), max(0, box[1] - pad),
                  min(width, box[2] + pad), min(height, box[3] + pad))
        if not _worth_reading(ink, padded, height):
            continue
        found.append({"box": padded, "png": _crop(page, padded)})
    return _number_rows(found[:max_lines])

_STACKED = 1.7

_VALLEY = 0.45

def _unstack(ink: Any, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """여백이 없어 붙어 버린 줄을 골짜기에서 가른다.

    기준이 되는 줄 높이는 **이 쪽에서 이미 찾아낸 칸들**로 정한다. 글씨
    크기는 사람마다 사진마다 다르니 고정값을 쓸 수 없다. 다른 칸들보다
    유난히 두꺼운 칸만 손댄다.
    """
    heights = sorted(bottom - top for _, top, _, bottom in boxes)
    if not heights:
        return boxes
    typical = heights[len(heights) // 2]
    result: list[tuple[int, int, int, int]] = []
    for box in boxes:
        result.extend(_cut_valley(ink, box, typical))
    return result

def _cut_valley(ink: Any, box: tuple[int, int, int, int], typical: int,
                depth: int = 0) -> list[tuple[int, int, int, int]]:
    left, top, right, bottom = box
    if depth > 4 or bottom - top < typical * _STACKED:
        return [box]

    profile = _scan(ink, box, vertical=True)
    if len(profile) < 8:
        return [box]
    middle = sorted(value for value in profile if value)
    if not middle:
        return [box]
    level = middle[len(middle) // 2] * _VALLEY

    # 가장자리는 보지 않는다. 줄의 위아래 끝은 원래 옅다.
    margin = max(2, len(profile) // 5)
    inner = list(enumerate(profile))[margin:len(profile) - margin]
    if not inner:
        return [box]
    index, value = min(inner, key=lambda pair: pair[1])
    if value > level:
        return [box]

    cut = top + index
    return (_cut_valley(ink, _tighten(ink, (left, top, right, cut)), typical, depth + 1)
            + _cut_valley(ink, _tighten(ink, (left, cut, right, bottom)), typical, depth + 1))

def _text_height(ink: Any, box: tuple[int, int, int, int], gap_y: int) -> int:
    """이 사진에서 글자 한 줄이 대략 얼마나 두꺼운지.

    가로줄로만 한 번 훑어 나온 띠들의 가운뎃값을 쓴다. 나란히 놓인 것들이
    한 띠로 묶여 있어도 상관없다 — 여기서 필요한 건 대략의 크기뿐이다.
    """
    bands = _runs(_scan(ink, box, vertical=True), gap_y)
    if not bands:
        return box[3] - box[1]
    heights = sorted(end - start + 1 for start, end in bands)
    return heights[len(heights) // 2]

def _worth_reading(ink: Any, box: tuple[int, int, int, int], page_height: int) -> bool:
    """글씨가 담긴 칸인지, 얼룩이나 배경인지."""
    left, top, right, bottom = box
    tall, wide = bottom - top, right - left
    area = tall * wide
    if not area:
        return False
    if not (page_height * _MIN_LINE <= tall <= page_height * _MAX_LINE):
        return False
    # 글자 하나는 대체로 정사각형에 가깝다. 줄 높이의 절반도 안 되게 좁으면
    # 종이 모서리에 남은 점이다(실측 9x35 · 16x37).
    if wide < tall * 0.5:
        return False
    density = sum(ink.crop(box).get_flattened_data()) / 255 / area
    return _MIN_INK <= density <= _MAX_INK

def _split(ink: Any, box: tuple[int, int, int, int], gap_x: int, gap_y: int,
           depth: int = 0) -> list[tuple[int, int, int, int]]:
    """빈 자리에서 가로·세로로 번갈아 가르기를 되풀이한다.

    문서 판독에서 오래 쓰인 방법이다(XY-cut). 글자를 읽지 않고 여백만 보므로
    어느 나라 글자든, 서식이든 줄글이든 똑같이 통한다.
    """
    left, top, right, bottom = _tighten(ink, box)
    if right <= left or bottom <= top or depth > 8:
        return [(left, top, right, bottom)] if right > left and bottom > top else []

    rows = _runs(_scan(ink, (left, top, right, bottom), vertical=True), gap_y)
    if len(rows) > 1:
        return [piece
                for start, end in rows
                for piece in _split(ink, (left, top + start, right, top + end + 1),
                                    gap_x, gap_y, depth + 1)]

    columns = _columns(_scan(ink, (left, top, right, bottom), vertical=False), gap_x)
    if len(columns) > 1:
        return [piece
                for start, end in columns
                for piece in _split(ink, (left + start, top, left + end + 1, bottom),
                                    gap_x, gap_y, depth + 1)]

    return [(left, top, right, bottom)]

def _scan(ink: Any, box: tuple[int, int, int, int], vertical: bool) -> list[int]:
    """상자 안의 행별(또는 열별) 잉크량. 한 줄짜리로 눌러서 얻는다."""
    from PIL import Image

    region = ink.crop(box)
    length = region.height if vertical else region.width
    size = (1, length) if vertical else (length, 1)
    return list(region.resize(size, Image.BOX).get_flattened_data())

def _runs(profile: list[int], gap: int) -> list[tuple[int, int]]:
    """잉크가 있는 구간들. `gap` 보다 좁은 틈은 같은 구간으로 본다."""
    # 종이 얼룩 한두 점으로 구간이 갈리지 않게 최소치를 둔다.
    floor = max(2, int(255 * 0.004))
    bands: list[list[int]] = []
    for index, value in enumerate(profile):
        if value < floor:
            continue
        if bands and index - bands[-1][1] <= gap:
            bands[-1][1] = index
        else:
            bands.append([index, index])
    return [(start, end) for start, end in bands]

_GAP_ODD = 2.0

def _columns(profile: list[int], gap: int) -> list[tuple[int, int]]:
    """세로로 가를 자리. 크기와 유별남을 **둘 다** 본다.

    `gap` 만 보면 안 되는 이유가 있다. 이름을 '정 윤 석' 처럼 띄어 쓰면 글자
    사이가 그만큼 벌어지는데, 그건 한 낱말이라 갈라선 안 된다. 반대로 한 줄에
    '부서 : 강원도팀' 과 '이름 : 감자밭' 이 나란히 놓이면 갈라야 한다.
    실측하면 둘 다 글자 높이의 1.1 배 안팎이라 크기만으로는 구분이 안 된다.

    갈라놓는 것은 **고르기**다. 띄어 쓴 이름은 글자 사이가 고르게 넓고
    ('정 윤 석' 은 127px 과 115px 로 1.1 배 차이), 두 항목 사이의 빈칸은
    그 줄의 다른 어떤 빈칸보다도 유별나게 넓다(221px 대 100px, 2.2 배).

    그래서 견주는 상대는 중앙값이 아니라 **두 번째로 넓은 빈칸**이다.
    중앙값으로 하면 덩이가 둘뿐일 때 그 하나가 곧 중앙값이라 영영 안 갈린다.
    """
    fine = _runs(profile, max(2, gap // 4))
    if len(fine) < 2:
        return fine
    spaces = sorted(fine[index + 1][0] - fine[index][1] for index in range(len(fine) - 1))
    # 견줄 것이 없으면(빈칸이 하나뿐이면) 크기만 본다.
    runner_up = spaces[-2] if len(spaces) > 1 else 0
    return _runs(profile, max(gap, round(runner_up * _GAP_ODD)))

def _tighten(ink: Any, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """상자를 잉크가 있는 데까지 좁힌다."""
    left, top, right, bottom = box
    rows = _runs(_scan(ink, box, vertical=True), 10 ** 6)
    if not rows:
        return (left, top, left, top)
    columns = _runs(_scan(ink, box, vertical=False), 10 ** 6)
    if not columns:
        return (left, top, left, top)
    return (left + columns[0][0], top + rows[0][0],
            left + columns[0][1] + 1, top + rows[0][1] + 1)

def _number_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """세로로 겹치는 칸끼리 같은 줄 번호를 붙인다.

    서식은 한 줄에 여러 칸을 가로로 늘어놓는다('부서 : ...    이름 : ...').
    반대로 제목은 혼자 한 줄을 차지한다. 이 차이는 무슨 글자인지 몰라도
    보이고, 어느 회사 서류에나 통한다 — 목록이 필요 없는 단서다.
    """
    ordered = sorted(cells, key=lambda cell: cell["box"][1])
    row = -1
    bottom = -1
    for cell in ordered:
        top = cell["box"][1]
        if top >= bottom:
            row += 1
            bottom = cell["box"][3]
        else:
            bottom = max(bottom, cell["box"][3])
        cell["row"] = row
    return ordered

def headline(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """제목으로 볼 칸. 혼자 한 줄을 쓰는 것 중 가장 큰 것.

    서류 제목은 종이 가운데에 크게, 혼자 한 줄로 적힌다. 라벨이 붙은 줄은
    '라벨 + 값' 이라 한 줄에 칸이 둘 이상이다.
    """
    counts: dict[int, int] = {}
    for cell in cells:
        counts[cell["row"]] = counts.get(cell["row"], 0) + 1
    alone = [cell for cell in cells if counts.get(cell["row"]) == 1]
    if not alone:
        return None
    return max(alone, key=lambda cell: cell["box"][3] - cell["box"][1])

def _crop(page: Any, box: tuple[int, int, int, int]) -> bytes:
    buffer = io.BytesIO()
    page.crop(box).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()
