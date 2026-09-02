"""한 칸을 **여러 번 다르게** 읽고 가운데 답을 고른다.

왜 필요한가. 판(checkpoint)마다 잘 읽는 사진이 엇갈린다. 실측:

    판              hand-01  hand-06  hand-09  hand-10
    v10a-15000        94.4     71.9     90.8     68.7
    v14               66.7     80.2     76.7     82.8

같은 자료로 조금 더 학습했을 뿐인데 어떤 칸은 살고 어떤 칸은 죽는다.
이건 모델이 그 칸을 **아슬아슬하게** 읽고 있다는 뜻이다. 학습을 더 해서
아슬아슬함을 없애는 것보다, 읽을 때 여러 번 흔들어 보고 **여러 번 같게 나온
답**을 고르는 편이 싸고 확실하다.

두 판을 모델 확신도로 합치는 것은 재 봤고 **안 됐다**(87.4%, 최저 그대로).
틀린 답을 더 확신하는 일이 있어서다. 그래서 확신도가 아니라 **서로 얼마나
닮았나**로 고른다 — 흔들어도 살아남는 답이 진짜에 가깝다.

흔드는 방법은 실제 사진이 실제로 달라지는 축과 같아야 한다. 사진마다 다른 것은
기울기, 밝기, 획 굵기, 그리고 **가로세로**다. 가로세로는 특히 중요한데,
모델이 글자 수를 그것으로 가늠하기 때문이다.
"""

from __future__ import annotations

import difflib
import math
import unicodedata

from PIL import Image, ImageFilter

from .cell import INK_EDGE

# 흔드는 방법. (이름, 함수) 로 두어 어떤 흔들기가 일했는지 셀 수 있다.
#
# 고르는 데 쓰는 것은 **다수결**이라 홀수가 좋다. 그리고 원본은 반드시 넣는다 —
# 흔든 것끼리만 견주면 원본이 맞은 칸에서 원본을 잃는다.


def _gamma(image: Image.Image, power: float) -> Image.Image:
    table = [min(255, round(255 * (v / 255) ** power)) for v in range(256)]
    return image.point(table)


def _widen(image: Image.Image, factor: float) -> Image.Image:
    """가로만 늘리거나 줄인다. 모델이 글자 수를 가늠하는 축을 흔든다."""
    width = max(8, round(image.width * factor))
    return image.resize((width, image.height), Image.Resampling.BICUBIC)


def _tilt(image: Image.Image, angle: float) -> Image.Image:
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True,
                        fillcolor=245)


def lean(cell: Image.Image) -> float | None:
    """줄이 얼마나 기울었나(도). 기둥마다 잉크 무게중심을 구해 직선을 맞춘다.

    글자 하나하나의 기울기가 아니라 **줄 전체가 앉은 각도**를 잰다. 그래서
    기울어 쓴 글씨(획이 비스듬한 것)에는 안 흔들리고 줄이 누운 것만 잡힌다.

    numpy 를 **여기서** 부른다. 이 모듈 머리에서 부르면 판독하는 사람이 numpy 를
    쓰게 되는데, 정작 판독 경로(`spread`, `middle`)는 PIL 과 difflib 만 쓴다.
    numpy 가 필요한 것은 이 함수와 `upright` 뿐이고 둘 다 읽을 때 안 쓴다.
    머리에 두면 numpy 없이 설치한 사람이 **불러올 때가 아니라 읽는 중에** 터진다.
    """
    import numpy as np

    grid = np.asarray(cell.convert("L"), dtype=np.float32)
    ink = np.clip(INK_EDGE - grid, 0.0, None)              # 어두울수록 크다
    mass = ink.sum(axis=0)
    if not mass.any():
        return None
    keep = mass > mass.max() * 0.05
    if keep.sum() < 8:
        return None
    columns = np.arange(grid.shape[1])[keep]
    if columns.max() - columns.min() < grid.shape[0]:      # 너무 짧으면 못 잰다
        return None
    rows = np.arange(grid.shape[0])[:, None]
    middle = (ink * rows).sum(axis=0)[keep] / mass[keep]
    return math.degrees(math.atan(np.polyfit(columns, middle, 1)[0]))


def upright(cell: Image.Image, cap: float = 8.0) -> Image.Image:
    """줄을 **눕힌 만큼 되돌려** 세운다.

    **지금은 읽기 경로에서 쓰지 않는다. 재 봤더니 이득이 없었다.**

    쓰려던 까닭은 이랬다. 실제 사진의 칸은 평균 -1.51도 누워 있는데(실측 37칸)
    합성은 +0.10 도에 앉아 있다. 게다가 읽기 전에 -1.2도만 돌리면 사진 11장 중
    넘긴 장이 7 -> 10 이 됐다. 그래서 칸마다 재서 세우면 되겠다고 봤다.

    안 된 까닭이 둘이다.

    하나. **칸 하나로는 각도를 못 잰다.** 0.78도로 나온 칸을 0.78도 돌렸더니
    잰 값이 7도로 튀었다. 음절마다 받침이 있고 없어서 기둥별 무게중심이 크게
    오르내리는데, 짧은 줄에서 거기에 직선을 맞추면 값이 안 선다. 37칸 중 하나는
    -30.8도로 나왔다.

    둘. 장 단위로 가운데값을 잡아 통째로 세워 봤지만 **88.6% -> 88.2%** 로
    이득이 없었다. 애초에 '-1.2도가 좋다'는 것부터 흔들림이었다 — v14 는
    +0.6도가 좋았고 v16 은 -1.2도가 좋았다(chasing-noise).

    남겨 두는 까닭: `lean` 은 **모아서 보면** 쓸 만하다(실제와 합성의 기울기
    분포를 견주는 데 썼고, 그건 맞았다). 다시 손대려는 사람이 같은 길을 또
    가지 않도록 결과를 여기 적어 둔다.
    """
    angle = lean(cell)
    # 부호는 재서 정했다: `rotate(t)` 는 잰 값을 **-t** 만큼 옮긴다(실측:
    # rotate(-3.0) -> +2.88, rotate(+3.0) -> -2.90). 그러니 기울기 a 를 없애려면
    # +a 로 돌린다. 반대로 넣으면 기울기가 두 배가 된다.
    #
    # 너무 큰 값은 **자르는 것이 아니라 버린다.** 8도를 넘는 값은 줄이 그렇게
    # 누운 것이 아니라 잰 것이 틀린 것이다(칸에 줄이 둘 들어갔거나 지운 자국이
    # 크거나). 실측 37칸 중 하나가 -30.8 도로 나왔다. 그런 칸을 6도로 잘라서
    # 돌리면 멀쩡한 줄을 6도 눕히게 된다.
    if angle is None or abs(angle) < 0.2 or abs(angle) > cap:
        return cell
    return _tilt(cell, angle)


WAYS: tuple[tuple[str, object], ...] = (
    ("원본", lambda im: im),
    ("가로 -7%", lambda im: _widen(im, 0.93)),
    ("가로 +7%", lambda im: _widen(im, 1.07)),
    ("연하게", lambda im: _gamma(im, 0.80)),
    ("진하게", lambda im: _gamma(im, 1.25)),
    ("획 굵게", lambda im: im.filter(ImageFilter.MinFilter(3))),
    ("기울기 -1.2도", lambda im: _tilt(im, -1.2)),
    ("기울기 +1.2도", lambda im: _tilt(im, 1.2)),
)


def spread(cell: Image.Image, ways=WAYS) -> list[Image.Image]:
    """한 칸 -> 흔든 칸 여러 개. 순서는 `WAYS` 와 같다."""
    return [make(cell) for _, make in ways]


def _jamo(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def agree(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, _jamo(left), _jamo(right)).ratio()


def middle(reads: list[str]) -> tuple[str, float]:
    """여럿 중 **나머지와 가장 닮은 답**과 그 닮음 정도를 돌려준다.

    평균이나 최빈값이 아니라 medoid 다. 글월은 더할 수 없으니 평균이 없고,
    최빈값은 완전히 같은 답이 둘 이상 있을 때만 쓸모가 있는데 손글씨에서는
    드물다. '나머지와 가장 닮은 하나' 는 언제나 뽑을 수 있고, 한두 개가
    엉뚱해도 흔들리지 않는다.

    닮음 정도를 같이 주는 까닭: 이 값이 낮으면 **읽을 때마다 다르게 읽었다**는
    뜻이고, 그건 그 칸을 사람이 봐야 한다는 신호다. 점수를 못 내는 대신
    '못 읽었다'를 말할 수 있게 된다.
    """
    kept = [text for text in reads if text]
    if not kept:
        return "", 0.0
    if len(kept) == 1:
        return kept[0], 1.0
    # **자리로 센다.** 예전엔 `for other in kept if other is not one` 이었는데,
    # 그러면 값이 같은 답끼리 서로를 빼 버린다. 파이썬은 같은 글월을 한 객체로
    # 묶는 일이 있어서(상수 접기·인터닝), 그럴 때 이 함수가 정확히 거꾸로 답한다:
    #
    #     middle(['가나다', '가나다', '가나라'])  ->  '가나라'      (틀림)
    #
    # 두 번 같게 나온 답이 가장 강한 신호인데 바로 그것을 깎았다. 판독에서 나오는
    # 글월은 그때그때 새로 만들어져 객체가 달랐던 덕에 아직 안 물렸을 뿐이다.
    scores = [(sum(agree(one, other) for at, other in enumerate(kept) if at != mine)
               / (len(kept) - 1), one)
              for mine, one in enumerate(kept)]
    scores.sort(key=lambda pair: -pair[0])
    return scores[0][1], scores[0][0]


def steady(reads: list[str], floor: float = 0.55) -> bool:
    """읽을 때마다 크게 다르면 못 읽은 것으로 본다."""
    return middle(reads)[1] >= floor
