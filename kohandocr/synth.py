"""손글씨 줄 이미지를 그려낸다.

목표는 예쁜 글씨가 아니라 **우리 파이프라인이 실제로 넘겨주는 것과 같은 모양**이다.
사진 판독 경로는 이런 순서를 거친다.

    사진 -> 세우기 -> 그늘 펴기 -> 종이 잘라내기 -> XY 자르기 -> 줄 한 칸

그러니까 모델이 보는 것은 이미 그늘이 펴지고 흰 바탕에 가깝게 눌린,
글자에 바싹 붙여 자른 조각이다. 종이 결이나 책상 그림자를 정성껏 그려 봐야
`flatten()` 이 지워 버리니 헛일이다. 대신 **펴고 난 뒤에도 남는 것**을 흉내낸다.

  - 고르지 않은 빛이 남긴 옅은 얼룩
  - 뒷장 글씨가 비쳐 보이는 흔적 (hand-07 에서 실제로 나왔다)
  - 폰 카메라의 흔들림과 잡티
  - 줄이 살짝 기운 것, 글자마다 다른 밑선과 기울기

글자를 한 자씩 따로 그려서 붙이는 이유는, 글꼴 한 벌로 한 줄을 통째로 그리면
자간과 밑선이 자로 잰 듯 똑같아져서 인쇄물처럼 보이기 때문이다.
사람 손은 글자마다 조금씩 흔들린다. 그 흔들림이 손글씨의 핵심이다.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageMath, ImageOps

from .cell import CELL, INK_EDGE, fit          # 규격은 판독 쪽과 같이 쓴다

# 획 밝기. 0 은 인쇄물이라 손글씨에는 안 쓴다.
#
# 실측: 실제 칸에서 **잉크로 친 픽셀만** 골라 재면 진한 쪽 10% 가
# 43~89 (중앙 63), 가운데가 79~134 (중앙 104) 다. 좁다.
# 예전 값 (5, 95, 16) 은 20~111 로 훨씬 넓게 퍼졌다 — 새까만 줄과
# 유령 같은 줄이 둘 다 나왔다. 사람이 쓴 획은 그렇게까지 다르지 않다.
INK_DARK = (32, 76, 50)      # 최소, 최대, 가장 흔한 값
PAPER_LIGHT = (232, 255)

# 획 굵기를 글자 높이에 대한 비율로 잡는다.
#
# 맞추는 눈금은 **잉크가 덮은 넓이**다. 모델이 실제로 보는 양이 그것이다.
# 실측: 실제 조각 34칸이 4.4%. 글꼴을 그대로 그리면 3.9% 로 이미 가깝다.
#
# 한 번 헛짚었다. 가로로 훑은 획 길이로 재서 "합성이 1.8 배 두껍다" 고 보고
# 얇게 깎았더니 덮은 넓이가 2.5% 로 떨어져 오히려 멀어졌다. 그 지표는 기운 획을
# 가로로 자른 길이라, 실제 사진과 글꼴 렌더에서 서로 다르게 나온다.
#
# 그래도 굵기를 손대는 이유는 **폭**이다. 예전에는 3x3 창을 45% 확률로만 걸어서
# 큰 글꼴에서는 한 픽셀 차이가 티도 안 났다 — 굵기가 사실상 한 값이었다.
_GLYPH_SHARE = 0.8           # 글꼴 크기 대비 실제 글자 높이
# `_solid` 로 속을 채우고, 글꼴마다 원래 굵기를 재게 된 뒤 다시 맞춘 값이다.
# 실측으로 고른다: 칸잉크와 글자속밀도의 분위수를 실제 34칸과 견줘 어긋남이
# 가장 작은 값. 결과는 실제와 거의 겹친다(칸잉크 1.1/3.2/4.4/9.1 대 1.2/3.0/4.2/9.2).
STROKE = (0.018, 0.058, 0.036)   # 최소, 최대, 가장 흔한 값
# 이 손잡이는 **크게 그리기 시작하고 나서야 살아났다.** 예전 크기(72~168)에서는
# 0.050 -> 0.037 로 내려도 밀도가 27.2 -> 26.8 로 꿈쩍도 안 했다 — 깎는 바닥이
# 1픽셀이라 손잡이보다 바닥이 먼저 걸렸기 때문이다. 크기를 96~184 로 올리니
# 같은 손잡이가 밀도를 11% -> 8% 로 움직인다. 손잡이가 죽어 있으면 손잡이가
# 아니라 **그 손잡이가 놓인 자리**를 의심할 것.
_ERODE_FLOOR = 0.35
# 한 번에 굵기의 이만큼 아래로는 안 깎는다. 처음에 0.50 으로 잡았다가 되돌렸다.
#
# 0.50 은 획 부서짐을 막았지만 **얇고 옅은 칸을 만들 길까지 막았다.**
# hand-01 의 '테스트'(가로세로 4.43, 잉크 3.0%, 밀도 6.8%) 같은 칸이 나오는
# 비율을 재 보니 실제 빈도(37칸 중 1칸 = 2.7%)에 한참 못 미쳤다:
#
#     바닥 0.50 -> 1.0%      바닥 0.35 -> 2.5%      바닥 0.30 -> 2.8%
#
# 그 사이 STROKE 를 올려서 보통 줄은 애초에 깎을 일이 적어졌고(어느 바닥에서든
# 중앙 잉크 100%), 바닥은 **꼬리에서만** 일한다. 0.35 로 뽑은 옅은 칸 여덟 장을
# 실제 칸과 나란히 놓고 봤더니 전부 온전한 '테스트' 였다. 과녁 어긋남도 18% 로
# 0.50(20%)보다 낫다. 막을 것을 막고 열 것은 열어 둔 자리가 여기다.
MIN_STROKE = 1.0             # 픽셀. 이보다 얇게 깎으면 획이 사라진다
# 여기가 바닥이다. 더 얇게 잡아도 3x3 창으로는 1px 아래로 못 깎아서
# 글자 속 밀도가 22% 위에서 멈춘다. 그 아래로 가려면 크게 그려서 줄여야 한다.

# 글자를 얼마나 넓게 쓰는가. 사람마다, 같은 사람도 줄마다 크게 다르다.
#
# 실측(사진 7장 12칸): 글자 하나가 차지하는 가로/세로가 중앙 0.73,
# 0.48 부터 1.22 까지 2.5 배로 벌어진다. 반면 글꼴로 그리면 0.59~0.67 로
# 거의 안 흔들린다. 이 차이를 놔두면 모델이 "같은 폭이면 글자 수는 이만큼"
# 이라고 잘못 배워서, 실제로 넓게 쓴 글씨를 읽을 때 **글자를 더 끼워 넣는다**
# (실측: 조직개편팀 -> 조직객텽편팀, 홍서말 -> 홍서서말).
#
# 그래서 줄마다 가로로 늘렸다 줄였다 한다. 값은 아래 `_measure_stretch` 로
# 실제 분포와 맞춰 가며 정했다.
# 위쪽을 2.20 -> 2.70 으로 넓혔다. 제목은 크고 헐겁게 쓴다:
# 실측 hand-01 의 '테스트' 는 글자당 폭 1.48~1.60 인데 합성은 1.45 에서 멈춰 있었고,
# 그 칸에서 모델이 한글 대신 라틴 글자를 뱉었다('EYZYZ S').
# 폭을 **글자마다** 뽑으면 안 된다. 사람은 짧은 줄은 넓게(제목), 긴 줄은 좁게
# (한 줄 안에 넣으려고) 쓴다. 길이와 폭이 서로 묶여 있다.
#
# 실측(사진 10장 34칸): 칸의 가로/세로가 최저 1.5, 중앙 3.9, 최고 7.5.
# hand-07 의 30자 문장조차 7.5 다 — 글자당 0.25 로 바싹 붙여 썼다.
# 그런데 길이와 폭을 따로 뽑았더니 합성은 중앙 6.1, 최고 **32.9** 가 나왔다.
# 40자를 넓게 쓴 줄 같은 건 세상에 없는데 그걸 배우고 있었다. 더 나쁜 것은
# 그런 줄이 640px 틀에 눌리면서 획이 문턱 아래로 사라져 **잉크 0.0%** —
# 즉 빈 칸을 정답과 함께 배웠다는 것이다(합성 칸의 10% 가 0.4% 이하였다).
#
# 그래서 이제 **칸의 가로/세로를 먼저 뽑고** 늘이기 배수를 거기서 역산한다.
# 그런데 한 분포에서 뽑는 것으로는 모자랐다. 실제 칸의 가로/세로는 글자 수를
# 따라 **완만하게 자란다.** 사진 11장의 칸과 정답을 짝지어 28쌍을 재니
#
#     가로세로 = 1.35 x 글자수^0.595        (흩어짐 x0.75 ~ x1.32)
#      2자 2.0 | 3자 2.6 | 5자 3.5 | 8자 4.6 | 12자 5.9 | 17자 7.3 | 25자 9.1
#
# 글자 수와 무관하게 뽑았더니 11글자 서식 줄이 1.7:1 로 짓뭉개졌다.
# 실제 칸과 나란히 놓고 나서야 보였다 — 합성만 칸을 못 채우고 있었다.
ASPECT_A, ASPECT_P = 1.35, 0.613
# 흩어짐은 **맞추는 게 아니라 덮는다.** 실측 31줄의 10~90% 구간은 0.74~1.39 인데
# 그 폭으로 잡았더니 실제로 있는 줄을 못 만들었다: hand-01 의 제목 '테스트' 는
# 곡선 대비 1.70 배로 넓은데(가로세로 4.43), 상한이 1.38 이라 300번을 그려도
# 3.70 이 최고였다. 그 칸에서 모델은 0% 였다.
#
# 그래서 한동안 triangular(0.58, 1.95, 0.97) 을 썼는데, **덮으려다 가운데를
# 옮겼다.** 이 분포는 기하평균이 1.13 이라 모든 줄이 13% 씩 넓게 나온다.
# 재 보니 합성 곡선이 1.74 x n^0.534 로, 실제(1.35 x n^0.613)와 어긋나 있었다.
# 짧은 줄이 특히 넓어서 **가로 길이가 글자 수를 말해 주지 못한다.**
# v8 이 일곱 글자 줄을 열일곱 글자로 읽은 것('부서 : 강원도팀' ->
# '부서 : 강일 2028년 10월 15일')이 바로 이 자리다.
#
# 로그 자리에서 좌우 대칭인 정규분포로 바꾼다. 기하평균이 정확히 1 이라
# 가운데가 안 밀리고, 꼬리는 저절로 멀리 간다 — 2.2 시그마가 1.70 배라
# 100줄에 세 줄쯤은 hand-01 같은 칸이 나온다. **가운데는 맞추고 끝은 덮는다.**
ASPECT_SIGMA = 0.246             # 실측 10~90% (0.74~1.39) 를 그대로 옮긴 값
ASPECT_TAIL = 3.0                # 이 시그마에서 자른다 (x0.48 ~ x2.09)
ASPECT_CAP = 7.6                 # 칸이 640x64 = 10:1 이지만 실측 최고가 7.5 다.
                                 # 8.5 로 두면 끝에서 맞춘 뒤 90% 값이 8.4 가 되어
                                 # 실제(5.8)를 크게 넘는다. 실측 최고에 붙인다.
WIDEN = (0.34, 2.60)             # 늘이기 한계. 넘기면 획이 뭉개지거나 사라진다.

# 표백 흐림 반지름. 글자 높이에 대한 비율로 잡는다.
# 실측: 2200px 판에서 반지름 44, 글줄 높이 123px -> 0.36 배.
# 절대값으로 박아 두면 글꼴 크기를 바꿀 때마다 결과가 달라진다.
BLEACH_BLUR = (0.30, 0.46)
# 칸 가장자리에 남는 여백(글자 높이 대비). 앱의 XY 자르기가 실제로 남기는 만큼이다.
# 실측 37칸: 위아래 0.000 / 0.022 / 0.048, 좌우 0.021 / 0.041 / 0.058 (10%/50%/90%).
# 합성은 사방 0.10 **고정**이었다 — 실제의 네 배다. 64px 틀에 맞추고 나면 실제
# 글자는 61px 인데 합성 글자는 53px 이라, 모델이 보는 글자 크기가 줄곧 15% 작았다.
# 위아래가 좌우보다 좁은 것은 XY 자르기가 줄을 행보다 열에서 헐겁게 끊기 때문이다.
EDGE_UP = (0.000, 0.052)
EDGE_SIDE = (0.016, 0.062)
# 글자 속 밀도(글자가 차지한 네모 안에서 잉크가 덮은 몫)의 바깥 울타리.
# 실제 37칸은 6.2% ~ 18.8% 안에 있다. 합성은 3.8% ~ 28.1% 로 양쪽이 다 넘쳤다.
# 울타리를 실측 그대로 두면 울타리에 짐이 쌓이므로(cap-piles-up) 조금 넓게 둔다.
DENSE = (0.052, 0.215)

# 줄 기울기. 자세한 까닭은 `render` 안의 쓰는 자리에 적어 두었다.
TILT_SHARE = 0.85            # 이 비율의 줄을 기울인다 (예전 0.25)
TILT_SPREAD = 1.9            # 표준편차(도)
TILT_CAP = 5.0               # 이보다 더는 안 기울인다

# 자간만 넓은 글씨. hand-01 의 '테스트' 는 세 글자인데 칸 가로세로가 4.43 —
# 곡선(2.61)의 1.70 배다. 그런데 그 칸을 실제로 놓고 보면 **획은 실오라기처럼 얇고
# 글자 사이만 벌어져** 있다. 합성은 이 모양을 `_stretch` 로만 만들고 있었는데,
# 늘이기는 자간과 글자를 **같이** 늘여서 굵고 넓적한 딴 글씨가 된다. 그 칸에서
# 모델은 0% 였다('테스트' -> 'ㄴㅡㅡ').
#
# 그래서 자간을 크게 벌리는 갈래를 따로 둔다. 이러면 `natural` 이 이미 넓어져
# 늘이기 배수가 1 에 가까워지고, 획은 얇은 채로 남는다.
WIDE_GAP = 0.14                  # 이 비율만큼은 자간을 크게 벌려 쓴 글씨로
WIDE_GAP_RANGE = (0.26, 0.86)

# 지워 놓은 자국. hand-10 의 '부서 : 강원▓▓도팀' 에는 까맣게 뭉갠 덩이가 있는데
# 모델이 그걸 **글자로 읽어 지어냈다**: '부서 : 강일 2028년 10월 15일'.
# 일곱 글자 줄을 열일곱 글자로 읽은 것이라 그 한 줄만으로 47% 였다.
#
# 빔의 길이 벌점(length_penalty 0.2~1.0)을 쓸어 봤지만 하나도 안 나아졌다
# (평균 79.7~81.0, 최저는 오히려 떨어짐). 빔이 긴 걸 좋아해서가 아니라
# **모델이 그 덩이를 진짜 글자로 친 것**이다. 그러니 합성에서 가르쳐야 한다.
#
# 가르치는 방법: 이름표에 없는 토막을 **더 그려 넣고 그 자리만 뭉갠다.**
# 이름표는 원래 글월 그대로 둔다. 그러면 '뭉갠 덩이는 글자가 아니다' 를 배운다.
STRIKE = 0.075

# 덧그은 글자. 지운 자국과 **반대**다 — 지운 것은 읽으면 안 되고, 덧그은 것은
# 읽어야 한다. 사람은 획이 흐리게 나오면 그 글자만 다시 눌러 쓴다.
#
# hand-06 에 둘이 같이 있다. 칸2 `데이터팀` 앞에는 지운 덩이가 있고(모델이 그걸
# 'Team' 으로 읽었다), 칸0 `전자금융TF서약` 의 '융' 은 덧그어 진해져 있다
# (모델이 '글륦' 으로 뭉갰다). 지운 것만 가르치면 덧그은 글자까지 버리게 된다.
RETRACE = 0.06               # 이 비율의 줄에서 한두 글자를 덧긋는다
# 뭉갤 토막에 쓸 글자. 한글이 대부분이고 숫자·영문도 섞는다 — 사람은
# 아무거나 잘못 쓰고 지운다. 글꼴이 다 그릴 수 있는 것만 골라 둔다.
_STRIKE_POOL = ("가나다라마바사아자차카타파하고노도로모보소오조구누두루무부수우주"
                "그느드르므브스으즈기니디리미비시이지0123456789ABCDEFGHIJ")
PROBE = "가나AZ09ㄱㅏ"          # 흔한 한글·영문·숫자·홀자모.
# 예전에는 "가힣..." 이었는데 **힣 을 넣으면 안 된다.** 한글을 2,350자만
# 담은 글꼴(구글 개구·독도·연성 …)은 힣 이 없어서 거기서 두부를 그리고,
# 그 두부가 획 굵기 재는 값을 흐린다.


def _thickness(canvas: Image.Image) -> float:
    """그려 놓은 글씨의 획 굵기를 픽셀로 잰다. 글자 높이 대비 비율로 돌려준다.

    넓이를 1픽셀 깎았을 때 줄어드는 양이 곧 둘레다. 굵기 w, 길이 L 인 획이면
    넓이는 wL, 둘레는 2L 이므로 **굵기 = 2 x 넓이 / 줄어든 양** 이 된다.
    뼈대를 뽑는 것보다 훨씬 싸고, 기운 획에도 흔들리지 않는다.

    가로로 훑은 런 길이로 재려다 한 번 크게 헛짚었다(v3 기록 참고).
    그 지표는 기운 획을 가로로 자른 길이라 글꼴과 사진에서 다르게 나온다.
    """
    solid = canvas.point([255 if v > 127 else 0 for v in range(256)])
    box = solid.getbbox()
    if box is None:
        return 0.0
    area = sum(1 for v in solid.crop(box).get_flattened_data() if v)
    inner = solid.filter(ImageFilter.MinFilter(3)).crop(box)
    eaten = area - sum(1 for v in inner.get_flattened_data() if v)
    tall = box[3] - box[1]
    if eaten <= 0 or tall <= 0:
        return 0.0
    return (2 * area / eaten) / tall


# 시험용으로 빼 둘 글꼴의 몫. 109벌 중 12벌쯤이 빠진다.
#
# 왜 빼는가. 시험지를 **학습과 같은 글꼴로** 만들면, 재는 것이 '손글씨를 읽나'가
# 아니라 '내 생성기를 외웠나'가 된다. 실제로 그렇게 됐다 — 합성 시험지는
# 88.9 -> 89.2 -> 90.3 으로 계속 올라가는데 같은 판들이 사진에서는 90.6 -> 87.8
# -> 88.5 로 안 올라갔다. 그 자만 봤으면 사진에서 더 나쁜 판을 골랐을 것이다.
#
# 가르는 기준을 **이름의 해시**로 둔다. 폴더를 읽는 순서나 글꼴을 더 넣는 것에
# 흔들리지 않아야 판끼리 견줄 수 있다.
# 시험용으로 빼는 글꼴은 **이름으로 못 박는다.** 해시로 갈랐더니 나눔손글씨가
# 섞여 들어갔는데, 지금 쓰는 판(v18 갈래)은 나눔손글씨 109벌을 이미 다 봤다.
# 그러면 시험지가 깨끗하지 않다. 그래서 **밖에서 새로 받아온 것 중에서만** 뺀다 —
# 이 여섯 벌은 어떤 판도 본 적이 없으므로 v18 갈래든 새 판이든 똑같이 처음이다.
#
# 처음부터 다시 학습해 깨끗한 판을 만들어 보기도 했다(v19, 20,000걸음). 그런데
# 84.0% 로 v18-5000 의 89.8% 에 한참 못 미쳤다 — v18 은 base 부터 이어 달려
# 누적 5만 걸음이 넘는다. **자를 깨끗이 하려고 모델을 버릴 필요는 없다.**
# 자를 옮기면 된다.
FONT_TEST = (
    "구글 나눔펜스크립트 NanumPenScript-Regular.ttf",
    "구글 감자꽃 GamjaFlower-Regular.ttf",
    "구글 하이멜로디 HiMelody-Regular.ttf",
    "구글 싱글데이 SingleDay-Regular.ttf",
    "구글 동해독도 EastSeaDokdo-Regular.ttf",
    "구글 기랑해랑 KirangHaerang-Regular.ttf",
)


def _for_test(name: str) -> bool:
    return name in FONT_TEST


def font_folder(given: str = "") -> Path:
    """손글씨 글꼴이 있는 폴더. **경로를 코드에 박지 않는다.**

    찾는 차례는 이렇다.

      1. 부르는 쪽이 준 값 (`--fonts`)
      2. 환경 변수 `KOHAND_FONTS`
      3. `~/.cache/ko-hand-ocr/fonts`   <- `tools/fetch_fonts.py` 가 받는 자리

    예전에는 만든 사람의 PC 경로가 여덟 파일에 그대로 박혀 있었다. 공개하면
    남의 PC 에서 안 돌고, 계정 이름도 같이 나간다. 글꼴은 저장소에 안 들어가니
    (PROVENANCE.md 를 보라) 받는 사람은 어차피 제 자리에 받아야 한다.
    """
    if given:
        return Path(given)
    from os import environ

    said = environ.get("KOHAND_FONTS")
    return Path(said) if said else Path.home() / ".cache/ko-hand-ocr/fonts"


class Fonts:
    """글꼴 꾸러미. 일꾼(worker)마다 따로 열리도록 게으르게 연다.

    `part` 는 "train"(시험용을 뺀 것) / "test"(시험용만) / "all" 이다.
    학습은 "train", `tools/holdout.py` 는 "test" 를 쓴다.

    `only` 를 주면 이름에 그 글자가 든 글꼴만 남긴다. **한 글씨체씩 따로 재는
    데 쓴다** — 합계만 보면 어떤 글씨체가 무너지는지 안 보인다. 실측으로 시험용
    여섯 벌 사이에 20%p 넘게 벌어졌다.
    """

    def __init__(self, folder,
                 sizes: tuple[int, ...] = (96, 116, 136, 160, 184),
                 part: str = "all", only: str = "") -> None:
        # 크게 그리는 이유가 둘이다.
        #
        # 하나. 획을 깎는 바닥이 1 픽셀이라 작은 글꼴에서는 그보다 가늘게 못
        # 만든다. 크게 그리면 같은 1 픽셀이 상대적으로 가늘어져서 **크고 가는
        # 글씨**(hand-01 의 제목)에 닿는다.
        #
        # 둘. 64px 틀에 **얼마나 줄여 넣느냐**가 또렷함을 정한다. 실제 칸의 글자
        # 높이는 93 / 121 / 156 px 인데(10/50/90%) 예전 크기로는 59 / 97 / 159
        # 이었다. 실제는 1.9배로 줄여 넣고 합성은 1.5배로 줄여 넣으니, 합성 쪽이
        # 덜 뭉개져서 획 사이 틈이 그대로 남는다. 실제로 '이어진 덩이 폭'이
        # 실제의 0.6배였다(가운데값 0.5 대 0.8). 줄여 넣는 배수를 맞추면
        # 틈이 실제만큼 메워진다.
        self.folder = Path(folder)
        self.sizes = sizes
        found = sorted(self.folder.glob("*.ttf")) + sorted(self.folder.glob("*.otf"))
        if not found:
            raise FileNotFoundError(f"글꼴이 없다: {self.folder}")
        # 나눔손글씨 109종 중 2종은 FreeType 이 "too many function definitions" 로 뱉는다.
        # 학습 도중에 터지면 원인을 찾기 어려우니 시작할 때 한 번 그려 보고 걸러 둔다.
        # 일꾼(worker)마다 109벌을 다시 시험하면 켤 때마다 몇 초씩 버리므로
        # 결과를 폴더에 적어 두고 다음부터는 그대로 쓴다.
        # 그리면서 **글꼴마다 원래 획 굵기도 잰다.** 예전에는 109벌 전부
        # 0.075 라고 가정했는데, 실제로는 벌마다 크게 달라서 굵기를 맞추려는
        # 계산이 출발점부터 틀렸다. 그 탓에 글자 속 밀도가 실제보다 훨씬
        # 넓게 퍼졌다(실측 퍼짐 31배 대 실제 3배).
        self.files, self.broken, self.weights, self.whole = [], [], [], []
        note = self.folder / "stroke.json"
        # 적어 둔 것을 쓰되, **새로 들어온 글꼴만 다시 잰다.**
        #
        # 예전에는 적어 둔 것에 없는 글꼴을 조용히 빼 버렸다. 그래서 글꼴을 14벌
        # 더 넣고도 계속 107벌로 돌았는데, 아무 말도 안 나오니 한참 몰랐다.
        # 적어 둔 것은 **빠른 길**이지 글꼴 목록이 아니다.
        try:
            known = json.loads(note.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            known = {}
        pairs, fresh = [], False
        for path in found:
            if path.name in known:
                if known[path.name]:
                    pairs.append((path, known[path.name]))
                else:
                    self.broken.append((path.name, "예전에 못 쓴다고 적어 둠"))
                continue
            fresh = True
            try:
                font = ImageFont.truetype(str(path), 64)
                canvas = Image.new("L", (600, 120), 0)
                ImageDraw.Draw(canvas).text((0, 0), PROBE, font=font, fill=255)
                thick = _thickness(canvas)
            except OSError as exc:
                self.broken.append((path.name, str(exc)))
                known[path.name] = 0
                continue
            if thick <= 0:
                self.broken.append((path.name, "획을 잴 수 없다"))
                known[path.name] = 0
                continue
            known[path.name] = round(thick, 5)
            pairs.append((path, thick))
        if fresh:
            try:
                note.write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass                      # 못 적어도 그냥 매번 재면 된다
        if part not in ("all", "train", "test"):
            raise ValueError(f"part 는 all/train/test 다: {part!r}")
        if part != "all":
            want = part == "test"
            pairs = [(f, t) for f, t in pairs if _for_test(f.name) == want]
        if only:
            pairs = [(f, t) for f, t in pairs if only in f.name]
            if not pairs:
                raise RuntimeError(f"'{only}' 가 든 글꼴이 없다: {folder}")
        self.part = part
        self.only = only
        self.files = [p for p, _ in pairs]
        self.weights = [t for _, t in pairs]
        self._cache: dict = {}
        self._gap: dict = {}
        self._know: dict = {}
        # '다 담은 글꼴'은 마지막에 기댈 자리다. 흔치 않은 글자 몇으로 가른다.
        self.full = [i for i in range(len(self.files))
                     if all(self.has(i, c) for c in "힣뷁쫑햏똠")]
        if not self.files:
            raise RuntimeError(f"쓸 수 있는 글꼴이 없다: {self.folder} ({part})")
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.files)

    def _font(self, which: int, size: int) -> ImageFont.FreeTypeFont:
        key = (which, size)
        font = self._cache.get(key)
        if font is None:
            font = ImageFont.truetype(str(self.files[which]), size)
            self._cache[key] = font
        return font

    def _missing(self, which: int) -> tuple:
        """그 글꼴이 '없는 글자'를 그리는 모양. 이것과 같으면 그 글자가 없다."""
        got = self._gap.get(which)
        if got is None:
            font = self._font(which, 40)
            mask = font.getmask("", mode="L")   # 아무도 안 쓰는 자리
            got = self._gap[which] = (mask.size, bytes(mask))
        return got

    def has(self, which: int, char: str) -> bool:
        """이 글꼴에 이 글자가 있나. 한 번 잰 것은 적어 둔다."""
        seen = self._know.setdefault(which, {})
        if char not in seen:
            mask = self._font(which, 40).getmask(char, mode="L")
            seen[char] = (mask.size, bytes(mask)) != self._missing(which)
        return seen[char]

    def pick(self, rng: random.Random,
             text: str = "") -> tuple[ImageFont.FreeTypeFont, float]:
        """글꼴 하나와 **그 글꼴의 원래 획 굵기**(글자 높이 대비)를 함께 준다.

        `text` 를 주면 **그 글월을 다 담은 글꼴**로 고른다. 왜 필요한가.
        새로 모은 손글씨 글꼴 여덟 벌은 한글을 2,350자만 담고 있다(흔한 글자는
        100% 인데 전체로는 19.5%). 없는 글자는 두부(.notdef)로 그려지므로,
        그대로 두면 **글자가 아닌 네모를 정답과 함께** 먹이게 된다.
        버리기엔 아까운 손씨라(여덟 벌이면 손씨 가짓수의 7%다) 글월에 맞춰 고른다.
        """
        want = {c for c in text if "가" <= c <= "힣"}
        for _ in range(6):
            which = rng.randrange(len(self.files))
            if not want or all(self.has(which, c) for c in want):
                return self._font(which, rng.choice(self.sizes)), self.weights[which]
        # 여섯 번을 골라도 못 담으면 **다 담은 글꼴**에서 고른다.
        pool = self.full or range(len(self.files))
        which = rng.choice(list(pool))
        return self._font(which, rng.choice(self.sizes)), self.weights[which]


# 손으로 빨리 쓴 구분자는 인쇄 글꼴과 딴판이다.
#
# 실측(사진 6장의 부서·성명 줄을 잘라 나란히 놓고 봄): 콜론이 동그란 점 두 개인
# 경우가 **한 번도 없었다.** 세로 획 하나, 비스듬한 틱, 짧은 두 줄로 나온다.
# 그런데 합성은 전부 교과서 ':' 였다. 그래서 모델이 그 획을 '|', '`', '%' 로
# 읽었는데, 그건 모델 잘못이 아니라 **우리가 그 모양을 안 가르친 것**이다.
# 사진 34줄 중 13줄에 이 자리가 있다.
#
# 처음에는 "칸 사이가 좁아서" 라고 짐작하고 빈칸 폭을 쟀다. 틀렸다 —
# 실제 빈칸이 오히려 좁았다. 잘라서 눈으로 보고서야 알았다.
HANDMARKS = ":;"
HANDMARK_SHARE = 0.72        # 이 비율만큼은 손으로 쓴 모양으로. 나머지는 글꼴 그대로.


def _handmark(char: str, font: ImageFont.FreeTypeFont, rng: random.Random):
    """콜론·세미콜론을 손으로 쓴 모양으로 그린다.

    글꼴을 안 쓰고 직접 긋는다. 나눔손글씨 107벌 어느 것도 이렇게 안 그린다.
    """
    size = int(font.size)
    tall = max(6, int(size * rng.uniform(0.28, 0.50)))
    wide = max(3, int(size * rng.uniform(0.08, 0.20)))
    thick = max(2, int(size * rng.uniform(0.040, 0.080)))
    pad = max(4, int(max(wide, tall) * 0.35))
    canvas = Image.new("L", (wide + pad * 2, tall + pad * 2), 0)
    pen = ImageDraw.Draw(canvas)

    lean = rng.uniform(-0.30, 0.60)              # 오른쪽으로 눕는 손이 많다
    top, low = pad, pad + tall
    mid = pad + wide // 2
    head = mid + int(lean * tall / 2)
    foot = mid - int(lean * tall / 2)

    shape = rng.random()
    if shape < 0.36:                             # 짧은 두 획 (점이 늘어난 꼴)
        gap = tall * rng.uniform(0.16, 0.40)
        cut = int((tall - gap) / 2)
        pen.line([(head, top), (head - int(lean * cut), top + cut)], fill=255, width=thick)
        pen.line([(foot + int(lean * cut), low - cut), (foot, low)], fill=255, width=thick)
    elif shape < 0.72:                           # 이어진 획 하나
        pen.line([(head, top), (foot, low)], fill=255, width=thick)
    else:                                        # 위를 눌러 찍고 아래로 흘린 꼴
        # 동그란 점을 찍으면 압정처럼 뭉툭해진다. 실제는 그냥 위쪽이 조금 굵을 뿐이다.
        head_low = top + max(2, int(thick * 0.8))
        pen.line([(head, top), (head, head_low)], fill=255, width=thick + 1)
        pen.line([(head, head_low), (foot, low)], fill=255, width=max(2, thick - 1))

    if char == ";":                              # 세미콜론은 아래가 더 길다
        pen.line([(foot, low), (foot - int(lean * tall * 0.3), low + int(tall * 0.3))],
                 fill=255, width=max(2, thick - 1))
    tilt = rng.uniform(-7.0, 7.0)
    if abs(tilt) > 0.5:
        canvas = canvas.rotate(tilt, resample=Image.Resampling.BILINEAR, expand=True)
    return canvas


# 한 글자 **안에서** 자모 사이를 벌린다.
#
# hand-01 의 '테스트' 를 홀자모까지 허용해 읽혀 보니 'ㅌㅔㅅ ㅌ' 이 나왔다.
# 획은 정확히 봤는데 **음절로 못 묶은 것**이다. 그 칸을 보면 까닭이 뚜렷하다 —
# 크게 쓴 글씨라 '테' 의 ㅌ 과 ㅔ 가 한참 떨어져 있다.
#
# 그런데 합성은 음절을 **글꼴 한 글자로** 그린다. 글자와 글자 사이는 `gap` 으로
# 벌릴 수 있어도 글자 **속**은 못 벌린다. 그러니 이 모양을 한 번도 못 보여 줬다.
# 그래서 그려 놓은 글자를 잉크가 가장 옅은 세로줄에서 갈라 밀어낸다.
# 획을 늘이는 게 아니라 밀어내는 것이라 굵기는 그대로다.
LOOSE = 0.10                 # 이 비율의 줄에서 글자 속을 벌린다
LOOSE_SHARE = 0.55           # 그 줄 안에서 이만큼의 글자를
LOOSE_GAP = (0.08, 0.26)     # 글자 높이 대비 벌리는 폭.
                             # 실제 '테스트' 칸의 가장 큰 빈 구간이 0.58 인데
                             # 0.32 로 두면 자간 넓힘과 겹쳐 1.0 을 넘겼다.


def _loosen(drawn: Image.Image, rng: random.Random) -> Image.Image:
    """글자 속 자모 사이를 벌린다. 가운데께에서 가장 빈 세로줄을 찾아 가른다."""
    width, height = drawn.size
    if width < 14 or height < 8:
        return drawn
    left, right = int(width * 0.28), int(width * 0.74)
    if right - left < 2:
        return drawn
    # 세로줄 합은 numpy 로 한 번에 낸다. 줄마다 crop 해서 더하면 글자 하나에
    # 수십 번 파이썬 반복이 돌아 렌더링이 눈에 띄게 느려진다(실측 2.6ms -> 0.3ms).
    columns = np.asarray(drawn, dtype=np.int32)[:, left:right].sum(axis=0)
    column = left + int(columns.argmin())
    room = int(height * rng.uniform(*LOOSE_GAP))
    if room < 2:
        return drawn
    out = Image.new("L", (width + room, height), 0)
    out.paste(drawn.crop((0, 0, column, height)), (0, 0))
    out.paste(drawn.crop((column, 0, width, height)), (column + room, 0))
    return out


def _glyph(char: str, font: ImageFont.FreeTypeFont, rng: random.Random,
           loose: bool = False):
    """글자 한 자를 제 크기에 맞게 그린다. 밝기값이 곧 잉크가 묻은 정도다."""
    if char in HANDMARKS and rng.random() < HANDMARK_SHARE:
        return _handmark(char, font, rng)
    box = font.getbbox(char)
    width, height = box[2] - box[0], box[3] - box[1]
    if width <= 0 or height <= 0:
        return None
    pad = max(4, int(max(width, height) * 0.35))
    canvas = Image.new("L", (int(width) + pad * 2, int(height) + pad * 2), 0)
    ImageDraw.Draw(canvas).text((pad - box[0], pad - box[1]), char, font=font, fill=255)
    tilt = rng.uniform(-7.0, 7.0)
    if abs(tilt) > 0.5:
        canvas = canvas.rotate(tilt, resample=Image.Resampling.BILINEAR, expand=True)
    if loose and rng.random() < LOOSE_SHARE:
        canvas = _loosen(canvas, rng)
    return canvas


def _compose(text: str, font: ImageFont.FreeTypeFont, rng: random.Random):
    """글자들을 밑선과 자간을 흔들어 가며 이어 붙인다.

    판과 함께 **글자마다 가로 자리**(왼끝, 오른끝)를 돌려준다. 지운 자국을
    그리려면 어느 칸을 뭉갤지 알아야 하는데, 붙여 그린 뒤에는 다시 못 찾는다.
    """
    size = int(font.size)
    # 이 줄이 어떤 손으로 쓰였는지 **먼저** 정한다. 자간을 크게 벌려 쓰는 손은
    # 글자 속도 같이 벌어진다. 글자를 그리면서 벌리므로 그리기 전에 정해야 한다.
    wide = rng.random() < WIDE_GAP
    loose = rng.random() < (LOOSE * 3 if wide else LOOSE)
    glyphs = []
    for char in text:
        if char == " ":
            glyphs.append(None)
            continue
        drawn = _glyph(char, font, rng, loose=loose)
        glyphs.append(None if drawn is None else (drawn, font.getbbox(char)))

    tight = max(size // 8, 2)
    space = int(size * rng.uniform(0.34, 0.56))
    drift = size * rng.uniform(0.02, 0.09)          # 밑선이 오르내리는 폭
    # 자간. 음수면 글자가 붙는다. 음수 쪽으로 크게 치우쳐 있는데 까닭이 있다.
    #
    # 실측: 칸 안에서 **이어진 잉크 덩이의 폭**(글자 높이 대비)이 실제는 중앙 0.65 인데
    # 합성은 0.46 이었다. 합성 글자가 1.7 배 잘게 쪼개진다는 뜻이다. 사람은 획을 이어
    # 쓰는데 글꼴은 떼어 그리고, 밀도를 맞추느라 얇게 깎아서 더 끊긴다.
    # 빈칸 자체는 이미 비슷했다(실제 0.21, 합성 0.15) — 문제는 **붙는 정도**였다.
    #
    # 그런데 좁히기만 하면 안 된다. 모델이 **없는 띄어쓰기를 넣기 시작한다**
    # ('홍길동' -> '홍길 동', '감독권' -> '검 독 궈'). 까닭은 이렇다:
    #
    #   띄어쓰기 없는 줄의 빈칸   실제 중앙 0.16, 90% **0.39**, 최고 0.53
    #                            합성 중앙 0.13, 90% **0.26**, 최고 0.52
    #
    # 사람은 띄어쓰기가 없어도 글자를 0.4 까지 벌려 쓴다. 합성이 0.26 에서
    # 멈추니 모델은 '이만큼 벌어지면 띄어쓰기' 라고 배운다. 그래서 자간의
    # **위쪽**을 열어 두고, 띄어쓰기는 그만큼 더 벌려서 둘이 겹치지 않게 한다.
    # 사람은 한 줄에서 한두 글자만 다시 긋는다. 개수를 안 막으면 줄 전체가
    # 겹쳐 그려져서 딴 글씨가 된다(눈으로 보고 알았다).
    retrace = rng.randint(1, 2) if rng.random() < RETRACE else 0
    # 여기를 좁히려다 **한 번 데었다.** '이어진 덩이 폭'이 실제의 0.6배라서
    # 원인을 찾다가, 글자 네모 안의 '잉크 없는 기둥'이 실제 16.5% / 합성 20.8%
    # 인 것을 보고 자간을 좁혔다. 숫자는 좋아졌다(17.2%, 덩이 폭 어긋남 11% -> 6%).
    #
    # 그런데 실제 칸과 합성 칸을 나란히 놓고 보니 **반대**였다. 실제 글씨는
    # 글자 사이가 시원하게 벌어져 있고 대신 **글자 안쪽 획이 붙어** 있다.
    # 합성은 글자 사이가 좁고 안쪽이 끊겨 있었다. '빈 기둥'이라는 눈금 하나가
    # 서로 다른 두 가지 틈(글자 사이 / 글자 안쪽)을 같이 세고 있어서, 숫자를
    # 맞추려고 **엉뚱한 쪽**을 좁힌 것이다. 그림을 안 봤으면 그대로 갔다.
    #
    # 그래서 자간은 실제대로 되돌린다. 안쪽이 끊기는 것은 따로 찾아야 한다.
    gap = rng.uniform(*WIDE_GAP_RANGE) if wide else rng.uniform(-0.14, 0.20)
    # 띄어쓰기는 자간보다 늘 더 벌어져야 한다. 자간을 벌린 줄에서 빈칸을 그대로
    # 두면 둘이 겹쳐서, 모델이 자간을 띄어쓰기로 읽는다(전에 겪은 그 오류다).
    space = int(space + size * max(gap, 0.0))
    slope = rng.uniform(-0.05, 0.05)                # 줄 전체가 비스듬히 오르내림

    margin = int(size * 1.2)
    canvas = Image.new(
        "L",
        (int(size * (len(text) + 2) * 1.3) + margin * 2, size * 3 + margin * 2),
        0,
    )
    pen = margin
    baseline = margin + size
    placed = 0
    spans: list[tuple[int, int] | None] = []
    for item in glyphs:
        if item is None:
            spans.append(None)
            pen += space
            continue
        drawn, box = item
        top = baseline + box[1] - (drawn.height - (box[3] - box[1])) // 2
        top += int(rng.gauss(0, drift) + slope * (pen - margin))
        canvas.paste(drawn, (int(pen), int(top)), drawn)
        if retrace and rng.random() < 0.30:
            # 같은 글자를 살짝 어긋나게 한 번 더. 손으로 덧그으면 정확히 안 겹친다.
            canvas.paste(drawn, (int(pen + rng.gauss(0, size * 0.035)),
                                 int(top + rng.gauss(0, size * 0.030))), drawn)
            retrace -= 1
        spans.append((int(pen), int(pen) + drawn.width))
        pen += drawn.width - int(size * 0.35) * 2 + int(size * gap) + tight
        placed += 1
    return (canvas, spans, wide) if placed else None


def _strike(canvas: Image.Image, box: tuple[int, int, int, int], size: int,
            rng: random.Random) -> None:
    """이미 그려 놓은 칸 위에 지운 자국을 덧그린다. 사람이 실제로 쓰는 세 꼴.

    잉크가 밝은 판이므로 밝게 긋는다. 굵기는 글자 획보다 조금 두껍게 —
    사람은 지울 때 힘을 더 준다.

    세로 자리는 **글자가 실제로 있는 높이**에서 받아야 한다. 판 좌표에 상수로
    박았더니 자국이 글자 위 허공에 떠서, 지운 게 아니라 딴 획이 되었다.
    """
    left, top, right, low = box
    if right - left < 4 or low - top < 4:
        return
    draw = ImageDraw.Draw(canvas)
    tall = low - top
    top -= int(tall * 0.10)                           # 글자 밖으로 조금 넘겨 긋는다
    low += int(tall * 0.10)
    wide = max(2, int(size * rng.uniform(0.045, 0.095)))
    shape = rng.random()
    if shape < 0.42:                                  # 지그재그로 마구 그은 것
        # 한 번만 그으면 밑 글자가 그대로 읽힌다. 실제 사진의 자국은 여러 번
        # 겹쳐 그어 **덩이째 까맣다.** 그래서 두세 번 덧긋는다.
        for _ in range(rng.randint(2, 3)):
            steps = rng.randint(5, 12)
            points = []
            for i in range(steps + 1):
                x = left + (right - left) * i / steps
                points.append((x + rng.gauss(0, tall * 0.06),
                               (top if i % 2 else low) + rng.gauss(0, tall * 0.07)))
            draw.line(points, fill=255, width=wide, joint="curve")
    elif shape < 0.72:                                # 가로로 한두 줄 죽 그은 것
        for _ in range(rng.randint(1, 2)):
            y = rng.uniform(top + tall * 0.25, low - tall * 0.25)
            draw.line([(left - tall * 0.06, y + rng.gauss(0, tall * 0.03)),
                       (right + tall * 0.06, y + rng.gauss(0, tall * 0.03))],
                      fill=255, width=wide)
    else:                                             # 동그랗게 말아 뭉갠 것
        turns = rng.randint(3, 6)
        points = []
        for i in range(turns * 12 + 1):
            phase = i / 12.0 * math.tau
            along = left + (right - left) * i / (turns * 12)
            points.append((along + math.cos(phase) * tall * 0.20,
                           (top + low) / 2 + math.sin(phase) * tall * 0.42))
        draw.line(points, fill=255, width=wide, joint="curve")


def _shear(ink: Image.Image, rng: random.Random) -> Image.Image:
    """글씨 전체를 눕힌다. 오른쪽으로 눕는 사람이 많지만 왼쪽도 있다."""
    slant = rng.gauss(0.10, 0.13)
    if abs(slant) < 0.02:
        return ink
    grow = int(abs(slant) * ink.height) + 1
    wide = Image.new("L", (ink.width + grow * 2, ink.height), 0)
    wide.paste(ink, (grow, 0))
    return wide.transform(
        wide.size,
        Image.Transform.AFFINE,
        (1, slant, -slant * wide.height / 2, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
    )


def _wobble(ink: Image.Image, rng: random.Random, cells: int = 6) -> Image.Image:
    """가로로 나눠 잡고 위아래로 흔든다. 종이가 안 평평했거나 손이 떨린 자국."""
    width, height = ink.size
    if width < cells * 4:
        return ink
    step = width / cells
    swing = height * rng.uniform(0.01, 0.055)
    shifts = [rng.uniform(-swing, swing) for _ in range(cells + 1)]
    mesh = []
    for index in range(cells):
        left, right = index * step, (index + 1) * step
        up, down = shifts[index], shifts[index + 1]
        mesh.append((
            (int(left), 0, int(right), height),
            (left, up, left, height + up, right, height + down, right, down),
        ))
    return ink.transform((width, height), Image.Transform.MESH, mesh, resample=Image.Resampling.BILINEAR)


def _solid(ink: Image.Image) -> Image.Image:
    """번져서 흐려진 잉크 자국을 다시 꽉 채운다.

    `_stretch`·`_shear`·`_wobble` 은 모두 쌍선형/삼차 보간이다. 세 번 거치면
    얇은 획의 마스크가 150 언저리까지밖에 안 차오른다. 그 마스크로 합치면
    진한 잉크(16)와 종이(245)가 섞여서 **획이 109 로 나온다.**

    실측: 합성 500칸 중 78칸(16%)이 잉크 1.2% 미만이었다. 실제 조각의 최저는
    1.2% 인데 합성 최저는 0.03% — 거의 빈 칸을 정답과 함께 먹이고 있었다.
    길이 탓이 아니었다. '봵' 한 자짜리도 그랬다.

    실제 가는 펜은 얇아도 속이 꽉 찬다. 그래서 자국의 위쪽 45% 가 완전히
    진해지도록 눈금을 늘린다. 이미 꽉 찬 획은 건드리지 않는다.
    """
    hist = ink.histogram()
    marks = sum(hist[16:])                        # 잉크가 조금이라도 묻은 자리
    if not marks:
        return ink
    want, seen, top = marks * 0.45, 0, 255
    for value in range(255, 15, -1):
        seen += hist[value]
        if seen >= want:
            top = value
            break
    if top >= 248:
        return ink
    boost = 255 / top
    return ink.point([min(255, round(v * boost)) for v in range(256)])


def _stretch(ink: Image.Image, factor: float) -> Image.Image:
    """글씨를 가로로 늘리거나 좁힌다. 넓게 쓰는 사람과 좁게 쓰는 사람."""
    if abs(factor - 1.0) < 0.03:
        return ink
    wide = max(8, int(ink.width * factor))
    return ink.resize((wide, ink.height), Image.Resampling.BICUBIC)


def _weight(ink: Image.Image, size: int, mine: float, widen: float,
            rng: random.Random, thin: bool = False) -> Image.Image:
    """획 굵기를 줄마다 다르게 한다.

    사람은 얇게도 굵게도 쓴다. 그런데 예전에는 3x3 창을 45% 확률로만 걸어서,
    큰 글꼴에서는 그 한 픽셀이 티도 안 나 굵기가 사실상 한 값이었다.

    여기서는 **글꼴 크기에 맞춰** 깎거나 붙인다. 그래서 56pt 로 그리든
    112pt 로 그리든 나오는 굵기의 폭이 같다.

    `mine` 은 **이 글꼴의 원래 굵기**다(`_thickness` 가 잰 값). 예전에는 109벌
    전부 0.075 라고 가정했는데, 그러면 두꺼운 글꼴은 더 두꺼워지고 얇은 글꼴은
    더 얇아져서 굵기를 맞추려는 계산이 거꾸로 벌어진다.

    깎기는 반드시 **가로로 늘린 뒤에** 한다. 3x3 창은 정사각인데 늘리기는
    가로만 늘리므로, 늘리기 전에 깎으면 세로획만 맞고 **가로획은 그 배수만큼
    어긋난다**(2.6배 늘린 줄은 가로획이 2.6배 얇아진다). 사람 손글씨는 두 방향
    굵기가 비슷하다. 순서를 바꾸니 밀도 퍼짐이 70배 -> 31배로 줄었고,
    게다가 더 빨랐다(좁힌 줄은 깎을 넓이도 좁다).

    그런데 순서만 바꾸고 `widen` 을 안 넘겨서, **지금 굵기를 누르기 전 값으로
    알고** 깎고 있었다. 가로로 0.62 배 누른 줄은 세로획이 이미 0.62 배로 얇은데
    `_weight` 는 그걸 원래 굵기로 보고 거기서 또 깎아 획을 지웠다.
    긴 줄일수록 많이 눌리므로 이 잘못은 길이를 따라 커진다. 실측:

        가장 옅은 10%   밀도 0.7~5.0%    가로세로 6.4   11자
        가운데 50%      밀도 7.6~14.6%   가로세로 4.5    7자
        가장 진한 10%   밀도 19.0~52.7%  가로세로 3.1    4자

    실제 사진은 길이와 상관없이 6.2~18.8% 안에 있다(3배 폭). 합성은 15배였다.
    가로로 w 배 누르면 세로획은 w 배 얇아지고 가로획은 그대로이므로,
    지금 굵기는 대략 `mine x (1 + w) / 2` 다.
    """
    tall = size * _GLYPH_SHARE
    now = tall * max(mine * (1.0 + widen) / 2.0, 0.01)
    # 자간을 크게 벌려 쓴 글씨는 획도 대체로 얇다. 사람이 크게 띄어 쓸 때는
    # 펜을 세워 가볍게 긋기 때문이다. 실제 hand-01 의 '테스트' 가 그렇다 —
    # 가로세로 4.43 인데 글자 속 밀도가 6.8% 다.
    #
    # 이 둘을 따로 뽑으면 **넓으면서 옅은 칸**이 거의 안 나온다(3글자 줄
    # 600장에 5장). 굵기와 자간은 사람 손에서 같이 움직이니 여기서도 묶는다.
    low, high, mode = STROKE
    if thin:
        high, mode = mode, (low + mode) / 2
    want = tall * rng.triangular(low, high, mode)
    step = min(round(abs(now - want) / 2), 4)     # 한쪽에서 깎거나 붙일 픽셀
    if want < now:
        # 너무 깎으면 획이 아예 사라진다(실측: 작은 글꼴에서 '김' 이 통째로 없어졌다).
        #
        # 절대 바닥(MIN_STROKE)만으로는 모자랐다. `mine` 은 글꼴의 **평균** 굵기라,
        # 평균이 6px 이어도 ㅅ 의 이음획처럼 3px 인 획이 같이 있다. 평균 기준으로
        # 2px 를 깎으면 평균 획은 살고 얇은 획만 사라진다.
        #
        # 실측('테스트' 120장, 단계마다 남은 잉크): 그린 것 대비 중앙 65%,
        # **가장 나쁜 10% 는 25%**. 눈으로 보니 스가 통째로 없는 칸이었다.
        # 그런데 실제 사진의 옅은 칸은 획이 얇을 뿐 **온전하다.** 이대로 두면
        # '넓고 옅은 칸 = 부서진 조각' 을 가르치는 셈이고, 실제로 hand-01 의
        # '테스트'(넓고 옅고 온전함)를 0% 로 읽었다.
        #
        # 그래서 상대 바닥을 같이 둔다 — 한 번에 굵기의 45% 넘게 깎지 않는다.
        floor = max(MIN_STROKE, now * _ERODE_FLOOR)
        step = min(step, max(0, int((now - floor) / 2)))
    if step < 1:
        return ink
    # 3x3 을 n 번 거는 것은 (2n+1)x(2n+1) 을 한 번 거는 것과 결과가 같고,
    # 값이 큰 창은 픽셀마다 창 넓이만큼 훑어서 훨씬 비싸다(실측 17 -> 32ms).
    shrink = want < now
    for _ in range(step):
        ink = ink.filter(ImageFilter.MinFilter(3) if shrink else ImageFilter.MaxFilter(3))
    if not shrink:
        return ink
    # 깎으면 획의 진한 속까지 같이 깎여서 잉크가 흐려진다. 실제 가는 펜은
    # 얇아도 진하다(실측: 실제 조각의 가장 어두운 1% 가 68, 되살리기 전 합성은 153).
    peak = int(ink.getextrema()[1] or 0)
    if peak and peak < 255:
        boost = 255 / peak
        ink = ink.point([min(255, round(v * boost)) for v in range(256)])
    return ink


def _paper(size, rng: random.Random) -> Image.Image:
    """그늘을 편 뒤에 남는 바탕. 거의 흰데 완전히 고르지는 않다."""
    width, height = size
    base = rng.randint(*PAPER_LIGHT)
    coarse = np.random.default_rng(rng.getrandbits(32)).normal(
        base, 5.0, (max(height // 8, 2), max(width // 24, 2)))
    small = Image.fromarray(np.clip(coarse, 0, 255).astype(np.uint8))
    sheet = small.resize(size, Image.Resampling.BICUBIC)
    if rng.random() < 0.35:                                   # 한쪽이 살짝 어두운 얼룩
        shade = Image.linear_gradient("L").resize(size, Image.Resampling.BILINEAR)
        if rng.random() < 0.5:
            shade = shade.transpose(Image.Transpose.ROTATE_180 if rng.random() < 0.5
                                    else Image.Transpose.FLIP_TOP_BOTTOM)
        soft = ImageOps.invert(shade).point([200 + v // 5 for v in range(256)])
        sheet = Image.blend(sheet, soft, rng.uniform(0.05, 0.22))
    return sheet


def _bleed(sheet: Image.Image, ink: Image.Image, rng: random.Random) -> Image.Image:
    """뒷장 글씨가 비친 자국. 좌우 뒤집힌 흐린 그림자로 들어온다."""
    ghost = ink.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    ghost = ghost.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 3.5)))
    faint = rng.uniform(0.05, 0.16)
    ghost = ghost.point([int(v * faint) for v in range(256)])
    offset = (rng.randint(-ink.width // 6, ink.width // 6),
              rng.randint(-ink.height // 4, ink.height // 4))
    layer = Image.new("L", sheet.size, 0)
    layer.paste(ghost, offset)
    return Image.composite(Image.new("L", sheet.size, rng.randint(150, 205)), sheet, layer)


def render(text: str, fonts: Fonts, rng: random.Random):
    """글월 한 줄 -> 흑백 이미지. 못 그리면 None.

    실제 경로와 순서를 맞춘다. 앱은 **여백이 넓은 판 전체를 표백한 뒤** 줄을 잘라낸다.
    바싹 자른 조각을 표백하면 흐림판이 글자로 물들어 획이 도로 하얘진다(실측함).
    그래서 여기서도 여백을 넉넉히 두고 표백한 다음, 마지막에 바싹 자른다.
    """
    # **글월을 같이 넘긴다.** 안 넘기면 그 글자가 없는 글꼴이 걸려 두부를 그린다.
    font, mine = fonts.pick(rng, text)
    size = int(font.size)
    # 이름표에 없는 토막을 더 그려 넣고 그 자리만 뭉갠다. 돌려주는 이미지에는
    # 지운 자국이 있지만 **이름표는 `text` 그대로**다. 그래야 '뭉갠 덩이는
    # 글자가 아니다' 를 배운다. `render` 를 부르는 쪽은 아무것도 안 바꿔도 된다.
    drawn_text, cross, apart = text, None, False
    if len(text) >= 2 and rng.random() < STRIKE:
        # 몇 글자를 지우나. 예전에는 1~4 를 고르게 뽑아 평균 2.5 글자였다.
        # 실제 사진의 자국을 합성 자국과 나란히 놓고 보니 **실제가 훨씬 작다** —
        # hand-06 의 '데이터팀' 앞 덩이도, hand-01 의 '융' 위 자국도 한 글자
        # 남짓이다. 우리 자국은 줄 절반을 덮는 큰 고리와 긴 가로줄이었다.
        #
        # 그래서 모델이 **작은 덩이를 지운 자국으로 못 알아본다.** 실제로
        # hand-06 의 그 덩이를 글자로 읽어 'Edge' 를 지어냈다(아는 라틴 낱말로
        # 메운 것이다). 사람은 잘못 쓴 **한 글자**를 지우지 줄을 지우지 않는다.
        # 떨어뜨릴 것인지 먼저 정한다. 길이가 거기에 달렸기 때문이다.
        # 실측(사진 11장 37칸의 자국 넷):
        #   06-2  떨어진 덩이   칸 높이의 0.58배  <- 한 글자보다 좁다
        #   06-0  붙은 덩이     한 글자 남짓
        #   10-2  붙은 덩이     한 글자 남짓
        #   01-0  칸 전체가 낙서 (읽을 것이 없는 칸)
        # 붙은 것은 지금 크기가 맞다. 떨어진 것만 작으니 그때만 한 글자로 둔다.
        apart = rng.random() < 0.45
        howmany = 1 if apart else rng.choices((1, 2, 3, 4), weights=(55, 30, 12, 3))[0]
        junk = "".join(rng.choice(_STRIKE_POOL) for _ in range(howmany))
        # 자리를 고르게 뽑으면 **줄 맨 앞**에 오는 일이 거의 없다(7글자 줄이면
        # 12%, 전체로는 0.9%). 그런데 실제 사진 둘이 다 맨 앞이다 — hand-06 의
        # '데이터팀' 앞, hand-10 의 '부서 :' 뒤. 사람은 첫 글자를 잘못 쓰고
        # 지우는 일이 잦다. 그래서 앞자리에 무게를 준다.
        at = 0 if rng.random() < 0.25 else rng.randint(0, len(text))
        # 덩이를 **떼어 놓기도 한다.** 실제 사진(hand-06 칸2)은 지운 덩이와
        # '데이터팀' 사이가 칸 높이의 0.48배로 벌어져 있다. 우리는 늘 글자에
        # 붙여 그렸다. 그래서 합성에서는 **틈 건너에 있는 덩어리가 언제나 읽을
        # 낱말**이었고, 모델은 그렇게 배웠다 — 그 덩이를 'Core' 로 옮겨 적었다.
        #
        # 틈 **크기**는 이미 맞다(실측: 칸 높이의 0.35배를 넘는 틈이 실제 42.4%,
        # 합성 41.6%). 모자란 것은 크기가 아니라 **틈 건너에 버릴 것이 있는
        # 경우**다. 그런 칸을 한 번도 안 보여 줬으니 0% 였다.
        #
        # 절반쯤으로 둔 근거는 약하다 — 실제로 확인한 떨어진 덩이는 한 개다.
        # 다만 0% 가 틀린 것은 분명하니, 붙은 것과 떨어진 것을 둘 다 보여 준다.
        if not apart:
            drawn_text = text[:at] + junk + text[at:]
            cross = (at, at + len(junk))
        elif at == len(text):
            drawn_text = text + " " + junk
            cross = (at + 1, at + 1 + len(junk))
        else:
            drawn_text = text[:at] + junk + " " + text[at:]
            cross = (at, at + len(junk))
    made = _compose(drawn_text, font, rng)
    if made is None:
        return None
    ink, spans, wide = made
    if cross is not None:
        marks = [span for span in spans[cross[0]:cross[1]] if span]
        if marks:
            left, right = marks[0][0], marks[-1][1]
            # 떨어뜨린 자국은 **좁혀서** 그린다. 실측: 실제 사진의 지운 덩이는
            # 칸 높이의 0.58배인데 우리 것은 중앙값 1.10배, 62%가 0.8배를 넘었다.
            # 글자 한 칸을 통째로 덮으니 그렇다. 사람은 잘못 쓴 자리만 좁게 뭉갠다.
            #
            # 붙은 자국에는 안 한다. 좁히려면 남는 글자 잉크를 지워야 하는데,
            # 이어 쓰는 글꼴은 옆 글자가 그 자리까지 넘어와 있어서 멀쩡한 획을
            # 깎게 된다. 떨어뜨린 쪽은 옆에 빈칸이 있어 안전하다.
            if apart:
                keep = rng.uniform(0.35, 0.75)
                mid, half = (left + right) / 2, (right - left) * keep / 2
                narrow = (int(mid - half), int(mid + half))
                blank = ImageDraw.Draw(ink)
                blank.rectangle([left, 0, narrow[0], ink.height], fill=0)
                blank.rectangle([narrow[1], 0, right, ink.height], fill=0)
                left, right = narrow
            band = ink.crop((left, 0, right, ink.height)).getbbox()
            if band:
                _strike(ink, (left, band[1], right, band[3]), size, rng)
    # 변형은 반드시 **자른 뒤에** 한다. `_compose` 의 판은 글자보다 서너 배 크고,
    # 그 위에서 기울이고 흔들면 빈 자리를 옮기느라 시간의 절반을 쓴다(실측 43%).
    box = ink.getbbox()
    if box is None:
        return None
    slack = max((box[3] - box[1]) // 3, 12)
    ink = ink.crop((max(0, box[0] - slack), max(0, box[1] - slack),
                    min(ink.width, box[2] + slack), min(ink.height, box[3] + slack)))
    # 이 줄이 남길 여백을 먼저 뽑는다. 뒤의 가로세로 셈이 이 값을 쓴다.
    up, side = rng.uniform(*EDGE_UP), rng.uniform(*EDGE_SIDE)
    # 글자 수에 맞는 목표 가로/세로를 뽑고, 지금 모양에서 거기까지 가는 배수를 구한다.
    letters = max(len(drawn_text.replace(" ", "")), 1)
    # 자간을 벌려 쓴 줄은 **칸도 실제로 넓다.** 여기서 폭을 따로 뽑으면
    # `widen = want / natural` 이 벌려 놓은 자간을 도로 눌러, 자간만 넓고 칸은
    # 보통인 딴 것이 된다(실측: 늘이기 배수가 1.63 -> 1.10 로 줄 뿐 목표는 그대로).
    # 그래서 자간을 벌린 줄은 흩어짐을 위쪽에서만 뽑는다.
    draw = abs(rng.gauss(0, 1)) if wide else rng.gauss(0, 1)
    spread = math.exp(max(-ASPECT_TAIL, min(ASPECT_TAIL, draw)) * ASPECT_SIGMA)
    want = min(ASPECT_A * letters ** ASPECT_P * spread, ASPECT_CAP)
    # 마지막에 사방으로 글자 높이의 `pad` 만큼 여백을 붙이므로 가로세로가
    # 줄어든다. 목표는 **여백까지 붙인 칸** 기준이라 여기서 되돌려 놓는다.
    #
    # 목표는 **다 만든 칸** 기준이다. 여기서는 대충 그 근처로 늘여 두기만 하고,
    # 정확히 맞추는 일은 맨 끝에서 한다(아래 '마지막에 한 번 더 잰다'를 보라).
    # 여백을 붙이면 (a + 2s) / (1 + 2u) 로 바뀌니 그만큼 미리 되돌린다.
    goal = want
    want = want * (1 + 2 * up) - 2 * side
    natural = (box[2] - box[0]) / max(box[3] - box[1], 1)
    widen = min(max(want / natural, WIDEN[0]), WIDEN[1])
    ink = _weight(_stretch(ink, widen), size, mine, widen, rng, thin=wide)
    ink = _solid(_wobble(_shear(ink, rng), rng))
    box = ink.getbbox()
    if box is None or box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None

    # 판 위의 한 줄처럼, 글자 높이만큼 사방에 흰 여백을 둔다.
    tall = box[3] - box[1]
    room = max(int(tall * 1.3), 40)
    ink = ink.crop((box[0] - room, box[1] - room, box[2] + room, box[3] + room))

    sheet = _paper(ink.size, rng)
    if rng.random() < 0.30:
        sheet = _bleed(sheet, ink, rng)
    dark = Image.new("L", ink.size, int(rng.triangular(*INK_DARK)))
    page = Image.composite(dark, sheet, ink)

    # 줄이 기운 것. 예전에는 0.25 확률로 표준편차 1.2도였다 — 줄의 75% 가 자로
    # 잰 듯 수평이었다는 뜻이다.
    #
    # **근거를 조심해서 읽을 것.** 처음에 이 손잡이를 올린 까닭은 사진을 각도별로
    # 다시 읽혔더니 점수가 크게 움직여서였다(0.0도 87.5% / +0.6도 90.4%). 그런데
    # 그건 **흔들림이었다.** v14 는 +0.6도가 가장 좋았는데 v16 은 -1.2도가 가장
    # 좋았다 — 같은 사진인데 방향이 반대다. 사진 11장 37칸의 평균은 ±5%p
    # 흔들린다(chasing-noise 를 보라). 사진마다 기울기를 재서 세워 읽는 것도
    # 해 봤지만 88.6% -> 88.2% 로 이득이 없었다.
    #
    # 그래도 이 값을 올려 둔 근거는 따로 있다. **실제 칸의 기울기를 직접 쟀다** —
    # 10%/50%/90% 가 -4.2 / -0.7 / +2.1 도다. 줄마다 이만큼 눕는데 합성이 75% 를
    # 수평으로 그리면 모델이 그 폭을 못 배운다. 종이를 펴는 단계는 페이지 전체
    # 기울기만 잡아 주고, 줄마다 남는 기울기와 사람이 비스듬히 쓴 것은 남는다.
    #
    # 가운데를 -1.5도로 옮기지 않는 까닭: 그 값은 **이 사진 11장을 찍은 버릇**이다.
    # 폭은 실측을 따르되 가운데는 0 에 둔다.
    if rng.random() < TILT_SHARE:
        lean = max(-TILT_CAP, min(TILT_CAP, rng.gauss(0, TILT_SPREAD)))
        page = page.rotate(lean, resample=Image.Resampling.BILINEAR,
                           fillcolor=rng.randint(*PAPER_LIGHT))
    blur = rng.uniform(0.0, 1.1)
    if blur > 0.25:
        page = page.filter(ImageFilter.GaussianBlur(blur))
    if rng.random() < 0.55:                                   # 폰 카메라 잡티
        grain = np.random.default_rng(rng.getrandbits(32)).normal(
            0.0, rng.uniform(2.0, 9.0), (page.height, page.width))
        page = Image.fromarray(
            np.clip(np.asarray(page, dtype=np.float32) + grain, 0, 255).astype(np.uint8))

    page = bleach(page, max(6, int(tall * rng.uniform(*BLEACH_BLUR))))

    # 이제 앱의 XY 자르기가 하는 만큼 바싹 자른다.
    # 기준은 앱과 **똑같아야** 한다. 우리가 더 옅은 것까지 글자로 세면
    # 뒷장 비침이 판 전체를 덮어, 글자가 칸의 27% 로 쪼그라든 채 학습된다(실측).
    tight = page.point([255 if v < INK_EDGE else 0 for v in range(256)]).getbbox()
    if tight is None:
        return None
    wide_edge, tall_edge = max(int(tall * side), 2), max(int(tall * up), 1)
    page = page.crop((max(0, tight[0] - wide_edge), max(0, tight[1] - tall_edge),
                      min(page.width, tight[2] + wide_edge), min(page.height, tight[3] + tall_edge)))
    if page.width < 8 or page.height < 8:
        return None

    # **다 만든 뒤에 한 번 더 잰다.** 이 한 줄이 앞의 손잡이 세 개보다 낫다.
    #
    # 목표 배수를 아무리 잘 뽑아도 늘이기 -> 굵기 -> 흔들기 -> 표백 -> 바싹
    # 자르기를 지나는 동안 칸이 목표에서 벗어난다. 벗어나는 방향이 줄 길이마다
    # 다르다는 것이 문제였다: 짧은 줄은 그대로인데(x1.00) 긴 줄은 눌린다
    # (x0.85). 그래서 앞에서 상수로 되돌리면 한쪽이 반드시 어긋난다
    # (실측: 3자 +7%, 17자 -12%). 사슬을 예측하는 대신 **끝에서 재고 맞춘다.**
    #
    # 안 맞추면 34자 줄이 12.4:1 까지 나온다(실제 34칸의 최대는 7.5:1). 그런
    # 칸은 640px 틀에 눌리면서 획이 문턱 아래로 사라져 **빈 칸을 정답과 함께**
    # 배우게 한다. 고치는 폭은 좁게 묶는다 — 크게 늘이거나 누르면 여기까지
    # 공들여 만든 획 굵기가 도로 망가진다.
    fix = min(max(min(goal, ASPECT_CAP) / (page.width / page.height), 0.72), 1.40)
    if abs(fix - 1.0) > 0.02:
        page = page.resize((max(8, round(page.width * fix)), page.height),
                           Image.Resampling.BICUBIC)

    # 획 굵기도 같은 이유로 **끝에서 잰다.** 굵기를 정하는 손잡이(STROKE)를
    # 쓸어 봐도 밀도가 꿈쩍하지 않았다(0.050 -> 0.037 로 내려도 98.5% 값이
    # 27.2 -> 26.8). 밀도를 실제로 움직이는 것은 굵기 손잡이가 아니라 **가로로
    # 누른 배수**였다(실측: 가장 옅은 10% 는 누름 0.86, 가장 진한 10% 는 1.38).
    # 그 배수는 글자 수와 목표 가로세로가 정하는 것이라 굵기 쪽에서 못 건드린다.
    #
    # 그래서 무엇이 원인이든 **나온 값을 보고** 울타리 안으로 한 칸 밀어 넣는다.
    # 종이 위에서는 어두운 것이 획이므로 MinFilter 가 굵게, MaxFilter 가 얇게다.
    #
    # numpy 한 번으로 센다. PIL 로 하면 `point` 로 마스크를 새로 그리고
    # `histogram` 으로 또 한 번 훑어서, 줄 하나에 7ms 를 썼다(전체 46ms 의 15%).
    # 합성이 학습의 병목이라 이 7ms 가 그대로 학습 속도다.
    for _ in range(2):
        mark = np.asarray(page) < INK_EDGE
        rows, cols = mark.any(axis=1), mark.any(axis=0)
        if not rows.any():
            break
        tall = int(rows.sum() and (np.flatnonzero(rows)[-1] - np.flatnonzero(rows)[0] + 1))
        wide = int(np.flatnonzero(cols)[-1] - np.flatnonzero(cols)[0] + 1)
        dense = mark.sum() / max(tall * wide, 1)
        if dense < DENSE[0]:
            page = page.filter(ImageFilter.MinFilter(3))
        elif dense > DENSE[1]:
            page = page.filter(ImageFilter.MaxFilter(3))
        else:
            break

    # 그리고 **칸 전체로 봐서** 너무 빈 것은 아예 안 내보낸다.
    #
    # 위의 울타리는 '글자가 앉은 네모 안'의 밀도라, 짧은 줄이 640px 틀에서
    # 차지하는 몫이 작으면 걸러지지 않는다. 새로 받은 얇은 손씨(개구 Light 등)가
    # 그런 칸을 만들었다 — 잉크가 칸의 0.31% 인 칸이 나왔다. 실제 최저는 0.4% 다.
    # 거의 빈 그림에 이름표를 붙여 먹이면 모델은 **빈 칸에서 글자를 지어내는
    # 법**을 배운다. 그래서 굵혀 보고, 그래도 안 되면 그 줄은 버린다.
    mark = np.asarray(page) < INK_EDGE
    for _ in range(2):
        if mark.mean() * min(page.width / page.height, 10.0) / 10.0 > 0.005:
            return page
        page = page.filter(ImageFilter.MinFilter(3))
        mark = np.asarray(page) < INK_EDGE
    return page if mark.mean() * min(page.width / page.height, 10.0) / 10.0 > 0.005 else None


def bleach(page: Image.Image, blur: int) -> Image.Image:
    """앱의 `photo.flatten()` 과 같은 표백.

    실제 판독 경로는 사진을 이 방법으로 편 **뒤에** 줄을 잘라낸다.
    그러니 모델이 보는 실제 조각은 이미 이 처리를 거친 것이다.
    합성 이미지도 같은 손질을 거쳐야 두 쪽이 같은 모양이 된다.
    """
    smooth = page.filter(ImageFilter.GaussianBlur(radius=blur))
    return ImageMath.lambda_eval(
        lambda a: a["convert"](a["min"](a["ink"] * 255 / a["max"](a["paper"], 1), 255), "L"),
        ink=page,
        paper=smooth,
    )


__all__ = ["Fonts", "render", "bleach", "fit", "CELL", "INK_EDGE"]
