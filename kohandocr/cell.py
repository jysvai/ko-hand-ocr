"""칸 규격. 그림 한 칸을 모델이 먹는 모양으로 맞춘다.

이 모듈은 **PIL 말고는 아무것도 쓰지 않는다.** 학습에는 numpy 와 torch 가
필요하지만 판독만 하는 쪽(앱)에는 필요 없다. 규격이 학습 쪽 모듈에 묻혀
있으면 앱이 학습 의존성까지 끌어와야 한다.

여기 값이 바뀌면 학습과 판독이 서로 다른 그림을 보게 된다. 한쪽만 고치면 안 된다.
"""

from __future__ import annotations

from PIL import Image

# 모델이 먹는 크기 (높이, 너비).
# 실측 27칸의 가로/세로는 중앙 3.5, 90% 7.0, 최대 21.9 였다.
# 10:1 이면 대부분 안 눌리고, 아주 긴 문장 한 줄만 홀쭉해진다.
CELL = (64, 640)

# 이 밝기보다 어두우면 글자로 친다. **줄 자르기(`kohandocr.page`)와 반드시
# 같은 값이어야 한다.** 어긋나면 자를 때와 학습할 때의 기준이 달라져, 뒷장
# 비침이 판 전체를 덮은 채로 학습된다.
INK_EDGE = 200


def fit(page: Image.Image, height: int = CELL[0], width: int = CELL[1]) -> Image.Image:
    """모델이 먹는 크기로 맞춘다.

    **높이는 언제나 꽉 채운다.** 비율을 지키느라 통째로 줄이면, 아주 긴 문장
    한 줄이 64px 틀 안에서 29px 로 쪼그라들어 자모가 뭉개진다(실측: hand-07).
    10:1 보다 긴 줄은 가로만 눌러 넣는다. 글씨가 홀쭉해지긴 해도,
    자모를 가르는 것은 위아래 획이라 통째로 줄이는 것보다 훨씬 잘 읽힌다.
    """
    scale = height / page.height
    wide = max(1, round(page.width * scale))
    small = page.resize((min(wide, width), height), Image.Resampling.BICUBIC)
    edge = small.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    canvas = Image.new("L", (width, height), max(int(edge or 0), 200))
    canvas.paste(small, (0, 0))
    return canvas
