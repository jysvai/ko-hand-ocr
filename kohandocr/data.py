"""학습에 먹일 묶음.

이미지를 디스크에 미리 만들어 두지 않는다. 한 장 그리는 데 17ms 라
일꾼 여럿이 붙으면 GPU 가 기다릴 일이 없고, 무엇보다 **같은 글월을 두 번
똑같이 보여주지 않는다**. 손글씨는 글꼴·기울기·굵기·번짐이 매번 달라야
모델이 글씨 모양이 아니라 글자를 배운다.

디스크에 굽지 않으니 학습 데이터가 남지도 않는다. 사내 글씨를 쓰는
2 단계에서 이 성질이 특히 중요하다.

**일꾼은 흑백 uint8 한 장만 보낸다.** 3채널 float32 로 만들어 보내면
한 장이 0.5MB 라 프로세스 사이를 건너는 데만 시간을 다 쓴다(실측: 일꾼을
8배로 늘려도 1.5배밖에 안 빨라졌다). 40KB 로 보내고 3채널·정규화는
GPU 에서 한다. 어차피 GPU 에서는 공짜에 가깝다.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import IterableDataset, get_worker_info

from . import corpus, synth
from .cell import CELL
from .reader import MEAN, STD    # 판독 쪽과 **같은 눈금**을 쓴다. 갈라지면 조용히 틀린다.


class Lines(IterableDataset):
    """끝없이 나오는 (그림, 글월) 짝."""

    def __init__(self, fonts_dir, vocab, cell: tuple[int, int] = CELL,
                 letters: int = 64, seed: int = 0, length: int = 100_000,
                 strike: float | None = None, warp: float | None = None,
                 digits: float = 0.0) -> None:
        self.fonts_dir = fonts_dir
        self.vocab = vocab
        self.cell = cell
        self.letters = letters
        self.seed = seed
        self.length = length            # 한 epoch 을 몇 장으로 볼지. 끝은 없다.
        # 지운 자국 비율. None 이면 synth.STRIKE. 모듈 값을 고쳐 넣으면 안 된다 —
        # 일꾼은 spawn 으로 뜨므로 synth 를 새로 읽어 원래 값으로 돌아간다.
        self.strike = strike
        self.warp = warp                # 글자 속 획을 휠 줄 비율. None 이면 synth.WARP(끔)
        # 날짜가 아닌 숫자 줄(`corpus.digit_line`)로 바꿀 비율. 0 이면 난수도 안 뽑는다.
        self.digits = digits
        self._fonts: synth.Fonts | None = None

    def __len__(self) -> int:
        return self.length

    def _ready(self) -> synth.Fonts:
        if self._fonts is None:         # 일꾼마다 제 것을 연다 (프로세스 넘어 못 간다)
            # **시험용 글꼴은 여기서 안 쓴다.** `part="train"` 이 `FONT_TEST`
            # 여섯 벌과 `FONT_WIDE` 스물네 벌을 뺀다. 앞의 여섯은
            # `tools/holdout.py` 가 시험지를 만드는 데, 뒤의 스물넷은
            # `tools/wide.py` 가 두루 읽나를 재는 데만 쓴다. 같은 글꼴로
            # 시험지를 만들면 '손글씨를 읽나'가 아니라 '내 생성기를 외웠나'를
            # 재게 된다(`synth.FONT_TEST` 머리말). **수는 여기 안 적는다** —
            # 글꼴은 늘어나고 적어 둔 수는 안 늘어난다.
            self._fonts = synth.Fonts(self.fonts_dir, part="train")
        return self._fonts

    def __iter__(self):
        info = get_worker_info()
        worker = 0 if info is None else info.id
        total = 1 if info is None else info.num_workers
        rng = random.Random(self.seed * 1000 + worker)
        fonts = self._ready()
        made = 0
        share = self.length // total + 1
        # 헛도는 횟수를 센다. 글월이 안 나오거나 그림이 안 그려지면 그냥 다시
        # 뽑는데, 그 자리가 **영영 안 되는 자리**가 되면 일꾼이 아무 말 없이
        # 돌기만 한다(오류도 로그도 없다. 밖에서는 '느리다'로만 보인다).
        # 만 번을 잇달아 헛돌면 그것은 뽑기 운이 아니라 고장이다.
        misses = 0
        while made < share:
            if misses > 10_000:
                raise RuntimeError(
                    "합성이 만 번을 잇달아 실패했다 (글꼴 %d벌, 일꾼 %d). "
                    "글꼴 폴더나 corpus 를 보라." % (len(fonts), worker))
            text = corpus.decorate(
                corpus.digit_line(rng) if self.digits and rng.random() < self.digits
                else corpus.line(rng), rng)
            if not text or (corpus.UNDRAWABLE & set(text)):
                misses += 1
                continue
            ids = self.vocab.encode(text)
            if len(ids) > self.letters:
                misses += 1
                continue
            page = synth.render(text, fonts, rng, strike=self.strike, warp=self.warp)
            if page is None:
                misses += 1
                continue
            yield {"gray": to_gray(synth.fit(page, *self.cell)),
                   "labels": torch.tensor(ids, dtype=torch.long),
                   "text": text}
            made += 1
            misses = 0


def to_gray(cell) -> torch.Tensor:
    """흑백 한 장 -> (1, 높이, 너비) uint8. 여기서는 손대지 않고 그대로 보낸다."""
    flat = torch.frombuffer(bytearray(cell.tobytes()), dtype=torch.uint8)
    return flat.reshape(1, cell.height, cell.width)


def to_pixels(gray: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) uint8 -> (B, 3, H, W) float. GPU 에 올린 뒤에 부른다."""
    x = gray.float().div_(255.0).sub_(MEAN).div_(STD)
    return x.expand(-1, 3, -1, -1)


def collate(batch):
    """길이가 다른 글월을 한 묶음으로. 남는 자리는 -100 이라 손실에서 빠진다."""
    longest = max(len(row["labels"]) for row in batch)
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for index, row in enumerate(batch):
        labels[index, : len(row["labels"])] = row["labels"]
    return {
        "gray": torch.stack([row["gray"] for row in batch]),
        "labels": labels,
        "text": [row["text"] for row in batch],
    }
