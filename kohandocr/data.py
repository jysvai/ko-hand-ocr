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
                 letters: int = 64, seed: int = 0, length: int = 100_000) -> None:
        self.fonts_dir = fonts_dir
        self.vocab = vocab
        self.cell = cell
        self.letters = letters
        self.seed = seed
        self.length = length            # 한 epoch 을 몇 장으로 볼지. 끝은 없다.
        self._fonts: synth.Fonts | None = None

    def __len__(self) -> int:
        return self.length

    def _ready(self) -> synth.Fonts:
        if self._fonts is None:         # 일꾼마다 제 것을 연다 (프로세스 넘어 못 간다)
            # **시험용 글꼴은 여기서 안 쓴다**(part="train", 107벌 중 91벌).
            # 나머지 16벌은 `tools/holdout.py` 가 시험지를 만드는 데만 쓴다.
            # 같은 글꼴로 시험지를 만들면 '손글씨를 읽나'가 아니라 '내 생성기를
            # 외웠나'를 재게 된다(synth.FONT_TEST_SHARE 의 설명을 보라).
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
        while made < share:
            text = corpus.decorate(corpus.line(rng), rng)
            if not text or (corpus.UNDRAWABLE & set(text)):
                continue
            ids = self.vocab.encode(text)
            if len(ids) > self.letters:
                continue
            page = synth.render(text, fonts, rng)
            if page is None:
                continue
            yield {"gray": to_gray(synth.fit(page, *self.cell)),
                   "labels": torch.tensor(ids, dtype=torch.long),
                   "text": text}
            made += 1


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
