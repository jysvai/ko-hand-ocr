"""판독기. 학습이 끝난 모델로 손글씨 줄을 읽는다.

앱에서 실제로 쓰는 쪽이다. 두 가지를 지킨다.

**한 장씩 읽지 않는다.** 사진 한 장에서 줄이 서너 칸 나오는데, 칸마다
`generate` 를 따로 부르면 모델을 올리고 내리는 비용을 칸 수만큼 낸다.
한 묶음으로 넣으면 같은 일을 한 번에 한다.

**가두는 판독은 필요할 때만 한다.** 명단으로 가둔 beam 판독은 그냥 읽기보다
서너 배 비싸다. 그냥 읽은 값이 이미 명단에 그대로 있으면 더 볼 것이 없다.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import torch

from .cell import CELL, fit
from .vocab import Vocab

# ImageNet 눈금. 인코더가 이걸로 배웠으니 같은 자로 재 줘야 첫 층부터 말이 통한다.
MEAN, STD = 0.449, 0.226

# 빔 탐색 너비. 실측(사진 10장, v3): 그리디 69.1% -> 빔 4 에서 76.4%,
# 빔 6 에서 76.8% 로 포화한다. 값싼 정확도다 — 한 장에 1.32 -> 1.49초.
BEAMS = 5

# 한 줄에서 만들 수 있는 자모 수. 디코더 자리(128)와 맞춘다.
# 64 로 두면 실제 문장(84 자모)이 중간에서 잘린다.
MAX_JAMO = 128

# 곁들이는 판이 들어가는 자리, 그리고 흔드는 방법.
ALSO = "also"
SHAKE = ("원본", "가로 -7%", "가로 +7%")

# 홀로 쓴 자모(ㄱ, ㅏ)를 답에서 뺀다.
#
# 어휘에는 한자도 일본어도 없다. 그러니 그쪽으로 튈 일은 애초에 없다.
# 그런데 **홀자모로는 튄다.** 모델이 음절을 확신하지 못하면 음절을 만들지 않고
# 낱자로 흘린다: '테스트' -> 'ㅌㅔ ㅅ ㅌ', '서약서' -> 'ㅅㅢ 약서'.
# 낱자는 어느 음절과도 조금씩 닮아서 **값싼 도피처**가 된다.
#
# 실측(v11@15,000, 사진 11장):
#     안 가둠            평균 87.0%  가장 나쁜 장 65.1%
#     홀자모만 금지        평균 88.1%  가장 나쁜 장 71.8%     <- 이것만 이득이다
#     장식기호(·「」…) 금지  변화 없음 (그림에 안 나오니 원래 안 뱉는다)
#     자모 문법 강제       86.9% — 오히려 손해. 모델은 이미 조립되는 자모열만
#                        뱉고 있어서, 문법은 빔의 탐색 경로만 흐트러뜨린다.
#
# 어휘에서 빼지는 않는다. 학습에서는 계속 나와야 죽은 자리가 안 된다(초성 퀴즈
# 같은 줄이 실제로 있다). **읽을 때만** 뺀다.
LETTERS_ONLY = True


class Reader:
    """모델 한 벌을 올려 두고 계속 쓴다."""

    def __init__(self, folder, device: str = "cpu", threads: int | None = None) -> None:
        from . import model as builder

        folder = Path(folder)
        if threads:
            torch.set_num_threads(threads)
        self.vocab = Vocab.load(folder / "vocab.json")
        self.model = builder.load(folder).to(device).eval()
        self.device = device
        # 시작 토큰이 몇 개 붙는지. 가두는 판독에서 이만큼은 건드리지 않는다.
        self.skip = 1
        # 곁들이는 판. `<모델>/also/<이름>/` 에 있으면 같이 읽고 **가운데 답**을 고른다.
        # 없으면 예전 그대로 한 판으로 읽는다.
        self.also = []
        for extra in sorted((folder / ALSO).glob("*")):
            if (extra / "config.json").exists() and (extra / "vocab.json").exists():
                self.also.append((Vocab.load(extra / "vocab.json"),
                                  builder.load(extra).to(device).eval()))

    # ── 그림 준비 ────────────────────────────────────────────────
    def _pixels(self, images) -> torch.Tensor:
        """흑백 칸 -> ViT 가 먹는 3채널 텐서.

        학습 쪽(`data.to_pixels`)과 **같은 눈금**이어야 한다. 여기만 바꾸면
        모델이 밝기를 다르게 읽어 조용히 못 맞힌다.
        """
        grays = []
        for image in images:
            one = fit(image.convert("L"), *CELL)
            flat = torch.frombuffer(bytearray(one.tobytes()), dtype=torch.uint8)
            grays.append(flat.reshape(1, one.height, one.width))
        batch = torch.stack(grays).to(self.device).float().div_(255.0)
        return batch.sub_(MEAN).div_(STD).expand(-1, 3, -1, -1)

    def _decode(self, ids) -> list[str]:
        return [unicodedata.normalize("NFC", self.vocab.decode(row.tolist())).strip()
                for row in ids]

    # ── 읽기 ────────────────────────────────────────────────────
    def _letters(self):
        """홀자모를 뺀 토큰만 허용하는 판. 한 번 만들어 두고 계속 쓴다."""
        if getattr(self, "_letters_fn", None) is None:
            self._letters_fn = letters_only(self.vocab)
        return self._letters_fn

    @torch.no_grad()
    def read(self, images, max_chars: int = MAX_JAMO, beams: int = BEAMS,
             letters: bool | None = None) -> list[str]:
        """묶음으로 읽는다. 아무 글자나 나올 수 있다.

        `letters` 가 참이면 홀자모(ㄱ, ㅏ)는 답에서 뺀다. 까닭은 `LETTERS_ONLY`
        의 머리말에 있다. 초성 퀴즈처럼 낱자가 실제로 있는 그림을 읽을 때는
        False 로 준다.
        """
        if not images:
            return []
        if letters is None:
            letters = LETTERS_ONLY
        if self.also:
            return self._together(images, max_chars, beams, letters)
        ids = self.model.generate(self._pixels(images), max_new_tokens=max_chars,
                                  num_beams=beams, early_stopping=beams > 1,
                                  prefix_allowed_tokens_fn=self._letters() if letters else None)
        return self._decode(ids)

    def read_photo(self, data: bytes, max_lines: int = 16, **rest) -> list[str]:
        """**사진 한 장을 통째로 읽는다.** 줄 자르기까지 여기서 한다.

            texts = reader.read_photo(open("scan.jpg", "rb").read())

        `read` 는 이미 잘린 한 줄을 받는다. 그것만 내놓으니 꾸러미를 받은
        사람은 사진에서 글자까지 갈 수가 없었다 — 줄 자르는 코드가 앱 안에만
        있었기 때문이다. 이제 `kohandocr.page` 에 같이 들어 있다.

        줄 자르기는 PIL 만 쓴다. 여기서 늦게 부르는 까닭은, 잘린 칸을 직접
        넘겨 읽는 사람에게 필요 없는 짐이기 때문이다.
        """
        from io import BytesIO

        from PIL import Image

        from . import page

        cut = page.lines(page.prepare(data), max_lines=max_lines)
        cells = [Image.open(BytesIO(one["png"])).convert("L") for one in cut]
        return self.read(cells, **rest)

    @torch.no_grad()
    def _together(self, images, max_chars: int, beams: int, letters: bool) -> list[str]:
        """판 여럿 x 흔들기 여럿으로 읽고 **서로 가장 닮은 답**을 고른다.

        왜. 판마다 잘 읽는 사진이 엇갈린다. 한 판이 무너지는 칸에서 다른 판이
        멀쩡한 일이 잦다. 실측(사진 11장):

            판                          평균   가장 나쁜 장   85% 넘긴 장
            v18-5000 혼자              90.6%     78.5%        8/11
            v20b 혼자                  87.9%     66.7%        6/11
            v19-17500 혼자             85.4%     63.5%        7/11
            v10a-15000 혼자            88.6%     68.7%        8/11
            **넷을 함께 (가운데 답)**   93.3%     81.2%        9/11

        확신도로 합치는 것은 예전에 재 봤고 **안 됐다**(87.4%, 가장 나쁜 장
        그대로) — 틀린 답을 더 확신하는 일이 있어서다. 그래서 확신도가 아니라
        **닮음**으로 고른다. 흔들어도, 판을 바꿔도 살아남는 답이 진짜에 가깝다.

        흔들기는 셋이면 된다. 여덟까지 늘려도 92.9% 로 같은데 세 배 느리다.
        """
        from . import decode

        how = tuple(way for way in decode.WAYS if way[0] in SHAKE)
        flat, span = [], []
        for cell in images:
            shaken = decode.spread(cell, how)
            span.append((len(flat), len(flat) + len(shaken)))
            flat += shaken
        pixels = self._pixels(flat)
        said: list[list[str]] = [[] for _ in flat]
        for vocab, model in [(self.vocab, self.model)] + self.also:
            allowed = letters_only(vocab) if letters else None
            ids = model.generate(pixels, max_new_tokens=max_chars, num_beams=beams,
                                 early_stopping=beams > 1,
                                 prefix_allowed_tokens_fn=allowed)
            for at, row in enumerate(ids):
                said[at].append(unicodedata.normalize("NFC", vocab.decode(row.tolist())).strip())
        return [decode.middle([one for at in range(a, b) for one in said[at]])[0]
                for a, b in span]

    @torch.no_grad()
    def read_within(self, images, options: list[str], max_chars: int = MAX_JAMO,
                    beams: int = BEAMS) -> list[str]:
        """명단 안에서만 고르게 가두고 읽는다.

        답이 명단에 있다는 것을 알면서 아무 글자나 만들게 둘 이유가 없다.
        다만 **명단에 없는 사람은 읽을 수 없다** — 가둔 결과는 언제나 명단
        안의 값이라, 그것만 보면 틀렸는지 알 수 없다. 그냥 읽은 값과 같이 봐야 한다.
        """
        if not images or not options:
            return [""] * len(images)
        trie = self.trie(options)
        ids = self.model.generate(
            self._pixels(images), max_new_tokens=max_chars, num_beams=beams,
            early_stopping=True, prefix_allowed_tokens_fn=self._walk(trie),
        )
        return self._decode(ids)

    def both(self, images, options: list[str] | None = None) -> list[dict]:
        """그냥 읽기 + (필요하면) 가둔 읽기.

        그냥 읽은 값이 이미 명단에 그대로 있으면 가두는 판독을 건너뛴다.
        비싼 쪽을 안 해도 되는 칸이 대부분이다.
        """
        free = self.read(images)
        if not options:
            return [{"text": t, "constrained": "", "agrees": False} for t in free]

        known = {unicodedata.normalize("NFC", o).strip() for o in options}
        need = [index for index, text in enumerate(free) if text not in known]
        tight = dict.fromkeys(range(len(free)), "")
        for index, text in enumerate(free):
            if text in known:
                tight[index] = text
        if need:
            got = self.read_within([images[i] for i in need], options)
            for index, text in zip(need, got):
                tight[index] = text
        return [{"text": free[i], "constrained": tight[i],
                 "agrees": bool(tight[i]) and _close(free[i], tight[i])}
                for i in range(len(free))]

    # ── 명단 가두기 ──────────────────────────────────────────────
    def trie(self, options: list[str]) -> dict:
        """허용된 글월들을 토큰 나무로 만든다."""
        root: dict = {}
        for text in options:
            if not text:
                continue
            node = root
            for token in self.vocab.encode(text, wrap=False):
                node = node.setdefault(token, {})
            node[self.vocab.eos] = {}
        return root

    def _walk(self, trie: dict):
        """나무를 따라갈 수 있는 토큰만 허용한다."""
        end, start, skip = self.vocab.eos, self.vocab.bos, self.skip

        def allowed(_batch: int, input_ids) -> list[int]:
            ids = input_ids.tolist()
            if len(ids) < skip:
                return [start]
            node = trie
            for token in ids[skip:]:
                node = node.get(token)
                if node is None:
                    return [end]
            return list(node) or [end]

        return allowed


def letters_only(vocab: Vocab):
    """홀자모를 뺀 토큰만 허용하는 `prefix_allowed_tokens_fn` 을 만든다.

    첫 걸음에서 `<s>` 를 반드시 허용해야 한다. 이 모델은 정답을 `<s>...</s>` 로
    감싸 배웠기 때문에 **시작 토큰 다음에 `<s>` 를 한 번 더 뱉는 것이 정상**이다
    (실측 로짓 29.7 대 그 다음 3.6). 그걸 막았더니 첫 글자부터 무너져
    87% -> 65% 가 됐다. 가두는 판을 만들 때 가장 먼저 확인할 자리다.
    """
    from .vocab import COMPAT_JAMO

    ban = {vocab.ids[t] for t in COMPAT_JAMO if t in vocab.ids}
    ban |= {vocab.pad, vocab.bos, vocab.unk}
    keep = sorted(set(range(len(vocab))) - ban)
    start = [vocab.bos]

    def allowed(_batch: int, input_ids) -> list[int]:
        return start if len(input_ids) == 1 else keep

    return allowed


def _close(left: str, right: str, cutoff: float = 0.5) -> bool:
    """두 판독이 같은 글씨를 본 것인지. 자모 단위로 견준다."""
    import difflib

    if not left or not right:
        return False
    a = unicodedata.normalize("NFD", left.lower())
    b = unicodedata.normalize("NFD", right.lower())
    lengths = sorted((len(a), len(b)))
    if lengths[0] < lengths[1] * 0.6:          # 짧은 글을 긴 값에 욱여넣은 것 걸러내기
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= cutoff
