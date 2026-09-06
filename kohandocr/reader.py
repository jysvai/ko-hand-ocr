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
# 한 칸을 이 가짓수로 흔들어 읽고 가운데 답을 고른다. `decode.WAYS` 여덟 가지
# 중에서 고른 것이고, 재는 쪽(`tools/pair.py` 의 FEW)이 여기를 그대로 따라온다.
#
# 많이 흔들수록 좋은 것이 아니다. 2026-09-04 에 여덟 가지를 다 걸어 보니 평균이
# 93.3%에서 92.1% 로 내려갔다 — 연하게·진하게·획 굵게는 틀린 답을 더 얹는다.
#
# 기울기 둘을 더한 자리(다섯 가지)는 한 조합에서 사진이 8/11 에서 9/11 로
# 올라 보여서 넣었다가 **되돌렸다.** 조합을 셋으로 바꿔 다시 재니 하나만 좋아지고
# 둘은 나빠졌다. 사진이 열한 장뿐이라 이 크기의 차이는 **조합이 바뀌는 폭에
# 묻힌다**(같은 값으로도 조합에 따라 7~10장을 오간다). 글씨체는 400줄로 재서
# 5/6, 최저 87.8% 로 어느 쪽이든 같았다. 그러면 남는 것은 1.2배 느려지는 것뿐이다.
#
# 다시 건드리려면 **조합 여럿에서 같은 방향이 나오는지**부터 볼 것.
SHAKE = ("원본", "가로 -7%", "가로 +7%")

# 이 아래면 **읽었다고 하지 않는다.**
#
# 판을 바꿔 읽고 흔들어 읽었을 때 답들이 서로 얼마나 닮았나가 `decode.middle`
# 이 같이 돌려주는 값이다. 이 값이 낮다는 것은 읽을 때마다 딴소리를 했다는
# 뜻이다. 그런데 지금까지는 그 값을 받아서 **버리고** 답만 내보냈다. 그래서
# 못 읽은 칸도 그럴듯한 글월로 나갔다 — 실측으로 '816883539' 를
# '2025-05-18' 로, '사번' 을 '신청인:' 으로 지어냈다. 서식에서 이것은 빈칸보다
# 나쁘다. 사람이 검토할 자리를 못 찾기 때문이다.
#
# 처음 0.80 을 고른 근거는 이랬다(한 조합, 실측):
#
#     손으로 쓴 서명 두 점                0.34, 0.35
#     깨진 저해상도 인쇄 이름              0.39, 0.67
#     동해독도 손글씨 60줄 **가운데값**      0.96
#     싱글데이 손글씨 60줄 **가운데값**      1.00
#
# 서명 0.35 와 손글씨 0.96 사이가 비어 보였다. **그 셈이 틀렸다.** 서명 쪽은
# 낱값인데 손글씨 쪽은 가운데값이라, 잘 읽은 줄이 어디까지 낮게 내려가는지를
# 아예 안 본 것이다. 가운데값은 꼬리에 대해 아무 말도 해 주지 않는다.
#
# 2026-09-06 에 `tools/trustsweep.py` 로 조합 넷을 재 보니 꼬리가 이렇다.
#
#     **글자 하나까지 맞게 읽은** 줄의 가장 낮은 믿음값
#         글씨체 시험지 (조합당 145~154줄)     0.75  0.75  0.79  0.81
#         사진        (조합당  18~ 21줄)     0.74  0.73  0.89  0.76
#     지어낸 줄(닮음 0.60 미만)의 믿음값
#         조합 1·2 에는 아예 없음
#         조합 3      0.76(닮음 0.50), 0.78(닮음 0.00)
#         조합 4      0.57(닮음 0.55), 0.76(닮음 0.50)
#
# 두 무리가 **0.73~0.79 에서 겹친다.** 지어낸 줄 0.78 과 완벽한 줄 0.74 사이에
# 그을 자리가 없다. 그래서 어떤 값을 골라도 한쪽을 잃는다.
#
#     0.80 이면  완벽하게 읽은 사진 줄을 조합 넷 중 셋에서 2~4줄 지운다
#     0.70 이면  지우는 것은 없어지는데, 조합 3 의 닮음 0.00 짜리가 그냥 나간다
#
# 그래서 **값을 옮기지 않는다.** 옮겨서 나아지는 것이 아니라 손해를 맞바꾸는
# 것뿐이고, 어느 쪽이 큰지는 조합마다 다르다(고리가 몇 시간마다 조합을 바꾼다).
# 고칠 것은 숫자가 아니라 믿음값 자체다 — 지금 값은 '흔들어도 살아남았나'라서
# **어려운 글씨**와 **틀린 글씨**를 못 가른다. 건드리기 전에 반드시
# `tools/trustsweep.py` 로 조합 셋 이상에서 같은 방향인지 볼 것.
TRUST_EDGE = 0.80

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
            self._letters_fn = letters_gate(self.vocab)
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
        return [text for text, _ in self.read_trust(images, max_chars, beams, letters)]

    @torch.no_grad()
    def read_trust(self, images, max_chars: int = MAX_JAMO, beams: int = BEAMS,
                   letters: bool | None = None) -> list[tuple[str, float]]:
        """읽은 값과 **얼마나 믿을 만한지**를 같이 준다.

            for text, trust in reader.read_trust(cells):
                if trust < reader.TRUST_EDGE:
                    text = ""          # 못 읽었다. 지어내지 않는다.

        믿음값은 판을 바꿔 읽고 흔들어 읽었을 때 답들이 서로 얼마나 닮았나이다.
        판이 하나뿐이면 견줄 것이 없어서 **잴 수 없다.** 그때는 1.0 을 주는
        대신 -1.0 을 준다 — 모르는 것을 안다고 하면 문턱이 조용히 무력해진다.
        곁들이는 판을 넣어 쓰라는 뜻이다(`ALSO`).
        """
        if not images:
            return []
        if letters is None:
            letters = LETTERS_ONLY
        if self.also:
            return self._together(images, max_chars, beams, letters)
        ids = self.model.generate(self._pixels(images), max_new_tokens=max_chars,
                                  num_beams=beams, early_stopping=beams > 1,
                                  logits_processor=self._letters() if letters else None)
        return [(text, -1.0) for text in self._decode(ids)]

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
    def _together(self, images, max_chars: int, beams: int,
                  letters: bool) -> list[tuple[str, float]]:
        """판 여럿 x 흔들기 여럿으로 읽고 **서로 가장 닮은 답**과 그 닮음을 준다.

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
            allowed = letters_gate(vocab) if letters else None
            ids = model.generate(pixels, max_new_tokens=max_chars, num_beams=beams,
                                 early_stopping=beams > 1,
                                 logits_processor=allowed)
            for at, row in enumerate(ids):
                said[at].append(unicodedata.normalize("NFC", vocab.decode(row.tolist())).strip())
        return [decode.middle([one for at in range(a, b) for one in said[at]])
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
        pairs = self.read_trust(images)
        free = [text for text, _ in pairs]
        # `unsure` 는 **믿음값을 잰 끝에** 문턱 아래인 칸이다. 잴 수 없었던
        # 칸(-1.0)은 참이라고 하지 않는다. 못 쟀다와 못 읽었다는 다르다.
        unsure = [0.0 <= trust < TRUST_EDGE for _, trust in pairs]
        if not options:
            return [{"text": free[i], "constrained": "", "agrees": False,
                     "trust": pairs[i][1], "unsure": unsure[i]}
                    for i in range(len(free))]

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
                 "agrees": bool(tight[i]) and _close(free[i], tight[i]),
                 "trust": pairs[i][1], "unsure": unsure[i]}
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


class _LettersMask:
    """`letters_only` 와 **똑같이** 가두되, 파이썬 호출 대신 마스크를 더한다.

    왜 이것이 필요한가. `letters_only` 는 첫 걸음만 빼면 늘 같은 목록을
    돌려준다. 그런데 `generate` 는 그것을 (묶음 x 빔 x 토큰)마다 파이썬으로
    부르고 그때마다 허용 목록에서 마스크를 새로 만든다. 빔이 5면 파이썬
    호출이 다섯 배다. 실측으로 **채점 시간의 3분의 2가 여기서 나갔다**
    (글씨체 한 벌 12.6초 -> 4.4초, 답은 162칸 전부 글자 하나까지 같았다).

    늘 같은 값이니 미리 만들어 두고 더하기만 하면 된다. 가두는 조건은
    그대로라 답이 바뀌지 않는다.

    마스크는 첫 부름 때 만든다. 그때라야 로짓의 너비·자리·정밀도를 알 수 있다.
    """

    def __init__(self, vocab: Vocab):
        from .vocab import COMPAT_JAMO

        ban = {vocab.ids[t] for t in COMPAT_JAMO if t in vocab.ids}
        ban |= {vocab.pad, vocab.bos, vocab.unk}
        self._keep = sorted(set(range(len(vocab))) - ban)
        self._bos = vocab.bos
        self._made: tuple | None = None
        self._like: tuple | None = None

    def _masks(self, scores):
        wide = scores.shape[-1]
        first = torch.full((wide,), float("-inf"),
                           device=scores.device, dtype=scores.dtype)
        first[self._bos] = 0.0
        rest = torch.full((wide,), float("-inf"),
                          device=scores.device, dtype=scores.dtype)
        rest[torch.tensor([i for i in self._keep if i < wide],
                          device=scores.device)] = 0.0
        return first, rest

    def __call__(self, input_ids, scores):
        like = (scores.shape[-1], scores.device, scores.dtype)
        if self._like != like:
            self._made, self._like = self._masks(scores), like
        first, rest = self._made
        # 첫 걸음에서 `<s>` 를 반드시 허용해야 한다. `letters_only` 머리말 참고.
        return scores + (first if input_ids.shape[1] == 1 else rest)


def letters_gate(vocab: Vocab):
    """`generate(logits_processor=...)` 에 넣을 가두는 판.

    `prefix_allowed_tokens_fn=letters_only(vocab)` 을 이것으로 바꾸면 답은
    그대로이고 세 배 가까이 빨라진다. 이쪽을 쓸 것.
    """
    from transformers import LogitsProcessorList

    return LogitsProcessorList([_LettersMask(vocab)])


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
