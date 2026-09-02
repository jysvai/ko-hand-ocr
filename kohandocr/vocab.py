"""ko-hand-ocr 어휘.

한 글자 = 한 토큰. 서브워드를 쓰지 않는다.
손글씨는 획이 애매해서 모델은 "이 자리에 무슨 자모가 왔나" 만 답하면 된다.
단어 단위 토큰을 주면 오히려 없는 단어를 그럴듯하게 지어낸다.

한글은 NFD 로 쪼갠다.  '감' -> U+1100 U+1161 U+11B7 (초성/중성/종성)
초성 19 + 중성 21 + 종성 27 = 67 개로 11,172 음절이 전부 나온다.
음절을 통째로 토큰에 두면 11,172 개가 필요하고, 학습에서 한 번도
못 본 음절은 영영 쓰지 못한다.

홀로 쓴 닿소리/홀소리(ㄱ, ㅏ)는 호환 자모 U+3131~U+3163 이라
위 67 개와 부호 자체가 겹치지 않는다. 그래서 같이 넣어도 헷갈리지 않는다.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
SPECIAL = (PAD, BOS, EOS, UNK)

# NFD 가 만들어내는 한글 자모 (U+1100 블록). 순서는 유니코드 순서 그대로 둔다.
CHOSEONG = tuple(chr(c) for c in range(0x1100, 0x1113))       # 19
JUNGSEONG = tuple(chr(c) for c in range(0x1161, 0x1176))      # 21
JONGSEONG = tuple(chr(c) for c in range(0x11A8, 0x11C3))      # 27

# 홀로 쓴 자모. 초성 퀴즈, 표 머리글, 손으로 그은 기호에 실제로 나온다.
COMPAT_JAMO = tuple(chr(c) for c in range(0x3131, 0x3164))    # 51

LATIN_LOWER = tuple("abcdefghijklmnopqrstuvwxyz")
LATIN_UPPER = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = tuple("0123456789")

# 사내 문서에서 실제로 보이는 것만. 장식 기호는 넣지 않는다.
PUNCT = tuple(r" .,:;·/\-_()[]{}<>'" + '"' + r"`~!?%&+=*#@^|" + "「」『』《》〈〉…°㎡№")

GROUPS = {
    "special": SPECIAL,
    "choseong": CHOSEONG,
    "jungseong": JUNGSEONG,
    "jongseong": JONGSEONG,
    "compat": COMPAT_JAMO,
    "lower": LATIN_LOWER,
    "upper": LATIN_UPPER,
    "digit": DIGITS,
    "punct": PUNCT,
}


def build() -> list[str]:
    """토큰 목록. 순서가 곧 id 라서 한 번 정하면 바꾸지 않는다."""
    tokens: list[str] = []
    for group in GROUPS.values():
        for token in group:
            if token not in tokens:
                tokens.append(token)
    return tokens


class Vocab:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.ids = {token: index for index, token in enumerate(tokens)}
        self.pad, self.bos, self.eos, self.unk = (self.ids[t] for t in SPECIAL)

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str, wrap: bool = True) -> list[int]:
        """글월 -> id. 모르는 글자는 <unk> 가 아니라 통째로 버린다.

        학습표에 없는 글자를 <unk> 로 남기면 모델이 '읽을 수 없음' 을
        정답으로 배운다. 애초에 그런 줄은 학습에서 빼는 편이 낫다."""
        body = [self.ids[ch] for ch in unicodedata.normalize("NFD", text) if ch in self.ids]
        return [self.bos, *body, self.eos] if wrap else body

    def decode(self, ids: list[int]) -> str:
        """id -> 글월. 자모를 도로 음절로 합친다."""
        drop = {self.pad, self.bos, self.eos}
        raw = "".join(self.tokens[i] for i in ids if i not in drop and 0 <= i < len(self.tokens))
        return unicodedata.normalize("NFC", raw)

    def covers(self, text: str) -> bool:
        return all(ch in self.ids for ch in unicodedata.normalize("NFD", text))

    def save(self, path) -> None:
        payload = {
            "name": "ko-hand-ocr",
            "scheme": "NFD jamo + latin + digit + punct, one char per token",
            "size": len(self.tokens),
            "special": {"pad": self.pad, "bos": self.bos, "eos": self.eos, "unk": self.unk},
            "groups": {name: len(group) for name, group in GROUPS.items()},
            "tokens": self.tokens,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["tokens"])

    @classmethod
    def default(cls) -> "Vocab":
        # 꾸러미 **안**을 본다. 예전엔 `with_name("..")` 로 저장소 뿌리를 봤는데,
        # 그러면 pip 로 설치했을 때 `site-packages/vocab.json` 을 찾게 되고 없으니
        # 조용히 `build()` 로 넘어갔다. 지금 둘은 토큰 229개가 차례까지 같아서
        # 티가 안 났을 뿐, 어휘를 한 번이라도 손보면 설치본만 다르게 읽게 된다.
        here = Path(__file__).with_name("vocab.json")
        return cls.load(here) if here.exists() else cls(build())
