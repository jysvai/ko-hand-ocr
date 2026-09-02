"""ko-hand-ocr 검사.

학습 환경(torch·numpy 가 있는 쪽)에서 돌린다.

    "<tt>/Scripts/python.exe" -m pytest ko-hand-ocr/tests -q

여기서 지키려는 것은 "모델이 잘 읽나" 가 아니다. 그건 실제 사진으로 재는
`tools/eval_photos.py` 몫이다. 여기서는 **그 재기가 뜻을 갖기 위한 전제**를 지킨다.
글월이 되돌아오는가, 그림이 실제와 같은 규격으로 나오는가, 가둔 판독이
명단 밖으로 새지 않는가.
"""

from __future__ import annotations

import random
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kohandocr import corpus, synth                      # noqa: E402
from kohandocr import vocab as corpus_vocab              # noqa: E402
from kohandocr.vocab import Vocab, build                 # noqa: E402

FONTS = Path(str(synth.font_folder()))


@pytest.fixture(scope="module")
def vocab() -> Vocab:
    # 꾸러미 안의 어휘를 쓴다. 예전엔 저장소 뿌리를 봤는데, 그러면 pip 로 설치한
    # 것을 시험할 때 여기만 딴 파일을 읽게 된다.
    return Vocab.default()


@pytest.fixture(scope="module")
def fonts():
    if not FONTS.exists():
        pytest.skip(f"글꼴이 없다: {FONTS}")
    return synth.Fonts(FONTS)


# ── 어휘 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "AI혁신TF", "조직개편팀", "홍서말", "보안점검표 2026-08-31",
    "ㄱㄴㄷㄹㅁ ㅏㅑㅓㅕㅗ", "부서 : 데이터팀", "Hello World 42%",
    "초성 퀴즈를 맞히면 튜립 한 송이와 야생 다람쥐, 표범, 여우를 고루 준다.",
])
def test_글월은_그대로_되돌아온다(vocab, text):
    assert vocab.decode(vocab.encode(text)) == text


def test_음절이_아니라_자모로_쪼갠다(vocab):
    """음절을 통째로 토큰에 두면 11,172개가 필요하고, 못 본 음절은 영영 못 쓴다."""
    assert len(vocab) < 300
    # '감' 한 자가 초성·중성·종성 세 토큰이 된다 (앞뒤 시작·끝 토큰 제외)
    assert len(vocab.encode("감", wrap=False)) == 3


def test_어휘_밖_글자는_버린다(vocab):
    """<unk> 를 남기면 모델이 '읽을 수 없음' 을 정답으로 배운다."""
    assert vocab.decode(vocab.encode("가★나")) == "가나"
    assert vocab.unk not in vocab.encode("가★나")


def test_토큰_차례는_바뀌면_안_된다(vocab):
    """차례가 곧 id 다. 한 번 배포한 뒤 바뀌면 예전 가중치를 못 읽는다."""
    assert vocab.tokens == build()


# ── 글월 만들기 ──────────────────────────────────────────────────
def test_못_그리는_기호는_안_나온다():
    rows = corpus.lines(3000, seed=1)
    assert not any(corpus.UNDRAWABLE & set(row) for row in rows)


def test_모든_토큰이_학습에_나온다(vocab):
    """한 번도 안 나오는 토큰은 죽은 자리다. 모델이 그 글자를 영영 못 쓴다."""
    rows = corpus.lines(40_000, seed=2)
    seen = {ch for row in rows for ch in unicodedata.normalize("NFD", row)}
    missing = [t for t in vocab.tokens
               if len(t) == 1 and t not in seen and t not in corpus.UNDRAWABLE]
    assert missing == []


def test_실제만큼_긴_줄도_만든다(vocab):
    """실제 사진(hand-07)의 문장은 자모 84개다. 그만한 줄을 안 만들면 못 배운다."""
    rows = corpus.lines(20_000, seed=3)
    lengths = [len(vocab.encode(row)) for row in rows]
    assert max(lengths) >= 84
    assert sum(1 for n in lengths if n >= 84) / len(lengths) > 0.01
    assert max(lengths) <= 128, "디코더 자리보다 긴 줄은 통째로 버려진다"


# ── 그림 ────────────────────────────────────────────────────────
def test_칸은_언제나_같은_규격이다(fonts):
    rng = random.Random(0)
    for text in ("김", "부서 : 데이터팀", "가" * 40):
        page = synth.render(text, fonts, rng)
        assert page is not None
        assert synth.fit(page).size == (synth.CELL[1], synth.CELL[0])


def test_긴_줄은_통째로_줄이지_않고_가로만_누른다(fonts):
    """통째로 줄이면 긴 문장이 64px 틀에서 29px 로 뭉개진다."""
    rng = random.Random(1)
    # 가로 늘이기는 줄마다 무작위라, 충분히 긴 것이 나올 때까지 몇 번 그려 본다.
    page = next(
        (drawn for _ in range(30)
         if (drawn := synth.render("가나다라마바사아자차카타파하" * 3, fonts, rng))
         and drawn.width / drawn.height > 6.5),
        None,
    )
    assert page is not None, "6.5:1 보다 긴 줄이 한 번도 안 나왔다"
    cell = synth.fit(page)
    ink = cell.point([255 if v < synth.INK_EDGE else 0 for v in range(256)]).getbbox()
    assert ink is not None
    assert (ink[3] - ink[1]) > synth.CELL[0] * 0.7, "글자가 칸 높이를 채워야 한다"


def test_칸의_가로세로가_실제_범위를_벗어나지_않는다(fonts):
    """길이와 폭을 따로 뽑으면 40자를 넓게 쓴 33:1 줄 같은 게 나온다.

    그런 줄은 세상에 없고, 640px 틀에 눌리면서 획이 문턱 아래로 사라져
    **빈 칸을 정답과 함께** 배우게 된다(실측: 합성 500칸 중 78칸).
    실제 34칸은 1.5 ~ 7.5 사이다.
    """
    rng = random.Random(7)
    pages = [drawn for text in corpus.lines(120, seed=11)
             if (drawn := synth.render(text, fonts, rng))]
    assert len(pages) > 100
    shapes = sorted(page.width / page.height for page in pages)
    assert shapes[len(shapes) // 2] < 6.0, "가운데가 실제(3.9)보다 훨씬 길면 안 된다"
    assert shapes[-1] < 12.0, f"가장 긴 줄이 {shapes[-1]:.1f}:1 이다"

    # 그리고 어느 칸도 비어 있으면 안 된다. 실제 최저는 4.2%(글자 속 밀도 기준).
    for page in pages:
        cell = synth.fit(page)
        data = cell.get_flattened_data()
        covered = sum(1 for value in data if value < synth.INK_EDGE) / len(data)
        assert covered > 0.004, "잉크가 거의 없는 칸을 정답과 함께 먹이고 있다"


def test_글자는_어둡고_바탕은_밝다(fonts):
    """실제 조각과 같은 분포여야 한다. 합성만 흐리면 학습이 헛돈다."""
    rng = random.Random(2)
    darkest, lightest = [], []
    for text in corpus.lines(40, seed=4):
        page = synth.render(text, fonts, rng)
        if page is None:
            continue
        values = sorted(page.get_flattened_data())
        darkest.append(values[len(values) // 100])
        lightest.append(values[len(values) // 2])
    assert sorted(darkest)[len(darkest) // 2] < 160
    assert sorted(lightest)[len(lightest) // 2] > 220


def test_망가진_글꼴은_걸러낸다(fonts):
    """109종 중 2종은 FreeType 이 뱉는다. 학습 도중에 터지면 원인 찾기가 어렵다."""
    assert len(fonts) >= 100
    for path in fonts.files:
        assert path.suffix.lower() in (".ttf", ".otf")


# ── 명단 가두기 ──────────────────────────────────────────────────
def test_가둔_판독은_명단_밖으로_안_샌다(vocab):
    torch = pytest.importorskip("torch")
    from kohandocr.reader import Reader

    options = ["데이터팀", "AI혁신TF", "인재팀"]
    walker = Reader.__new__(Reader)
    walker.vocab = vocab
    walker.skip = 1
    allowed = walker._walk(walker.trie(options))

    start = torch.tensor([vocab.bos])
    first = allowed(0, start)
    heads = {vocab.encode(o, wrap=False)[0] for o in options}
    assert set(first) == heads

    # '데이터팀' 을 끝까지 따라가면 마지막에는 끝 토큰만 남는다
    ids = [vocab.bos, *vocab.encode("데이터팀", wrap=False)]
    assert allowed(0, torch.tensor(ids)) == [vocab.eos]


def test_길이가_안_맞으면_같은_글씨로_안_본다():
    """7자를 3자 이름에 욱여넣은 것을 걸러내야 한다(실측: 전자금융TF서약 -> 홍서말)."""
    from kohandocr.reader import _close

    assert _close("데이터팀", "데이더팀")
    assert not _close("전자금융TF서약", "홍서말")


# ── 기억한 과녁 ──────────────────────────────────────────────────
def test_합성이_기억해_둔_과녁에서_벗어나지_않는다(fonts):
    """`kohandocr/measured.json` 의 실측 분포와 지금 합성을 견준다.

    이 프로젝트에서 정확도를 올린 것은 대부분 모델이 아니라 **합성 그림을
    실제와 같은 분포로 맞춘 것**이었다. 그런데 그 맞춤은 한 상수만 건드려도
    조용히 풀린다(실측: 사진이 10장에서 11장으로 늘자 글자 속 밀도가 26% 어긋났다).

    그래서 과녁을 파일에 적어 두고 여기서 지킨다. 깨지면 `tools/match.py` 로
    어느 항목이 어긋났는지 보고, 고친 뒤에 학습을 건다.
    """
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("match", ROOT / "tools" / "match.py")
    assert spec and spec.loader
    match = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(match)

    target = json.loads((ROOT / "kohandocr/measured.json")
                    .read_text(encoding="utf-8"))["measured"]
    if not target:
        pytest.skip("과녁이 비어 있다: tools/match.py --update")
    got = match.from_synth(200, 23, str(FONTS))
    off = match.gaps(target, got)
    worst = max(off, key=lambda key: off[key])
    assert off[worst] < 0.40, (
        f"{match.NAMES[worst][0]} 이 과녁에서 {100 * off[worst]:.0f}% 벗어났다. "
        "tools/match.py 로 보라"
    )

def test_읽을_때_홀자모를_빼되_시작토큰은_막지_않는다():
    """가두는 판이 지켜야 할 두 가지.

    하나. 홀자모(ㄱ, ㅏ)는 답에서 뺀다. 모델이 음절을 확신 못 할 때 낱자로
    흘리는 도피처라, 막으면 실제 사진 점수가 오른다(87.0 -> 88.1%).

    둘. **첫 걸음의 `<s>` 는 반드시 허용한다.** 이 모델은 정답을 `<s>...</s>`
    로 감싸 배워서 시작 토큰 다음에 `<s>` 를 한 번 더 뱉는다. 그걸 막으면
    첫 글자부터 무너진다(87% -> 65%). 이 시험이 그 자리를 지킨다.
    """
    from kohandocr.reader import letters_only

    vocab = Vocab(build())
    allowed = letters_only(vocab)

    assert allowed(0, [vocab.bos]) == [vocab.bos], "첫 걸음에는 <s> 만"

    later = set(allowed(0, [vocab.bos, vocab.bos]))
    for token in corpus_vocab.COMPAT_JAMO:
        assert vocab.ids[token] not in later, f"홀자모 {token} 이 남아 있다"
    for token in ("ᄀ", "ᅡ", "A", "0", " ", ":"):
        assert vocab.ids[token] in later, f"{token} 이 막혔다"
    assert vocab.eos in later, "끝낼 수 없으면 줄이 안 끝난다"

    # 정답 줄이 이 판을 통과해야 한다. 안 그러면 맞는 답을 막고 있는 것이다.
    for text in ("성명 : 홍서말", "AI혁신TF", "2026-01-31", "Codex 서약"):
        ids = vocab.encode(text)
        for token in ids[1:]:
            assert token in later, f"{text!r} 의 {vocab.tokens[token]!r} 이 막혔다"


def test_글꼴은_그_글월을_담은_것으로_고른다(fonts):
    """새로 모은 손글씨 여덟 벌은 한글을 2,350자만 담고 있다.

    없는 글자는 두부(.notdef)로 그려진다. 그대로 두면 **글자가 아닌 네모를
    정답과 함께** 먹이게 되므로, `pick` 은 글월을 담은 글꼴을 골라야 한다.
    """
    rng = random.Random(5)
    hard = "뷁쫑햏똠힣쒫"
    for _ in range(40):
        font, _ = fonts.pick(rng, hard)
        which = fonts.files.index(Path(font.path))
        for char in hard:
            assert fonts.has(which, char), f"{Path(font.path).name} 에 {char} 가 없다"

def test_가운데_답은_같은_답이_겹칠수록_세진다():
    """값이 같은 답끼리 서로를 빼면 안 된다.

    `other is not one` 으로 세던 때 이것이 '가나라' 를 골랐다 — 두 번 같게 나온
    답이 가장 강한 신호인데 그것만 깎였다. 글월 리터럴이라 파이썬이 두 '가나다'
    를 한 객체로 묶어서 드러났고, 판독에서는 글월이 그때그때 새로 만들어져
    객체가 달랐던 덕에 안 물렸다. 자리로 세면 두 경우가 같아진다.
    """
    from kohandocr import decode

    same = ["가나다", "가나다", "가나라"]
    apart = ["".join("가나다"), "".join("가나다"), "".join("가나라")]
    assert decode.middle(same)[0] == "가나다"
    assert decode.middle(same) == decode.middle(apart)
