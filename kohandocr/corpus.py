"""학습에 쓸 글월을 우리가 만들어 낸다.

외부 말뭉치를 가져오지 않는 이유는 두 가지다.
하나는 라이선스가 깨끗해야 해서고, 하나는 우리가 원하는 분포가 따로 있어서다.

읽어야 하는 것은 대부분 **이름과 부서명**이다. 이건 언어 모델이 도와줄 수 없다.
'홍서말'은 문법적으로 이상하지만 실제 사람 이름일 수 있다.
그래서 모델이 말이 되는 쪽으로 고쳐 읽으면 오히려 틀린다.

반대로 문장은 앞뒤 맥락이 도움이 된다.
그래서 세 종류를 섞어 먹인다.

  무작위  40%  — 언어 지식 없이 획만 보고 읽는 힘. 이름을 지켜준다.
  서식    40%  — 사내 문서에서 실제로 보이는 줄 모양.
  문장    20%  — 문장이 나올 때 앞뒤를 쓸 수 있게.

사내 실제 명단은 여기에 한 글자도 들어가지 않는다. 형태만 본뜬다.
"""

from __future__ import annotations

import random

SYLLABLES = [chr(c) for c in range(0xAC00, 0xD7A4)]
# 지어낸 말을 만들 때 음절을 **고르게** 뽑으면 안 된다. 11,172자에서 고르게
# 뽑으면 받침이 27/28 = 96% 로 붙고, 겹받침(ㄳㄵㄶㄺ…)이 11.9%, 겹모음(ㅘㅚㅢ…)이
# 16.0% 나온다. 실제 사진의 글월은 받침 53%, **겹받침 0%**, 겹모음 4.8% 다
# (실측 167음절). 고르게 뽑은 음절은 획이 촘촘해서 그림이 실제보다 빽빽해지고,
# 모델은 평생 만나지 않을 모양에 힘을 쓴다.
#
# 그래서 초·중·종성을 각각 **실제 비율에 가깝게** 뽑아 음절을 짜 맞춘다.
# 겹받침·겹모음도 아주 없애지는 않는다 — 어휘 시험(모든 토큰이 학습에 나온다)이
# 걸리고, 실제로 드물게 오기도 한다.
_HEAD = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_HEAD_W = (9, 1, 7, 6, 1, 7, 6, 5, 1, 9, 2, 12, 8, 1, 3, 2, 2, 2, 5)
_MID = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_MID_W = (17, 5, 2, 1, 9, 5, 4, 1, 8, 1, .5, 1, 2, 6, .5, .5, .5, 1, 9, 1, 15)
# 맨 앞이 **받침 없음**이다. 57 대 64.8 이라 받침이 53% 로 붙는다(실측과 같다).
_TAIL = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_TAIL_W = (57, 9, 1, .16, 12, .16, .16, 1, 9, .16, .16, .16, .16, .16, .16, .16,
           5, 2, .16, 4, 1, 14, 1, 1, .5, 1, .5, 1)
assert len(_HEAD) == len(_HEAD_W) and len(_MID) == len(_MID_W) and len(_TAIL) == len(_TAIL_W)


def _syllable(rng: random.Random) -> str:
    """실제 글월에 가까운 비율로 초·중·종성을 뽑아 음절 하나를 만든다."""
    head = _HEAD.index(rng.choices(_HEAD, _HEAD_W)[0])
    mid = _MID.index(rng.choices(_MID, _MID_W)[0])
    tail = _TAIL.index(rng.choices(_TAIL, _TAIL_W)[0])
    return chr(0xAC00 + (head * 21 + mid) * 28 + tail)

SURNAME = ("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민"
           "진지엄채원천방공현함변염여추도소석선설마길연위표명기반라왕금옥육인맹제모탁국여진")

GIVEN = ("가강건경계고관광구규근기길나남노다단대덕도동두라람래로록루리린마만명무문미민바박발방배"
         "백범별보복본봉부빈사산삼상새서석선설섭성세소솔수숙순숭슬승시식신실심아안애야양어언에여연"
         "열영예오옥온완요용우운웅원월위유윤율은음의이익인일임자작장재정제조종주준중지진찬창채천철"
         "청초최추춘충치태택하한해향헌혁현형혜호홍화환회효훈휘흥희")

ORG_HEAD = ("경영 인사 재무 회계 총무 법무 감사 전략 기획 홍보 대외 구매 물류 영업 마케팅 상품 개발 연구 "
            "기술 정보 보안 정보보안 데이터 인공지능 디지털 플랫폼 클라우드 인프라 네트워크 품질 안전 환경 "
            "고객 서비스 운영 생산 설비 자재 수출 해외 국내 신사업 미래 혁신 조직 교육 채용 노무 복지 "
            "리스크 준법 내부통제 자금 투자 여신 심사 채권 결제 정산 카드 수신 파생 자산 세무").split()
ORG_TAIL = ("팀 본부 부문 사업부 사업부문 센터 그룹 파트 실 담당 총괄 지점 지사 연구소 랩 유닛 "
            "스쿼드 챕터 TF 태스크포스 위원회 사무국 스튜디오 조직 조직팀 부서").split()

DOC_HEAD = ("보안 개인정보 정보보호 내부통제 자금세탁 준법 윤리 청렴 인사 근태 휴가 출장 교육 채용 "
            "성과 평가 승진 급여 복리후생 자산 재고 구매 계약 발주 검수 정산 세금 회계 결산 예산 "
            "위험 사고 장애 점검 감사 실사 재해 대응 비상 운영 개발 배포 변경 접근권한 계정 "
            "이해상충 하도급 외주 반출 반입 폐기 이관 조직개편 겸직 부업").split()
DOC_TAIL = ("서약서 동의서 확인서 신청서 승인서 요청서 보고서 결의서 계획서 제안서 명세서 내역서 "
            "점검표 기준표 관리대장 대장 목록 명단 현황 지침 규정 매뉴얼 절차서 체크리스트 "
            "결과서 회의록 의사록 각서 협약서 위임장 증명서 통지서").split()

LABEL = ("성명 이름 부서 소속 조직 팀명 본부 직위 직급 사번 연락처 작성일 작성자 담당자 확인자 "
         "결재자 신청인 제출일 시행일 문서번호 비고 구분 항목 내용").split()

# 문장 재료. 앞뒤가 이어지는 느낌만 배우면 되므로 많을 필요는 없다.
SUBJECT = ("담당자 부서장 신청인 관리자 작성자 확인자 우리 회사 본부 팀원 직원 사용자 고객 "
           "튜립 다람쥐 표범 여우 고양이 강아지 참새 나비 소나무 바람 구름 하늘 바다").split()
VERB = ("확인한다 검토한다 승인한다 제출한다 보관한다 폐기한다 통보한다 기록한다 점검한다 관리한다 "
        "준수한다 보고한다 요청한다 반려한다 이관한다 열람한다 서명한다 날인한다 "
        "피어난다 흐른다 맑았다 고루 준다 지나간다 자란다").split()
OBJECT = ("서류를 문서를 자료를 기록을 내용을 항목을 결과를 사유를 근거를 절차를 권한을 "
          "한 송이를 열매를 그림자를 이야기를").split()
ADVERB = ("반드시 즉시 지체 없이 사전에 사후에 정기적으로 수시로 빠짐없이 신중히 고루 천천히 "
          "함께 조용히 부지런히").split()


# 문서에 실제로 나오는 라틴 토막. 사내 서식의 영문은 대개 약어이거나 제품·조직 이름이다.
#
# 예전에는 무작위 글자를 이어 붙여 낱말 흉내를 냈다('Dqpt', 'Bkzf'). 그런데 v7 을
# 재 보니 모델이 한글 첫 글자를 **바로 그런 모양의 라틴으로** 읽고 있었다:
#   '테스트' -> 'Ell 소 E'   '서약서' -> 'Xfl 약서'   'Codex' -> 'Cwlex'
# 크게 쓴 ㅌ 은 E 를 닮고 ㅅ 은 X 를 닮는다. 사람은 뒷글자로 가려내지만 디코더는
# 첫 글자를 만들 때 뒤를 모른다. 그때 '아무 라틴 글자열이나 그럴듯하다' 고 배워
# 두었으면 그쪽으로 샌다. 글자 수프를 가르쳐 놓고 글자 수프를 뱉는다고 할 수 없다.
LATIN_SHORT = ("AI IT DX ESG R&D CX TF PM QA HR RPA API DB UI UX KPI OKR SLA MOU NDA "
               "OCR ERP CRM SCM BI ML NLP LLM IR PR CS GA SI SM PoC B2B B2C").split()
LATIN_WORD = ("Codex Claude Copilot Cloud Data Team Lab Studio Core Next Alpha Beta "
              "Gamma Delta Prime Plus Pro Max Mini Nano Edge Hub Link Flow Sync Track "
              "Vision Insight Bridge Gateway Portal Center Global Local Smart Auto").split()


# 글꼴 109종이 못 그리는 장식 기호. 어휘에는 남겨 두되(인쇄물 판독이 쓴다)
# 손글씨 학습 이미지에는 넣지 않는다. 그릴 수 없는 것을 가르칠 수는 없다.
UNDRAWABLE = frozenset("·「」『』《》〈〉…°㎡№")

# 손으로 쓸 때 실제로 같이 나오는 기호들. 어휘의 나머지를 골고루 만나게 한다.
WRAPPERS = (("(", ")"), ("(", ")"), ("(", ")"), ("[", "]"), ("<", ">"),
            ("{", "}"), ("'", "'"), ('"', '"'))
TRAILERS = (".", ",", "!", "?", ";", ":", "~")
JOINERS = ("/", "-", "_", "+", "=", "&", "*", "#", "@", "%", "^", "|", "\\", "`")

# 줄 **가운데**에 놓아도 되는 이음표. 위의 JOINERS 전부를 쓰면 안 된다.
#
# 실측(v8 판독 34줄): 모델이 없는 기호를 콜론 뒤에 지어냈다.
#   '성명 : 홍길동'   -> '성명: ` 홍길동'
#   '이름 : 감자밭'   -> '미름: ` 강자발'
#   '부서: 인재육성팀' -> '부서: ` 인재육설팀'
# 세 줄 모두 **백틱 하나**로 깎였다. 까닭은 `{앞} {JOINERS} {뒤}` 를 6% 나
# 만들고 있었기 때문이다 — '성명 ` : 홍길동' 같은 줄을 배웠으니 그대로 뱉는다.
# 사람은 서식 줄 가운데에 백틱·역슬래시·꺾쇠를 쓰지 않는다.
#
# 나머지 기호를 어휘에서 빼지는 않는다. `random_line` 안에서는 그대로 나온다.
# 거기서는 앞뒤가 무작위라 '이름표 뒤에 오는 것' 으로 굳지 않는다.
MID_JOINERS = ("/", "-", "+", "&", "~")


# 지어낸 라틴을 만드는 재료. 자음과 모음을 번갈아 놓으면 읽히는 꼴이 된다
# ('Marex', 'Voleta'). 무작위 글자를 이어 붙인 수프('Dqpt')와는 다르다.
# 알파벳 26자가 **하나도 빠짐없이** 나와야 한다. 어휘에 있는데 학습에서 한 번도
# 안 나오는 글자는 죽은 자리가 된다(tests 의 `test_모든_토큰이_학습에_나온다`
# 가 'q' 와 'Y' 를 잡았다). 그래서 자음 목록에 q·x·y 를, 약어 머리에 전체를 둔다.
_LAT_CONS = "bcdfghjklmnpqrstvwxyz"
_LAT_VOW = "aeiou"
_LAT_HEAD = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _latin_word(rng: random.Random) -> str:
    """읽히는 꼴의 지어낸 낱말."""
    out = []
    for _ in range(rng.randint(2, 3)):
        out.append(rng.choice(_LAT_CONS) + rng.choice(_LAT_VOW))
    if rng.random() < 0.55:
        out.append(rng.choice("nrstxlm"))
    return "".join(out).capitalize()


def _latin(rng: random.Random) -> str:
    """문서에 나올 법한 라틴 토막 하나.

    절반쯤은 **지어낸 것**으로 둔다. 까닭이 있다.

    v11 을 15,000 걸음과 30,000 걸음에서 재 보니, 학습이 길어질수록 라틴으로
    바꿔 읽는 것이 심해졌다:
        'Codex 서약'     100% -> 53%  ('Codex Adge')
        '부서 - OCR조직'  100% -> 70%  ('OCR73')
        '데이터팀'  'Team 데이터팀' -> 'Track 데이터팀'
    재 보니 말뭉치의 라틴은 **양이 문제가 아니었다** — 라틴이 든 줄이 9.8% 로
    실제(20.6%)보다 오히려 적었다. 문제는 **쏠림**이었다. 177종뿐인데 흔한
    12개가 전체의 27% 라, 모델이 '라틴 다음엔 아는 라틴' 을 외워 버렸다.

    한글 쪽에서 통한 방법과 같다(`_made`, `_slot`). 목록을 줄이거나 늘리는 게
    아니라 **자리를 흔든다.** 라틴이 예측 불가능해지면 라틴 길의 빔 점수가
    낮아져서, 한글 그림 앞에서 라틴을 덜 고른다.
    """
    roll = rng.random()
    if roll < 0.30:
        return rng.choice(LATIN_SHORT)                    # 실제 약어
    if roll < 0.52:
        return rng.choice(LATIN_WORD)                     # 실제 낱말
    if roll < 0.72:
        return _latin_word(rng)                           # 지어낸 낱말
    if roll < 0.88:                                       # 지어낸 약어
        return "".join(rng.choice(_LAT_HEAD) for _ in range(rng.randint(2, 4)))
    if roll < 0.96:                                       # 이름 + 번호
        word = rng.choice(LATIN_WORD) if rng.random() < 0.5 else _latin_word(rng)
        return f"{word}{rng.choice((' ', '-', ''))}{rng.randint(1, 99)}"
    return _latin_word(rng).lower()


def _name(rng: random.Random) -> str:
    surname = rng.choice(SURNAME)
    if rng.random() < 0.04:                      # 남궁, 선우 같은 두 자 성
        surname += rng.choice(SURNAME)
    return surname + "".join(rng.choice(GIVEN) for _ in range(rng.choice((2, 2, 2, 3))))


def _org(rng: random.Random) -> str:
    """조직 이름.

    라틴 약어가 앞에 붙는 꼴을 늘렸다. v8 은 'AI혁신TF' 는 두 번 다 맞혔는데
    'OCR팀' 은 '인사얼' 로, 'OCR조직' 은 'OCRZ직' 으로 읽었다. 앞의 것은
    `_org` 가 실제로 만드는 모양이고(라틴 + 한글 + ORG_TAIL), 뒤의 것은
    **한글 없이 라틴이 곧바로 꼬리에 붙는 모양**이라 한 번도 못 봤다.
    약어 목록도 여섯 개뿐이라 OCR 은 그 자리에 서 본 적이 없었다.
    """
    head = _made(rng) if rng.random() < 0.15 else rng.choice(ORG_HEAD)
    if rng.random() < 0.25:
        head += rng.choice(ORG_HEAD)
    roll = rng.random()
    if roll < 0.10:
        head = rng.choice(LATIN_SHORT) + head        # AI혁신팀
    elif roll < 0.18:
        head = rng.choice(LATIN_SHORT)               # OCR팀 — 한글 없이 바로 꼬리
    elif roll < 0.21:
        head = rng.choice(LATIN_SHORT) + " "         # OCR 개발팀
    return head + rng.choice(ORG_TAIL)


def _made(rng: random.Random, least: int = 2, most: int = 4) -> str:
    """뜻 없는 한글 토막. 이름·상호·지어낸 말이 실제로 이렇게 생겼다."""
    return "".join(_syllable(rng) for _ in range(rng.randint(least, most)))


def _doc(rng: random.Random) -> str:
    """서류명.

    앞머리를 가끔 지어낸 말로 바꾼다. v10@15,000 이 '포테토뭉부서' 를
    '근태토명세서' 로 읽었는데 '근태' 도 '명세서' 도 우리가 넣어 준 말이다.
    꼬리(서약서·점검표)는 실제로 정해진 말이지만 **앞머리는 아무 말이나 온다.**
    그걸 안 가르치면 모델이 아는 앞머리로 바꿔 읽는다.
    """
    head = _made(rng) if rng.random() < 0.18 else rng.choice(DOC_HEAD)
    if rng.random() < 0.3:
        head += " " + rng.choice(DOC_HEAD)
    return head + ("" if rng.random() < 0.5 else " ") + rng.choice(DOC_TAIL)


def _date(rng: random.Random) -> str:
    y, m, d = rng.randint(2019, 2030), rng.randint(1, 12), rng.randint(1, 28)
    shape = rng.random()
    if shape < 0.4:
        return f"{y}-{m:02d}-{d:02d}"
    if shape < 0.7:
        return f"{y}.{m:02d}.{d:02d}"
    return f"{y}년 {m}월 {d}일"


def random_line(rng: random.Random) -> str:
    """언어 지식이 안 통하는 줄. 획만 보고 읽는 힘을 기른다."""
    kind = rng.random()
    if kind < 0.55:
        return "".join(_syllable(rng) for _ in range(rng.randint(1, 9)))
    if kind < 0.70:                              # 한글 + 영문/숫자 섞임
        parts = []
        for _ in range(rng.randint(2, 5)):
            pick = rng.random()
            if pick < 0.5:
                parts.append("".join(_syllable(rng) for _ in range(rng.randint(1, 3))))
            elif pick < 0.8:
                parts.append(_latin(rng))
            else:
                parts.append("".join(rng.choice("0123456789") for _ in range(rng.randint(1, 4))))
        return ("" if rng.random() < 0.6 else " ").join(parts)
    if kind < 0.82:                              # 홀로 쓴 자모 (초성 퀴즈 같은 것)
        jamo = [chr(c) for c in range(0x3131, 0x3164)]
        return " ".join("".join(rng.choice(jamo) for _ in range(rng.randint(1, 3)))
                        for _ in range(rng.randint(2, 5)))
    if kind < 0.86:                              # 라틴 토막
        return " ".join(_latin(rng) for _ in range(rng.randint(1, 2)))
    if kind < 0.93:                              # 기호가 섞인 토막
        # 이 기호들(` \ | ^ # @ % _ = *)은 서식 줄에서 뺐다. 사람이 '성명 : 값'
        # 같은 줄 가운데에 쓰지 않는데, 거기 두었더니 모델이 콜론 뒤에 지어냈다.
        #
        # 그렇다고 어디에도 안 쓰면 **어휘의 죽은 자리**가 된다. 학습에서 한 번도
        # 못 본 토큰은 영영 못 쓰는데, 인쇄물 판독은 그 글자를 만난다.
        # 그래서 여기, 앞뒤가 무작위라 '이름표 뒤' 로 굳지 않는 자리에 둔다.
        bits = []
        for _ in range(rng.randint(2, 5)):
            if rng.random() < 0.45:
                bits.append(rng.choice(JOINERS))
            else:
                bits.append("".join(_syllable(rng) for _ in range(rng.randint(1, 3))))
        return ("" if rng.random() < 0.5 else " ").join(bits)
    return "".join(rng.choice("0123456789") for _ in range(rng.randint(2, 10)))


def form_line(rng: random.Random) -> str:
    """사내 문서에서 실제로 보이는 줄 모양."""
    kind = rng.random()
    value = None
    if kind < 0.22:
        return _doc(rng)
    if kind < 0.40:
        value, label = _org(rng), rng.choice(("부서", "소속", "조직", "팀", "본부"))
    elif kind < 0.60:
        value, label = _name(rng), rng.choice(("성명", "이름", "작성자", "담당자", "신청인"))
    elif kind < 0.70:
        value, label = _date(rng), rng.choice(("작성일", "제출일", "시행일", "일자"))
    elif kind < 0.78:
        return _org(rng)
    elif kind < 0.86:
        return _name(rng)
    elif kind < 0.90:
        return _date(rng)
    elif kind < 0.95:
        return rng.choice(LABEL)
    else:
        return f"{rng.choice(LABEL)} : {rng.choice(('해당없음', '없음', '이상없음', '완료', '미비', 'N/A', '-'))}"

    # 이름표와 값 사이. **실측대로** 뽑는다.
    #
    # 실제 정답 34줄 가운데 콜론이 있는 15줄을 세어 보니 앞뒤가 이랬다:
    #     앞뒤 다 띄움 13   앞만 붙임 2   **뒤를 붙인 것 0**
    # 그런데 예전에는 여섯 가지를 고르게 뽑아서 `":"`(양쪽 다 붙음)가 17% 였다.
    # 붙여 그린 콜론은 앞 글자의 **받침처럼** 보인다. 실제로 가장 나쁜 줄 여섯 중
    # 둘이 이 잘못이었다:
    #     '이름 : 감자밭' -> '미릮 감자낱'
    #     '이름 : 김과박' -> '이릵길과박'
    # 둘 다 '름 :' 이 한 글자로 뭉쳤다. 우리가 그렇게 가르치고 있었다.
    sep = rng.choices((" : ", ": ", " ", "  ", " - "),
                      weights=(56, 10, 20, 10, 4))[0]
    return f"{label}{sep}{value}"


ODD_WORD = 0.30            # 문장 자리 하나를 지어낸 말로 바꾸는 비율


def _slot(pool: tuple, rng: random.Random, keep: int = 0) -> str:
    """문장의 한 자리. 가끔은 **틀에 없는 말**이 온다.

    v9 를 재 보니 모델이 문장 틀만 보고 아는 낱말을 넣어 읽었다.
        '초성 퀴즈를 맞히면 튜립 한 송이와' -> '정기적으로 밀해 튜립 한 송이준'
    '정기적으로' 는 ADVERB 에 있는 말이다. 사진에는 없다.

    문장 몫을 줄이는 것으로는 안 된다 — 문장을 못 읽게 된다. 틀은 그대로 두고
    **그 자리에 아무 말이나 올 수 있다**는 것을 같이 가르쳐야 한다. 조사와
    어미(`keep` 만큼의 뒷글자)는 남겨서 문장 꼴은 유지한다.
    """
    word = rng.choice(pool)
    if rng.random() >= ODD_WORD:
        return word
    tail = word[len(word) - keep:] if keep else ""
    made = "".join(_syllable(rng) for _ in range(rng.randint(2, 3)))
    return made + tail


def sentence_line(rng: random.Random) -> str:
    """문장. 앞뒤를 쓸 수 있게 하되, 이름을 고쳐 읽을 만큼 세지 않게 조금만 먹인다."""
    shape = rng.random()
    if shape < 0.45:
        return f"{_slot(SUBJECT, rng)}{rng.choice(('는', '은', '이', '가'))} "                f"{_slot(OBJECT, rng, 1)} {_slot(VERB, rng, 2)}"
    if shape < 0.75:
        return f"{_slot(ADVERB, rng)} {_slot(SUBJECT, rng)}{rng.choice(('는', '은'))} "                f"{_slot(OBJECT, rng, 1)} {_slot(VERB, rng, 2)}"
    return f"{_slot(SUBJECT, rng)}와 {_slot(SUBJECT, rng)}, {_slot(SUBJECT, rng)}를 "            f"{_slot(ADVERB, rng)} {_slot(VERB, rng, 2)}"


def long_sentence(rng: random.Random) -> str:
    """마디를 여럿 이어 붙인 긴 줄.

    실제 사진(hand-07)의 문장은 42자, 자모로 84개다. 30자짜리만 먹이면
    모델은 그렇게 긴 줄을 만들어 본 적이 없게 되고, 길이가 모자라 중간에서 끊는다.
    길이도 배워야 한다.
    """
    head = f"{_slot(SUBJECT, rng)}{rng.choice(('를', '을'))} {_slot(VERB, rng, 2)[:-2]}면"
    middle = ", ".join(_slot(SUBJECT, rng) for _ in range(rng.randint(2, 4)))
    tail = f"{_slot(ADVERB, rng)} {_slot(VERB, rng, 2)}"
    return (f"{head} {_slot(SUBJECT, rng)} {_slot(OBJECT, rng, 1)} "
            f"{middle}{rng.choice(('와', '과'))} {tail}.")


def title_line(rng: random.Random) -> str:
    """제목 줄. 서식의 맨 윗줄에 크고 짧게 쓰는 것.

    v3 를 실제 사진으로 재 보니 실패가 한쪽으로 몰려 있었다.

        제목 줄      0, 29, 34, 47, 53, 60%
        부서·이름 줄  67~95%

    까닭은 **짧고 넓다** 는 조합이 합성에 거의 없어서였다. 실측 첫 줄은 전부
    3~9 글자인데(테스트, 서약서, 보안점검표, Codex 서약, 포테토뭉부서)
    합성은 3 글자 이하가 10.8% 뿐이었다(실제 23.3%).
    """
    kind = rng.random()
    if kind < 0.26:
        return rng.choice(DOC_TAIL)                          # 서약서, 기준표
    if kind < 0.52:
        return rng.choice(DOC_HEAD) + rng.choice(DOC_TAIL)   # 보안점검표
    if kind < 0.68:
        return _org(rng)                                     # AI본부, 강원도팀
    if kind < 0.78:                                          # Codex 서약
        # 예전에는 여기서 **무작위 글자**를 이어 붙였다('Xflqk 서약'). 그래서
        # 실제 'Codex 서약' 을 만나면 앞은 맞히고 뒤를 'API' 로 바꿔 읽었다 —
        # 라틴 낱말 뒤에 한글 서류명이 오는 줄을 한 번도 못 봤기 때문이다.
        word = rng.choice(LATIN_WORD if rng.random() < 0.72 else LATIN_SHORT)
        tail = rng.choice(DOC_TAIL if rng.random() < 0.7 else ORG_TAIL)
        return f"{word} {tail}" if rng.random() < 0.75 else f"{word}{tail}"
    if kind < 0.95:                                          # 지어낸 말 (테스트, 포테토뭉부서)
        return _made(rng) + (rng.choice(ORG_TAIL + DOC_TAIL) if rng.random() < 0.55 else "")
    return rng.choice(DOC_HEAD) + rng.choice(("TF", "팀", "부", "실")) + rng.choice(DOC_TAIL)


# 갈래 섞는 비율. 실제 칸의 **글자 수**에 맞춘다.
#
#     글자수        10%  25%  50%  75%  90%  최대
#     실제 34줄      3    4    6    7    8    17
#     고치기 전      3    5    7   10   16    38   <- 열에 셋이 실제 최장보다 길다
#
# 긴 줄을 아주 없애지는 않는다. 이 사진 11장은 서식 한 종류라, 자유 기입란이
# 있는 다른 서류에서는 긴 줄이 온다. 꼬리를 **줄이되 남긴다.**
MIX = ((0.34, random_line), (0.68, form_line), (0.89, title_line),
       (0.97, sentence_line), (1.00, long_sentence))


def line(rng: random.Random) -> str:
    roll = rng.random()
    for edge, maker in MIX:
        if roll < edge:
            return maker(rng)
    return random_line(rng)






def decorate(text: str, rng: random.Random) -> str:
    """기호를 조금 얹는다. 사람이 손으로 쓸 때 하는 만큼만."""
    roll = rng.random()
    if roll < 0.04:
        open_mark, close_mark = rng.choice(WRAPPERS)
        return f"{open_mark}{text}{close_mark}"
    if roll < 0.10:
        return f"{text}{rng.choice(TRAILERS)}"
    if roll < 0.12:                              # 줄머리 표시. 사람이 실제로 쓰는 것만.
        return f"{rng.choice(('-', '*'))} {text}"
    if roll < 0.16:
        # 띄어쓰기가 없으면 넣을 자리가 없다. 그냥 둔다.
        # 여기서 `return text` 를 빼먹으면 아래 번호 갈래로 흘러내려서,
        # **띄어쓰기 없는 줄에만** 번호가 여섯 배로 붙는다(실측 0.8% -> 4.8%).
        # 모델이 '9. 릜 길 과 댁', '7. 자금윰 TF 서약' 을 뱉던 게 이것이다.
        if " " not in text:
            return text
        head, _, tail = text.partition(" ")
        return f"{head} {rng.choice(MID_JOINERS)} {tail}"
    if roll < 0.168:
        return f"{rng.randint(1, 20)}{rng.choice(('.', ')', '-'))} {text}"
    return text


# 줄머리를 꾸미는 몫을 줄인 까닭.
#
# 실측(v7 판독): 모델이 없는 것을 줄 앞에 지어냈다 — '18. 팀 : 이설',
# '17. 자긍륨 TF 서약', '9. 뢼길라먹'. 재 보니 합성 줄의 11.3% 가 숫자로,
# 9.8% 가 기호로 시작하고 있었다(실제 34줄은 0줄).
#
# 특히 나빴던 것은 `{JOINERS} {글월}` 이었다. '% 명세서', '` 인재팀', '\ 시행일'
# 같은 것을 4% 나 만들고 있었는데, 사람은 줄 앞에 그런 기호를 쓰지 않는다.
# 모델이 뱉던 쓰레기가 정확히 그 기호들이었다.
#
# 번호는 남겨 둔다 — 사내 서식에는 실제로 '1. 성명' 같은 줄이 있다.
# 다만 4% 에서 2.5% 로 낮췄다. 사진 11장에 없다고 0 으로 만들면 그 11장에
# 과적합된다.


def lines(count: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    made = []
    while len(made) < count:
        text = decorate(line(rng), rng)
        if text and not (UNDRAWABLE & set(text)):
            made.append(text)
    return made
