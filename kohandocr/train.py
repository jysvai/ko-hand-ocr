"""ko-hand-ocr 학습.

    python -m kohandocr.train --fonts <글꼴폴더> --steps 30000 --out runs/base

디스크에 미리 구운 데이터가 없다. 매 걸음마다 새로 그린 그림이 들어온다.
그래서 '몇 epoch' 이 아니라 '몇 걸음' 으로 센다.

되돌아볼 눈금은 두 가지를 따로 본다.
  - **글자 맞춤**  한 자 한 자가 맞았나 (자모 단위)
  - **줄 맞춤**    한 줄이 통째로 맞았나

줄 맞춤이 진짜 쓸모의 눈금이다. 이름 세 자 중 한 자가 틀리면 그 서류는 못 찾는다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import data, model as builder, synth
from .vocab import SPECIAL, Vocab


def one_per_core(workers: int) -> tuple[int, int]:
    """Windows 에서 물리 코어마다 논리 CPU 를 **하나만** 남긴다. (남긴 수, 원래 수).

    학습은 그림을 그리는 일꾼(CPU)이 GPU 보다 느리다. 그런데 Windows 가 일꾼 둘을
    **한 코어의 두 스레드**(SMT)에 앉히는 일이 있다(2026-09-26, 논리 CPU 2·3 과
    6·7 이 짝으로 찼다). 그러면 두 일꾼이 한 코어를 나눠 써서 저마다 절반쯤으로
    느려진다. 실측으로 46장/초이던 v86 을 짝수 CPU 로만 묶자 곧바로 80장/초 넘게
    올랐다(예전 판들은 64~73). 운에 맡기던 자리라 판마다 속도가 달랐다.

    일꾼은 이 프로세스가 **나중에** 띄우므로 여기서 묶으면 그대로 물려받는다.
    일꾼이 물리 코어 수보다 많거나(묶으면 오히려 모자라다) SMT 가 없거나 Windows 가
    아니면 아무것도 안 한다. 코어를 못 읽으면 조용히 그대로 둔다 — 속도 문제일
    뿐 틀린 답을 내지는 않는다.
    """
    total = os.cpu_count() or 0
    if sys.platform != "win32" or total < 2:
        return 0, total
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        size = wintypes.DWORD(0)
        k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # 0 = 코어
        buf = ctypes.create_string_buffer(size.value)
        if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
            return 0, total
        raw, at, mask, cores = buf.raw, 0, 0, 0
        while at + 8 <= size.value:
            kind, length = struct.unpack_from("<II", raw, at)
            if length == 0:
                break
            if kind == 0:
                # PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20],
                # GroupCount, 그 뒤 GROUP_AFFINITY(Mask, Group) — 8 바이트 자리 맞춤.
                core, group = struct.unpack_from("<QH", raw, at + 32)
                cores += 1
                if group == 0 and core:
                    mask |= core & -core                 # 그 코어의 첫 논리 CPU
            at += length
        kept = bin(mask).count("1")
        if kept == total or kept != cores or workers + 1 > kept:
            return 0, total
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            return 0, total
        return kept, total
    except (OSError, AttributeError, struct.error):
        return 0, total


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ko-hand-ocr 학습")
    ap.add_argument("--fonts", required=True, help="손글씨 글꼴 폴더 (.ttf)")
    ap.add_argument("--out", default="runs/base", help="저장 위치")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder-lr", type=float, default=3e-5,
                    help="인코더는 이미 배운 게 있어 살살 건드린다")
    ap.add_argument("--warmup", type=int, default=1_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--letters", type=int, default=64, help="한 줄 최대 자모 수")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--check-every", type=int, default=1_000)
    ap.add_argument("--save-every", type=int, default=2_000)
    ap.add_argument("--keep", type=int, default=15_000,
                    help="이 걸음마다 판을 따로 남긴다 (덮어쓰지 않고). 0 이면 안 남김")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="이어서 학습할 저장 폴더")
    ap.add_argument("--strike", type=float, default=None,
                    help="지운 자국을 넣을 줄 비율. 안 주면 synth.STRIKE. 학습 자료만 바꾼다")
    ap.add_argument("--warp", type=float, default=None,
                    help="글자 속 획을 휠 줄 비율. 안 주면 synth.WARP(끔). 학습 자료만 바꾼다")
    ap.add_argument("--digits", type=float, default=0.0,
                    help="날짜가 아닌 숫자 줄(corpus.digit_line)로 바꿀 비율. 0 이면 끔. 학습 자료만 바꾼다")
    ap.add_argument("--free", type=float, default=0.0,
                    help="뜻 없는 줄(corpus.random_line)로 더 바꿀 비율. 0 이면 끔. 학습 자료만 바꾼다")
    ap.add_argument("--hard", type=float, default=0.0,
                    help="못 읽는 글꼴을 더 자주 뽑을 몫(synth.Fonts.lean). 0 이면 고르게. "
                         "--hard-from 과 같이 준다")
    ap.add_argument("--hard-from", default="",
                    help="글꼴마다 잰 점수 파일(tools/hardfonts.py 가 쓴다)")
    ap.add_argument("--tangle", type=float, default=0.0,
                    help="지운 자국 가운데 제멋대로 엉킨 고리(synth._tangle)로 그릴 몫. "
                         "0 이면 끔. 학습 자료만 바꾼다")
    ap.add_argument("--layers", type=int, default=0,
                    help="이어받은 판의 디코더가 이보다 얕으면 이만큼 늘려서 학습한다 "
                         "(`model.grow`). 0 이면 그대로")
    # 아래 넷은 **배우는 방식**을 바꾼다. 자료는 그대로라 시험지가 안 바뀌고,
    # 끈 채로는(기본값) 손실을 예전과 똑같이 낸다. 2026-10-06 에 넣은 까닭은
    # `blur_inputs` 머리말.
    ap.add_argument("--noise", type=float, default=0.0,
                    help="디코더에 먹이는 앞 글자 가운데 엉뚱한 글자로 바꿀 몫. 0 이면 끔")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="라벨 스무딩. 0 이면 끔")
    ap.add_argument("--freeze-decoder", action="store_true",
                    help="디코더를 얼리고 인코더(눈)만 배운다")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="학습하는 동안만 디코더 드롭아웃을 이 값으로. 0 이면 그대로. "
                         "저장하는 설정은 안 바꾼다")
    ap.add_argument("--sam", type=float, default=0.0,
                    help="SAM 반경. 걸음마다 기울기 쪽으로 이만큼 밀어 본 자리의 기울기로 "
                         "간다(걸음이 두 배 느리다). 0 이면 끔")
    ap.add_argument("--teacher", default="",
                    help="증류: 이 모델들의 평균 분포를 따라 배운다. 쉼표로 가른 폴더들, "
                         "또는 'ensemble'(runs/ENSEMBLE.json 의 식구). 비면 끔")
    ap.add_argument("--distill", type=float, default=0.5,
                    help="증류 손실의 몫(0~1). 나머지는 정답 손실")
    ap.add_argument("--temp", type=float, default=2.0, help="증류 온도")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="무게를 걸음마다 지수 평균한 것(EMA)으로 **저장한다**. 값은 "
                         "감쇠(예: 0.999 = 대략 마지막 천 걸음). 0 이면 끔")
    return ap.parse_args()


class Shadow:
    """학습 궤적의 무게를 지수 평균한 것(EMA). 저장은 이 무게로 한다.

    왜 있나(2026-10-06). 이어받은 판과 새 판을 **반씩 섞은** 판이 혼자 재서
    부모 둘보다 나았다(v81-5000 과 그 자식 v101: 글씨체 평균 95.1/94.9 ->
    섞은 판 95.3). 섞기는 두 점만 평균하지만 EMA 는 지나온 길 전체를 평균한다.
    학습 자체(손실, 걸음)는 안 바꾸고 **남기는 무게만** 바꾼다.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay, self.count = decay, 0
        self.held = {k: v.detach().float().clone()
                     for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.count += 1
        # 처음 몇 걸음은 짧게 평균한다. 이어받은 판에 너무 오래 묶이지 않게.
        rate = min(self.decay, (1 + self.count) / (10 + self.count))
        now = model.state_dict()
        for key, kept in self.held.items():
            kept.mul_(rate).add_(now[key].detach().float(), alpha=1 - rate)

    @contextlib.contextmanager
    def worn(self, model: torch.nn.Module):
        """잠깐 평균 무게를 입힌다(저장하는 동안). 나오면 학습 무게로 돌린다."""
        now = model.state_dict()
        back = {k: now[k].detach().clone() for k in self.held}
        model.load_state_dict({k: v.to(back[k].dtype) for k, v in self.held.items()},
                              strict=False)
        try:
            yield
        finally:
            model.load_state_dict(back, strict=False)


ROOT = Path(__file__).resolve().parent.parent


def teachers_of(spec: str, vocab: Vocab, device: str) -> tuple[list, list[str]]:
    """증류의 선생들. 'ensemble' 이면 지금 앙상블(`runs/ENSEMBLE.json`)의 식구다.

    어휘가 한 자라도 다르면 분포를 못 견주므로 세운다.
    """
    if spec == "ensemble":
        book = json.loads((ROOT / "runs" / "ENSEMBLE.json").read_text(encoding="utf-8"))
        paths = list(book["members"])
    else:
        paths = [p for p in spec.split(",") if p]
    models = []
    for path in paths:
        folder = Path(path) if Path(path).is_absolute() else ROOT / path
        if Vocab.load(folder / "vocab.json").tokens != vocab.tokens:
            raise SystemExit("선생 %s 의 어휘가 다르다 — 증류할 수 없다" % path)
        teacher = builder.load(str(folder)).to(device).eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        models.append(teacher)
    return models, paths


def teach(teachers: list, pixels: torch.Tensor, feed: torch.Tensor, temp: float) -> torch.Tensor:
    """선생들의 다음 글자 분포를 **확률로 평균**한다(온도 `temp`). 기울기 없음."""
    with torch.no_grad():
        probs = [torch.softmax(t(pixel_values=pixels, decoder_input_ids=feed).logits.float() / temp, -1)
                 for t in teachers]
    return torch.stack(probs).mean(0)


def distill_loss(logits: torch.Tensor, soft: torch.Tensor, labels: torch.Tensor,
                 temp: float) -> torch.Tensor:
    """KL(선생 평균 || 학생). 정답 자리(-100 아닌 곳)만 센다. 온도 제곱을 곱해
    기울기 크기를 정답 손실과 맞춘다(Hinton 2015)."""
    keep = labels != -100
    logq = torch.log_softmax(logits.float() / temp, -1)
    kl = (soft * (torch.log(soft.clamp_min(1e-9)) - logq)).sum(-1)
    return kl[keep].mean() * temp * temp


def lean(params: list, rho: float) -> list:
    """SAM 의 첫 걸음. 무게를 지금 기울기 쪽으로 `rho` 만큼 밀고, 민 몫을 돌려준다.

    왜 있나(2026-10-06). 배우는 것은 글꼴에서 찍은 합성인데 시험은 사람 손씨
    사진이다. 날카로운 골짜기에 앉은 무게는 그 옮김에 약하고, 둘레가 평평한
    자리는 덜 흔들린다는 것이 SAM 이다. 민 자리에서 다시 잰 기울기로 간다.
    """
    live = [p for p in params if p.grad is not None]
    if not live:
        return []
    norm = torch.norm(torch.stack([p.grad.detach().float().norm(2) for p in live]), 2)
    scale = rho / (norm + 1e-12)
    pushed = []
    with torch.no_grad():
        for p in live:
            step = (p.grad * scale).to(p.dtype)
            p.add_(step)
            pushed.append((p, step))
    return pushed


def unlean(pushed: list) -> None:
    """`lean` 이 민 몫을 되돌린다. 기울기는 민 자리에서 잰 것이 남는다."""
    with torch.no_grad():
        for p, step in pushed:
            p.sub_(step)


def blur_inputs(labels: torch.Tensor, start: int, pad: int, share: float,
                low: int, high: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """디코더에 먹일 앞 글자(라벨을 한 칸 민 것). `share` 만큼은 엉뚱한 글자로.

    왜. 가장 낮은 글씨체(동해독도)의 큰 손실은 잘못 본 자모가 아니라 **지어낸
    줄**이다 — '816883539' -> '2025-03-14', '사번' -> '신청인', '근태 규정' ->
    '구매 규정'(2026-10-05 `tools/confuse.py`). 흐린 글씨에서 디코더가 그림 대신
    앞 글자를 보고 학습 글월의 틀을 이어 쓴다. 학습 때 앞 글자를 가끔 틀리게
    주면 앞 글자만 믿고 이어 갈 수 없어 그림을 더 본다.

    첫 자리(시작 토큰)와 빈자리는 건드리지 않는다. 바꾸는 글자는 특수 토큰을 뺀
    `[low, high)` 에서 고른다. 정답(`labels`)은 그대로라 맞혀야 할 것은 같다.
    """
    shifted = labels.new_full(labels.shape, pad)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = start
    shifted = shifted.masked_fill(shifted == -100, pad)
    if share <= 0:
        return shifted
    roll = torch.rand(labels.shape, device=labels.device, generator=gen)
    hit = (roll < share) & (shifted != pad)
    hit[:, 0] = False
    other = torch.randint(low, high, labels.shape, device=labels.device, generator=gen)
    return torch.where(hit, other, shifted)


def loosen(decoder, rate: float) -> int:
    """학습하는 동안만 디코더 드롭아웃을 `rate` 로. 바꾼 자리 수.

    설정(`config`)은 안 건드린다 — 저장하는 판의 설정이 바뀌면 그 판을 읽는
    쪽이 다른 모델을 보게 된다. 드롭아웃은 읽을 때(`eval`) 어차피 꺼진다.
    """
    hit = 0
    for module in decoder.modules():
        for key in ("dropout", "activation_dropout"):
            if isinstance(getattr(module, key, None), float):
                setattr(module, key, rate)
                hit += 1
        if isinstance(module, torch.nn.Dropout):
            module.p = rate
            hit += 1
    return hit


def groups(model, lr: float, encoder_lr: float):
    """인코더와 디코더에 다른 배움 속도를 준다.

    인코더는 ImageNet 에서 이미 눈을 떴다. 세게 흔들면 그 눈이 망가진다.
    디코더는 백지라 빨리 배워야 한다.
    """
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        thin = param.ndim <= 1 or name.endswith(".bias") or "layernorm" in name.lower()
        (plain if thin else decay).append((name, param))

    def split(items, is_encoder):
        return [p for n, p in items if n.startswith("encoder.") == is_encoder]

    return [
        {"params": split(decay, True), "lr": encoder_lr, "weight_decay": 0.01},
        {"params": split(plain, True), "lr": encoder_lr, "weight_decay": 0.0},
        {"params": split(decay, False), "lr": lr, "weight_decay": 0.01},
        {"params": split(plain, False), "lr": lr, "weight_decay": 0.0},
    ]


def pace(step: int, warmup: int, total: int) -> float:
    """천천히 올렸다가 부드럽게 내린다."""
    if step < warmup:
        return step / max(warmup, 1)
    done = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(done, 1.0)))


@torch.no_grad()
def check(model, vocab, batch, device) -> dict:
    """지금 실력을 잰다. 그리디로 읽고 글자·줄 단위로 센다."""
    model.eval()
    pixels = data.to_pixels(batch["gray"].to(device))
    ids = model.generate(pixels, max_new_tokens=vocab_room(batch))
    got = [vocab.decode(row.tolist()) for row in ids]
    want = [text for text in batch["text"]]
    letters_ok = letters_all = lines_ok = 0
    for guess, truth in zip(got, want):
        import unicodedata
        a = unicodedata.normalize("NFD", guess)
        b = unicodedata.normalize("NFD", truth)
        letters_ok += sum(1 for x, y in zip(a, b) if x == y)
        letters_all += len(b)
        lines_ok += guess == truth
    model.train()
    return {
        "글자": letters_ok / max(letters_all, 1),
        "줄": lines_ok / len(want),
        "보기": list(zip(want[:4], got[:4])),
    }


def vocab_room(batch) -> int:
    return int(batch["labels"].shape[1]) + 4


def main() -> None:
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("경고: GPU 가 없다. CPU 로는 사실상 학습이 안 된다.")
    torch.manual_seed(args.seed)

    vocab = Vocab.default()
    vocab.save(out / "vocab.json")

    model = builder.load(args.resume) if args.resume else builder.build(
        vocab, letters=args.letters)
    if args.layers:
        added = builder.grow(model, args.layers)
        if added:
            print("디코더를 %d층 늘렸다 -> %d층" % (added, args.layers))
    model.to(device)
    print("모델(M):", {k: round(v, 1) for k, v in builder.sizes(model).items()},
          "| 어휘", len(vocab), "| 입력", synth.CELL, "|", device)

    stream = data.Lines(args.fonts, vocab, letters=args.letters,
                        seed=args.seed, length=args.steps * args.batch,
                        strike=args.strike, warp=args.warp, digits=args.digits,
                        free=args.free, hard=args.hard, hard_from=args.hard_from,
                        tangle=args.tangle)
    if args.hard and not args.hard_from:
        raise SystemExit("--hard 에는 --hard-from 이 있어야 한다 (tools/hardfonts.py)")
    if args.strike is not None:
        print("지운 자국: 줄의 %.1f%% (기본 %.1f%%)" % (100 * args.strike, 100 * synth.STRIKE))
    if args.warp is not None:
        print("획 휘기: 줄의 %.1f%% (기본 %.1f%%)" % (100 * args.warp, 100 * synth.WARP))
    if args.digits:
        print("날짜 아닌 숫자 줄: %.1f%%" % (100 * args.digits))
    if args.free:
        print("뜻 없는 줄 더: %.1f%%" % (100 * args.free))
    if args.hard:
        print("못 읽는 글꼴 몰기: 뽑기의 %.0f%% (%s)" % (100 * args.hard, args.hard_from))
    if args.tangle:
        print("엉킨 지운 자국: 지운 자국의 %.0f%%" % (100 * args.tangle))
    if args.noise:
        print("디코더 입력 흐리기: 앞 글자의 %.0f%%" % (100 * args.noise))
    if args.smooth:
        print("라벨 스무딩: %g" % args.smooth)
    if args.freeze_decoder:
        for name, param in model.named_parameters():
            if not name.startswith("encoder."):
                param.requires_grad_(False)
        print("디코더 얼림: 인코더(눈)만 배운다")
    if args.dropout:
        print("디코더 드롭아웃: %g (학습 중에만, %d자리)"
              % (args.dropout, loosen(model.decoder, args.dropout)))
    if args.sam:
        print("SAM: 반경 %g (걸음마다 앞뒤로 두 번 — 두 배 느리다)" % args.sam)
    teachers: list = []
    if args.teacher:
        teachers, names = teachers_of(args.teacher, vocab, device)
        print("증류: 선생 %d (%s), 몫 %g, 온도 %g"
              % (len(teachers), ", ".join(n.replace("runs/", "") for n in names),
                 args.distill, args.temp))
    shadow = Shadow(model, args.ema) if args.ema else None
    if shadow:
        print("무게 지수 평균(EMA): %g — 저장은 평균 무게로" % args.ema)
    wear = (lambda: shadow.worn(model)) if shadow else contextlib.nullcontext
    kept, total = one_per_core(args.workers)
    if kept:
        print("코어마다 하나씩: 논리 CPU %d -> %d (일꾼이 한 코어를 나눠 쓰지 않게)"
              % (total, kept))
    loader = DataLoader(
        stream, batch_size=args.batch, num_workers=args.workers,
        collate_fn=data.collate,
        pin_memory=(device == "cuda"), persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    held = data.Lines(args.fonts, vocab, letters=args.letters, seed=9_999, length=64)
    holdout = data.collate(list(held))

    optimizer = torch.optim.AdamW(groups(model, args.lr, args.encoder_lr), betas=(0.9, 0.98))
    base = [g["lr"] for g in optimizer.param_groups]
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else None

    history, running, seen, started = [], 0.0, 0, time.perf_counter()
    for step, batch in enumerate(loader, start=1):
        share = pace(step, args.warmup, args.steps)
        for group, start in zip(optimizer.param_groups, base):
            group["lr"] = start * share

        pixels = data.to_pixels(batch["gray"].to(device, non_blocking=True))
        labels = batch["labels"].to(device, non_blocking=True)
        feed = None
        if args.noise or args.smooth or teachers:
            # 흐리기가 0 이면 원래 밀기와 같다(시험이 있다).
            feed = blur_inputs(labels, model.config.decoder_start_token_id,
                               model.config.pad_token_id, args.noise,
                               len(SPECIAL), len(vocab))
        soft = None
        if teachers:
            with (amp if amp is not None else contextlib.nullcontext()):
                soft = teach(teachers, pixels, feed, args.temp)

        def forward() -> torch.Tensor:
            with (amp if amp is not None else contextlib.nullcontext()):
                if feed is None:
                    return model(pixel_values=pixels, labels=labels).loss
                logits = model(pixel_values=pixels, decoder_input_ids=feed).logits
                truth = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                    ignore_index=-100, label_smoothing=args.smooth)
                if soft is None:
                    return truth
                return ((1 - args.distill) * truth
                        + args.distill * distill_loss(logits, soft, labels, args.temp))

        loss = forward()
        loss.backward()
        if args.sam:
            pushed = lean(list(model.parameters()), args.sam)
            optimizer.zero_grad(set_to_none=True)
            forward().backward()
            unlean(pushed)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if shadow:
            shadow.update(model)

        running += loss.item()
        seen += pixels.shape[0]
        # 첫 걸음은 **반드시** 찍는다. 띄우는 쪽(tools/train.ps1)이 이 줄을
        # '살아서 돈다'는 신호로 쓰는데, log_every 가 500 이면 그 줄이 4분 뒤에야
        # 나온다. 지켜보는 쪽은 4분을 못 기다리고 멀쩡한 학습을 죽인 뒤 다시
        # 띄웠고, 죽은 부모의 일꾼들이 GPU 를 쥔 채 남아 **학습이 둘 동시에**
        # 떴다. 'CUDA out of memory' 로 다섯 번 실패하던 것의 진짜 뿌리다.
        if step == 1 or step % args.log_every == 0:
            rate = seen / (time.perf_counter() - started)
            print("%6d  손실 %.4f  lr %.2e  %.0f장/초" % (
                step, running / (1 if step == 1 else args.log_every),
                optimizer.param_groups[2]["lr"], rate), flush=True)
            running = 0.0
        if step % args.check_every == 0:
            score = check(model, vocab, holdout, device)
            print("       └ 글자 %.1f%%  줄 %.1f%%" % (score["글자"] * 100, score["줄"] * 100))
            for truth, guess in score["보기"][:2]:
                print("           %r -> %r" % (truth, guess))
            history.append({"step": step, **{k: v for k, v in score.items() if k != "보기"}})
            (out / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=1), encoding="utf-8")
        if step % args.save_every == 0 or step == args.steps:
            with wear():
                model.save_pretrained(out)
            print("       └ 저장:", out)
        # 마지막 판이 가장 좋은 판이 아니다. 실측(v8): 30,000 걸음이 81.4% 인데
        # 끝까지 간 45,000 걸음은 80.1% 였다. 낮은 학습률에서 합성 쪽으로 더
        # 붙는다. 덮어쓰기만 하면 좋았던 판을 잃는다.
        #
        # 이 갈래는 `save_every` **안에 두면 안 된다.** 안에 두면 두 값의
        # 공배수에서만 걸린다 — `--keep 3000 --save-every 2500` 으로 돌렸더니
        # 15,000 걸음에서 딱 한 번 남았다. 조용히 안 남으니 다 끝나고 나서야
        # 알게 된다.
        if args.keep and step % args.keep == 0 and step < args.steps:
            spare = out.parent / f"{out.name}-{step}"
            with wear():
                model.save_pretrained(spare)
            vocab.save(spare / "vocab.json")
            print("       └ 따로 남김:", spare)
        if step >= args.steps:
            break

    with wear():
        model.save_pretrained(out)
    print("끝. 저장:", out)


if __name__ == "__main__":
    main()
