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
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import data, model as builder, synth
from .vocab import Vocab


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
    return ap.parse_args()


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
    model.to(device)
    print("모델(M):", {k: round(v, 1) for k, v in builder.sizes(model).items()},
          "| 어휘", len(vocab), "| 입력", synth.CELL, "|", device)

    stream = data.Lines(args.fonts, vocab, letters=args.letters,
                        seed=args.seed, length=args.steps * args.batch)
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
        if amp is not None:
            with amp:
                loss = model(pixel_values=pixels, labels=labels).loss
        else:
            loss = model(pixel_values=pixels, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
            model.save_pretrained(spare)
            vocab.save(spare / "vocab.json")
            print("       └ 따로 남김:", spare)
        if step >= args.steps:
            break

    model.save_pretrained(out)
    print("끝. 저장:", out)


if __name__ == "__main__":
    main()
