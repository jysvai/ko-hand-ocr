"""다 그린 줄에 **합성이 아직 안 하는** 사람 손의 변이를 입힌다 (2026-10-11).

사용자 말: 「어느정도 학습된 케이스가 있으면 웹 서치를 하든 다른 변이를 일으키든 해서 손글씨체 다른
방안으로도 1차 시도 해보고 90% 정합성을 넘기지 못하면 다 학습」, 「지금 당장 학습된 데이터셋으로만
유지하는 건 좋은 케이스가 아닌거같아」.

그래서 둘로 나눈다. `tools/probe.py` 가 변이마다 지금 모델로 **먼저 읽혀 보고**, 90% 를 못 넘는 변이만
학습에 넣는다(`train.py --vary-<이름> 비율`). 이미 읽는 변이를 먹이면 걸음만 쓴다.

`synth.render` 가 이미 하는 것(줄 기울기·흐림·카메라 잡티·번짐·종이 얼룩·굵기·기울여 쓰기·흔들기·
획 휘기·손버릇·지운 자국)은 여기 없다. 모두 `synth.render` 가 내준 그림(흰 종이, 어두운 잉크)에 입힌다.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .cell import INK_EDGE

NAMES = {
    "pencil": "연필",
    "ruled": "줄 공책·모눈",
    "gaps": "획 끊김",
    "press": "필압",
    "link": "이어 쓰기",
    "crowd": "글자 겹침",
    "lowres": "멀리서 찍음",
}
KINDS = tuple(NAMES)


def _paper(arr: np.ndarray) -> float:
    blank = arr[arr >= INK_EDGE]
    return float(np.median(blank)) if blank.size else 245.0


def _stroke(arr: np.ndarray) -> int:
    """획 굵기(px). 잉크 칸마다 가로로 이어진 길이의 가운데값."""
    ink = arr < INK_EDGE
    runs = []
    for row in ink[:: max(ink.shape[0] // 12, 1)]:
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(np.int8), [0]))))
        runs += list(edges[1::2] - edges[::2])
    return max(int(np.median(runs)) if runs else 2, 1)


def _segments(arr: np.ndarray) -> list[tuple[int, int]]:
    """잉크가 있는 열의 덩이들 (왼쪽, 오른쪽+1). 글자 하나이거나 글자의 한 조각이다."""
    cols = (arr < INK_EDGE).any(axis=0)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], cols.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def pencil(page: Image.Image, rng: random.Random) -> Image.Image:
    """연필: 옅고(잉크 95~150) 결이 거칠고 군데군데 비는 획."""
    arr = np.asarray(page, dtype=np.float32)
    ink = arr < INK_EDGE
    noise = np.random.default_rng(rng.getrandbits(32))
    tone = rng.uniform(95, 150)
    out = arr.copy()
    out[ink] = tone + noise.normal(0, rng.uniform(12, 28), int(ink.sum()))
    drop = ink & (noise.random(arr.shape) < rng.uniform(0.05, 0.18))
    out[drop] = _paper(arr)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def ruled(page: Image.Image, rng: random.Random) -> Image.Image:
    """줄 공책·모눈: 글씨 밑(가끔 위)에 줄, 셋에 하나는 세로줄까지."""
    out = page.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    arr = np.asarray(page)
    rows = np.flatnonzero((arr < INK_EDGE).any(axis=1))
    top, bottom = (int(rows[0]), int(rows[-1])) if rows.size else (0, height - 1)
    tone = int(rng.uniform(150, 205))
    thick = rng.choice((1, 1, 2))
    lines = [min(height - 1, bottom + rng.randint(-max((bottom - top) // 8, 1), 3))]
    if rng.random() < 0.5:
        lines.append(max(0, top - rng.randint(0, 4)))
    for y in lines:
        draw.line([(0, y), (width, y)], fill=tone, width=thick)
    if rng.random() < 0.33:
        step = max(int((bottom - top) * rng.uniform(0.5, 1.0)), 6)
        for x in range(rng.randint(0, step), width, step):
            draw.line([(x, 0), (x, height)], fill=tone, width=1)
    # 잉크 위에 줄이 덮이지 않게 — 펜이 줄 위를 지나간다.
    return Image.fromarray(np.minimum(np.asarray(out), arr))


def gaps(page: Image.Image, rng: random.Random) -> Image.Image:
    """획 끊김: 펜이 안 나와 획 몇 군데가 짧게 빈다."""
    arr = np.asarray(page).copy()
    ink = np.argwhere(arr < INK_EDGE)
    if not len(ink):
        return page
    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    paper = int(_paper(arr))
    stroke = _stroke(arr)
    count = max(2, int(arr.shape[1] / max(arr.shape[0], 1) * rng.uniform(1.0, 2.5)))
    for _ in range(count):
        y, x = ink[rng.randrange(len(ink))]
        reach = stroke * rng.uniform(1.5, 3.0)
        angle = rng.uniform(0, np.pi)
        dx, dy = reach * np.cos(angle), reach * np.sin(angle)
        draw.line([(x - dx, y - dy), (x + dx, y + dy)], fill=paper,
                  width=max(1, int(stroke * rng.uniform(0.6, 1.2))))
    return out


def press(page: Image.Image, rng: random.Random) -> Image.Image:
    """필압: 줄을 따라 진하기와 굵기가 천천히 바뀐다."""
    arr = np.asarray(page, dtype=np.float32)
    width = arr.shape[1]
    x = np.linspace(0, 1, width)
    wave = np.zeros(width)
    for _ in range(3):
        wave += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * (rng.uniform(0.5, 3.0) * x + rng.random()))
    level = (wave - wave.min()) / max(np.ptp(wave), 1e-6)          # 0..1
    paper = _paper(arr)
    light = paper - (paper - arr) * (0.45 + 0.55 * level)[None, :]
    bold = np.asarray(Image.fromarray(np.clip(light, 0, 255).astype(np.uint8))
                      .filter(ImageFilter.MinFilter(3)), dtype=np.float32)
    heavy = (level > 0.7)[None, :]
    out = np.where(heavy, bold, light)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def link(page: Image.Image, rng: random.Random) -> Image.Image:
    """이어 쓰기: 앞 덩이의 끝에서 다음 덩이의 머리로 가는 가는 획."""
    arr = np.asarray(page)
    parts = _segments(arr)
    if len(parts) < 2:
        return page
    out = page.copy()
    draw = ImageDraw.Draw(out)
    ink = arr < INK_EDGE
    tone = int(np.median(arr[ink])) if ink.any() else 60
    stroke = max(1, int(_stroke(arr) * rng.uniform(0.4, 0.8)))
    share = rng.uniform(0.35, 0.8)
    for (_, a1), (b0, _) in zip(parts, parts[1:]):
        if b0 - a1 > arr.shape[0] * 0.6 or rng.random() > share:
            continue                       # 띄어 쓴 자리는 잇지 않는다
        left = np.flatnonzero(ink[:, a1 - 1])
        right = np.flatnonzero(ink[:, b0])
        if not left.size or not right.size:
            continue
        start = (a1 - 1, int(left[-1] if rng.random() < 0.7 else left[len(left) // 2]))
        end = (b0, int(right[0] if rng.random() < 0.6 else right[len(right) // 2]))
        bend = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + rng.uniform(-1, 1) * arr.shape[0] * 0.15)
        draw.line([start, bend, end], fill=tone, width=stroke, joint="curve")
    return out


def crowd(page: Image.Image, rng: random.Random) -> Image.Image:
    """글자 겹침: 덩이 사이를 좁혀 글자끼리 닿게 한다. 띄어 쓴 자리는 덜 좁힌다."""
    arr = np.asarray(page)
    parts = _segments(arr)
    if len(parts) < 2:
        return page
    paper = int(_paper(arr))
    out = np.full_like(arr, paper)
    at = parts[0][0]
    out[:, : parts[0][0]] = arr[:, : parts[0][0]]
    # 낱말 사이는 남긴다. 처음에는 그림 높이로 갈랐는데 그림에는 위아래 여백이 있어서 띄어 쓴
    # 자리까지 다 붙었다(「정보보호서약서제출」). 글자 높이와, 틈들의 가운데값의 두 배로 가른다.
    rows = np.flatnonzero((arr < INK_EDGE).any(axis=1))
    tall = int(rows[-1] - rows[0] + 1) if rows.size else arr.shape[0]
    holes = [b - a for (_, a), (b, _) in zip(parts, parts[1:])]
    word = max(0.3 * tall, 2.0 * float(np.median(holes)))
    for i, (left, right) in enumerate(parts):
        if i:
            gap = left - parts[i - 1][1]
            keep = rng.uniform(0.6, 0.9) if gap > word else rng.uniform(-0.3, 0.3)
            at = max(at + int(gap * keep), 0)
        piece = arr[:, left:right]
        span = min(piece.shape[1], out.shape[1] - at)
        if span <= 0:
            break
        out[:, at:at + span] = np.minimum(out[:, at:at + span], piece[:, :span])
        at += span
    end = min(out.shape[1], at + (arr.shape[1] - parts[-1][1]))
    return Image.fromarray(out[:, :max(end, 8)])


def lowres(page: Image.Image, rng: random.Random) -> Image.Image:
    """멀리서 찍음: 해상도를 3~5할로 떨어뜨렸다 되돌리고 jpeg 로 뭉갠다."""
    width, height = page.size
    scale = rng.uniform(0.3, 0.5)
    small = page.resize((max(8, int(width * scale)), max(8, int(height * scale))),
                        Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    small.save(buf, "JPEG", quality=rng.randint(30, 65))
    buf.seek(0)
    return Image.open(buf).convert("L").resize((width, height), Image.Resampling.BICUBIC)


def apply(page: Image.Image, kind: str, rng: random.Random) -> Image.Image:
    return globals()[kind](page, rng)


def maybe(page: Image.Image, shares: dict[str, float], rng: random.Random) -> Image.Image:
    """학습용. `shares` 의 차례대로 그 비율만큼 입힌다. 비어 있으면 난수를 안 뽑는다."""
    for kind in KINDS:
        if shares.get(kind) and rng.random() < shares[kind]:
            page = apply(page, kind, rng)
    return page
