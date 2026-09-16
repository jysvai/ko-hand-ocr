<div align="center">

# ko-hand-ocr

**Reads one line of handwritten Korean + English.**
A 31M model trained from scratch on synthetic data — no handwriting dataset, no inherited terms.

[![PyPI](https://img.shields.io/pypi/v/ko-hand-ocr)](https://pypi.org/project/ko-hand-ocr/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/jysvai/ko-hand-ocr/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/jysvai/ko-hand-ocr)
[![Params](https://img.shields.io/badge/params-31M-1baf7a)](https://github.com/jysvai/ko-hand-ocr)
[![Weights](https://img.shields.io/badge/weights-116MB-1baf7a)](https://github.com/jysvai/ko-hand-ocr/releases/tag/v0.2.0)
[![CPU](https://img.shields.io/badge/runs%20on-CPU%20only-eda100)](https://github.com/jysvai/ko-hand-ocr)
[![Data](https://img.shields.io/badge/training%20data-synthetic-e87ba4)](https://github.com/jysvai/ko-hand-ocr/blob/main/PROVENANCE.md)

[한국어](https://github.com/jysvai/ko-hand-ocr/blob/main/README.md) · **English**

</div>

```
photo  ->  line cutting  ->  ko-hand-ocr  ->  "부서 : 포테토뭉부서"
```

---

## At a glance

| | |
|---|---|
| **What it reads** | One line of handwritten Korean + English, from a photo |
| **Model** | ViT-Small encoder (ImageNet, Apache-2.0) + 4-layer TrOCR decoder trained from scratch |
| **Vocabulary** | 229 jamo tokens — not 11,172 syllables, so unseen characters are still writable |
| **Parameters** | **31M** — one seventh of `ko-trocr` (213.7M) |
| **Weights** | **116 MB** single · 465 MB four-way ensemble *(downloaded separately)* |
| **Package** | 83 KB — the code only |
| **Speed** | **0.16 s per cell** · 1.37 s for a whole photo — **CPU only, no GPU** |
| **Memory** | 890 MB single · 1.35 GB ensemble |
| **Accuracy** | **93.5%** on handwriting fonts never seen in training (93.9% ensemble) |
| **Training data** | Synthetic — drawn on the fly from OFL fonts. Nothing is stored on disk |
| **License** | Apache-2.0, weights included — **no dataset terms inherited** |
| **Python** | 3.10+ · PyTorch 2.5+ |

## What it does — and does not

**Does**

- Reads a **whole photo**: finds the lines, cuts them, reads each one (`read_photo`)
- Reads **pre-cut line images**, batched (`read`)
- **Constrained decoding** — restrict the output to a candidate list, and report whether the free and constrained readings agree (`both`)
- **Ensemble reading** — run several checkpoints and take the answer they agree on
- Runs **fully offline on CPU.** Nothing is sent anywhere

**Does not**

- Split a table cell into label and value — `read_photo` only cuts down to lines
- Handle vertical writing or merged cells
- Stay silent on an empty cell — a cell that is pure scribble still produces something

## Why it exists

In practice the only publicly available Korean handwriting OCR model is
`ddobokki/ko-trocr`, and its training data comes from AI Hub, which places conditions
on purpose of use and on redistribution. That blocks it from being embedded in an
in-house tool or shipped as a public package. So **the same capability was rebuilt
from scratch** — with a provenance chain that can be audited part by part.

![ko-hand-ocr vs ko-trocr](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-compare.svg)

## Install

```bash
pip install ko-hand-ocr
```

Weights ship separately — they are too large for the repository, so they are attached
to [Releases](https://github.com/jysvai/ko-hand-ocr/releases/tag/v0.2.0).

> Weights are versioned separately from the package. The current weights are on the
> **v0.2.0** release; a package bump does not mean the weights changed.

| Download | Size | What it is |
|---|---|---|
| `ko-hand-ocr-v46.zip` | 116MB | **A single checkpoint.** Enough for most uses. 0.16 s/cell |
| `ko-hand-ocr-ensemble.zip` | 465MB | Four checkpoints. More accurate, 10x slower |

```bash
curl -LO https://github.com/jysvai/ko-hand-ocr/releases/download/v0.2.0/ko-hand-ocr-v46.zip
```

Pass the unzipped folder straight to `Reader()`. It must contain `config.json`,
`vocab.json` and `model.safetensors`.

## Quick start

```python
from kohandocr.reader import Reader

reader = Reader("ko-hand-ocr-v46", device="cpu")   # the unzipped folder

# A whole photo. Line cutting is included.
reader.read_photo(open("scan.jpg", "rb").read())
# -> ['보안점검표', '부서 : 포테토뭉부서', '이름 : 감자밭']

# If the lines are already cut, hand the images in directly. Batch them for speed.
reader.read([cell_a, cell_b, cell_c])

# Given a candidate list, the output can be constrained to it.
reader.both([cell_b], options=["포테토뭉부서", "감자밭", "김클로드"])
# -> [{'text': '부서 : 포테토뭄부서', 'constrained': '포테토뭉부서', 'agrees': True}]
```

**Do not trust the constrained result on its own.** Because it is constrained, the
model *must* return something from the list. If the true answer is not in the list you
get a confident wrong answer and nothing looks off. When `agrees` is false, hand it
to a human.

### Reading with several checkpoints

Different checkpoints fail on different cells. Train a little longer on the same data
and some cells survive while others die — which tells you those cells were being read
by a thin margin. So keep several checkpoints and take **the answer they agree on**.

```
unzipped/
  config.json  vocab.json  model.safetensors
  also/
    v20b/  config.json  vocab.json  model.safetensors
    v19/   ...
```

If `also/` is present, `Reader` uses it automatically. The call site does not change.

Combining by confidence was measured and **it did not work** — the model is sometimes
more confident about a wrong answer. So selection is by **how much the outputs agree**,
not by confidence.

## How it is built

```
  64 x 640 line image
          |
   ViT-Small encoder          facebook/deit-small-patch16-224 (ImageNet-1k, Apache-2.0)
   patch 16                   pretrained weights, continued at a lower learning rate
          |
   TrOCR decoder              4 layers, 6 heads — trained from scratch
          |
   229 jamo tokens            ㄱ ㅏ ㅁ ... assembled into 감
          |
      "부서 : 감자밭"
```

Two decisions carry most of the result.

**Jamo, not syllables.** Memorising all 11,172 Hangul syllables needs a large output
layer and still fails on anything rare. 229 jamo compose into any syllable, so a
character the model never saw in training is still writable — which matters for names.

**The encoder is not frozen, but it is not shaken either.** It already learned to see
on ImageNet, so it continues at a lower learning rate while the blank decoder learns
fast. Training both at the same rate destroys what the encoder knew.

## Accuracy

Measured two ways. **Both matter.** All figures below come from
`python tools/verify.py` (400 lines per font, 11 photos).

| | unseen fonts (mean) | worst font | 11 handwriting photos | worst photo |
|---|---|---|---|---|
| single (v46) | 93.5% | 86.4% | 89.6% | 66.7% |
| four | **93.9%** | **88.1%** | **95.6%** | **83.1%** |

Similarity is measured per jamo; on the photo side each photo gets one vote.

**Do not take the photo number at face value.** It is measured on 11 photos (37 cells)
taken by the author, so it swings ±5%p. The figure closer to what you would see on
someone else's handwriting is the **unseen-fonts** column — tested only on six font
families never used in training, where the swing is half as wide (±2-3%p).

Four checkpoints open a **large gap on photos** (95.6 vs 89.6) and a small one on
fonts (93.9 vs 93.5). The photo gap is large because where one checkpoint collapses
another carries the cell — the worst photo goes from 66.7% with one checkpoint to
83.1% with four. The cost is being ten times slower.

### Per font (400 lines each)

![Accuracy by handwriting font](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-accuracy.svg)

Look only at the average and **the font that collapses is hidden.** The real spread
is 12%p.

| Font | single (v46) | four |
|---|---|---|
| **EastSeaDokdo (동해독도)** | **86.4%** | **88.1%** |
| KirangHaerang (기랑해랑) | 92.4% | 92.6% |
| HiMelody (하이멜로디) | 93.2% | 93.2% |
| NanumPenScript (나눔펜스크립트) | 94.0% | 94.2% |
| GamjaFlower (감자꽃) | 96.9% | 97.0% |
| SingleDay (싱글데이) | 97.9% | 98.1% |

What has to clear the bar is not the mean but **the lowest font.** The goal is 90% on
all six; EastSeaDokdo has not got there yet.

All six are fonts **never used in training**, and all are SIL Open Font License.

```bash
python tools/holdout.py <checkpoint-dir> --per-font --sample sample.png
```

### What the test sheet actually looks like

![unseen-font test sheet](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/holdout-sample.png)

Three lines per font, with the ground truth, what the model read, and the similarity.
The pale band on the right is the leftover space from fitting into a 64x640 frame —
this is exactly what the model sees.

It shows why the gap opens. **SingleDay** has well-separated characters and intact
strokes, close to print. **EastSeaDokdo** smears and drops strokes, so `안전사무국`
looks like a different word almost entirely — a human cannot read it without context
either.

This is **an image, not a font file.** Drawing glyphs with a font is what the OFL
permits; what is not redistributed is the `.ttf` files themselves
([PROVENANCE.md](https://github.com/jysvai/ko-hand-ocr/blob/main/PROVENANCE.md)).

## Speed and footprint (measured)

No GPU needed. Emitting jamo one at a time is the bottleneck, so more cores do not help
much — and by the same token **it does not get slower on a weak machine.**

CPU only, median over 11 photos (37 cells). Time is split into **cutting and reading** —
shrinking the model does not shrink cutting, and when a photo holds only three or four
lines, cutting is more than half the cost.

| | cut | read | one photo (3.4 cells) | per cell |
|---|---|---|---|---|
| single, beam 5 | 0.82s | 0.55s | 1.37s | 0.16s |
| single, greedy | 0.82s | 0.32s | 1.13s | 0.10s |
| four, beam 5 | 0.82s | 5.73s | 6.55s | 1.70s |

Memory is about 890MB for one checkpoint, about 1.35GB for four.

Produced with `python tools/bench.py --all --device cpu`. That harness walks exactly the
path the application walks — a speed measured along a different path is not the speed
the user gets.

## Where it runs

**Built and tested on Windows.** Training and inference were both run on
Windows 11 with Python 3.13/3.14.

| | |
|---|---|
| Windows | Where it was built. Training and inference both verified |
| macOS | **Untested.** Inference is pure PyTorch + PIL so it should run, but it has not been confirmed |
| Linux | Not tried yet |

`tools/train.ps1` is PowerShell, so it is Windows-only. Elsewhere, call
`python -m kohandocr.train` directly — all that script does is wait for the previous
run to release the GPU, launch it, and watch the first few steps.

## Training

No data is baked ahead of time. A corpus line is generated, drawn with a handwriting
font, and fed in **on the spot**. Nothing is left on disk, and a change to the
distribution takes effect from the next step.

```bash
python tools/fetch_fonts.py                       # fetch OFL handwriting fonts
python -m kohandocr.train --out runs/v1 \
       --steps 20000 --batch 40 --workers 3 --letters 128
```

Fonts are looked up in this order. **No path is hard-coded.**

1. whatever `--fonts` was given
2. the `KOHAND_FONTS` environment variable
3. `~/.cache/ko-hand-ocr/fonts`

The fonts are not in this repository and are **not redistributed.** The 109 Nanum
handwriting fonts come from [clova.ai/handwriting](https://clova.ai/handwriting).
Details in [PROVENANCE.md](https://github.com/jysvai/ko-hand-ocr/blob/main/PROVENANCE.md).

How closely the synthetic images match real handwriting is checked against the measured
distributions recorded in `kohandocr/measured.json` — nine of them (ink coverage,
density, aspect ratio, margins and so on).

```bash
python tools/match.py        # target vs current synthesis, side by side
```

**Running longer does not make it better.** After changing the synthesis distribution,
run short, keep intermediate checkpoints, and pick between them (measured: 15,000 steps
was the peak and 30,000 steps was worse).

```bash
python tools/pick.py runs/v1-15000 runs/v1-30000    # score the kept checkpoints and pick
```

## How this was built

The engineering behind this model is written up as a 12-page document — why it exists,
the four decisions that shaped it, how the training data was produced at zero labelling
cost, how the evaluation was designed, and what is still missing.

**[Engineering write-up — PDF, 12 pages](https://github.com/jysvai/ko-hand-ocr/releases/download/v0.2.0/ko-hand-ocr-portfolio-en.pdf)** &nbsp;·&nbsp;
[한국어판](https://github.com/jysvai/ko-hand-ocr/releases/download/v0.2.0/ko-hand-ocr-portfolio-ko.pdf)

It includes the parts that are usually left out: the measurement that was wrong, the
approach that was tried and failed, and the numbers that have not been measured yet.

## License

Apache-2.0, weights included. The licenses of the fonts used in training do not attach
to the weights — the weights are not a derivative of the glyph artwork, they are values
learned from images. The reasoning is recorded part by part in
[PROVENANCE.md](https://github.com/jysvai/ko-hand-ocr/blob/main/PROVENANCE.md).
