<div align="center">

# ko-hand-ocr

**Reads one line of handwritten Korean + English.**
A 41M model trained from scratch on synthetic data — no handwriting dataset, no inherited terms.

[![PyPI](https://img.shields.io/pypi/v/ko-hand-ocr)](https://pypi.org/project/ko-hand-ocr/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/jysvai/ko-hand-ocr/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/jysvai/ko-hand-ocr)
[![Params](https://img.shields.io/badge/params-41M-1baf7a)](https://github.com/jysvai/ko-hand-ocr)
[![Weights](https://img.shields.io/badge/weights-156MB-1baf7a)](https://github.com/jysvai/ko-hand-ocr/releases/tag/v0.4.1)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-localdeel%2Fko--hand--ocr-ffd21e)](https://huggingface.co/localdeel/ko-hand-ocr)
[![CPU](https://img.shields.io/badge/runs%20on-CPU%20only-eda100)](https://github.com/jysvai/ko-hand-ocr)
[![Data](https://img.shields.io/badge/training%20data-synthetic-e87ba4)](https://github.com/jysvai/ko-hand-ocr/blob/main/PROVENANCE.md)

[한국어](https://github.com/jysvai/ko-hand-ocr/blob/main/README.md) · **English**

</div>

```
photo  ->  line cutting  ->  ko-hand-ocr  ->  "부서 : 포테토뭉부서"
```

![ko-hand-ocr scoreboard](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-board.en.svg)

The top table puts us next to **six other public OCR models** on the same test sheets, on
the same PC at the same moment (`tools/vs.py` — who was run and how is
[below](#measured-against-other-ocr-on-the-same-ruler)). Below it are our two models item
by item, from `tools/verify.py` at about 960 lines per font. The **single model** reads with
one 41M model. The **ensemble** runs three models and picks the answer they agree on —
more accurate, seven times slower.

---

## At a glance

| | |
|---|---|
| **What it reads** | One line of handwritten Korean + English, from a photo |
| **Model** | ViT-Small encoder (ImageNet, Apache-2.0) + 8-layer TrOCR decoder trained from scratch |
| **Vocabulary** | 229 jamo tokens — not 11,172 syllables, so unseen characters are still writable |
| **Parameters** | **41M** — one fifth of `ko-trocr` (213.7M) |
| **Weights** | **156 MB** single model · 450 MB ensemble (3 models) *(downloaded separately — GitHub Releases · [Hugging Face](https://huggingface.co/localdeel/ko-hand-ocr))* |
| **Package** | 119 KB — the code only |
| **Plugs into** | a server that speaks the OpenAI and Ollama APIs (`ko-hand-ocr-serve`) · an MCP tool for LM Studio (`ko-hand-ocr-mcp`) |
| **Speed** | **0.18 s per line** · 1.04 s for a whole photo — **CPU only, no GPU** |
| **Memory** | 1.07 GB single model · 1.51 GB ensemble |
| **Accuracy** | **95.6%** mean on six handwriting fonts never seen in training (95.6% ensemble) · worst font 90.4% (90.7% ensemble) · 11 handwriting photos 94.9% (96.6% ensemble) · 11 of 17 items over 95% (ensemble 12) |
| **Against others** | On the same sheets, ahead of the best of six public OCR models by **+17.7 points** on the six fonts (ko-trocr) and **+15.7 points** on the photos (ko-trocr) — single model |
| **Training data** | Synthetic — drawn on the fly from OFL fonts. Nothing is stored on disk |
| **License** | Apache-2.0, weights included — **no dataset terms inherited** |
| **Python** | 3.10+ · PyTorch 2.5+ |

## What it does — and does not

**Does**

- Reads a **whole photo**: finds the lines, cuts them, reads each one (`read_photo`)
- Reads **pre-cut line images**, batched (`read`)
- **Constrained decoding** — restrict the output to a candidate list, and report whether the free and constrained readings agree (`both`)
- **Ensemble reading** — run several models and take the answer they agree on
- Runs **fully offline on CPU.** Nothing is sent anywhere
- **Plugs into LM Studio, Ollama and vLLM tooling** — a server speaking their API, and an MCP tool ("Use it with LM Studio, Ollama and vLLM" below)

**Does not**

- Split a table cell into label and value — `read_photo` only cuts down to lines
- Handle vertical writing or merged cells
- Stay silent on an empty cell — a cell that is pure scribble still produces something

## Why it exists

The only public model aimed at Korean **handwriting** that we could find is
`ddobokki/ko-trocr`, and its training data comes from AI Hub, which places conditions on purpose of use and on
redistribution. That blocks it from being embedded in an in-house tool or shipped as a
public package. General OCR engines (PaddleOCR, EasyOCR, Tesseract) and large VLMs fall
well short on a line of handwriting — on the same sheets the best of them reads
77.6% of the six fonts and 78.5% of the photos (below). So
**a model for this job was built from scratch** — with a provenance chain that can be
audited part by part.

## Measured against other OCR on the same ruler

There are six rivals — one from each kind of thing people actually reach for to read Korean.

| Rival | Kind | Size | How it was run |
|---|---|---|---|
| `ddobokki/ko-trocr` | Korean TrOCR (trained on AI Hub) | 214M | transformers, float32, beam 5, length cap raised 16 -> 64 |
| PaddleOCR PP-OCRv5 | line recogniser (`korean_PP-OCRv5_mobile_rec`) | 3.3M | paddleocr 3.7.0 / paddle 3.4.0, `TextRecognition` |
| EasyOCR | line recogniser (`korean_g2`) | 4.0M | easyocr 1.7.2, `Reader(["ko", "en"]).recognize()` |
| Tesseract 5 | line recogniser (LSTM, no parameter count published) | 6MB file | tesseract 5.5.3, `--oem 1 --psm 7 -l kor+eng` |
| PaddleOCR-VL-1.6 | OCR-specific VLM | 906M | transformers 5.16.1, bfloat16, prompt "OCR:" (model card) |
| Qwen3-VL-2B-Instruct | general VLM | 2.1B | transformers 5.16.1, bfloat16, Korean prompt — the best of three we tried |

Every rival runs **exactly as its model card says**, each in its own environment
(`tools/rival_worker.py`), and only the time the engine itself reports is counted (handing
images across is not).

A comparison only holds if **the model is the only thing that differs**. Photos were cut
once with our line cutter (`kohandocr.page`) and the **same cells** were handed to
everyone. Font test sheets were handed over **raw**, exactly as `synth.render` drew them —
putting our preprocessing (64×640 letterbox) on them would squash the others twice.
Metric, GPU and the moment of measurement are all shared. What was measured and how is
spelled out in the header of `tools/vs.py`, and every number below comes from the
`runs/VS.json` it leaves behind.

![Accuracy](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-vs-accuracy.svg)

#### Real handwriting photos · 34 lines, 209 characters
|  | Jamo similarity | Character error rate | Exact line match | Worst photo |
|---|---|---|---|---|
| ko-hand-ocr ensemble (118M) | **96.52%** | **8.61%** | **67.6%** | **90.28%** |
| ko-hand-ocr single model (41M) | 94.24% | 13.40% | 61.8% | 83.03% |
| ddobokki/ko-trocr (214M) | 78.50% | 39.23% | 26.5% | 58.56% |
| PaddleOCR PP-OCRv5 (3.3M) | 71.75% | 39.71% | 8.8% | 34.81% |
| Qwen3-VL-2B (2.1B) | 69.86% | 38.28% | 8.8% | 46.86% |
| PaddleOCR-VL-1.6 (906M) | 61.71% | 44.02% | 2.9% | 18.52% |
| EasyOCR (4.0M) | 66.88% | 55.02% | 2.9% | 46.19% |
| Tesseract 5 (6MB) | 13.21% | 98.56% | 0.0% | 0.00% |

#### Six unseen handwriting fonts · 670 lines, 4708 characters
|  | Jamo similarity | Character error rate | Exact line match | Worst font |
|---|---|---|---|---|
| ko-hand-ocr ensemble (118M) | **95.65%** | **8.28%** | **73.0%** | **89.83%** |
| ko-hand-ocr single model (41M) | 95.36% | 8.45% | 72.2% | 89.02% |
| ddobokki/ko-trocr (214M) | 77.62% | 41.67% | 29.4% | 65.24% |
| PaddleOCR PP-OCRv5 (3.3M) | 72.28% | 40.40% | 28.4% | 52.43% |
| Qwen3-VL-2B (2.1B) | 66.02% | 79.89% | 19.9% | 50.02% |
| PaddleOCR-VL-1.6 (906M) | 62.97% | 62.34% | 18.1% | 43.05% |
| EasyOCR (4.0M) | 56.35% | 64.21% | 9.7% | 36.95% |
| Tesseract 5 (6MB) | 17.42% | 98.13% | 0.9% | 7.29% |

#### Breadth — 24 more fonts · 648 lines, 3768 characters
|  | Jamo similarity | Character error rate | Exact line match | Worst font |
|---|---|---|---|---|
| ko-hand-ocr ensemble (118M) | **97.71%** | **3.42%** | **85.5%** | **92.46%** |
| ko-hand-ocr single model (41M) | 97.20% | 3.98% | 84.7% | 89.14% |
| ddobokki/ko-trocr (214M) | 79.94% | 32.38% | 42.0% | 46.52% |
| PaddleOCR PP-OCRv5 (3.3M) | 75.27% | 30.02% | 39.5% | 18.66% |
| Qwen3-VL-2B (2.1B) | 67.81% | 62.98% | 26.1% | 24.46% |
| PaddleOCR-VL-1.6 (906M) | 66.69% | 45.94% | 24.8% | 27.33% |
| EasyOCR (4.0M) | 64.38% | 48.91% | 19.1% | 19.28% |
| Tesseract 5 (6MB) | 19.98% | 95.12% | 2.2% | 3.07% |

![All 480 fonts](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-vs-fonts.svg)

#### All 480 fonts · 12 lines each, 4320 lines, 22560 characters
|  | held out (6)<br>6 | breadth (24)<br>24 | seen in training<br>450 | All<br>480 | fonts ≥ 95% | fonts < 80% |
|---|---|---|---|---|---|---|
| ko-hand-ocr ensemble (118M) | 99.15% | 99.00% | 98.22% | **98.27%** | 436 | 5 |
| ko-hand-ocr single model (41M) | 98.49% | 98.58% | 98.06% | **98.09%** | 434 | 4 |
| ddobokki/ko-trocr (214M) | 85.29% | 83.29% | 82.48% | **82.56%** | 46 | 158 |
| PaddleOCR PP-OCRv5 (3.3M) | 80.07% | 78.48% | 78.02% | **78.07%** | 86 | 198 |
| Qwen3-VL-2B (2.1B) | 65.17% | 67.35% | 69.32% | **69.17%** | 50 | 289 |
| PaddleOCR-VL-1.6 (906M) | — | — | — | — | — | — |
| EasyOCR (4.0M) | 67.73% | 68.17% | 67.15% | **67.21%** | 9 | 319 |
| Tesseract 5 (6MB) | 21.56% | 19.28% | 20.50% | **20.45%** | 0 | 480 |

- **Even the best of the six rivals is far behind.** On the six unseen fonts the best
  rival is ko-trocr at 77.62%; our single model (41M) reads
  95.36%. On the twenty-four it is ko-trocr 79.94% against
  97.20%.
- **Nobody has ever seen the eleven photos.** That is the cleanest ground here: our
  ensemble reads 96.52% and the single model 94.24%, against
  78.50% for the best rival (ko-trocr). Lines read exactly right:
  67.6% against 26.5% (the best rival value).
- **A large VLM does not read a handwritten line well just by being large.**
  Qwen3-VL-2B (2.1B) reads 66.02% of the six fonts and
  PaddleOCR-VL-1.6 (906M) 62.97%. They are twenty to fifty times our
  size but were not trained for this job. Reading whole documents is another matter (not
  measured).
- **Tesseract barely reads handwriting** (17.42% on the fonts,
  13.21% on the photos). It is a printed-text engine, as expected. It stays as
  a baseline.
- **Run every single font in the folder — all 480 of them** — and we read
  98.27% (ensemble) and 98.09% (single) against
  82.56% for the best rival (ko-trocr). **The 24 fonts we have never
  seen (99.00%) score higher than the 450 used in training
  (98.22%)** — so this is not a number propped up by
  memorisation. PaddleOCR-VL takes about a second per line at its cheapest batch, so it was
  not run on all 480 (blank in the table).
- **Some fonts still collapse.** The worst of the 480 sits at 47.31% for our single
  model and 63.21% for the ensemble. Six fonts would never have shown that. With
  only 12 lines per font a single font's score swings hard, so read the herd totals and
  the distribution, not one row.
- **The single model has a lower photo floor than 0.3.0.** Its worst photo is hand-06 at
  83.03% — one line (`전자금융TF서약`) of a three-line photo. In the
  ensemble another model carries it and it stays at 90.28%.
- **The 0.4.1 single model gained on fonts only.** Against 0.4.0 (v83) it moves
  from 95.20 to 95.36% on the six fonts, 96.87 to 97.20% on the twenty-four and
  98.00 to 98.09% on all 480 — up on all three rulers. The photo row in this table
  went from 94.95 to 94.24%: 22 -> 21 of 34 lines read exactly, **one line's
  worth** (`tools/verify.py` gives 95.0 vs 94.9%).

### Where the gap opens

![CER by line content](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-vs-kind.svg)

#### CER by line content — lower is better
|  | Latin mixed in<br>66 lines | form label<br>60 lines | digits mixed in<br>84 lines | Hangul word / name<br>304 lines | Hangul sentence<br>156 lines |
|---|---|---|---|---|---|
| ko-hand-ocr ensemble (118M) | 14.1% | 8.3% | 9.6% | 4.7% | 8.1% |
| ko-hand-ocr single model (41M) | 14.5% | 7.6% | 8.7% | 5.5% | 8.6% |
| ddobokki/ko-trocr (214M) | 59.5% | 30.3% | 49.4% | 25.0% | 46.1% |
| PaddleOCR PP-OCRv5 (3.3M) | 53.7% | 32.1% | 43.7% | 33.5% | 40.9% |
| Qwen3-VL-2B (2.1B) | 58.7% | 96.5% | 49.1% | 104.5% | 82.1% |
| PaddleOCR-VL-1.6 (906M) | 75.7% | 65.9% | 45.6% | 81.7% | 51.4% |
| EasyOCR (4.0M) | 68.5% | 55.6% | 64.8% | 64.7% | 64.1% |
| Tesseract 5 (6MB) | 90.8% | 103.3% | 82.3% | 114.5% | 96.0% |

![CER by line length](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-vs-span.svg)

#### CER by line length — lower is better
|  | 1-5 chars<br>286 lines | 6-10 chars<br>270 lines | 11-20 chars<br>108 lines | 21+ chars<br>6 lines |
|---|---|---|---|---|
| ko-hand-ocr ensemble (118M) | 6.1% | 8.4% | 8.3% | 17.6% |
| ko-hand-ocr single model (41M) | 6.5% | 8.9% | 8.1% | 15.7% |
| ddobokki/ko-trocr (214M) | 30.7% | 35.0% | 50.7% | 92.1% |
| PaddleOCR PP-OCRv5 (3.3M) | 42.9% | 33.8% | 40.1% | 92.1% |
| Qwen3-VL-2B (2.1B) | 101.2% | 84.0% | 59.0% | 85.6% |
| PaddleOCR-VL-1.6 (906M) | 96.3% | 50.8% | 55.5% | 58.8% |
| EasyOCR (4.0M) | 70.4% | 58.5% | 63.0% | 97.2% |
| Tesseract 5 (6MB) | 114.6% | 98.9% | 86.0% | 98.6% |

- **Latin abbreviations split them.** Character error rate is 14.5% for our single
  model against 53.7% for the best rival there (PaddleOCR PP-OCRv5). The Latin
  that turns up on forms is mostly abbreviations (`TF`, `OCR`, `Codex`); reading Hangul
  and Latin together on one line is where they part.
- **Longer lines split them further.** At 21+ characters: 15.7% against
  58.8% for the best rival (PaddleOCR-VL-1.6). ko-trocr's encoder is a 384×384
  square, so a long line is squashed whole into it; ours is 64×640. **The 21+ bucket holds
  only 6 lines, though** — read it as a direction, not a result.

### What it costs

![Speed and size](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-vs-speed.svg)

#### Speed and size
|  | Parameters | Download | lines/s (GPU) | lines/s (CPU) | 30-line page (GPU) | 30-line page (CPU) | Peak VRAM |
|---|---|---|---|---|---|---|---|
| ko-hand-ocr ensemble (118M) | 118M | 450MB | 6.9 | 0.66 | 4.76s | 45.8s | 1772MB |
| ko-hand-ocr single model (41M) | 41M | 156MB | 34.2 | 3.08 | 1.31s | 10.2s | 1052MB |
| ddobokki/ko-trocr (214M) | 214M | 408MB | 5.3 | 0.52 | 6.09s | 58.1s | 1798MB |
| PaddleOCR PP-OCRv5 (3.3M) | 3.3M | 13MB | **336.5** | 14.68 | **0.52s** | 2.5s | 111MB |
| Qwen3-VL-2B (2.1B) | 2.1B | 4058MB | 6.5 | 0.47 | 5.02s | 64.1s | 5724MB |
| PaddleOCR-VL-1.6 (906M) | 906M | 1828MB | 0.9 | 0.04 | 32.80s | 680.7s | 1937MB |
| EasyOCR (4.0M) | 4.0M | 15MB | 32.5 | 14.88 | 1.36s | 2.5s | **70MB** |
| Tesseract 5 (6MB) | — | 6MB | — | **22.25** | — | **1.8s** | — |

- **We are not the fastest.** On CPU, Tesseract 5 (22.2 lines/s), EasyOCR (14.9 lines/s), PaddleOCR PP-OCRv5 (14.7 lines/s)
  beat our single model (3.1 lines/s). They read
  Tesseract 5 17.4%, EasyOCR 56.4%, PaddleOCR PP-OCRv5 72.3% of the six fonts.
  **Read accuracy and speed together** — on a form a wrong answer is worse than a blank.
- **Against a model of similar purpose**, ko-trocr (214M), we are
  5.9× faster on CPU (a thirty-line page in 10.2s against
  58.1s). Whether it fits on an office PC with no GPU is decided here.
- **CPU figures ride hard on this laptop's state at the time.** Same day, same cells, same
  method: the single model ran at 4.6 lines/s in the morning, 3.1 in this run and 2.6 in
  the evening, and its lead over ko-trocr moved through 6.3×, 5.9× and 4.4×. That is why
  the 0.18 s per line under "Speed and footprint" below (`bench.py`, measured separately)
  does not match. Read **the order within one run** rather than lines/s itself, and the
  lead as 4–6× — being faster than ko-trocr held all three times.
- **The large VLMs need a GPU.** On CPU Qwen3-VL-2B does 0.47 lines/s and
  PaddleOCR-VL-1.6 0.04, so a thirty-line page takes 64s and
  681s. Even on GPU they run at 6.5 and 0.9 lines/s, slower than
  our single model (34.2). On CPU they take 2s and 23s per line,
  so only the one-line batch was timed, once.
- **Batch size is measured, not guessed.** Each model reads at its own cheapest batch. A
  batch that does not fit this card is not timed at all — ko-trocr's 32-line batch is one,
  and the tool's own reason reads: "앞 묶음 4642MB(가중치 848MB)에서 활성값을 두 배 하면 8437MB 로 카드 8151MB 를 넘는다". **A number that was not
  measured is not written down as if it were.**

### What is *not* equal — and which side it favours

| What | How | Favours |
|---|---|---|
| Test fonts | The six and the twenty-four were held out of **our** training only. What the rivals were trained on is unknown to us | rivals |
| Line cutting | Our cutter was lent to every rival (they all receive one line image) | rivals |
| Length cap | ko-trocr ships `max_length: 16` (character tokenizer). Like the two VLMs it was given 64 tokens. Every line where Qwen3-VL hit the cap was a runaway repeat (`11 11 11 …`); no long answer was cut short (re-reading the 670 font lines, all 24 that hit the cap were like that) | ko-trocr |
| Prompt | Qwen3-VL has no fixed OCR prompt; three were tried and the one it reads best with was used | Qwen3-VL |
| Precision | The two VLMs run in their released precision (bfloat16), the rest in float32. ko-trocr's float16 speed was also timed | — |
| VLM on CPU | Seconds to tens of seconds per line, so only the one-line batch was timed, once. An 8-line batch might be faster | ko-hand-ocr |
| All 480 fonts | PaddleOCR-VL takes about a second per line, so it was not run on all 480 (blank) | — |
| Design intent | The three line recognisers target mainly printed and scene text, the VLMs whole documents, ko-trocr AI Hub's handwriting and administrative documents. **Only this model targets a single handwritten line** | ko-hand-ocr |
| Number of models | Our ensemble is three models (one 36M + two 41M, 118M). For a size-matched view read the "single model" row | ko-hand-ocr |
| Training fonts | 450 of the 480 fonts are ones **we have seen**. That is why the herds are reported apart | ko-hand-ocr |

**None of this says the rivals are bad models.** Models built for printed documents,
scene text or whole pages were handed a line of handwriting, and on that job this one does
better. The other direction — whole printed documents, tables, scene text — was not
measured, and if it were, this model would probably lose.

Measured on: NVIDIA GeForce RTX 5060 Laptop GPU · torch 2.14.0+cu130 · 2026-10-07 17:32. Accuracy for our two models
and ko-trocr matches the same morning's run to the character — same models, same ruler.
Speed rides on the machine's state, so **compare only figures taken at the same moment**.

## Install

```bash
pip install ko-hand-ocr
```

Weights ship separately — they are too large for the repository, so they are attached
to [Releases](https://github.com/jysvai/ko-hand-ocr/releases/tag/v0.4.1).
**The same files are on [Hugging Face](https://huggingface.co/localdeel/ko-hand-ocr)** — the server and the MCP tool below fetch them from there.

> Weights are versioned separately from the package. The current weights are on the
> **v0.4.1** release (the package on PyPI is 0.5.0 — 0.5.0 adds the server and the MCP
> tool below; the reading code and the weights are unchanged). Earlier weights stay where they are — the
> previous single model (v83) is on v0.4.0, and **the 6-layer v62 on
> v0.3.0** is faster (about 40% when measured the same day) and has a higher photo
> floor (worst photo 90.3%); the 4-layer weights on v0.2.0 are faster still.

| Download | Size | What it is |
|---|---|---|
| `ko-hand-ocr-single.zip` | 144MB | **Single model.** Enough for most uses. 0.18 s/line |
| `ko-hand-ocr-ensemble.zip` | 417MB | Ensemble (3 models). Raises the floor, 7x slower |

```bash
curl -LO https://github.com/jysvai/ko-hand-ocr/releases/download/v0.4.1/ko-hand-ocr-single.zip
```

Pass the unzipped folder straight to `Reader()`. It must contain `config.json`,
`vocab.json` and `model.safetensors`.

## Quick start

```python
from kohandocr.reader import Reader

reader = Reader("ko-hand-ocr-single", device="cpu")   # the unzipped folder

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

### Ensemble reading (several models)

Different models fail on different cells. Train a little longer on the same data
and some cells survive while others die — which tells you those cells were being read
by a thin margin. So keep several models and take **the answer they agree on**.

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

## Use it with LM Studio, Ollama and vLLM

**It cannot be loaded into them as a model.** All three are engines for LLMs that continue text:
LM Studio and Ollama run llama.cpp GGUF files, and vLLM runs the architectures it knows. This
model is a TrOCR-style image encoder with a cross-attention decoder, plus its own jamo
vocabulary and line cutter, so it neither converts to GGUF nor fits vLLM's list. Instead the
package ships **a server that speaks the same APIs** and **an MCP tool**.

```bash
pip install "ko-hand-ocr[mcp]"      # drop [mcp] if you only need the server
```

### Server — the OpenAI and Ollama APIs

```bash
ko-hand-ocr-serve                   # fetches the single model (156MB) from Hugging Face, serves 127.0.0.1:8765
ko-hand-ocr-serve --ensemble        # the ensemble (450MB)
```

| Caller | How |
|---|---|
| `openai` client, code written for vLLM | set `base_url="http://127.0.0.1:8765/v1"`; send the image as base64 in `image_url` |
| `ollama` CLI | set `OLLAMA_HOST=127.0.0.1:8765`, then `ollama run ko-hand-ocr "C:\scan.jpg"` · `ollama list` |
| curl | `curl --data-binary @scan.jpg http://127.0.0.1:8765/read` |

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="none")
picture = "data:image/jpeg;base64," + base64.b64encode(open("scan.jpg", "rb").read()).decode()
reply = client.chat.completions.create(model="ko-hand-ocr", messages=[
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": picture}}]}])
print(reply.choices[0].message.content)     # one line per handwritten line
```

- **Text in the prompt is ignored.** The image is cut into lines and each line comes back as one
  line of text. It cannot summarise or tidy up — that is the chat model's job (MCP, below).
- **Image URLs (`http://…`) are not fetched.** Send base64. Nothing leaves the machine.
- By default it listens on this PC only (`127.0.0.1`). There is no password, so use
  `--host 0.0.0.0` only on a network you trust.

### LM Studio — as an MCP tool

Since 0.3.17, chat models in LM Studio can call outside tools (MCP). In `~/.lmstudio/mcp.json`:

```json
{"mcpServers": {"ko-hand-ocr": {"command": "ko-hand-ocr-mcp"}}}
```

If `ko-hand-ocr-mcp` is not on PATH, give the full path (`Scripts\ko-hand-ocr-mcp.exe` in the
virtual environment). Then ask a chat model that can call tools to "read C:\scan.jpg and put it in
a table": it calls `read_handwriting` for the text and does the tidying itself. **This model
recognises the characters; the chat model organises them.** An image pasted into the chat window
goes to the chat model, not the tool, so pass a file path. The first call downloads the model
(156MB, 20–30 s here); after that a photo takes about a second.

**What was checked and what was not.** The `ollama` CLI 0.34 (`list`, `show`, `run`), the official
`openai` client (whole and streamed replies) and an MCP SDK 2.3 client were all run against it. A full
run inside the LM Studio chat window or Open WebUI has not been done yet.

Putting it inside vLLM itself would mean writing the architecture as a plugin. For a 41M model a GPU
server buys little, so that was not done — the server above exposes the same OpenAI address vLLM does.

## How it is built

```
  64 x 640 line image
          |
   ViT-Small encoder          facebook/deit-small-patch16-224 (ImageNet-1k, Apache-2.0)
   patch 16                   pretrained weights, continued at a lower learning rate
          |
   TrOCR decoder              8 layers, 6 heads — trained from scratch
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
`python tools/verify.py` (about 960 lines per font, 11 photos).

| | unseen fonts (mean) | worst font | 11 handwriting photos | worst photo | items over 95% |
|---|---|---|---|---|---|
| single model (0.4.1) | 95.6% | 90.4% | 94.9% | 84.2% | 11 / 17 |
| ensemble (3 models) | 95.6% | **90.7%** | **96.6%** | **90.3%** | **12 / 17** |
| *0.4.0 single model (v83)* | *95.5%* | *90.1%* | *95.0%* | *83.0%* | *10 / 17* |
| *0.3.0 single model (v62)* | *94.6%* | *88.5%* | *95.4%* | *90.3%* | |
| *0.3.0 ensemble* | *95.3%* | *90.3%* | *95.8%* | *90.3%* | *10 / 17* |

Fonts are measured at about 960 lines each (0.4.0 shipped with 380). Italic rows are
earlier releases: the 0.4.0 single model was re-measured with the same 960-line ruler,
while the 0.3.0 rows used the old 380-line ruler, so part of those differences is the ruler.

**0.4.1 changes the single model only.** It was trained to follow the ensemble's
next-character probabilities (distillation), then averaged half-and-half with 0.4.0's
v83. Size and architecture are unchanged, so speed is too. Items over 95% go from 10 to
11 of 17 (hand-07 94.4 -> 95.8%), all six fonts move up a little (worst font 90.1 ->
90.4%), and the worst photo, hand-06, goes from 83.0 to 84.2%. In exchange hand-08 drops
from 100 to 96.7%. **This is a small step, not a jump** — the photo mean is the same
(95.0 vs 94.9%). Re-running the same method at other strengths (0.25, 1.0), and trying
dropout, weight EMA and SAM, all failed to beat v83.

**What 0.4.0 changed over 0.3.0.** The single model (v83) moved from 94.6 to 95.5% on the
font mean and from 88.5 to 90.1% on the worst font. In exchange **its photo floor
dropped** (hand-06 90.3 -> 83.0%, one line of a three-line photo). The ensemble holds
the floor and moves the photo mean from 95.8 to 96.6%. Of the 17 items (6 fonts + 11
photos), the ensemble clears 95% on 12, up from 10.

Similarity is measured per jamo; on the photo side each photo gets one vote.

**Do not take the photo number at face value.** It is measured on 11 photos (35 cells)
taken by the author, so it swings ±5%p. The figure closer to what you would see on
someone else's handwriting is the **unseen-fonts** column — tested only on six font
families never used in training, where the swing is half as wide (±2-3%p).

**What the ensemble buys is the floor, not the mean.** The font mean is now the same
(95.6 vs 95.6), but the worst font goes from 90.4% to 90.7%, and a photo the single
model dropped to 84.2% (hand-06) comes back at 90.3%. Where one model reads
a cell by a hair, another carries it.

**Not every page improves, though.** A page the single model cleared at 95.2% can
come down to 92.9% (hand-10) — when two models agree on the wrong answer, the one
that was right is outvoted. And it is seven times slower. Use the ensemble where
a single page reaching a human is costly, the single model where you need to sweep a lot of pages.

### Per font (about 960 lines each)

![Accuracy by handwriting font](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/bench-accuracy.svg)

Look only at the average and **the font that collapses is hidden.** The real spread
is close to 9%p.

| Font | single model (0.4.1) | ensemble | *0.4.0 single model (v83)* |
|---|---|---|---|
| **EastSeaDokdo (동해독도)** | **90.4%** | **90.7%** | *90.1%* |
| KirangHaerang (기랑해랑) | 94.7% | 94.7% | *94.7%* |
| HiMelody (하이멜로디) | 95.6% | 95.4% | *95.4%* |
| NanumPenScript (나눔펜스크립트) | 95.9% | 95.9% | *95.7%* |
| GamjaFlower (감자꽃) | 98.0% | 98.0% | *97.9%* |
| SingleDay (싱글데이) | 99.0% | 98.9% | *98.9%* |

What has to clear the bar is not the mean but **the lowest font.** Once all six passed
90% (2026-09-18) the goal was raised to 95%. Four are over it now; EastSeaDokdo and
KirangHaerang are not. Both are hands where strokes merge and get dropped.

All six are fonts **never used in training**, and all are SIL Open Font License.

```bash
python tools/holdout.py <checkpoint-dir> --per-font --sample sample.png
```

### What the test sheet actually looks like

![unseen-font test sheet](https://raw.githubusercontent.com/jysvai/ko-hand-ocr/main/holdout-sample.png)

Three lines per font, with the ground truth, what the model read, and the similarity.
The pale band on the right is the leftover space from fitting into a 64x640 frame —
this is exactly what the model sees.

**Every one of the six gets these short form lines right.** When the same picture was
produced on 2026-09-02, EastSeaDokdo missed `안전사무국`; it does not any more. Where
the fonts pull apart is not lines like these but **long sentences and lines no language
knowledge helps with** — a third of the sheet is meaningless syllables strung together
(put there on purpose, to train reading from strokes alone), and there a single wrong
jamo has nothing to fall back on.

By shape, **SingleDay** has well-separated characters and intact strokes, close to
print, while **EastSeaDokdo** and **KirangHaerang** smear and drop them. That is the
difference the table above measures.

This is **an image, not a font file.** Drawing glyphs with a font is what the OFL
permits; what is not redistributed is the `.ttf` files themselves
([PROVENANCE.md](https://github.com/jysvai/ko-hand-ocr/blob/main/PROVENANCE.md)).

## Speed and footprint (measured)

No GPU needed. Emitting jamo one at a time is the bottleneck, so more cores do not help
much — and by the same token **it does not get slower on a weak machine.**

CPU only, median over 11 photos (35 cells). Time is split into **cutting and reading** —
shrinking the model does not shrink cutting, and when a photo holds only three or four
lines, cutting is more than half the cost.

| | cut | read | one photo (3.2 lines) | per line | 30-line page |
|---|---|---|---|---|---|
| single model, beam 5 | 0.48s | 0.57s | 1.04s | 0.18s | 6.9s |
| single model, greedy | 0.48s | 0.30s | 0.78s | 0.09s | 2.0s |
| ensemble, beam 5 | 0.48s | 4.36s | 4.83s | 1.37s | 47.1s |

Measured on 2026-10-07. The same day, 0.4.0's single model (v83) took 1.00 s per photo
and 0.17 s per line — size and architecture are identical, so **0.4.1 is neither heavier
nor slower** (differences under 5% are day-to-day noise). **The machine itself varies by
day**: on 2026-09-22 the same v83 measured 0.25 s per line. Only compare figures taken
on the same day.

Memory, as the peak while reading the 11 photos, is about 1.07GB for the single model
and about 1.51GB for the ensemble. Measured on 2026-09-22 (the 0.4.1 single model has
the same architecture).

**Eight layers are not free.** On 2026-09-22, re-measuring 0.3.0's 6-layer v62 with the
same harness gave 0.18 s per line (beam 5), 0.09 s (greedy) and 1.03GB, against 0.25 s
for the 8-layer model that day. The 8-layer model is **about 40% slower** at beam 5. If speed comes first, take v62 from
v0.3.0.

**The right-hand column is not the photo figure multiplied out.** Our test photos hold
three or four lines each, while the "page" other people quote is a thirty-line
document. Quoting our seconds-per-page as if it were theirs flatters us by roughly ten
times. So the batch size is raised step by step to measure **how far seconds-per-line
falls**, and that value is what gets converted (`python tools/bench.py --throughput`).

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
python -m kohandocr.train --out runs/v1 --layers 8 \
       --steps 20000 --batch 40 --workers 5 --letters 128 \
       --strike 0.15 --warp 0.5 --digits 0.04
```

The last three change **the training data only.** Test sheets are always drawn with
the defaults, so switching them on or off leaves the ruler alone (switched off, they
do not even draw a random number — a test holds that).

- `--strike` — share of lines that get a crossed-out mark. Four of the 37 photo cells
  have one.
- `--warp` — share of lines whose strokes are bent **inside** each character
  (`synth._warp`). Every earlier distortion moved whole characters, so jamo kept the
  font's exact shape — yet every photo miss was a single jamo (금->글, 감->갈, 피->회).
- `--digits` — share of lines replaced by digit strings that are **not** dates
  (`corpus.digit_line`). The corpus had twice as many date lines as bare digit
  strings, so a faint digit string got filled in as a date
  (`'816883539' -> '2025-05-19'`).

Fonts are looked up in this order. **No path is hard-coded.**

1. whatever `--fonts` was given
2. the `KOHAND_FONTS` environment variable
3. `~/.cache/ko-hand-ocr/fonts`

Without `--layers` you get a 4-layer decoder. The single model that ships has
8 layers (the ensemble mixes in one 6-layer model). At 4 layers, thirteen rounds
of changing the recipe left the font mean stuck between 92.8 and 93.4%, and 6 layers
was the first to clear that band. At 6 layers the lowest item then sat at 90.1% for
thirteen rounds; the ensemble only started changing again with 8 layers plus
in-character stroke warping. Going from 6 to 8 layers costs about 40% on CPU
(measured on the same ruler). The encoder and the input size were left alone.

`fetch_fonts.py` fetches 482 handwriting fonts. Only **452 of them are trained on**;
six are the test above and twenty-four are held back for "does it read widely". A font
with the same design as a test font would quietly inflate the score, so they are
compared **as images**, not by name, and anything scoring 0.7 or above is set aside.

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
