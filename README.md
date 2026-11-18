# Detecting the Detector

Code and data for an INT4096 group project: **do AI-text detectors wrongly flag essays by non-native English writers,
and does it help to train detectors on non-native writing?**

We trained three detector families on identical data in four training sets ("arms"), changing only the human essays:

| arm | human essays (4,000) | AI essays (4,000) |
|---|---|---|
| A | native only (PERSUADE 2.0) | daigt + HC-Var ChatGPT |
| B | half native + school-age non-native (ELLIPSE, PERSUADE ELL) | same |
| C | half native + exam-style non-native (FCE, W&I) | same |
| D (control) | half native + other native writing (HC-Var human) | same |

Detectors: TF-IDF + logistic regression (5 seeds), RoBERTa-base full fine-tune (5 seeds), Qwen3.8-27B + LoRA
(2 seeds); for comparison, the open-source zero-shot detectors Binoculars and Fast-DetectGPT, and the commercial
detectors archived by Liang et al. (2023). Main test set: Liang et al.'s 91 real TOEFL essays vs native essays.

## Notebooks

Open any notebook in Colab with the badge at its top (or *File > Open notebook > GitHub* and paste this repo's
address), then *Runtime > Run all*. The first cell finds or downloads the repository; nothing needs editing.
Each notebook has a CONFIG cell with three modes: SMOKE (seconds, for automated tests), FAST (default) and FULL
(paper setting). Times below were measured on a laptop CPU (Apple M-series); expect Colab's CPU to be slower.

| notebook | what it does | re-run or replayed | FAST time | GPU |
|---|---|---|---|---|
| `00_setup_and_data` | gets the data, checks file fingerprints, split sizes, zero overlap between training and test essays, text cleaning | checks run on the data | ~30 s | no |
| `01_audit_commercial_and_opensource` | audit of the 7 commercial and 2 open-source detectors: false-flag rates, AUCs, F1, length control, why predictable text looks like AI, whether GPT-4 polishing removes the false flags; Binoculars and Fast-DetectGPT written out from the papers | numbers **replayed** from saved scores; live demo with small stand-in models (gpt2 pair on CPU) | ~2 min (CPU) | optional |
| `02_ml_logistic_regression` | trains TF-IDF + logistic regression for arms A-D x 5 seeds, false-flag rates, precision/recall/F1, comparison with the archived numbers | **re-run** from scratch | ~1.5 min (FULL ~2.5 min) | no |
| `03_dl_roberta` | RoBERTa-base fine-tuning, arms A-D | all 40 archived runs **replayed** in every mode; **re-training** in SMOKE (64 essays, ~1 min on CPU), FAST (arms A, C, 1 seed) and FULL (4 arms x 5 seeds) | SMOKE ~1 min; FAST/FULL need a GPU (estimates on a T4: FAST 30-50 min, FULL 5-8.5 h, **not measured**) | T4 for FAST/FULL |
| `04_lora_qwen` | Qwen3.8-27B + LoRA: replay of the 27B results, a look inside the published LoRA adapters (reads only the file header, checks 124,741,120 parameters), and a small **stand-in** LoRA training on Qwen3-0.6B | 27B numbers **replayed**; adapter inspection reads one adapter from Hugging Face; stand-in is **re-run** (SMOKE ~5 min on CPU; FAST/FULL on a T4 are estimates, not measured) | SMOKE ~5 min | T4 for FAST/FULL |
| `06_lime_word_explanations` | exploratory side analysis (**not in our slides**): which words push our logistic-regression detector toward AI, with the official LIME package, for two example essays | **re-run** (trains arms A and C, then LIME) and compared with the archived values | ~20-45 s | no |
| `05_metrics_and_figures` | recomputes all headline tables for all detectors and draws the main figures | **replayed** from saved scores in `data/preds/` and `data/commercial/` | ~5 s | no |

What is replayed rather than re-run (notebook 05): RoBERTa and Qwen 27B scores (trained on the A40 server),
Binoculars and Fast-DetectGPT scores (falcon-7b pair, run on the A40 server), commercial detector scores (archived
by Liang et al. in 2023; the 2023 detector versions can no longer be queried). The logistic regression scores in
`data/preds/lr/` are also the archived ones; notebook 02 re-trains the model and compares against them.

### Which experiment is in which notebook

| experiment | notebook | how |
|---|---|---|
| 7 commercial detectors (2023 scores of Liang et al.) | 01, 05 | replayed |
| Binoculars, Fast-DetectGPT (falcon-7b pair) | 01, 05 | replayed; formulas written out and run live on small stand-in models |
| word predictability / perplexity (falcon-7b, GPTZero) | 01 | replayed from `data/ppl/` |
| length control | 01 | replayed from saved scores |
| GPT-4 polishing of TOEFL essays | 01 (section 1.7) | replayed from the commercial scores of the polished essays |
| logistic regression, arms A-D x 5 seeds | 02 | **re-run** |
| RoBERTa-base, arms A-D x 5 seeds | 03 | 40 archived runs replayed; re-training needs a GPU (not tested) |
| Qwen3.8-27B + LoRA, arms A-D x 2 seeds | 04 | replayed (cannot be re-run: the base model exists only on our server); adapters inspected from Hugging Face; Qwen3-0.6B stand-in re-run |
| precision / recall / F1, bias AUC, paired tests, figures | 05 | replayed |
| LIME word explanations (side analysis, not in the slides) | 06 | **re-run** |
| FP8 to bf16 conversion of the 27B base model and its error check | not in a notebook | scripts in `a40/` (need the 27B weights) |

The eight Qwen LoRA adapters (3.99 GB) are on Hugging Face (private):
`SheridanLu/detecting-the-detector-qwen-lora`. Notebook 04 reads a token from the Colab secret `HF_TOKEN`
(or the environment variable). Cloning this repository while it is private needs a GitHub token in the Colab
secret `GH_TOKEN`; otherwise upload the repository as a zip (the first cell explains how).

## Data

| folder | content | size |
|---|---|---|
| `data/splits/` | test essays (E1-E7), training pool, validation set, training-set id lists, split summary with SHA-256 | 55 MB |
| `data/followup/` | training sets of arms C and D, ids of exam essays used for training (excluded from evaluation) | 106 MB |
| `data/preds/` | per-essay scores of every trained run and of the two open-source detectors | 19 MB |
| `data/ppl/` | per-essay falcon-7b perplexity and GPTZero perplexity of the TOEFL, Hewlett, college and CS224N essays | <1 MB |
| `data/commercial/` | scores of 7 commercial detectors on the Liang et al. essays | 0.1 MB |
| `data/results/` | result tables computed by the original run (the reference for all checks) | 0.3 MB |

Every file is described (columns, rows, origin) in [`data/README.md`](data/README.md). The original scripts are in
[`a40/`](a40/README.md), unchanged; `ddlib/` is the small helper package the notebooks import (paths, text cleaning
copied from the original, metrics, plots). To run outside Colab:
`python3.12 -m venv .venv && .venv/bin/pip install -r requirements-local.txt -r requirements-colab.txt jupyter`.

## Limits: what cannot be reproduced on free Colab

* **Qwen3.8-27B + LoRA** exists only on the A40 server (weights dequantised there from FP8); training took about 4 h
  on two 46 GB GPUs per run. Its numbers can only be replayed from the saved scores.
* **Binoculars / Fast-DetectGPT** need falcon-7b and falcon-7b-instruct together (about 28 GB in bf16), which does not
  fit a 16 GB T4. Replayed only.
* **Commercial detectors**: the 2023 versions are gone; we use the scores archived by Liang et al.
* **Logistic regression** re-runs agree with the archived numbers within the stated tolerances but are not
  bit-identical: the solver stops after ~20 iterations, and floating-point differences between machines move that
  point slightly. On our test machine 0.26% of individual decisions at threshold 0.5 changed; all 40 headline
  checks (arms A-D) stayed within 0.5 percentage points / 0.004 F1, and the learning-curve runs within 1.1 points.
* The notebooks were tested locally on CPU (macOS, Python 3.12, clean virtual environments): 00, 02 and 05 in SMOKE,
  FAST and FULL; 01 in SMOKE and FAST; 03 and 04 in SMOKE only. **Nothing has been run on a GPU or on a real Colab
  runtime**: RoBERTa re-training (03, FAST/FULL) and the Qwen3-0.6B stand-in training (04, FAST/FULL) are untested on
  a GPU, and their T4 times are estimates. A FAST attempt of notebook 03 on a laptop GPU was abandoned (arm A alone
  took 37 minutes).
* The Qwen LoRA adapters attach LoRA matrices to every linear layer, including the image encoder; 110 `lora_B`
  matrices there are exactly zero (never updated; the task is text only), so slightly fewer than the 124,741,120
  parameters marked trainable actually changed. No prediction is affected.
* Study limits (small TOEFL sample of 91 essays, only 2 Qwen seeds, a degenerate 90%-detection threshold for Qwen,
  corpora that differ in age and genre) are stated in the notebooks where they matter.

## Data sources, licence and ethics

The essays come from public third-party corpora: PERSUADE 2.0, ELLIPSE, daigt_v4, HC-Var, Write & Improve +
LOCNESS, the Cambridge FCE dataset, ArguGPT, and the TOEFL / Hewlett / CS224N / college-essay benchmark of Liang et
al. (2023, "GPT detectors are biased against non-native English writers"). They are included only as the
pre-split subsets needed to reproduce this coursework, remain under their original licences and terms of use, and
must not be redistributed for other purposes; please obtain them from the original sources for any other use. The
corpora are anonymised by their publishers; we added no personal data. Our own code is for teaching and assessment.
