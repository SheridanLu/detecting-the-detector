# INT4096 unified_v4 — final report (run of 2026-11-16 19:13 UTC → 2026-11-17 00:43 UTC)

Environment: 111.48.160.138, 8×A40 (46GB), production GLM (SGLang) stopped for the whole run and restored at 00:43 UTC.
Primary endpoints were frozen before the GPU run: `PREREG.md`, sha256 `51e8219d…2139`, `logs/prereg_sha256.txt`.
Disclosure: classical-ML (LR) numbers were viewed before the endpoints were written down.

## 1. Data (all verified on the server; sha256 in `logs/build_splits.log`)

| Pool | Source | n | Notes |
|---|---|---|---|
| Human, native | PERSUADE 2.0 `ell_status=No` | 21,345 | minus exam/validation overlap |
| Human, non-native | ELLIPSE (HF `tcapelle/…`, rev `9a7a03c8`) + PERSUADE `ell_status=Yes` | 1,008 + 1,000 | 452 PERSUADE ELL essays are also inside ELLIPSE and were removed from the PERSUADE pool |
| AI | daigt_v4 (Llama fine-tuned on PERSUADE) + HC-Var ChatGPT (`writing` domain) | 7,708 + 3,950 | daigt is **AI only** (label all 1) |
| Validation (shared) | PERSUADE native 300 + (daigt 150 + HC-Var 150) | 600 | used only for the 90%-TPR threshold |

Design: arm A human = 4,000 native; arm B human = 2,000 native + 1,000 ELLIPSE + 1,000 PERSUADE-ELL; identical AI 4,000 in both arms; sizes 1,000/2,000/4,000 nested (learning curve); 5 seeds (LR, RoBERTa), 2 seeds (Qwen 27B).

Exams (never trained on; gate `final_overlap_train_val_vs_exams = 0`, `SPLIT_GATE PASS`):
E1 Liang 925 · E2 PERSUADE same-prompt ELL 792 vs non-ELL 800 · E3 W&I A/B/C 3,300 vs LOCNESS 50 · E4 FCE 2,469 · E5 ELLIPSE 2,901 · E6 daigt 400 + HC-Var 400 · E7 ArguGPT TOEFL-prompt AI 1,635.

**Text normalisation**: every text (train, validation, exams) passed through `common.clean()` — newlines collapsed, curly quotes/dashes unified, CP-1252 mojibake repaired, Cyrillic homoglyphs transliterated then removed. `FORMAT_GATE PASS` (no residual formatting difference between any pool). **Reason**: an earlier attempt without normalisation let RoBERTa learn "no paragraph breaks + curly quotes = AI"; that attempt is archived in `archive_v1/attempt2_format_contaminated/` and is not used in any number below.

## 2. Primary endpoint 1+2 — TOEFL bias (E1)

Bias AUC = P(random TOEFL essay scores more AI-like than random Hewlett essay); 0.5 = no bias.

| Detector | Bias AUC [95% CI] | TOEFL flagged @90%-TPR threshold |
|---|---|---|
| Crossplag | 0.880 [0.825, 0.933] | — (no val set) |
| GPTZero | 0.985 [0.969, 0.997] | — |
| HFOpenAI | 0.902 [0.845, 0.949] | — |
| OriginalityAI | 0.963 [0.934, 0.985] | — |
| Quil | 0.931 [0.894, 0.966] | — |
| Sapling | 0.836 [0.773, 0.892] | — |
| ZeroGPT | 0.895 [0.848, 0.936] | — |
| Binoculars (ICML'24) | 0.868 [0.816, 0.918] | 6.6% |
| Fast-DetectGPT (ICLR'24) | 0.819 [0.756, 0.875] | 35.2% |
| **LR, arm A** (native only) | **0.987 ± 0.003** | **80.7% ± 2.9** |
| **LR, arm B** (+ non-native) | **0.972 ± 0.004** | **41.8% ± 3.4** |
| **RoBERTa, arm A** | **0.999 ± 0.001** | **66.2% ± 13.2** |
| **RoBERTa, arm B** | **0.999 ± 0.001** | **63.3% ± 9.6** |
| **Qwen 27B, arm A** | **0.920 ± 0.040** | 1.1% (metric degenerate, see §5) |
| **Qwen 27B, arm B** | **0.945 ± 0.005** | 1.1% (metric degenerate) |

Paired A-vs-B on the same 91 TOEFL essays (`metrics/paired_toefl_A_vs_B.csv`):
- LR: B − A = −0.33 to −0.43 at the 90%-TPR threshold, **all 5 seeds p < 0.001**.
- RoBERTa: B − A = −0.03 to −0.12, **not consistent** (seed 3 reverses, p = 0.001; others p = 0.11–0.55).
- Qwen: at 0.5, seed 0 −0.25 (p < 0.001) but seed 1 +0.055 (p = 0.33) → **no reliable effect**; at the 90%-TPR threshold both seeds are degenerate (1.1% in both arms).

**Reading**: adding non-native essays to the human side clearly lowers the false-flag rate of the simple classical model, does nothing measurable for RoBERTa or Qwen 27B, and does **not** move the threshold-free bias AUC for any of them.

## 3. Primary endpoint 3 — detection ability (must not degrade in arm B)

| Detector | Liang AI 352 (AUC) | Same-prompt TOEFL AI, ArguGPT (AUC) |
|---|---|---|
| Commercial (range over 7) | **0.439 – 0.603** | — |
| Binoculars | 0.931 | 0.989 |
| Fast-DetectGPT | 0.944 | 0.993 |
| LR, arm A → B | 0.987 → 0.988 | 0.739 → 0.773 |
| RoBERTa, arm A → B | 0.967 → 0.977 | 0.908 → 0.914 |
| Qwen 27B, arm A → B | 0.996 → 0.997 | 0.990 → 0.989 |

No arm lost detection ability. Two facts worth reporting: (i) **the 2023 commercial detectors are at chance on Liang's AI essays**; (ii) every model detects Liang's GPT-3 essays better than AI essays written on the *same prompts as* the TOEFL human essays — the gap is largest for LR (0.99 vs 0.74), i.e. topic/style mismatch matters more than the AI-vs-human distinction for a simple model.

## 4. Length control (`metrics/length_control.csv`)

TOEFL median 104 words, Hewlett 378 words. Recomputing bias over length-matched pairs only (±30% word count, 99 matched pairs per run):

| Detector | Bias, all pairs | Bias, length-matched |
|---|---|---|
| Binoculars | 0.868 | 0.786 |
| Fast-DetectGPT | 0.819 | 0.760 |
| LR | 0.980 | 0.996 |
| Qwen 27B | 0.933 | 0.999 |
| RoBERTa | 0.999 | 1.000 |

Length explains ~0.06–0.08 of the open-source detectors' bias; nothing for the fine-tuned models. This control is conservative (matching against short Hewlett essays selects the subset that detectors find most AI-like), so 0.76–0.79 is a lower bound.

## 5. Limitations (must be stated in the report)

1. **Qwen's 90%-TPR threshold is degenerate**: Qwen's scores are nearly saturated, so the threshold sits near 1.0 and both arms flag 1.1% of TOEFL. Endpoint 1 is not usable for Qwen; endpoint 2 (bias AUC) is.
2. **Qwen is unstable across seeds** (arm A: 0.920 ± 0.040; TOEFL flag rate at 0.5: 56% seed 0 vs 23% seed 1). Only 2 seeds.
3. Arm B's training data is same-source as E2 and E5, so improvements there are expected and are reported as secondary only.
4. PERSUADE "native" = `ell_status=No` (not verified native speakers); daigt and HC-Var carry no demographics.
5. Exam corpora differ in age, genre and provenance (TOEFL adult exam essays; PERSUADE/ELLIPSE grades 6–12; W&I mixed genres; FCE exam answers); LOCNESS has only 50 essays.
6. 27B weights were dequantised from FP8 (≈2.7% weight error; verified against an original bf16 shard, `logs/check_dequant.log`).
7. Single forward pass per text, threshold 0.5 / 90%-TPR / off-the-shelf; no prompt-based or judge baselines.

## 6. Artifacts

`splits/` (pools, exams, val, manifests, sha256) · `preds/{lr,roberta,qwen,binoculars,fastdetect}/` · `adapters/` (trained models, reusable for re-scoring) · `metrics/{metrics_runs,metrics_agg,paired_toefl_A_vs_B,length_control}.csv` · `logs/` (per-job logs, ETA, restore log) · `prod_snapshot/` (pre/post stop state, crontab diff, restore generation output) · `archive_v1/` (contaminated attempt, excluded).

Production restore: `RESTORE START 00:39:59 → RESTORE END 00:43:05 UTC`, `CRONTAB_RESTORED_IDENTICAL`, `CMDLINE_IDENTICAL pid=2209165`, real generation request returned `RESTORE_OK`, health 200.

All numbers: 请独立复核.
