# data/

Everything the notebooks read. Total about 175 MB; the largest single file is `splits/pool.parquet` (41 MB).
All essays come from public third-party corpora (see the main README) and are included only as the pre-split
subsets needed to reproduce this coursework. All texts are stored **raw** except where noted; the notebooks apply
`ddlib.clean()` (identical to `a40/common.py::clean`) before use.

Origin: copied unchanged from the project folder on the A40 server used for the original run (16-17 Nov 2026),
except `commercial/` (converted, see below). `splits/split_summary.json` stores the SHA-256 of the split files;
notebook 00 checks them.

## splits/ (built by `a40/build_splits.py`)

| file | rows | columns | what |
|---|---|---|---|
| `exams.parquet` | 13,672 | exam, corpus, essay_id, role, text, prompt, grade, score, level, cefr, script_score, l1, age | Test essays, never trained on (except the FCE / W&I half listed in `followup/train_exam_ids.json`, which arm C trains on and every evaluation removes). E1 Liang et al. 2023 (925: TOEFL_real_91, HewlettStudentEssay_real_88, CollegeEssay_real_70, CS224N_real_145, 4 GPT-3 sets = 352 AI, plus TOEFL_gpt4polished_91 and HewlettStudentEssay_GPTsimplify_88); E2 PERSUADE same-prompt ELL 792 / non-ELL 800; E3 W&I A/B/C 3,300 + LOCNESS 50; E4 FCE 2,469; E5 ELLIPSE test 2,901; E6 daigt Llama 400 + HC-Var ChatGPT 400; E7 ArguGPT TOEFL-prompt AI 1,635. Metadata columns are strings, empty where the corpus has none. |
| `pool.parquet` | 35,011 | id, text, label (0 human, 1 AI), source | Essays the training sets are drawn from: persuade_nonELL 21,345; persuade_ELL 1,000; ellipse 1,008; daigt_ai 7,708; hcvar_ai 3,950. |
| `val.parquet` | 600 | id, text, label, source | Shared validation set (PERSUADE native 300, daigt 150, HC-Var 150). Used only to set the 90%-detection threshold. |
| `manifests.json` | 30 runs | {run: {human: [pool ids], ai: [pool ids]}} | Training sets of arms A and B: `seed{0-4}_{A,B}_n{1000,2000,4000}` (n = essays per class). |
| `split_summary.json` | - | - | Counts, overlap removed, final overlap gate (0) and SHA-256 of the four files above. |

## followup/ (built by `a40/followup_ml.py`)

| file | rows | columns | what |
|---|---|---|---|
| `trainsets/seed{0-4}_{C,D}.parquet` (10 files, 8-12 MB each) | 8,000 each | text, label, part | Training sets of arms C and D, **already cleaned**. part = native (2,000 PERSUADE native, the first half of `seed{s}_A_n4000`'s human ids), C (2,000 FCE + W&I essays) or D (2,000 HC-Var human "writing" essays), ai (the 4,000 AI essays of `seed{s}_A_n4000`). |
| `train_exam_ids.json` | 2,577 ids | list of essay_id | The FCE / W&I essays in arm C's training half (a fixed 50/50 split, seed 0); removed from every evaluation (all arms). |
| `overlap_gate.json` | - | - | Overlap gate of the follow-up run (0 overlaps). |

## preds/ (scores saved by the original run)

One row per scored essay: every exam essay plus the 600 validation essays (exam = "VAL"), 14,272 rows, same order
as `ddlib.data.score_frame()`. Columns: exam, corpus, essay_id, score (probability of AI, 0-1), model, run.

| folder | files | produced by |
|---|---|---|
| `lr/` | 40: `seed{0-4}_{A,B}_n{1000,2000,4000}` (30) + `seed{0-4}_{C,D}` (10) | `a40/train_lr.py`, `a40/followup_ml.py` (CPU). Re-run in notebook 02. |
| `roberta/` | 40, same names | `a40/train_roberta.py` via `run_gpu.py` / `run_gpu_followup.py` (1x A40 per run) |
| `qwen/` | 8: `seed{0,1}_{A,B}_n4000`, `seed{0,1}_{C,D}` | `a40/train_qwen.py` (Qwen3.8-27B + LoRA, 2x A40 per run) |
| `binoculars/binoculars.parquet` | 1 | `a40/score_opensource.py` (falcon-7b / falcon-7b-instruct). Extra column `raw` = official Binoculars score; `score` is a monotone transform with the official low-FPR threshold mapped to 0.5. |
| `fastdetect/fastdetect.parquet` | 1 | `a40/score_opensource.py` (same model pair). `raw` = sampling discrepancy; `score` = official probability (0.5 = official threshold). |

## commercial/

| file | rows | columns | what |
|---|---|---|---|
| `liang2023_detector_scores.parquet` | 6,185 | exam, corpus, essay_id, detector, score | Scores of 7 commercial detectors (Crossplag, GPTZero, HFOpenAI, OriginalityAI, Quil, Sapling, ZeroGPT) on the 10 E1 corpora, archived by Liang et al. 2023 (`Data_and_Results/` of github.com/Weixin-Liang/ChatGPT-Detector-Bias). Converted to one table by `tools/build_commercial_scores.py` using the same reading rule as `a40/metrics.py` (essay i of a corpus = `corpus#i`). Quil and Sapling have no scores for CS224N_real_145 (no other missing values). |

## ppl/ (per-essay perplexities, used by notebook 01)

One JSON file per script; each maps a corpus name to a list of records (one per essay, in the corpus order of `exams.parquet`).
Record fields: `i` (essay index in the corpus), `words` (whitespace-separated word count of the raw text), `has_at` (text contains an `@`
placeholder), `falcon_ppl` (perplexity under falcon-7b, text cut at 512 tokens, a start token prepended), `gptzero_ppl` and `gptzero_score`
(GPTZero's own perplexity and score, copied from Liang et al. 2023's `GPTZero.json` of that corpus).

| file | corpora (records) | made by |
|---|---|---|
| `all_ppl.json` | TOEFL_real_91 (91), HewlettStudentEssay_real_88 (88), TOEFL_gpt4polished_91 (91) | `a40/all_ppl.py`, run on the A40 server |
| `all_ppl_native.json` | CollegeEssay_real_70 (70), CS224N_real_145 (145) | `a40/all_ppl_native.py`, run on the A40 server |

Hewlett has `@` placeholders in 85 of 88 essays, so it must not be used for perplexity comparisons.

## results/ (tables computed by the original run; used as the reference in notebooks 02 and 05)

| file | rows | made by | content |
|---|---|---|---|
| `metrics_runs.csv` | 73 | `a40/metrics.py` | Every run and external detector: bias AUCs, detection AUCs (with 95% CIs), flag rates at 0.5 and at the 90%-detection threshold per corpus. |
| `metrics_agg.csv` | 16 | `a40/metrics.py` | Same, mean/SD over seeds per model, arm (A/B) and training size n. |
| `paired_toefl_A_vs_B.csv` | 64 | `a40/metrics.py` | Paired arm A vs B on the 91 TOEFL essays (McNemar, bootstrap CI). |
| `followup_runs.csv`, `followup_runs_roberta.csv`, `followup_runs_qwen.csv` | 20 / 20 / 8 | `a40/followup_metrics.py`, `a40/followup_metrics_gpu.py` | Per run, arms A-D: flag rates (0.5 and 90%-detection threshold) per corpus, pre-registered endpoints P1-P3, bias AUCs. |
| `followup_paired*.csv` | 15 / 15 / 6 | same | Paired C-A, C-D, B-A on TOEFL per seed. |
| `prf1_runs.csv`, `prf1_summary.csv` | 57 / 21 | `a40/prf1.py` | Accuracy, precision, recall, F1 (and balanced F1 for TOEFL / Hewlett) on the Liang test set, threshold 0.5. |
| `model_comparison_all.csv` | 14 | ad-hoc aggregation on the server (script not saved); notebook 05 reproduces it exactly from `preds/` | Mean over seeds of flag rates at 0.5 per corpus, detection rates, AUCs, SD of the TOEFL rate. |
| `length_control.csv` | 66 | `a40/length_control.py` | Bias AUC over all pairs vs over length-matched TOEFL / Hewlett pairs. |
