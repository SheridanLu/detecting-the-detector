# a40/ - original scripts of the A40 run (unchanged)

These are verbatim copies of the scripts that produced every number in this project. They ran on a shared GPU
server with 8x NVIDIA A40 (46 GB each) on 16-17 November 2026, with Python 3.11, scikit-learn 1.9.1, pandas 3.1.1,
numpy 2.4.6. They contain the server's absolute paths (`/data/int4096_unified_v4/...`) and will not run elsewhere
without editing; the notebooks in `notebooks/` are the portable version. The dequantised bf16 copy of Qwen3.8-27B
exists only on that server (the FP8 original is public: `Qwen/Qwen3.8-27B-FP8`), so `train_qwen.py` cannot be re-run elsewhere
as written; notebook 04 (Part 4) is the portable single-GPU version for a GPU with at least 80 GB.

| file | role |
|---|---|
| `download_data.sh`, `download_models.sh`, `download_falcon.sh` | fetch public corpora and model weights |
| `dequant_fp8_to_bf16.py`, `check_dequant_vs_bf16.py` | convert Qwen3.8-27B FP8 weights to bf16 and check the error |
| `make_venv.sh`, `install_kernels.sh` | environment |
| `build_splits.py` | test sets, training pool, validation set, manifests of arms A/B, overlap gate |
| `common.py` | `clean()` text normalisation and shared loaders (copied into `ddlib/textclean.py`) |
| `train_lr.py`, `followup_ml.py` | TF-IDF + logistic regression, arms A/B and C/D (CPU); `followup_ml.py` also builds the C/D training sets |
| `train_roberta.py`, `train_qwen.py` | RoBERTa-base full fine-tune; Qwen3.8-27B + LoRA |
| `run_gpu.py`, `run_gpu_followup.py`, `run_all.sh`, `run_all_followup.sh` | GPU job orchestration |
| `score_opensource.py` | Binoculars and Fast-DetectGPT (falcon-7b pair), official thresholds |
| `metrics.py`, `followup_metrics.py`, `followup_metrics_gpu.py`, `prf1.py`, `length_control.py` | metrics |
| `surprisal.py`, `all_ppl.py`, `all_ppl_native.py` | per-token surprisal / perplexity under falcon-7b (for the figures on which words look AI-like) |
| `PREREG.md`, `PREREG_followup.md`, `PREREG_followup_gpu.md` | pre-registered endpoints, written before the GPU runs |
| `prereg*_sha256.txt` | time and SHA-256 of each PREREG file when it was frozen (they match the files here) |

Not included: scripts that stop and restart the server's shared production service around the GPU runs
(`run_all*.sh` call one of them, `restore_prod.sh`), a progress monitor and a scheduling helper. They contain no
experimental logic.

`dequant_fp8_to_bf16.py` and `check_dequant_vs_bf16.py` convert the Qwen3.8-27B FP8 weights to bf16 once and measure the weight error (about 2.7%). Notebook 04 (Part 4) contains a ported, tested version of the conversion (it downloads the public FP8 repo).
