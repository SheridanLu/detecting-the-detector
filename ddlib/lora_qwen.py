"""Helpers for notebook 04 (Qwen + LoRA). Four jobs:

1. Hugging Face access: find a token without ever printing it (Colab secret HF_TOKEN, env var HF_TOKEN, or an
   existing `hf auth login` on this machine).
2. "Look inside" the archived LoRA adapters of the Qwen3.8-27B runs: read the safetensors header (a small JSON block
   at the start of the file) WITHOUT loading the 27B base model (16 GB T4s and laptops cannot hold it).
3. A small stand-in LoRA run (default Qwen/Qwen3-0.6B) with the recipe of a40/train_qwen.py, so that the LoRA
   training code can actually be executed on a free Colab T4 or on a laptop CPU.
4. The REAL Qwen3.8-27B (Part 4 of notebook 04, needs a GPU with >= 80 GB): fetch the public FP8 base model
   `Qwen/Qwen3.8-27B-FP8`, dequantise it to bf16 with the exact rule of a40/dequant_fp8_to_bf16.py, load it on one
   GPU, attach the archived adapters, score essays, compare with the archived scores; optional re-training.

torch / transformers / peft are imported inside the functions that need them, so the replay part of the notebook
works without them.
"""
import contextlib
import fnmatch
import json
import os
import pathlib
import re
import shutil
import struct
import time
import zlib

import numpy as np
import pandas as pd

ADAPTER_REPO = "SheridanLu/detecting-the-detector-qwen-lora"
ADAPTER_RUNS = ["seed0_A_n4000", "seed0_B_n4000", "seed0_C", "seed0_D",
                "seed1_A_n4000", "seed1_B_n4000", "seed1_C", "seed1_D"]
EXPECTED_TRAINABLE = 124_741_120          # printed by peft's print_trainable_parameters() in the original run (0.476%)

# The recipe of a40/train_qwen.py. Only the base model, the text length and (in SMOKE/FAST) the number of training
# essays change in the stand-in.
RECIPE = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="SEQ_CLS",
              learning_rate=1e-4, per_device_train_batch_size=8, gradient_accumulation_steps=2, num_train_epochs=1,
              gradient_checkpointing=True, eval_batch_size=16)


# ---- 1. token ---------------------------------------------------------------------------------------------------
def hf_token():
    """Return (token or None, where it came from). The token itself must never be printed or saved."""
    try:
        from google.colab import userdata   # only exists on Colab
        try:
            tok = userdata.get("HF_TOKEN")
            if tok:
                return tok, "Colab secret HF_TOKEN"
        except Exception:
            pass                             # secret missing or notebook not granted access
    except ImportError:
        pass
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "environment variable HF_TOKEN"
    try:
        from huggingface_hub import get_token
        tok = get_token()
        if tok:
            return tok, "existing Hugging Face login on this machine"
    except Exception:
        pass
    return None, None


# ---- 2. adapter inspection ----------------------------------------------------------------------------------------
_DT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def read_safetensors_header(path):
    """Parse a .safetensors file's header by hand: 8 bytes = header length (little-endian uint64), then a JSON object
    {tensor name: {dtype, shape, data_offsets}, '__metadata__': {...}}. No tensor data is read."""
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    tensors = {k: dict(dtype=v["dtype"], shape=list(v["shape"]), data_offsets=list(v["data_offsets"])) for k, v in header.items()}
    return dict(tensors=tensors, metadata=meta, header_bytes=n, file_bytes=path.stat().st_size)


def remote_safetensors_header(filename, token, repo=ADAPTER_REPO):
    """Same information, read from the Hub with HTTP range requests (only the header, a few hundred kB)."""
    from huggingface_hub import HfApi
    m = HfApi(token=token).parse_safetensors_file_metadata(repo, filename)
    tensors = {k: dict(dtype=t.dtype, shape=list(t.shape), data_offsets=list(t.data_offsets)) for k, t in m.tensors.items()}
    return dict(tensors=tensors, metadata=m.metadata or {}, header_bytes=None, file_bytes=None)


def tensor_table(header):
    """One row per tensor: where it sits in the model, what kind it is, shape, dtype, number of parameters."""
    rows = []
    for name, t in header["tensors"].items():
        kind = "lora_A" if ".lora_A." in name else "lora_B" if ".lora_B." in name else "head (trained in full)"
        part = "vision tower" if ".visual." in name else "language model" if ".language_model." in name or ".layers." in name else "classifier head"
        m = re.search(r"(?:layers|blocks)\.(\d+)\.", name)
        module = re.sub(r"^.*?(?:layers|blocks)\.\d+\.", "", name)
        module = re.sub(r"\.lora_[AB]\.weight$", "", module)
        if ".visual.merger." in name:
            module = "merger." + re.sub(r"\.lora_[AB]\.weight$", "", name.split(".visual.merger.")[1])
        rows.append(dict(name=name, part=part, layer=int(m.group(1)) if m else -1, module=module, kind=kind,
                         shape=tuple(t["shape"]), dtype=t["dtype"], params=int(np.prod(t["shape"])) if t["shape"] else 1,
                         nbytes=int(np.prod(t["shape"])) * _DT_BYTES.get(t["dtype"], 0)))
    return pd.DataFrame(rows).sort_values(["part", "layer", "module", "kind"]).reset_index(drop=True)


def lora_rank_from_shapes(tt):
    """LoRA stores each update as B @ A with A of shape (r, in) and B of shape (out, r); r is read off the shapes."""
    a = tt[tt.kind == "lora_A"]["shape"].map(lambda s: s[0])
    b = tt[tt.kind == "lora_B"]["shape"].map(lambda s: s[1])
    return sorted(set(a) | set(b))


def tensor_value_checks(path, tt, max_rank_checks=None):
    """Read the adapter's own tensors (NOT the base model) and check, per tensor, whether it is all zeros, and the
    numerical rank of lora_A (r x in) and lora_B (out x r). LoRA initialises B to zero, so a B that is still all zero
    after training means that this layer never received a gradient."""
    import torch
    from safetensors import safe_open
    rows = []
    with safe_open(str(path), "pt") as f:
        names = tt.name.tolist()
        for i, n in enumerate(names):
            t = f.get_tensor(n).float()
            r = dict(name=n, all_zero=bool((t == 0).all()))
            if (max_rank_checks is None or i < max_rank_checks) and ".lora_" in n:
                r["numerical_rank"] = int(torch.linalg.matrix_rank(t)) if not r["all_zero"] else 0
            rows.append(r)
    return tt.merge(pd.DataFrame(rows), on="name")


# ---- 3. stand-in LoRA run -----------------------------------------------------------------------------------------
def pick_precision(force_fp32=False):
    """Device and number format. A40 run: bf16 weights + bf16 training. T4 has no bf16: keep the weights in fp32 and
    train with fp16 mixed precision (the Trainer then applies loss scaling automatically). CPU: plain fp32.
    force_fp32=True: plain fp32 on the GPU as well (slower; use it if fp16 gives NaN losses)."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if force_fp32:
            return dict(device="cuda", gpu=name, load_dtype=torch.float32, bf16=False, fp16=False, label="fp32 on GPU (forced)")
        if torch.cuda.is_bf16_supported():
            return dict(device="cuda", gpu=name, load_dtype=torch.bfloat16, bf16=True, fp16=False, label="bf16 (as on the A40)")
        return dict(device="cuda", gpu=name, load_dtype=torch.float32, bf16=False, fp16=True,
                    label="fp32 weights + fp16 mixed precision with loss scaling (GPU without bf16, e.g. T4)")
    return dict(device="cpu", gpu=None, load_dtype=torch.float32, bf16=False, fp16=False, label="fp32 on CPU")


def paired_subsample_A_C(XA, yA, XC, yC, partC, n_per_class, seed):
    """Smaller versions of arms A and C that keep the design of the full study: both arms get the SAME AI essays and
    the SAME native essays for half of their human side; only the other half differs (more native essays in A,
    exam-style non-native essays in C). Relies on facts of the saved data that are asserted here: C's AI essays are
    A's AI essays in the same order, and C's native essays are the first 2,000 human essays of A."""
    yA, yC, partC = np.asarray(yA), np.asarray(yC), np.asarray(partC)
    aiA, aiC = np.where(yA == 1)[0], np.where(yC == 1)[0]
    hA = np.where(yA == 0)[0]
    natC, exC = np.where(partC == "native")[0], np.where(partC == "C")[0]
    assert [XA[i] for i in aiA] == [XC[i] for i in aiC], "AI essays of arms A and C differ"
    assert [XA[i] for i in hA[:len(natC)]] == [XC[i] for i in natC], "native half of arm C is not the first half of arm A"
    rng = np.random.default_rng(seed)
    k_ai = rng.choice(len(aiA), n_per_class, replace=False)
    half = n_per_class // 2
    k_nat = rng.choice(len(natC), half, replace=False)                        # shared native essays
    k_A2 = rng.choice(np.arange(len(natC), len(hA)), n_per_class - half, replace=False)   # A only: more native
    k_C2 = rng.choice(len(exC), n_per_class - half, replace=False)            # C only: exam-style non-native
    iA = np.r_[hA[k_nat], hA[k_A2], aiA[k_ai]]
    iC = np.r_[natC[k_nat], exC[k_C2], aiC[k_ai]]
    return ([XA[i] for i in iA], yA[iA].tolist()), ([XC[i] for i in iC], yC[iC].tolist())


def score_subset(score_df, which, per_corpus=None, seed=0):
    """Texts a stand-in run is asked to score. 'all' = every exam essay + validation set (as in the original run).
    'core' = what the headline numbers need: the Liang test set (TOEFL, Hewlett, college essays, CS224N, 352 GPT-3
    essays), LOCNESS (third unfamiliar native set) and the AI essays of the validation set (for the 90%-detection
    threshold). per_corpus: keep at most this many essays per corpus (SMOKE only)."""
    from . import metrics as M
    if which == "all":
        return score_df.reset_index(drop=True)
    keep = set(M.E1_HUMAN + M.E1_AI + ["LOCNESS_native"])
    s = score_df[score_df.corpus.isin(keep) | ((score_df.exam == "VAL") & score_df.corpus.isin(M.AI_VAL))]
    if per_corpus:
        s = pd.concat([g.sample(min(len(g), per_corpus), random_state=seed) for _, g in s.groupby("corpus")]).sort_index()
    return s.reset_index(drop=True)


def run_id(model_name, arm, seed, n_train, max_len):
    return f"{model_name.split('/')[-1]}_{arm}_seed{seed}_n{n_train}_len{max_len}"


def finished(out_dir, rid, cfg):
    """A run counts as finished if its predictions file exists and was made with the same settings."""
    p, j = pathlib.Path(out_dir) / f"{rid}.parquet", pathlib.Path(out_dir) / f"{rid}.json"
    if not (p.exists() and j.exists()):
        return False
    try:
        old = json.load(open(j))["config"]
    except Exception:
        return False
    return all(old.get(k) == v for k, v in cfg.items())


def train_and_score(X, y, score_df, cfg, out_dir, rid, log_every=None, verbose=True, *, device_map=None, batch=None,
                    save_steps=100, ckpt_root=None, adapter_dir=None, model_label=None, simulate_oom_above=None):
    """Train one LoRA run with the recipe of a40/train_qwen.py and score `score_df`.
    cfg: model (Hub name or local folder), arm, seed, n_train, max_len, n_scored. Writes <rid>.parquet (one score per
    essay, same columns as data/preds/) and <rid>.json (settings and timings). Mid-run checkpoints go to
    <ckpt_root or out_dir/ckpt>/<rid> so that a restarted notebook resumes instead of starting the run again.
    Used by Part 3 (stand-in, defaults below) and by Part 4c (real 27B):
      device_map   e.g. {"": 0}: load the weights straight onto one GPU (the 27B does not fit in a T4 or in RAM twice)
      batch        None = recipe (8 x 2); "auto" = largest of 4 / 2 / 1 that fits (probe_train_batch), with gradient
                   accumulation 16 / 4 / 8 / 16 so that 16 essays still go into every optimizer step; or (bs, accum)
      adapter_dir  also save the trained adapter there (as a40/train_qwen.py does)
      simulate_oom_above  TESTS ONLY: pretend that batches larger than this run out of memory"""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback,
                              TrainingArguments)
    from peft import LoraConfig, get_peft_model
    from . import seeds

    import transformers, warnings
    transformers.logging.set_verbosity_error()        # hide the expected "lm_head unexpected / score missing" report
    warnings.filterwarnings("ignore", category=FutureWarning)
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prec = pick_precision(cfg.get("force_fp32", False))
    if prec["device"] == "cpu":
        # On a busy multi-core CPU, PyTorch's default thread count was several times slower in our test than 3-4 threads.
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    seeds.set_seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    load_kw = dict(num_labels=2, dtype=prec["load_dtype"])
    if device_map is not None:
        load_kw["device_map"] = device_map
    model, load_info = AutoModelForSequenceClassification.from_pretrained(cfg["model"], output_loading_info=True, **load_kw)
    model.config.pad_token_id = tok.pad_token_id
    try:
        model.config.get_text_config().pad_token_id = tok.pad_token_id
    except Exception:
        pass
    model = get_peft_model(model, LoraConfig(task_type=RECIPE["task_type"], r=RECIPE["r"], lora_alpha=RECIPE["lora_alpha"],
                                             lora_dropout=RECIPE["lora_dropout"], target_modules=RECIPE["target_modules"]))
    trainable, total = model.get_nb_trainable_parameters()    # the numbers peft's print_trainable_parameters() shows
    model.enable_input_require_grads()
    if prec["device"] == "cuda" and device_map is None:
        model.to("cuda")
    load_s = time.time() - t0
    L = cfg["max_len"]
    probe_rows = []
    if batch is None:
        bs, accum = RECIPE["per_device_train_batch_size"], RECIPE["gradient_accumulation_steps"]
    elif batch == "auto":
        bs, accum, probe_rows = probe_train_batch(model, L, pad_id=tok.pad_token_id, simulate_oom_above=simulate_oom_above,
                                                  log=print if verbose else (lambda *a, **k: None))
        seeds.set_seed(cfg["seed"]); torch.manual_seed(cfg["seed"])    # the probe used random numbers (dropout)
    else:
        bs, accum = batch

    class DS(torch.utils.data.Dataset):
        def __init__(self, X, y): self.X, self.y = X, y
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            e = dict(tok(self.X[i], truncation=True, max_length=L)); e["labels"] = self.y[i]; return e

    def collate(fs):                                  # verbatim logic of a40/train_qwen.py: right padding
        n = max(len(f["input_ids"]) for f in fs); pad = tok.pad_token_id
        return {"input_ids": torch.tensor([f["input_ids"] + [pad] * (n - len(f["input_ids"])) for f in fs]),
                "attention_mask": torch.tensor([[1] * len(f["input_ids"]) + [0] * (n - len(f["input_ids"])) for f in fs]),
                "labels": torch.tensor([f["labels"] for f in fs])}

    losses = []

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                losses.append((state.global_step, float(logs["loss"])))

        def on_step_end(self, args, state, control, **kw):
            s = state.global_step
            if verbose and (s in (1, 5, 20) or (log_every and s % log_every == 0) or s == state.max_steps):
                el = time.time() - t_train
                print(f"    step {s}/{state.max_steps}  elapsed {el:5.0f} s  projected training time "
                      f"{el / s * state.max_steps:5.0f} s", flush=True)

    ckpt = pathlib.Path(ckpt_root or out_dir / "ckpt") / rid
    save = cfg["n_train"] >= 1600                     # checkpoints only for runs of 100+ steps
    args = TrainingArguments(output_dir=str(ckpt), per_device_train_batch_size=bs,
                             gradient_accumulation_steps=accum,
                             num_train_epochs=RECIPE["num_train_epochs"], learning_rate=RECIPE["learning_rate"],
                             seed=cfg["seed"], bf16=prec["bf16"], fp16=prec["fp16"], use_cpu=prec["device"] == "cpu",
                             logging_steps=1 if cfg["n_train"] < 1600 else 10,
                             save_strategy="steps" if save else "no", save_steps=save_steps, save_total_limit=1,
                             report_to="none", gradient_checkpointing=RECIPE["gradient_checkpointing"],
                             gradient_checkpointing_kwargs={"use_reentrant": False},
                             remove_unused_columns=False, label_names=["labels"], disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=DS(X, y), data_collator=collate, callbacks=[Progress()])
    resume = sorted(ckpt.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) if save else []
    if resume and verbose:
        print(f"    resuming from {resume[-1].name}")
    if verbose:
        print(f"    {cfg['model']} loaded in {load_s:.0f} s ({prec['label']}); its language-model output layer is dropped and a "
              f"new 2-class head 'score' is created, as for the 27B. Trainable: {trainable:,} of {total:,} ({trainable / total:.2%})", flush=True)
        if batch is not None:
            print(f"    batch {bs} x gradient accumulation {accum} = {bs * accum} essays per optimizer step "
                  f"(recipe: {RECIPE['per_device_train_batch_size']} x {RECIPE['gradient_accumulation_steps']})", flush=True)
    t_train = time.time()
    trainer.train(resume_from_checkpoint=str(resume[-1]) if resume else None)
    train_s = time.time() - t_train
    if adapter_dir is not None:
        model.save_pretrained(str(pathlib.Path(adapter_dir) / rid)); tok.save_pretrained(str(pathlib.Path(adapter_dir) / rid))

    # score: sort by length so that each batch has similar lengths (faster), then restore the original order
    model.eval()
    t_eval = time.time()
    scores = score_texts(model, tok, score_df.text.tolist(), L, batch_size=RECIPE["eval_batch_size"], sort_by_length=True,
                         fp16_autocast=prec["fp16"])
    eval_s = time.time() - t_eval

    out = score_df[["exam", "corpus", "essay_id"]].copy(); out["score"] = scores
    out["model"] = model_label or f"STAND-IN {cfg['model']}"; out["run"] = rid
    out.to_parquet(out_dir / f"{rid}.parquet", index=False)
    import peft
    info = dict(config=cfg, trainable_params=int(trainable), total_params=int(total), device=prec["device"],
                gpu=prec["gpu"], precision=prec["label"], load_s=round(load_s, 1), train_s=round(train_s, 1),
                eval_s=round(eval_s, 1), optimizer_steps=int(trainer.state.global_step),
                loss_first=losses[0][1] if losses else None, loss_last=losses[-1][1] if losses else None,
                nan_scores=int(np.isnan(scores).sum()), per_device_batch=int(bs), grad_accum=int(accum),
                batch_probe=probe_rows, missing_keys=sorted(load_info.get("missing_keys", [])),
                peak_gpu_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2) if prec["device"] == "cuda" else None,
                versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__))
    json.dump(info, open(out_dir / f"{rid}.json", "w"), indent=1)
    del trainer, model
    if prec["device"] == "cuda":
        torch.cuda.empty_cache()
    shutil.rmtree(ckpt, ignore_errors=True)            # the run is finished; checkpoints are no longer needed
    return info


def score_texts(model, tok, texts, max_len, batch_size=16, sort_by_length=False, fp16_autocast=False, log_every=None,
                log=print):
    """AI probability of each text: softmax over the 2 logits, column 1 (as a40/train_qwen.py). sort_by_length=False
    keeps the original order and therefore the original batches of 16 (used to reproduce archived scores);
    True groups similar lengths (faster; used after training). If the GPU runs out of memory, the batch size is
    halved for the rest of the texts (the scores do not depend on the batch, up to floating-point noise)."""
    import torch
    dev = model.get_input_embeddings().weight.device
    order = np.argsort([len(t) for t in texts]) if sort_by_length else np.arange(len(texts))   # same sort as before
    scores = np.full(len(texts), np.nan); i = 0; bs = batch_size; t0 = time.time(); n_batches = 0
    with torch.no_grad():
        while i < len(texts):
            idx = order[i:i + bs]
            try:
                enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(dev)
                amp = torch.autocast(device_type="cuda", dtype=torch.float16) if fp16_autocast else contextlib.nullcontext()
                with amp:
                    logits = model(**enc).logits
                scores[idx] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
            except torch.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2); torch.cuda.empty_cache()
                log(f"    out of GPU memory while scoring: batch size lowered to {bs}")
                continue
            i += len(idx); n_batches += 1
            if log_every and n_batches % log_every == 0:
                log(f"    scored {i:,}/{len(texts):,} essays in {time.time() - t0:.0f} s", flush=True)
    return scores


# ---- 4. the REAL Qwen3.8-27B (notebook 04, Part 4; needs a GPU with >= 80 GB) --------------------------------------
# Base model of all 8 adapters: the public, non-gated Hub repo below, dequantised once from FP8 to bf16 with
# a40/dequant_fp8_to_bf16.py and then loaded with AutoModelForSequenceClassification (a40/train_qwen.py).
BASE_REPO = "Qwen/Qwen3.8-27B-FP8"
BASE_REVISION = "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"   # Hub revision read on 2026-11-19. Our server's copy has
#                     the same config, shard count and size; its file hashes were NOT compared with this revision.
FP8_BLOCK = 128                         # 128 x 128 blocks share one scale (config: weight_block_size [128, 128])
BASE_FP8_BYTES = 30_890_041_705         # whole Hub repo (66 safetensors shards + small files)
BASE_BF16_BYTES = 55_562_855_904        # 27,781,427,952 numbers x 2 bytes after dequantisation (whole checkpoint)
LARGEST_SHARD_BYTES = 6_007_102_112     # outside.safetensors (embeddings, lm_head, vision tower; already bf16)
SEQCLS_BASE_PARAMS = 26_085_340_400     # numbers the sequence-classification model keeps on the GPU (no lm_head, no MTP)
GiB = 2 ** 30
# Reproduction-by-inference tolerances (justified in notebook 04, Part 4b; fixed before any 27B result was seen).
REPRO_TOL = dict(pearson=0.99, spearman=0.98, mean_abs_diff=0.02, agree=0.98, toefl_flag_essays=3)
_SMALL_FILES = ["*.json", "*.txt", "*.jinja"]   # config, index, tokenizer, crc32.txt (no weights)


def dequant_fp8_block(w, s, block=FP8_BLOCK):
    """W_bf16 = W_fp8.float() * block-expanded(scale_inv): the two lines of a40/dequant_fp8_to_bf16.py, unchanged.
    w: (out, in) float8_e4m3fn; s: (ceil(out/128), ceil(in/128)) scales. Each scale is repeated over its 128 x 128
    block and the edge blocks are cut to the matrix size, so shapes that are not multiples of 128 work too."""
    import torch
    o, i = w.shape
    se = s.float().repeat_interleave(block, 0)[:o].repeat_interleave(block, 1)[:, :i]
    return (w.float() * se).to(torch.bfloat16)


def dequant_reference_loop(w, s, block=FP8_BLOCK):
    """Independent, slow reference for tests: multiply block by block with explicit loops."""
    import torch
    o, i = w.shape; out = torch.empty(o, i, dtype=torch.float32)
    for bi in range((o + block - 1) // block):
        for bj in range((i + block - 1) // block):
            r, c = slice(bi * block, min(o, (bi + 1) * block)), slice(bj * block, min(i, (bj + 1) * block))
            out[r, c] = w[r, c].to(torch.float32) * s[bi, bj].to(torch.float32)
    return out.to(torch.bfloat16)


def dequant_plan(weight_map, dtypes=None):
    """What a40/dequant_fp8_to_bf16.py does with each tensor: 'drop' (name ends in _scale_inv), 'dequant' (stored as
    float8_e4m3fn; needs '<name>_scale_inv', else the script stops) or 'keep' (copied unchanged). dtypes: {name:
    safetensors dtype string} from the shard headers; without it, 'has a <name>_scale_inv' stands in for 'is FP8'."""
    plan = {}
    for k in weight_map:
        if k.endswith("_scale_inv"):
            plan[k] = "drop"; continue
        fp8 = dtypes[k].upper().startswith("F8_E4M3") if dtypes is not None else (k + "_scale_inv") in weight_map
        if fp8 and (k + "_scale_inv") not in weight_map:
            raise ValueError(f"missing scale for {k}")
        plan[k] = "dequant" if fp8 else "keep"
    return plan


def check_plan(weight_map, headers, block=FP8_BLOCK):
    """Checks on the real checkpoint (index + shard headers, no weights): returns (checks list, kept-tensor table)."""
    dtypes = {k: v["dtype"] for h in headers.values() for k, v in h.items()}
    shapes = {k: v["shape"] for h in headers.values() for k, v in h.items()}
    by_name, by_dtype = dequant_plan(weight_map), dequant_plan(weight_map, dtypes)
    deq = [k for k, a in by_dtype.items() if a == "dequant"]
    scales = [k for k in weight_map if k.endswith("_scale_inv")]
    bad_shape = [k for k in deq if shapes[k + "_scale_inv"] != [-(-shapes[k][0] // block), -(-shapes[k][1] // block)]]
    checks = [
        ("every index entry is in a shard header, same shard", set(weight_map) == set(dtypes)
         and all(k in headers[f] for k, f in weight_map.items())),
        ("name rule (has *_scale_inv) == dtype rule (stored as FP8)", by_name == by_dtype),
        ("every FP8 tensor is a 2-D '*.weight' with a '*.weight_scale_inv'", all(k.endswith(".weight") and len(shapes[k]) == 2 for k in deq)),
        ("every *_scale_inv belongs to an FP8 tensor", sorted(k[:-len("_scale_inv")] for k in scales) == sorted(deq)),
        ("scale shapes = ceil(out/128) x ceil(in/128)", not bad_shape),
        ("kept tensors are not FP8", all(not dtypes[k].upper().startswith("F8") for k, a in by_dtype.items() if a == "keep")),
    ]
    rows = []
    for k, a in by_dtype.items():
        if a != "keep":
            continue
        part = "vision tower" if ".visual." in k else "MTP head (unused)" if k.startswith("mtp.") else \
               "lm_head (unused)" if k.startswith("lm_head") else "language model"
        kind = k.split(".")[-1] if k.split(".")[-1] not in ("weight", "bias") else ".".join(k.split(".")[-2:])
        rows.append(dict(part=part, tensor=re.sub(r"\.\d+\.", ".N.", kind), dtype=dtypes[k]))
    kept = pd.DataFrame(rows).groupby(["part", "tensor", "dtype"]).size().rename("tensors").reset_index()
    summary = dict(index_entries=len(weight_map), dequant=len(deq), keep=sum(a == "keep" for a in by_dtype.values()),
                   drop=len(scales), non_multiple_of_128=sum(1 for k in deq if shapes[k][0] % block or shapes[k][1] % block))
    return checks, kept, summary


def shard_headers_from_hub(repo=BASE_REPO, revision=BASE_REVISION, token=None):
    """Header (name, dtype, shape) of every shard, read with HTTP range requests (a few kB each; no weights)."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    wm = json.load(open(hf_hub_download(repo, "model.safetensors.index.json", revision=revision, token=token)))["weight_map"]
    heads = {}
    for fn in sorted(set(wm.values())):
        m = api.parse_safetensors_file_metadata(repo, fn, revision=revision)
        heads[fn] = {k: dict(dtype=t.dtype, shape=list(t.shape)) for k, t in m.tensors.items()}
    return wm, heads


def _needed_files(fn, weight_map):
    keys = [k for k, f in weight_map.items() if f == fn and not k.endswith("_scale_inv")]
    return {fn} | {weight_map[k + "_scale_inv"] for k in keys if (k + "_scale_inv") in weight_map}


def dequant_shard(src_dir, fn, weight_map, device="cpu", block=FP8_BLOCK):
    """One shard, exactly as the loop body of a40/dequant_fp8_to_bf16.py (the scale may sit in another shard).
    device='cuda' only moves the multiplication to the GPU (a single float32 multiply and one rounding to bf16 per
    number, so the result should be bit-identical to the CPU; notebook 04 checks this on the first shard)."""
    import torch
    from safetensors import safe_open
    src_dir = pathlib.Path(src_dir); tensors = {}; n_deq = 0
    with safe_open(str(src_dir / fn), "pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith("_scale_inv"):
                continue
            t = f.get_tensor(k); sk = k + "_scale_inv"
            if t.dtype == torch.float8_e4m3fn:
                if sk in keys:
                    s = f.get_tensor(sk)
                else:
                    sf = weight_map.get(sk)
                    assert sf is not None, f"missing scale for {k}"
                    with safe_open(str(src_dir / sf), "pt") as g:
                        s = g.get_tensor(sk)
                t = dequant_fp8_block(t.to(device), s.to(device), block).cpu()
                n_deq += 1
            tensors[k] = t.contiguous()
    return tensors, n_deq


def read_crc32_list(path):
    out = {}
    for line in open(path):
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0].lower()
    return out


def file_crc32(path, chunk=64 * 2**20):
    c = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return f"{c & 0xFFFFFFFF:08x}"
            c = zlib.crc32(b, c)


def hub_fetcher(local_dir, repo=BASE_REPO, revision=BASE_REVISION, token=None):
    """fetch(patterns): download the matching files of the base-model repo into local_dir (skips files already there)."""
    from huggingface_hub import snapshot_download

    def fetch(patterns):
        snapshot_download(repo, revision=revision, allow_patterns=list(patterns), local_dir=str(local_dir), token=token)
    return fetch


def folder_fetcher(src, local_dir):
    """Same interface as hub_fetcher, but copies from a local folder (used by the tests with a tiny checkpoint)."""
    src, local_dir = pathlib.Path(src), pathlib.Path(local_dir)

    def fetch(patterns):
        local_dir.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_file() and any(fnmatch.fnmatch(p.name, pat) for pat in patterns) and not (local_dir / p.name).exists():
                shutil.copy2(p, local_dir / p.name)
    return fetch


def prepare_bf16_base(fp8_dir, bf16_dir, fetch, delete_fp8=False, device="cpu", verify_crc=True, log=print, log_every=10):
    """Download the FP8 base and write the bf16 folder that a40/train_qwen.py loaded, shard by shard.
    Same rule and output as a40/dequant_fp8_to_bf16.py (same tensors dequantised / kept / dropped, same index note,
    config without quantization_config, other files copied). Additions, none of which changes a number:
      * each bf16 shard is written to a temporary name and renamed when complete, so after a disconnect the
        conversion resumes at the first missing shard (a half-written shard is never taken as finished);
      * delete_fp8=True: weights are downloaded one shard at a time and each FP8 shard is deleted as soon as no
        remaining shard needs it (peak disk ~ bf16 folder + largest shard instead of bf16 + all of FP8);
      * every FP8 shard is checked against the CRC32 list published in the repo before it is used."""
    from safetensors.torch import save_file
    fp8_dir, bf16_dir = pathlib.Path(fp8_dir), pathlib.Path(bf16_dir)
    fp8_dir.mkdir(parents=True, exist_ok=True); bf16_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fetch(_SMALL_FILES)
    wm = json.load(open(fp8_dir / "model.safetensors.index.json"))["weight_map"]
    files = sorted(set(wm.values()))
    need = {fn: _needed_files(fn, wm) for fn in files}
    crc = read_crc32_list(fp8_dir / "crc32.txt") if verify_crc and (fp8_dir / "crc32.txt").exists() else {}
    todo = [fn for fn in files if not (bf16_dir / fn).exists()]
    if todo and not delete_fp8:
        log(f"downloading {len(set().union(*(need[f] for f in todo)))} FP8 shards ...", flush=True)
        fetch(sorted(set().union(*(need[f] for f in todo))))
    checked, n_deq, crc_fail = set(), 0, []
    for k, fn in enumerate(files):
        out = bf16_dir / fn
        if out.exists():
            continue
        fetch(sorted(need[fn]))
        for g in need[fn] - checked:
            if crc:
                got = file_crc32(fp8_dir / g)
                if got != crc.get(g):
                    crc_fail.append(g)
                    raise RuntimeError(f"{g}: CRC32 {got} does not match the published {crc.get(g)}; delete it and re-run")
            checked.add(g)
        tensors, n = dequant_shard(fp8_dir, fn, wm, device=device)
        tmp = out.with_name(out.name + ".tmp")
        save_file(tensors, str(tmp), metadata={"format": "pt"}); os.replace(tmp, out)
        n_deq += n; del tensors
        if delete_fp8:
            still = set().union(*[need[r] for r in files[k + 1:] if not (bf16_dir / r).exists()]) if k + 1 < len(files) else set()
            for g in need[fn] - still:
                (fp8_dir / g).unlink(missing_ok=True)
        if k == 0 or k + 1 == len(files) or (k + 1) % log_every == 0:
            log(f"[{k + 1}/{len(files)}] {fn}: {n} FP8 tensors dequantised, elapsed {time.time() - t0:.0f} s", flush=True)
    # index, config and other files: as the end of a40/dequant_fp8_to_bf16.py
    new_map = {k: f for k, f in wm.items() if not k.endswith("_scale_inv")}
    json.dump({"metadata": {"note": "dequantized from /data/qwen38fn/model (fp8 e4m3, block 128) by int4096 unified_v4"},
               "weight_map": new_map}, open(bf16_dir / "model.safetensors.index.json", "w"), indent=1)
    cfg = json.load(open(fp8_dir / "config.json"))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    if "text_config" in cfg:
        cfg["text_config"]["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(bf16_dir / "config.json", "w"), indent=2)
    for fn in os.listdir(fp8_dir):
        if fn.endswith(".safetensors") or fn in ("config.json", "model.safetensors.index.json") or fn.startswith("."):
            continue
        p = fp8_dir / fn
        if p.is_file():
            shutil.copy2(p, bf16_dir / fn)
    plan = dequant_plan(wm)
    missing = [fn for fn in files if not (bf16_dir / fn).exists()]
    return dict(shards=len(files), shards_missing=len(missing), fp8_tensors_in_plan=sum(a == "dequant" for a in plan.values()),
                dequantised_now=n_deq, total_tensors=len(new_map), crc_checked=len(checked) if crc else 0,
                bf16_bytes=sum((bf16_dir / fn).stat().st_size for fn in files if (bf16_dir / fn).exists()),
                seconds=round(time.time() - t0, 1))


def disk_free(path):
    p = pathlib.Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def disk_needed(fp8_dir, bf16_dir, delete_fp8):
    """Bytes still to be written: rest of the bf16 folder, plus (keep mode) the FP8 shards not yet downloaded, plus
    one largest shard of head room. Files already on disk are subtracted, so a resumed run asks for less."""
    have_bf16 = sum(p.stat().st_size for p in pathlib.Path(bf16_dir).glob("*.safetensors")) if pathlib.Path(bf16_dir).exists() else 0
    have_fp8 = sum(p.stat().st_size for p in pathlib.Path(fp8_dir).glob("*.safetensors")) if pathlib.Path(fp8_dir).exists() else 0
    need = max(0, BASE_BF16_BYTES - have_bf16) + LARGEST_SHARD_BYTES + 2 * GiB
    if not delete_fp8:
        need += max(0, BASE_FP8_BYTES - have_fp8)
    return need


def gpu_check(min_total_gib=75):
    """(ok, message, info). 80 GB cards report about 79 GiB in total; the 27B weights alone take ~48.6 GiB."""
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed.", {}
    if not torch.cuda.is_available():
        return False, "no CUDA GPU on this machine.", {}
    p = torch.cuda.get_device_properties(0); free, total = torch.cuda.mem_get_info(0)
    info = dict(gpu=p.name, total_gib=round(total / GiB, 1), free_gib=round(free / GiB, 1),
                capability=f"{p.major}.{p.minor}", bf16=torch.cuda.is_bf16_supported())
    if total / GiB < min_total_gib:
        return False, f"GPU {p.name} has {total / GiB:.1f} GiB; Part 4 needs >= {min_total_gib} GiB (an 80 or 96 GB card).", info
    if free / GiB < min_total_gib - 8:
        return False, f"GPU {p.name}: only {free / GiB:.1f} GiB free (something else holds memory; restart the runtime).", info
    if not info["bf16"]:
        return False, f"GPU {p.name} has no bf16 support (the 27B runs used bf16).", info
    return True, f"GPU {p.name}: {total / GiB:.1f} GiB total, {free / GiB:.1f} GiB free, compute capability {info['capability']}", info


def probe_train_batch(model, max_len, candidates=(4, 2, 1), effective=16, pad_id=None, margin_gib=3.0,
                      simulate_oom_above=None, log=print):
    """Pick the per-device batch size for training: try 4, then 2, then 1 on one worst-case batch (every essay at
    max_len tokens; forward + backward with gradient checkpointing, as in training) and keep the first that leaves
    room for the optimizer state (3 x 4 bytes per trainable number: gradient + two Adam moments) plus margin_gib.
    Gradient accumulation = effective / batch, so every optimizer step still sees 16 essays as in the recipe.
    Returns (batch, accumulation, probe table). simulate_oom_above: TESTS ONLY, fakes out-of-memory above that size."""
    import gc
    import torch
    dev = model.get_input_embeddings().weight.device
    on_cuda = dev.type == "cuda"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    extra = 3 * 4 * trainable
    total = torch.cuda.get_device_properties(dev).total_memory if on_cuda else None
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass
    model.train(); rows = []
    for bs in candidates:
        assert effective % bs == 0, (effective, bs)
        try:
            if simulate_oom_above is not None and bs > simulate_oom_above:
                raise torch.OutOfMemoryError(f"simulated out-of-memory at batch {bs} (test only)")
            if on_cuda:
                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
            g = torch.Generator().manual_seed(0)
            ids = torch.randint(1000, 150000, (bs, max_len), generator=g)
            if pad_id is not None:
                ids[ids == pad_id] = 1000
            amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if on_cuda else contextlib.nullcontext()
            with amp:
                loss = model(input_ids=ids.to(dev), attention_mask=torch.ones_like(ids).to(dev),
                             labels=torch.zeros(bs, dtype=torch.long).to(dev)).loss
            loss.backward()
            peak = torch.cuda.max_memory_allocated(dev) if on_cuda else 0
            model.zero_grad(set_to_none=True)
            fits = (not on_cuda) or peak + extra + margin_gib * GiB <= total
            rows.append(dict(batch=bs, accumulation=effective // bs, peak_gib=round(peak / GiB, 2) if on_cuda else None,
                             result="fits" if fits else "too close to the limit"))
            log(f"    batch probe: {bs} x {max_len} tokens -> " + (f"peak {peak / GiB:.1f} GiB + optimizer {extra / GiB:.1f} GiB "
                f"of {total / GiB:.1f} GiB: " if on_cuda else "CPU (no memory limit measured): ") + rows[-1]["result"], flush=True)
            if fits:
                return bs, effective // bs, rows
        except torch.OutOfMemoryError as e:
            model.zero_grad(set_to_none=True); loss = None
            if on_cuda:
                gc.collect(); torch.cuda.empty_cache()
            rows.append(dict(batch=bs, accumulation=effective // bs, peak_gib=None, result=f"out of memory ({str(e)[:60]})"))
            log(f"    batch probe: {bs} x {max_len} tokens -> out of memory", flush=True)
    raise RuntimeError("Even batch 1 does not fit. Lower MAX_LEN in the Part 4 CONFIG cell (e.g. 384 or 256).")


def load_base_seqcls(path, device_map=None, dtype=None):
    """Base model exactly as a40/train_qwen.py loads it: AutoModelForSequenceClassification, 2 classes, bf16, pad token
    = the tokenizer's pad token. Returns (tokenizer, model, loading_info); loading_info lists the checkpoint tensors
    the model did not use (lm_head, MTP head) and the model tensors not in the checkpoint (only the new head 'score')."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(num_labels=2, dtype=dtype or torch.bfloat16, output_loading_info=True)
    if device_map is not None:
        kw["device_map"] = device_map
    model, info = AutoModelForSequenceClassification.from_pretrained(str(path), **kw)
    model.config.pad_token_id = tok.pad_token_id
    try:
        model.config.get_text_config().pad_token_id = tok.pad_token_id
    except Exception:
        pass
    model.eval()
    return tok, model, info


def download_adapter(run, token, repo=ADAPTER_REPO):
    """Local folder with adapter_config.json + adapter_model.safetensors of one archived run (499 MB)."""
    from huggingface_hub import hf_hub_download
    st = hf_hub_download(repo, f"{run}/adapter_model.safetensors", token=token)
    hf_hub_download(repo, f"{run}/adapter_config.json", token=token)
    return pathlib.Path(st).parent


def attach_adapters(model, adapter_dirs):
    """Attach several archived adapters to one base model (PEFT, task SEQ_CLS from adapter_config.json); the
    first becomes active. adapter_dirs: {name: folder}. Each adapter brings its own LoRA matrices and its own copy of
    the head 'score' (modules_to_save), so switching with set_adapter(name) switches both."""
    from peft import PeftModel
    names = list(adapter_dirs)
    pm = PeftModel.from_pretrained(model, str(adapter_dirs[names[0]]), adapter_name=names[0], is_trainable=False)
    for n in names[1:]:
        pm.load_adapter(str(adapter_dirs[n]), adapter_name=n, is_trainable=False)
    pm.set_adapter(names[0]); pm.eval()
    return pm


def verify_adapter_loaded(pm, name, adapter_dir):
    """Every tensor in the adapter file must be in the model, with exactly the same values, and the model must not
    hold any LoRA tensor of this adapter that the file does not have. A wrong key mapping (e.g. another
    transformers version naming layers differently) would make PEFT keep random/zero matrices: this catches it."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    f = load_file(str(pathlib.Path(adapter_dir) / "adapter_model.safetensors"))
    m = get_peft_model_state_dict(pm, adapter_name=name)
    equal = sum(1 for k, v in f.items() if k in m and torch.equal(m[k].detach().to("cpu", v.dtype), v))
    return dict(adapter=name, tensors_in_file=len(f), tensors_in_model=len(m), identical=equal,
                missing_in_model=len(set(f) - set(m)), extra_in_model=len(set(m) - set(f)),
                result="PASS" if equal == len(f) == len(m) else "FAIL")


def compare_scores(new, old, tol=None):
    """Per-essay comparison of two score files (joined on corpus + essay_id). Returns one dict with the measures and
    PASS/FAIL against the tolerances (REPRO_TOL by default)."""
    from scipy.stats import pearsonr, spearmanr
    from .metrics import TOEFL
    tol = tol or REPRO_TOL
    m = new[["corpus", "essay_id", "score"]].merge(old[["corpus", "essay_id", "score"]], on=["corpus", "essay_id"], suffixes=("_new", "_old"))
    a, b = m.score_new.to_numpy(float), m.score_old.to_numpy(float)
    tf = (m.corpus == TOEFL).to_numpy()
    r = dict(essays=len(m), nan=int(np.isnan(a).sum() + np.isnan(b).sum()))
    ok = r["nan"] == 0 and len(m) > 2
    r["pearson"] = float(pearsonr(a, b)[0]) if ok else np.nan
    r["spearman"] = float(spearmanr(a, b)[0]) if ok else np.nan
    r["mean_abs_diff"] = float(np.abs(a - b).mean()) if ok else np.nan
    r["max_abs_diff"] = float(np.abs(a - b).max()) if ok else np.nan
    r["same_decision_at_0.5"] = float(((a >= 0.5) == (b >= 0.5)).mean()) if ok else np.nan
    r["TOEFL_flagged_new_%"] = 100 * float((a[tf] >= 0.5).mean()) if tf.any() else np.nan
    r["TOEFL_flagged_archived_%"] = 100 * float((b[tf] >= 0.5).mean()) if tf.any() else np.nan
    r["TOEFL_flag_diff_essays"] = int(abs(int((a[tf] >= 0.5).sum()) - int((b[tf] >= 0.5).sum()))) if tf.any() else -1
    passed = ok and tf.any() and r["pearson"] >= tol["pearson"] and r["spearman"] >= tol["spearman"] \
        and r["mean_abs_diff"] <= tol["mean_abs_diff"] and r["same_decision_at_0.5"] >= tol["agree"] \
        and r["TOEFL_flag_diff_essays"] <= tol["toefl_flag_essays"]
    r["result"] = "PASS" if passed else "FAIL"
    return r


def meta_model_checks(config_dir, adapter_keys=None):
    """Build the REAL 27B sequence-classification model on PyTorch's 'meta' device (shapes only, no memory, no weights),
    wrap it with the LoRA recipe and compare with the archived run: number of trainable parameters (124,741,120
    printed in the original run) and, if given, the tensor names stored in an archived adapter file."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    cfg = AutoConfig.from_pretrained(str(config_dir), num_labels=2)
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config                  # what a40/dequant_fp8_to_bf16.py removes from config.json
    with torch.device("meta"):
        m = AutoModelForSequenceClassification.from_config(cfg, dtype=torch.bfloat16)
    wm = json.load(open(pathlib.Path(config_dir) / "model.safetensors.index.json"))["weight_map"]
    ckpt = {k for k in wm if not k.endswith("_scale_inv")}
    model_keys = set(m.state_dict())
    base_params = sum(p.numel() for p in m.parameters())
    pm = get_peft_model(m, LoraConfig(task_type=RECIPE["task_type"], r=RECIPE["r"], lora_alpha=RECIPE["lora_alpha"],
                                      lora_dropout=RECIPE["lora_dropout"], target_modules=RECIPE["target_modules"]))
    trainable, total = pm.get_nb_trainable_parameters()
    peft_keys = set(get_peft_model_state_dict(pm))
    out = dict(model_class=type(m).__name__, base_params=int(base_params), trainable=int(trainable), total=int(total),
               model_not_in_checkpoint=sorted(model_keys - ckpt), checkpoint_not_in_model=sorted(ckpt - model_keys),
               peft_keys=len(peft_keys))
    if adapter_keys is not None:
        out["adapter_keys_equal"] = set(adapter_keys) == peft_keys
        out["adapter_keys_only_in_file"] = len(set(adapter_keys) - peft_keys)
        out["adapter_keys_only_in_model"] = len(peft_keys - set(adapter_keys))
    return out


# ---- 4t. TEST ONLY: a tiny random Qwen3.5 checkpoint in the layout of the FP8 Hub repo -----------------------------
_HUB_FP8_MODULES = re.compile(r"^(model\.language_model\.layers|mtp\.layers)\.\d+\.(mlp\.(gate|up|down)_proj|"
                              r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.(q|k|v|o)_proj)\.weight$")


def quantize_fp8_block(w, block=FP8_BLOCK):
    """Block-wise FP8 e4m3 quantisation (scale = block max / 448, stored as bf16 'weight_scale_inv'), the format of
    the Hub repo. Only used to BUILD test checkpoints; the notebook never quantises."""
    import torch
    o, i = w.shape; nb0, nb1 = -(-o // block), -(-i // block)
    wp = torch.zeros(nb0 * block, nb1 * block); wp[:o, :i] = w.float()
    amax = wp.reshape(nb0, block, nb1, block).abs().amax(dim=(1, 3)).clamp(min=1e-8)
    s = (amax / 448.0).to(torch.bfloat16)
    se = s.float().repeat_interleave(block, 0)[:o].repeat_interleave(block, 1)[:, :i]
    q = (w.float() / se).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, s


def make_tiny_fp8_checkpoint(dst, small_files_dir, seed=0):
    """TEST ONLY. A randomly initialised Qwen3_5ForConditionalGeneration with tiny sizes (4 text layers: 3 linear-
    attention + 1 full-attention, 1 vision block) but the real vocabulary and tokenizer, saved like the Hub repo:
    FP8 + weight_scale_inv for the same kinds of layers, bf16 for the rest, an unused 'mtp.*' tensor, several shards
    (one scale deliberately stored in a different shard than its weight), index, config with quantization_config,
    tokenizer files and crc32.txt. small_files_dir: folder with the real config.json and tokenizer files."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    dst = pathlib.Path(dst); dst.mkdir(parents=True, exist_ok=True)
    real = json.load(open(pathlib.Path(small_files_dir) / "config.json"))
    c = json.loads(json.dumps(real)); tc, vc = c["text_config"], c["vision_config"]
    tc.update(hidden_size=64, intermediate_size=160, num_hidden_layers=4, layer_types=tc["layer_types"][:4],
              num_attention_heads=2, num_key_value_heads=1, head_dim=64, linear_num_key_heads=2, linear_num_value_heads=4,
              linear_key_head_dim=16, linear_value_head_dim=16, mtp_num_hidden_layers=0)
    tc["rope_parameters"] = dict(tc["rope_parameters"], mrope_section=[3, 3, 2])     # rotary dim 64 x 0.25 = 16 -> 8 pairs
    vc.update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64)
    qc = c.pop("quantization_config")
    tmp_cfg = dst / "_cfg"; tmp_cfg.mkdir(exist_ok=True); json.dump(c, open(tmp_cfg / "config.json", "w"))
    cfg = AutoConfig.from_pretrained(str(tmp_cfg)); shutil.rmtree(tmp_cfg)
    torch.manual_seed(seed)
    m = Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16)
    sd = {k: v.detach().clone().contiguous() for k, v in m.state_dict().items()}
    sd["mtp.fc.weight"] = (torch.randn(64, 128) * 0.02).to(torch.bfloat16)        # mimics the unused MTP head
    shards = {}
    for k, v in sd.items():
        fn = "outside.safetensors" if not k.startswith(("model.language_model.layers.", "mtp.")) else \
             "mtp.safetensors" if k.startswith("mtp.") else f"layers-{k.split('.')[3]}.safetensors"
        if _HUB_FP8_MODULES.match(k) or k == "mtp.fc.weight":
            q, s = quantize_fp8_block(v)
            shards.setdefault(fn, {})[k] = q
            sfn = "outside.safetensors" if k == "model.language_model.layers.0.mlp.down_proj.weight" else fn   # cross-shard scale
            shards.setdefault(sfn, {})[k + "_scale_inv"] = s
        else:
            shards.setdefault(fn, {})[k] = v
    wm = {}
    for fn, t in shards.items():
        save_file(t, str(dst / fn), metadata={"format": "pt"})
        wm.update({k: fn for k in t})
    json.dump({"metadata": {"total_size": 0}, "weight_map": dict(sorted(wm.items()))}, open(dst / "model.safetensors.index.json", "w"), indent=2)
    c["quantization_config"] = qc; c["text_config"]["dtype"] = "bfloat16"
    json.dump(c, open(dst / "config.json", "w"), indent=2)
    for p in pathlib.Path(small_files_dir).iterdir():
        if p.is_file() and p.name not in ("config.json", "model.safetensors.index.json", "crc32.txt") and not p.name.endswith(".safetensors"):
            shutil.copy2(p, dst / p.name)
    with open(dst / "crc32.txt", "w") as f:
        for p in sorted(dst.iterdir()):
            if p.is_file() and p.name != "crc32.txt":
                f.write(f"{file_crc32(p)}  {p.name}\n")
    return dict(files=len(shards), fp8=sum(1 for k in wm if k.endswith("_scale_inv")), tensors=len(wm))


def make_tiny_adapters(bf16_dir, adapter_root, archive_dir, runs, score_df, max_len, seed=0):
    """TEST ONLY. For each run name: wrap the tiny bf16 base with the LoRA recipe, give the LoRA B matrices and the head
    random non-zero values (a different seed per run), save the adapter with PEFT (same layout as the archived ones)
    and score score_df with the verbatim scoring loop of a40/train_qwen.py: these scores play the 'archived' role."""
    import torch
    from peft import LoraConfig, get_peft_model
    adapter_root, archive_dir = pathlib.Path(adapter_root), pathlib.Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    texts = score_df.text.tolist()
    for j, run in enumerate(runs):
        tok, model, _ = load_base_seqcls(bf16_dir)
        torch.manual_seed(seed + 100 * j)
        pm = get_peft_model(model, LoraConfig(task_type=RECIPE["task_type"], r=RECIPE["r"], lora_alpha=RECIPE["lora_alpha"],
                                              lora_dropout=RECIPE["lora_dropout"], target_modules=RECIPE["target_modules"]))
        with torch.no_grad():
            for n, p in pm.named_parameters():
                if "lora_B" in n or "modules_to_save" in n:
                    p.normal_(0, 0.3 if "lora_B" in n else 2.0)
        pm.save_pretrained(str(adapter_root / run))
        pm.eval(); scores = []
        with torch.no_grad():                                     # verbatim scoring loop of a40/train_qwen.py
            for i in range(0, len(texts), 16):
                enc = tok(texts[i:i + 16], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
                scores += torch.softmax(pm(**enc).logits.float(), -1)[:, 1].cpu().tolist()
        out = score_df[["exam", "corpus", "essay_id"]].copy(); out["score"] = scores
        out["model"] = "TINY TEST MODEL"; out["run"] = run
        out.to_parquet(archive_dir / f"{run}.parquet", index=False)
    return {r: adapter_root / r for r in runs}
