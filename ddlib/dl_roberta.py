"""RoBERTa-base fine-tuning for notebook 03 (portable version of a40/train_roberta.py).

The original script used the Hugging Face `Trainer`. This file writes the same training out as a short, plain
PyTorch loop, so that it needs only `torch` + `transformers` (no `accelerate`) and runs the same way on a Colab T4,
on a laptop CPU and on an Apple-silicon GPU. The settings are the original ones (a40/train_roberta.py) plus the
`Trainer` defaults that the original script did not change:

  model                 roberta-base, new 2-class head, every weight trained ("full fine-tune")
  learning rate         2e-5, linear decay to 0, no warm-up             (Trainer default schedule)
  batch size            16 essays per step                              (per_device_train_batch_size=16, 1 GPU)
  epochs                2                                               (num_train_epochs=2)
  max length            512 tokens; longer essays are cut               (truncation=True, max_length=512)
  optimiser             AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0   (Trainer default "adamw_torch")
  gradient clipping     1.0                                             (Trainer default max_grad_norm)
  padding               each batch padded to its longest essay          (DataCollatorWithPadding)
  scoring               probability of class 1 (AI) from a softmax over the 2 logits, max length 512

Differences that cannot be avoided, and why re-runs are not bit-identical to the archive:
  * precision: the A40 run used bf16 autocast. A T4 has no fast bf16, so on CUDA we use fp16 autocast with a gradient
    scaler (GPUs that support bf16, e.g. A100/L4, use bf16 like the original). CPU and Apple GPUs use fp32.
  * random numbers: the order of the training essays and the dropout masks come from different random-number streams
    than in `Trainer`, and GPU kernels differ between GPU models. The seeds (0-4) are the same, the draws are not.
  * scoring: essays are scored in length-sorted batches (faster); with attention masks this changes scores only at
    floating-point noise level.
"""
import json
import math
import os
import pathlib
import platform
import time

import numpy as np
import pandas as pd

MODEL_NAME = "FacebookAI/roberta-base"       # = "roberta-base" on huggingface.co
PAPER = dict(lr=2e-5, batch_size=16, epochs=2, max_len=512, weight_decay=0.0, warmup_steps=0, max_grad_norm=1.0,
             adam_betas=(0.9, 0.999), adam_eps=1e-8)


def device_and_precision(prefer="auto", force_device=None):
    """Pick where to compute and in which number format.
    CUDA: bf16 if the GPU supports it natively (compute capability >= 8, e.g. A100/L4), else fp16 (T4).
    Apple GPU (MPS): fp32. CPU: fp32. `prefer` may force 'fp16', 'bf16' or 'fp32' on CUDA.
    `force_device='cpu'` uses the CPU even if a GPU is present (used by the automated tests)."""
    import torch
    if force_device == "cpu":
        return "cpu", "fp32"
    if torch.cuda.is_available():
        if prefer in ("fp16", "bf16", "fp32"):
            return "cuda", prefer
        major, _ = torch.cuda.get_device_capability(0)
        return "cuda", ("bf16" if major >= 8 else "fp16")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", "fp32"
    return "cpu", "fp32"


def device_name(device):
    import torch
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple GPU (MPS), {platform.machine()}"
    return f"CPU ({platform.machine()}, {os.cpu_count()} threads)"


def load_tokenizer(name=MODEL_NAME):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def load_model(seed, device, name=MODEL_NAME):
    """Pretrained RoBERTa-base with a freshly initialised 2-class head. The head's random start depends on `seed`."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from transformers.utils import logging as hf_logging
    from .seeds import set_seed
    set_seed(seed)                                   # as in the original: torch.manual_seed(seed) before loading
    hf_logging.set_verbosity_error()                 # hide the expected "classifier weights newly initialised" note
    hf_logging.disable_progress_bar()
    m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
    return m.to(device)


def tokenize(tok, texts, max_len):
    """Token ids of every text, cut at max_len tokens (no padding yet)."""
    return tok(list(texts), truncation=True, max_length=max_len)["input_ids"]


def _pad(tok, ids_list, device):
    import torch
    L = max(len(x) for x in ids_list)
    pad = tok.pad_token_id
    ids = torch.full((len(ids_list), L), pad, dtype=torch.long)
    att = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long); att[i, :len(x)] = 1
    return ids.to(device), att.to(device)


def _autocast(device, precision):
    import torch
    if device == "cuda" and precision in ("fp16", "bf16"):
        return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)
    import contextlib
    return contextlib.nullcontext()


def _grad_scaler(enabled):
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):            # torch < 2.3
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train(model, tok, texts, labels, *, seed, device, precision, lr, batch_size, epochs, max_len,
          micro_batch=None, log_every=100, weight_decay=0.0, max_grad_norm=1.0, adam_betas=(0.9, 0.999),
          adam_eps=1e-8, log=print):
    """Fine-tune `model` in place. Returns dict(train_s, steps, final_loss).
    micro_batch < batch_size splits each step into several smaller pieces (gradient accumulation): same maths, less
    GPU memory. Used automatically if a T4 runs out of memory."""
    import torch
    micro_batch = micro_batch or batch_size
    assert batch_size % micro_batch == 0
    accum = batch_size // micro_batch
    ids = tokenize(tok, texts, max_len)
    y = torch.tensor(list(labels), dtype=torch.long)
    n = len(ids)
    steps_per_epoch = math.ceil(n / batch_size)
    total = steps_per_epoch * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, (total - s) / total))   # linear, no warm-up
    scaler = _grad_scaler(device == "cuda" and precision == "fp16")
    g = torch.Generator().manual_seed(seed)
    model.train()
    t0 = time.time(); step = 0; run_loss = []
    for ep in range(epochs):
        order = torch.randperm(n, generator=g).tolist()
        for b in range(0, n, batch_size):
            idx = order[b:b + batch_size]
            loss_sum = 0.0
            for m0 in range(0, len(idx), micro_batch):
                part = idx[m0:m0 + micro_batch]
                inp, att = _pad(tok, [ids[i] for i in part], device)
                with _autocast(device, precision):
                    out = model(input_ids=inp, attention_mask=att, labels=y[part].to(device))
                loss = out.loss.float() * (len(part) / len(idx))       # mean over the whole step
                scaler.scale(loss).backward()
                loss_sum += float(loss.detach())
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            step += 1; run_loss.append(loss_sum)
            if step % log_every == 0 or step == total:
                el = time.time() - t0
                log(f"    step {step:4d}/{total}  epoch {ep + 1}  loss (last {min(log_every, len(run_loss))} steps) "
                    f"{np.mean(run_loss[-log_every:]):.4f}  elapsed {el / 60:5.1f} min  "
                    f"left ~{el / step * (total - step) / 60:5.1f} min")
    return dict(train_s=time.time() - t0, steps=total, final_loss=float(np.mean(run_loss[-min(50, len(run_loss)):])))


def score(model, tok, ids, *, device, precision, batch_size=64):
    """Probability of 'AI' for every essay. `ids` = output of tokenize(). Essays are batched by length (faster)."""
    import torch
    model.eval()
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = np.empty(len(ids), dtype=np.float64)
    with torch.no_grad():
        for b in range(0, len(ids), batch_size):
            sel = order[b:b + batch_size]
            inp, att = _pad(tok, [ids[i] for i in sel], device)
            with _autocast(device, precision):
                logits = model(input_ids=inp, attention_mask=att).logits
            out[sel] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return out


def is_oom(err):
    return "out of memory" in str(err).lower()


def free_memory(device):
    import gc
    import torch
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def versions():
    import torch
    import transformers
    return dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__)


def save_run(run_dir, run, preds, meta):
    """Write predictions + a small JSON with settings and timings. Written to temporary names first and renamed at
    the end, so a run that is interrupted half-way never looks finished."""
    run_dir = pathlib.Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    tmp_p, tmp_m = run_dir / f".{run}.parquet.tmp", run_dir / f".{run}.json.tmp"
    preds.to_parquet(tmp_p, index=False)
    tmp_m.write_text(json.dumps(meta, indent=1, default=str))
    os.replace(tmp_p, run_dir / f"{run}.parquet"); os.replace(tmp_m, run_dir / f"{run}.json")


def load_finished(run_dir, run, settings):
    """Saved predictions of a finished run, or None if missing or made with different settings."""
    p, m = pathlib.Path(run_dir) / f"{run}.parquet", pathlib.Path(run_dir) / f"{run}.json"
    if not (p.exists() and m.exists()):
        return None, None
    meta = json.loads(m.read_text())
    if meta.get("settings") != json.loads(json.dumps(settings)):
        return None, None
    return pd.read_parquet(p), meta
