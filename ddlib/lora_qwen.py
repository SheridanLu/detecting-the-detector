"""Helpers for notebook 04 (Qwen + LoRA). Three jobs:

1. Hugging Face access: find a token without ever printing it (Colab secret HF_TOKEN, env var HF_TOKEN, or an
   existing `hf auth login` on this machine).
2. "Look inside" the archived LoRA adapters of the Qwen3.8-27B runs: read the safetensors header (a small JSON block
   at the start of the file) WITHOUT loading the 27B base model, which is not published.
3. A small stand-in LoRA run (default Qwen/Qwen3-0.6B) with the recipe of a40/train_qwen.py, so that the LoRA
   training code can actually be executed on a free Colab T4 or on a laptop CPU.

torch / transformers / peft are imported inside the functions that need them, so the replay part of the notebook
works without them.
"""
import contextlib
import json
import os
import pathlib
import re
import struct
import time

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


def train_and_score(X, y, score_df, cfg, out_dir, rid, log_every=None, verbose=True):
    """Train one LoRA stand-in run with the recipe of a40/train_qwen.py and score `score_df`.
    cfg: model, arm, seed, n_train, max_len, n_scored. Writes <rid>.parquet (one score per essay, same columns as
    data/preds/) and <rid>.json (settings and timings). Mid-run checkpoints go to <out_dir>/ckpt/<rid> so that a
    restarted notebook resumes instead of starting the run again."""
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
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=2, dtype=prec["load_dtype"])
    model.config.pad_token_id = tok.pad_token_id
    try:
        model.config.get_text_config().pad_token_id = tok.pad_token_id
    except Exception:
        pass
    model = get_peft_model(model, LoraConfig(task_type=RECIPE["task_type"], r=RECIPE["r"], lora_alpha=RECIPE["lora_alpha"],
                                             lora_dropout=RECIPE["lora_dropout"], target_modules=RECIPE["target_modules"]))
    trainable, total = model.get_nb_trainable_parameters()    # the numbers peft's print_trainable_parameters() shows
    model.enable_input_require_grads()
    if prec["device"] == "cuda":
        model.to("cuda")
    load_s = time.time() - t0
    L = cfg["max_len"]

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

    ckpt = out_dir / "ckpt" / rid
    save = cfg["n_train"] >= 1600                     # checkpoints only for runs of 100+ steps
    args = TrainingArguments(output_dir=str(ckpt), per_device_train_batch_size=RECIPE["per_device_train_batch_size"],
                             gradient_accumulation_steps=RECIPE["gradient_accumulation_steps"],
                             num_train_epochs=RECIPE["num_train_epochs"], learning_rate=RECIPE["learning_rate"],
                             seed=cfg["seed"], bf16=prec["bf16"], fp16=prec["fp16"], use_cpu=prec["device"] == "cpu",
                             logging_steps=1 if cfg["n_train"] < 1600 else 10,
                             save_strategy="steps" if save else "no", save_steps=100, save_total_limit=1,
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
    t_train = time.time()
    trainer.train(resume_from_checkpoint=str(resume[-1]) if resume else None)
    train_s = time.time() - t_train

    # score: sort by length so that each batch has similar lengths (faster), then restore the original order
    model.eval(); dev = model.get_input_embeddings().weight.device
    texts = score_df.text.tolist(); order = np.argsort([len(t) for t in texts]); scores = np.empty(len(texts))
    t_eval = time.time(); bs = RECIPE["eval_batch_size"]
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=L, padding=True, return_tensors="pt").to(dev)
            amp = torch.autocast(device_type="cuda", dtype=torch.float16) if prec["fp16"] else contextlib.nullcontext()
            with amp:
                logits = model(**enc).logits
            scores[idx] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    eval_s = time.time() - t_eval

    out = score_df[["exam", "corpus", "essay_id"]].copy(); out["score"] = scores
    out["model"] = f"STAND-IN {cfg['model']}"; out["run"] = rid
    out.to_parquet(out_dir / f"{rid}.parquet", index=False)
    import peft
    info = dict(config=cfg, trainable_params=int(trainable), total_params=int(total), device=prec["device"],
                gpu=prec["gpu"], precision=prec["label"], load_s=round(load_s, 1), train_s=round(train_s, 1),
                eval_s=round(eval_s, 1), optimizer_steps=int(trainer.state.global_step),
                loss_first=losses[0][1] if losses else None, loss_last=losses[-1][1] if losses else None,
                nan_scores=int(np.isnan(scores).sum()),
                versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__))
    json.dump(info, open(out_dir / f"{rid}.json", "w"), indent=1)
    del trainer, model
    if prec["device"] == "cuda":
        torch.cuda.empty_cache()
    import shutil
    shutil.rmtree(ckpt, ignore_errors=True)            # the run is finished; checkpoints are no longer needed
    return info
