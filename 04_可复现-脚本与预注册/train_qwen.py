# int4096 unified_v4 — Qwen3.8-27B (bf16, dequantized from local FP8) + LoRA sequence classification.
# Usage: train_qwen.py <run> [--dry]   (--dry: CPU load + LoRA wrap + 2-text forward, no training)
# Same LoRA hyper-parameters as v1: r=16, alpha=32, dropout=0.05, all-linear, lr 1e-4, bs 8 x accum 2, 1 epoch.
import sys, time, json, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, QWEN, score_frame, train_set, save_preds
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model

run = sys.argv[1]; dry = "--dry" in sys.argv

if not dry:
    # Wait until every assigned physical GPU is (nearly) free: a manually started job (e.g. Binoculars) may still hold one.
    import os, subprocess
    _gpus = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != ""]
    _t0 = time.time()
    while _gpus:
        _used = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.split("\n")
        _busy = [int(l.split(",")[0]) for l in _used if l.strip() and int(l.split(",")[1]) > 2000 and int(l.split(",")[0]) in _gpus]
        if not _busy:
            break
        print(f"waiting for GPUs {_busy} to free up ({time.time()-_t0:.0f}s)", flush=True)
        if time.time() - _t0 > 3 * 3600:
            raise SystemExit("GPU never freed within 3h")
        time.sleep(30)
seed = int(run.split("_")[0].replace("seed", ""))
out_dir = ROOT / ("preds_dry" if dry else ("followup/preds_gpu" if run.endswith(("_C", "_D")) else "preds")) / "qwen"
if (out_dir / f"{run}.parquet").exists():
    print("SKIP exists", run); sys.exit(0)
torch.manual_seed(seed)
tok = AutoTokenizer.from_pretrained(QWEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
t0 = time.time()
if dry:
    model = AutoModelForSequenceClassification.from_pretrained(QWEN, num_labels=2, dtype=torch.bfloat16,
                                                               device_map="cpu", output_loading_info=False)
else:
    ngpu = torch.cuda.device_count()
    model = AutoModelForSequenceClassification.from_pretrained(
        QWEN, num_labels=2, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: "41GiB" for i in range(ngpu)})
print(f"loaded in {time.time()-t0:.0f}s class={type(model).__name__}", flush=True)
model.config.pad_token_id = tok.pad_token_id
try:
    model.config.get_text_config().pad_token_id = tok.pad_token_id
except Exception:
    pass
model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules="all-linear"))
model.print_trainable_parameters()

if dry:
    enc = tok(["This is a short test essay about school.", "Another short text."], padding=True, return_tensors="pt")
    with torch.no_grad():
        p = torch.softmax(model(**enc).logits.float(), -1)[:, 1]
    print("DRY forward ok, probs:", p.tolist(), flush=True)
    print("DRY_QWEN_OK", flush=True)
    sys.exit(0)

model.enable_input_require_grads()
X, y = train_set(run)

class DS(torch.utils.data.Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        e = dict(tok(self.X[i], truncation=True, max_length=512)); e["labels"] = self.y[i]; return e

def collate(fs):
    L = max(len(f["input_ids"]) for f in fs)
    pad = tok.pad_token_id
    return {"input_ids": torch.tensor([f["input_ids"] + [pad] * (L - len(f["input_ids"])) for f in fs]),
            "attention_mask": torch.tensor([[1] * len(f["input_ids"]) + [0] * (L - len(f["input_ids"])) for f in fs]),
            "labels": torch.tensor([f["labels"] for f in fs])}

eta_file = ROOT / "logs" / f"qwen_eta_{run}.json"
class ETA(TrainerCallback):
    def on_step_end(self, args, state, control, **kw):
        if state.global_step in (5, 20, 50) or state.global_step % 100 == 0:
            el = time.time() - t_train
            eta = el / state.global_step * state.max_steps
            json.dump({"step": state.global_step, "max_steps": state.max_steps, "elapsed_s": el,
                       "est_train_total_s": eta}, open(eta_file, "w"))
            print(f"ETA step {state.global_step}/{state.max_steps} elapsed {el:.0f}s est_total {eta:.0f}s", flush=True)

args = TrainingArguments(output_dir=str(ROOT / "ckpt/qwen" / run), per_device_train_batch_size=8,
                         gradient_accumulation_steps=2, num_train_epochs=1, learning_rate=1e-4, seed=seed,
                         bf16=True, logging_steps=10, save_strategy="steps", save_steps=100, save_total_limit=1,
                         report_to="none", gradient_checkpointing=True,
                         remove_unused_columns=False, label_names=["labels"],
                         gradient_checkpointing_kwargs={"use_reentrant": False})
trainer = Trainer(model=model, args=args, train_dataset=DS(X, y), data_collator=collate, callbacks=[ETA()])
t_train = time.time()
ck = sorted((ROOT / "ckpt/qwen" / run).glob("checkpoint-*"))
trainer.train(resume_from_checkpoint=str(ck[-1]) if ck else None)
train_s = time.time() - t_train
model.save_pretrained(str(ROOT / "adapters/qwen" / run)); tok.save_pretrained(str(ROOT / "adapters/qwen" / run))

sf = score_frame()
model.eval(); dev = model.get_input_embeddings().weight.device
texts = sf.text.tolist(); scores = []
t_eval = time.time()
with torch.no_grad():
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
        scores += torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().tolist()
        if i % 1600 == 0:
            print(f"eval {i}/{len(texts)} {time.time()-t_eval:.0f}s", flush=True)
out = sf[["exam", "corpus", "essay_id"]].copy(); out["score"] = scores
save_preds(out, "Qwen27B", run, out_dir)
print(f"DONE qwen {run} n_train={len(X)} train_s={train_s:.0f} eval_s={time.time()-t_eval:.0f}", flush=True)
