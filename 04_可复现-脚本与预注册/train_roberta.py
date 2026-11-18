# int4096 unified_v4 — RoBERTa-base full fine-tune. Usage: train_roberta.py <run> [--dry]
import sys, time, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, MODELS, score_frame, train_set, save_preds
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding)

run = sys.argv[1]; dry = "--dry" in sys.argv
seed = int(run.split("_")[0].replace("seed", ""))
out_dir = ROOT / ("preds_dry" if dry else ("followup/preds_gpu" if run.endswith(("_C", "_D")) else "preds")) / "roberta"
if (out_dir / f"{run}.parquet").exists():
    print("SKIP exists", run); sys.exit(0)
torch.manual_seed(seed)
import os
if os.environ.get("ROB_MEMFRAC") and torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(float(os.environ["ROB_MEMFRAC"]))
tok = AutoTokenizer.from_pretrained(MODELS / "roberta-base")
X, y = train_set(run)
if dry:
    X, y = X[:8] + X[-8:], y[:8] + y[-8:]

class DS(torch.utils.data.Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        e = tok(self.X[i], truncation=True, max_length=512)
        e["labels"] = self.y[i]
        return e

model = AutoModelForSequenceClassification.from_pretrained(MODELS / "roberta-base", num_labels=2)
cuda = torch.cuda.is_available()
args = TrainingArguments(output_dir=str(ROOT / "ckpt/roberta" / run), per_device_train_batch_size=16,
                         num_train_epochs=2, learning_rate=2e-5, seed=seed, bf16=cuda, save_strategy="no",
                         report_to="none", logging_steps=50, dataloader_num_workers=2,
                         max_steps=2 if dry else -1)
t0 = time.time()
Trainer(model=model, args=args, train_dataset=DS(X, y), data_collator=DataCollatorWithPadding(tok)).train()
train_s = time.time() - t0
if not dry:
    model.save_pretrained(str(ROOT / "adapters/roberta" / run)); tok.save_pretrained(str(ROOT / "adapters/roberta" / run))

sf = score_frame()
if dry:
    sf = sf.groupby("exam").head(3)
model.eval(); dev = next(model.parameters()).device
scores = []
with torch.no_grad():
    texts = sf.text.tolist()
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], truncation=True, max_length=512, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            scores += torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().tolist()
out = sf[["exam", "corpus", "essay_id"]].copy(); out["score"] = scores
save_preds(out, "RoBERTa", run, out_dir)
print(f"DONE roberta {run} n_train={len(X)} train_s={train_s:.0f} total_s={time.time()-t0:.0f}", flush=True)
