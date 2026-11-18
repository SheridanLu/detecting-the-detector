# Accuracy / precision / recall / F1 on the Liang test set (E1), threshold 0.5, every detector the same essays.
# Test set: human = TOEFL 91 + Hewlett 88 + CollegeEssay 70 + CS224N 145 (394); AI = 352 (4 GPT corpora). Positive class = AI.
# Per-group F1 uses BALANCED precision: P_bal = TPR / (TPR + FPR_group), i.e. as if AI and that human group were equally common,
# so groups of different size (91 TOEFL vs 88 Hewlett vs 352 AI) can be compared fairly.
import json, glob, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path("/data/int4096_unified_v4"); LIANG = ROOT / "data/liang_local/Data_and_Results"
HUMAN = ["TOEFL_real_91", "HewlettStudentEssay_real_88", "CollegeEssay_real_70", "CS224N_real_145"]
AI = ["CS224N_gpt3_145", "CS224N_gpt3PromptEng_145", "CollegeEssay_gpt3_31", "CollegeEssay_gpt3PromptEng_31"]
E1 = set(HUMAN + AI)
DETS = ["Crossplag", "GPTZero", "HFOpenAI", "OriginalityAI", "Quil", "Sapling", "ZeroGPT"]

def prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, (tp + tn) / (tp + fp + fn + tn)

def metrics(df):
    """df: corpus, pred (1 = flagged AI). Returns dict."""
    h = df[df.corpus.isin(HUMAN)]; a = df[df.corpus.isin(AI)]
    tp = int(a.pred.sum()); fn = len(a) - tp; fp = int(h.pred.sum()); tn = len(h) - fp
    p, r, f, acc = prf(tp, fp, fn, tn)
    out = dict(n_human=len(h), n_ai=len(a), accuracy=acc, precision=p, recall=r, f1=f, false_flag_rate=fp / max(1, len(h)))
    for g, key in (("TOEFL_real_91", "TOEFL"), ("HewlettStudentEssay_real_88", "Hewlett")):
        hg = df[df.corpus == g]; ffr = hg.pred.mean() if len(hg) else np.nan
        pb = r / (r + ffr) if (r + ffr) > 0 else 0.0
        out[f"f1_bal_{key}"] = 2 * pb * r / (pb + r) if pb + r else 0.0; out[f"prec_bal_{key}"] = pb
    return out

def own_runs():
    rows = []
    for model, d in (("LR", "lr"), ("RoBERTa", "roberta"), ("Qwen27B", "qwen")):
        for arm in "ABCD":
            fs = sorted(glob.glob(f"{ROOT}/preds/{d}/seed*_{arm}_n4000.parquet")) if arm in "AB" else \
                 sorted(glob.glob(f"{ROOT}/followup/preds/seed*_{arm}.parquet" if model == "LR" else f"{ROOT}/followup/preds_gpu/{d}/seed*_{arm}.parquet"))
            for f in fs:
                p = pd.read_parquet(f); p = p[p.corpus.isin(E1)].copy(); p["pred"] = (p.score >= 0.5).astype(int)
                rows.append(dict(model=model, arm=arm, run=pathlib.Path(f).stem, **metrics(p)))
    return rows

def external_runs():
    rows = []
    for m, f in (("Binoculars", "preds/binoculars/binoculars.parquet"), ("FastDetectGPT", "preds/fastdetect/fastdetect.parquet")):
        p = pd.read_parquet(ROOT / f); p = p[p.corpus.isin(E1)].copy(); p["pred"] = (p.score >= 0.5).astype(int)
        rows.append(dict(model=m, arm="-", run=m, **metrics(p)))
    for det in DETS:
        recs = []
        for c in sorted(E1):
            sub = "Human_Data" if (LIANG / "Human_Data" / c).exists() else "GPT_Data"
            f = LIANG / sub / c / f"{det}.json"
            if not f.exists(): continue
            for i, d in enumerate(json.loads(f.read_text(encoding="utf-8"))):
                s = d.get("score")
                if s is not None: recs.append(dict(corpus=c, pred=int(float(s) >= 0.5)))
        rows.append(dict(model="commercial", arm="-", run=det, **metrics(pd.DataFrame(recs))))
    return rows

R = pd.DataFrame(own_runs() + external_runs())
R.to_csv(ROOT / "metrics/prf1_runs.csv", index=False)
num = ["accuracy", "precision", "recall", "f1", "false_flag_rate", "f1_bal_TOEFL", "f1_bal_Hewlett", "prec_bal_TOEFL", "prec_bal_Hewlett"]
own = R[R.model.isin(["LR", "RoBERTa", "Qwen27B"])]
agg = own.groupby(["model", "arm"])[num].agg(["mean", "std"]); agg.columns = [f"{a}__{b}" for a, b in agg.columns]
agg["n_seeds"] = own.groupby(["model", "arm"]).size(); agg = agg.reset_index()
ext = R[~R.model.isin(["LR", "RoBERTa", "Qwen27B"])].copy(); ext["n_seeds"] = 1
for c in num: ext[c + "__mean"] = ext[c]; ext[c + "__std"] = np.nan
agg = pd.concat([agg, ext[["model", "arm", "run"] + [c for c in agg.columns if c not in ("model", "arm")]]], ignore_index=True)
agg.to_csv(ROOT / "metrics/prf1_summary.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("n_human/n_ai per row (first rows):", R[["model", "arm", "n_human", "n_ai"]].drop_duplicates(["model", "n_human", "n_ai"]).to_string(index=False))
show = agg[["model", "arm", "run", "n_seeds"] + [c + "__mean" for c in ("accuracy", "precision", "recall", "f1", "f1_bal_TOEFL", "f1_bal_Hewlett")]].copy()
print(show.round(3).to_string(index=False))
