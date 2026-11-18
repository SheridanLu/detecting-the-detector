# int4096 unified_v4 — metrics for every detector run, same code for all models.
# Primary (threshold-free): bias AUC (non-native vs native; 0.5 = no bias) and detection AUC (human vs AI).
# Secondary: flag rates at 0.5 and at the threshold giving 90% TPR on the shared validation AI set.
import json, pathlib, math, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

ROOT = pathlib.Path("/data/int4096_unified_v4")
S = ROOT / "splits"
LIANG = ROOT / "data/liang_local/Data_and_Results"
OUTD = ROOT / "metrics"; OUTD.mkdir(exist_ok=True)
exams = pd.read_parquet(S / "exams.parquet")
rng = np.random.default_rng(0)
B = 1000

AI_VAL = {"daigt_ai", "hcvar_ai"}
AI352 = ["CS224N_gpt3_145", "CS224N_gpt3PromptEng_145", "CollegeEssay_gpt3_31", "CollegeEssay_gpt3PromptEng_31"]
BIAS_PAIRS = {
    "bias_TOEFL_vs_Hewlett": (["TOEFL_real_91"], ["HewlettStudentEssay_real_88"]),
    "bias_PERSUADE_ELL_vs_nonELL": (["PERSUADE_ELL"], ["PERSUADE_nonELL"]),
    "bias_WI_vs_LOCNESS": (["WI_A", "WI_B", "WI_C"], ["LOCNESS_native"]),
    "bias_ELLIPSE_vs_Hewlett": (["ELLIPSE_test"], ["HewlettStudentEssay_real_88"]),
    "bias_FCE_vs_Hewlett": (["FCE"], ["HewlettStudentEssay_real_88"]),
}
DET_PAIRS = {
    # (positive = AI, negative = human): AUC = P(AI essay scores higher than human essay)
    "det_AI352_vs_LiangHuman2": (AI352, ["TOEFL_real_91", "HewlettStudentEssay_real_88"]),
    "det_E6_vs_LiangHuman2": (["daigt_llama_AI", "HCVar_ChatGPT_AI", "GLM_modern_AI"], ["TOEFL_real_91", "HewlettStudentEssay_real_88"]),
    # same TOEFL prompts on both sides: removes topic differences from the detection check
    "det_ArguGPT_TOEFL_AI_vs_TOEFL_human": (["ArguGPT_TOEFL_AI"], ["TOEFL_real_91"]),
}
RATE_CORPORA = ["TOEFL_real_91", "HewlettStudentEssay_real_88", "CollegeEssay_real_70", "CS224N_real_145",
                "PERSUADE_ELL", "PERSUADE_nonELL", "WI_A", "WI_B", "WI_C", "LOCNESS_native", "FCE", "ELLIPSE_test",
                "TOEFL_gpt4polished_91", "HewlettStudentEssay_GPTsimplify_88", "daigt_llama_AI", "HCVar_ChatGPT_AI", "GLM_modern_AI", "ArguGPT_TOEFL_AI"]

def auc_ci(pos, neg, boot=True):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return (np.nan, np.nan, np.nan, len(pos), len(neg))
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; s = np.r_[pos, neg]
    a = roc_auc_score(y, s)
    if not boot:
        return (a, np.nan, np.nan, len(pos), len(neg))
    bs = []
    for _ in range(B):
        pp = rng.choice(pos, len(pos)); nn = rng.choice(neg, len(neg))
        bs.append(roc_auc_score(np.r_[np.ones(len(pp)), np.zeros(len(nn))], np.r_[pp, nn]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (a, lo, hi, len(pos), len(neg))

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def run_metrics(df, run_meta, val_ai_scores=None):
    """df: columns corpus, essay_id, score (exam rows). Returns dict row."""
    by = {c: g.score.to_numpy(float) for c, g in df.groupby("corpus")}
    get = lambda cs: np.concatenate([by[c] for c in cs if c in by]) if any(c in by for c in cs) else np.array([])
    row = dict(run_meta)
    for k, (p, n) in {**BIAS_PAIRS, **DET_PAIRS}.items():
        a, lo, hi, npos, nneg = auc_ci(get(p), get(n))
        row[k] = a; row[k + "_lo"] = lo; row[k + "_hi"] = hi; row[k + "_n"] = f"{npos}/{nneg}"
    thrs = {"t05": 0.5}
    if val_ai_scores is not None and len(val_ai_scores):
        thrs["tpr90"] = float(np.quantile(val_ai_scores, 0.10))
        row["thr_tpr90"] = thrs["tpr90"]
    for tn, t in thrs.items():
        for c in RATE_CORPORA:
            if c not in by: continue
            v = by[c][~np.isnan(by[c])]
            k = int((v >= t).sum()); n = len(v)
            row[f"rate_{tn}_{c}"] = k / n if n else np.nan
            lo, hi = wilson(k, n)
            row[f"rate_{tn}_{c}_lo"] = lo; row[f"rate_{tn}_{c}_hi"] = hi
    return row

rows = []
# ---- commercial baseline (E1 only, threshold 0.5) ----
DETS = ["Crossplag", "GPTZero", "HFOpenAI", "OriginalityAI", "Quil", "Sapling", "ZeroGPT"]
e1 = exams[exams.exam == "E1"]
for det in DETS:
    recs = []
    for c in e1.corpus.unique():
        sub = "Human_Data" if (LIANG / "Human_Data" / c).exists() else "GPT_Data"
        f = LIANG / sub / c / f"{det}.json"
        if not f.exists():
            continue
        sc = json.loads(f.read_text(encoding="utf-8"))
        for i, d in enumerate(sc):
            s = d.get("score")
            recs.append(dict(corpus=c, essay_id=f"{c}#{i}", score=np.nan if s is None else float(s)))
    df = pd.DataFrame(recs)
    rows.append(run_metrics(df, dict(model="commercial", detector=det, arm="-", n="-", seed="-", run=det)))
    print("commercial", det, flush=True)

# ---- our runs + open-source (any parquet under preds/) ----
for f in sorted((ROOT / "preds").rglob("*.parquet")):
    p = pd.read_parquet(f)
    model = p.model.iloc[0]; run = p.run.iloc[0]
    meta = dict(model=model, detector=model, run=run)
    parts = run.split("_")
    if run.startswith("seed"):
        meta.update(seed=parts[0].replace("seed", ""), arm=parts[1], n=parts[2].replace("n", ""))
    else:
        meta.update(seed="-", arm="-", n="-")
    val_ai = p[(p.exam == "VAL") & (p.corpus.isin(AI_VAL))].score.to_numpy(float)
    val_h = p[(p.exam == "VAL") & (~p.corpus.isin(AI_VAL))].score.to_numpy(float)
    r = run_metrics(p[p.exam != "VAL"], meta, val_ai)
    r["selfcheck_val_det_auc"] = auc_ci(val_ai, val_h, boot=False)[0]
    # orientation self-check: a trained detector must rank its own validation AI above validation human
    assert r["selfcheck_val_det_auc"] > 0.7, (run, r["selfcheck_val_det_auc"])
    rows.append(r)
    print(model, run, flush=True)

M = pd.DataFrame(rows)
M.to_csv(OUTD / "metrics_runs.csv", index=False)

# ---- paired arm A vs B on TOEFL (same seed, same n, same threshold rule) ----
pair_rows = []
for model in M.model.unique():
    if model in ("commercial",): continue
    _dir = {"LR": "lr", "RoBERTa": "roberta", "Qwen27B": "qwen"}.get(model, model.lower())
    for f in sorted((ROOT / "preds" / _dir).glob("seed*_A_*.parquet")) if (ROOT / "preds" / _dir).exists() else []:
        fb = pathlib.Path(str(f).replace("_A_", "_B_"))
        if not fb.exists(): continue
        a = pd.read_parquet(f); b = pd.read_parquet(fb)
        for tn in ["t05", "tpr90"]:
            def flags(d):
                t = 0.5 if tn == "t05" else float(np.quantile(d[(d.exam == "VAL") & (d.corpus.isin(AI_VAL))].score, 0.10))
                x = d[d.corpus == "TOEFL_real_91"].set_index("essay_id").score
                return (x >= t).astype(int)
            fa, fbb = flags(a), flags(b)
            fbb = fbb.loc[fa.index]
            n10 = int(((fa == 1) & (fbb == 0)).sum()); n01 = int(((fa == 0) & (fbb == 1)).sum())
            p_mc = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 > 0 else 1.0
            diffs = []
            idx = np.arange(len(fa)); va, vb = fa.to_numpy(), fbb.to_numpy()
            for _ in range(B):
                s = rng.choice(idx, len(idx)); diffs.append(vb[s].mean() - va[s].mean())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            pair_rows.append(dict(model=model, run=f.stem.replace("_A_", "_AB_"), threshold=tn,
                                  fpr_A=va.mean(), fpr_B=vb.mean(), diff_B_minus_A=vb.mean() - va.mean(),
                                  diff_lo=lo, diff_hi=hi, A_only=n10, B_only=n01, mcnemar_p=p_mc))
pd.DataFrame(pair_rows).to_csv(OUTD / "paired_toefl_A_vs_B.csv", index=False)

# ---- aggregate over seeds ----
key_cols = [c for c in M.columns if c.startswith(("bias_", "det_", "rate_", "thr_")) and not c.endswith(("_lo", "_hi", "_n"))]
ours = M[M.model != "commercial"]
if len(ours):
    agg = ours.groupby(["model", "arm", "n"])[key_cols].agg(["mean", "std"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(OUTD / "metrics_agg.csv", index=False)
print("METRICS_DONE runs=%d paired=%d" % (len(M), len(pair_rows)), flush=True)
