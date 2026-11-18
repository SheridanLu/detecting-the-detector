# int4096 follow-up metrics for classical ML arms A, B (existing) and C, D (new). Pre-registered endpoints P1-P3.
import json, pathlib, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
MODEL = sys.argv[1]; SEEDS = list(range(int(sys.argv[2])))   # e.g. roberta 5 | qwen 2
ROOT = pathlib.Path("/data/int4096_unified_v4"); F = ROOT / "followup"
train_ids = set(json.load(open(F / "train_exam_ids.json")))
AI_VAL = {"daigt_ai", "hcvar_ai"}
OOD = ["CollegeEssay_real_70", "CS224N_real_145", "LOCNESS_native"]
rng = np.random.default_rng(0); B = 1000

def load(arm, s):
    f = (ROOT / f"preds/{MODEL}" / f"seed{s}_{arm}_n4000.parquet") if arm in "AB" else (F / f"preds_gpu/{MODEL}" / f"seed{s}_{arm}.parquet")
    p = pd.read_parquet(f); return p[~p.essay_id.isin(train_ids)]

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])

rows, flags = [], {}
for arm in "ABCD":
    for s in SEEDS:
        p = load(arm, s); by = {c: g.set_index("essay_id").score for c, g in p.groupby("corpus")}
        v = p[(p.exam == "VAL")]; thr = float(np.quantile(v[v.corpus.isin(AI_VAL)].score, 0.10))
        r = dict(arm=arm, seed=s, thr=thr)
        for c in ["TOEFL_real_91", "HewlettStudentEssay_real_88", *OOD, "FCE", "WI_A", "WI_B", "WI_C", "PERSUADE_ELL", "PERSUADE_nonELL", "ELLIPSE_test"]:
            r[f"flag90_{c}"] = float((by[c] >= thr).mean()); r[f"flag05_{c}"] = float((by[c] >= 0.5).mean())
        ood_rates = [r[f"flag90_{c}"] for c in OOD]
        r["P1_toefl_flag90"] = r["flag90_TOEFL_real_91"]
        r["P2_excess90"] = r["flag90_TOEFL_real_91"] - float(np.mean(ood_rates))
        r["native_ood_mean90"] = float(np.mean(ood_rates))
        r["bias_auc_vs_hewlett"] = auc(by["TOEFL_real_91"], by["HewlettStudentEssay_real_88"])
        r["bias_auc_vs_native_ood"] = auc(by["TOEFL_real_91"], np.concatenate([by[c].to_numpy() for c in OOD]))
        r["P3_arguGPT_vs_toefl"] = auc(by["ArguGPT_TOEFL_AI"], by["TOEFL_real_91"])
        r["P3_ai352_vs_hewlett"] = auc(np.concatenate([by[c].to_numpy() for c in ["CS224N_gpt3_145", "CS224N_gpt3PromptEng_145", "CollegeEssay_gpt3_31", "CollegeEssay_gpt3PromptEng_31"]]), by["HewlettStudentEssay_real_88"])
        rows.append(r); flags[(arm, s)] = (by["TOEFL_real_91"] >= thr).astype(int)
M = pd.DataFrame(rows); M.to_csv(F / f"followup_runs_{MODEL}.csv", index=False)
pd.set_option("display.width", 250)
cols = ["P1_toefl_flag90", "P2_excess90", "native_ood_mean90", "bias_auc_vs_hewlett", "bias_auc_vs_native_ood", "P3_arguGPT_vs_toefl", "P3_ai352_vs_hewlett"]
agg = M.groupby("arm")[cols].agg(["mean", "std"]); agg.columns = [f"{a}|{b}" for a, b in agg.columns]
print("=== PRIMARY (mean|std over 5 seeds; flags at each run's own 90%-TPR threshold) ==="); print(agg.round(3).T.to_string())
print("\n=== flag rate by corpus (90%-TPR threshold, mean over seeds) ===")
fc = [c for c in M.columns if c.startswith("flag90_")]
print(M.groupby("arm")[fc].mean().round(2).rename(columns=lambda c: c[7:]).T.to_string())
print("\n=== flag rate by corpus at threshold 0.5 (use this when the 90%-TPR threshold is degenerate, e.g. Qwen) ===")
f5 = [c for c in M.columns if c.startswith("flag05_")]
print(M.groupby("arm")[f5].mean().round(2).rename(columns=lambda c: c[7:]).T.to_string())
print("\n=== paired on the same 91 TOEFL essays, per seed (diff = X - Y flag rate; exact McNemar p) ===")
out = []
for x, y in (("C", "A"), ("C", "D"), ("B", "A")):
    for s in SEEDS:
        fx, fy = flags[(x, s)], flags[(y, s)].loc[flags[(x, s)].index]
        n_xy = int(((fx == 1) & (fy == 0)).sum()); n_yx = int(((fx == 0) & (fy == 1)).sum())
        p = binomtest(n_xy, n_xy + n_yx, 0.5).pvalue if n_xy + n_yx else 1.0
        d = [fx.to_numpy()[i].mean() - fy.to_numpy()[i].mean() for i in (rng.integers(0, len(fx), len(fx)) for _ in range(B))]
        out.append(dict(pair=f"{x}-{y}", seed=s, rate_x=fx.mean(), rate_y=fy.mean(), diff=fx.mean() - fy.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=p))
O = pd.DataFrame(out); O.to_csv(F / f"followup_paired_{MODEL}.csv", index=False); print(O.round(3).to_string(index=False))
print("FOLLOWUP_METRICS_DONE")
