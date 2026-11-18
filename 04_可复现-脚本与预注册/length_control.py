# int4096 unified_v4 - length-controlled bias check.
# Question: is the TOEFL-vs-Hewlett bias driven by essay length rather than language background?
# Two analyses, both computed for every detector run already scored:
#   (1) matched analysis: every TOEFL essay is paired with Hewlett essays whose length is within +/-30%
#       (word count). AUC computed on matched pairs only (weighted by number of partners).
#   (2) band analysis: restrict to the overlapping length band and recompute AUC.
# Reference (no control): plain AUC on all pairs.
import json, sys, pathlib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = pathlib.Path("/data/int4096_unified_v4")
exams = pd.read_parquet(ROOT / "splits/exams.parquet")
wc = lambda s: len(str(s).split())
exams["w"] = exams.text.map(wc)
rng = np.random.default_rng(0); B = 1000

def pairs_auc(t, h, weights=None, boot=True):
    t = np.asarray(t, float); h = np.asarray(h, float)
    auc = roc_auc_score(np.r_[np.ones(len(t)), np.zeros(len(h))], np.r_[t, h])
    if not boot: return auc, np.nan, np.nan
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(t), len(t)); j = rng.integers(0, len(h), len(h))
        try: bs.append(roc_auc_score(np.r_[np.ones(len(i)), np.zeros(len(j))], np.r_[t[i], h[j]]))
        except ValueError: pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return auc, lo, hi

rows = []
for f in sorted((ROOT / "preds").rglob("*.parquet")):
    p = pd.read_parquet(f)
    model, run = p.model.iloc[0], p.run.iloc[0]
    d = p.merge(exams[["essay_id", "w"]], on="essay_id")
    t = d[d.corpus == "TOEFL_real_91"]; h = d[d.corpus == "HewlettStudentEssay_real_88"]
    if len(t) < 20 or len(h) < 20: continue
    plain = pairs_auc(t.score, h.score, boot=False)[0]
    # (1) length-matched: for each TOEFL essay take Hewlett essays within +/-30% length; pool matched pairs
    tt, hh = [], []
    for _, r in t.iterrows():
        lo, hi = 0.7 * r.w, 1.3 * r.w
        m = h[(h.w >= lo) & (h.w <= hi)]
        if len(m):
            tt += [r.score] * len(m); hh += list(m.score)
    matched = pairs_auc(tt, hh)[0] if tt and len(set(np.sign(np.array(tt) - np.median(tt)))) > 1 else np.nan
    matched_ci = pairs_auc(tt, hh)[1:] if tt else (np.nan, np.nan)
    # (2) common length band
    lo = max(t.w.min(), h.w.min()); hi = min(t.w.max(), h.w.max())
    tb = t[(t.w >= lo) & (t.w <= hi)]; hb = h[(h.w >= lo) & (h.w <= hi)]
    band = pairs_auc(tb.score, hb.score, boot=False)[0] if len(tb) >= 10 and len(hb) >= 10 else np.nan
    rows.append(dict(model=model, run=run, n_pairs_matched=len(tt),
                     bias_all_pairs=plain, bias_length_matched=matched,
                     matched_lo=matched_ci[0], matched_hi=matched_ci[1],
                     band_lo=int(lo), band_hi=int(hi), n_t_band=len(tb), n_h_band=len(hb), bias_band=band,
                     toefl_words_median=float(t.w.median()), hewlett_words_median=float(h.w.median())))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "metrics/length_control.csv", index=False)
sel = out[(out.run.str.contains("n4000")) | (out.model.isin(["Binoculars", "FastDetectGPT"]))]
print(sel.groupby("model").agg(n=("bias_all_pairs", "size"),
                               bias_all=("bias_all_pairs", "mean"),
                               bias_matched=("bias_length_matched", "mean"),
                               bias_band=("bias_band", "mean")).round(3).to_string())
print("\ntypical lengths: TOEFL", int(sel.toefl_words_median.iloc[0]), "words | Hewlett", int(sel.hewlett_words_median.iloc[0]))
print("common band:", int(sel.band_lo.iloc[0]), "-", int(sel.band_hi.iloc[0]), "words; matched pairs per run:", int(sel.n_pairs_matched.iloc[0]))
print("LENGTH_CONTROL_DONE")
