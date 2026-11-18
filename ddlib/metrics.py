"""Metric functions shared by all notebooks. Formulas follow the original scripts in a40/ (metrics.py, prf1.py,
followup_metrics.py, followup_metrics_gpu.py); the source of each function is named in its docstring.

Words used:
  score            the detector's probability that an essay is AI-written (0-1)
  flagged          score >= threshold. Threshold 0.5 unless stated. "tpr90" = the threshold at which the detector
                   catches 90% of the AI essays in the shared validation set (each run gets its own).
  false-flag rate  share of HUMAN essays that are flagged as AI (lower is better)
  bias AUC         probability that a random non-native essay gets a higher (more AI-like) score than a random native
                   essay. 0.5 = no bias, 1.0 = every non-native essay looks more AI-like than every native essay.
  detection AUC    probability that a random AI essay scores higher than a random human essay (1.0 = perfect).
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

# ---- corpus groups (names as in data/splits/exams.parquet) -----------------------------------------------------
AI_VAL = {"daigt_ai", "hcvar_ai"}                                  # AI part of the shared validation set
TOEFL = "TOEFL_real_91"                                            # 91 real TOEFL essays (non-native), Liang et al.
HEWLETT = "HewlettStudentEssay_real_88"                            # 88 US 8th-grade essays (native, like training data)
NATIVE_OOD = ["CollegeEssay_real_70", "CS224N_real_145", "LOCNESS_native"]   # native essays of unfamiliar kinds
AI352 = ["CS224N_gpt3_145", "CS224N_gpt3PromptEng_145", "CollegeEssay_gpt3_31", "CollegeEssay_gpt3PromptEng_31"]
E1_HUMAN = [TOEFL, HEWLETT, "CollegeEssay_real_70", "CS224N_real_145"]       # Liang test set, human side (394)
E1_AI = AI352                                                                # Liang test set, AI side (352)
COMMERCIAL = ["Crossplag", "GPTZero", "HFOpenAI", "OriginalityAI", "Quil", "Sapling", "ZeroGPT"]

# column name in data/results/model_comparison_all.csv -> corpus (flag rate at 0.5)
FP_COLS = {"FP_TOEFL": TOEFL, "FP_Hewlett": HEWLETT, "FP_CollegeEss": "CollegeEssay_real_70", "FP_CS224N": "CS224N_real_145",
           "FP_LOCNESS": "LOCNESS_native", "FP_FCE": "FCE", "FP_WI_A": "WI_A", "FP_WI_C": "WI_C",
           "FP_P_ELL": "PERSUADE_ELL", "FP_P_native": "PERSUADE_nonELL", "FP_ELLIPSE": "ELLIPSE_test"}
TPR_COLS = {"TPR_Liang352": AI352, "TPR_ArguGPT": ["ArguGPT_TOEFL_AI"], "TPR_Llama": ["daigt_llama_AI"], "TPR_ChatGPT": ["HCVar_ChatGPT_AI"]}


# ---- basic quantities ------------------------------------------------------------------------------------------
def auc(pos, neg):
    """AUC = P(score of a random `pos` item > score of a random `neg` item). NaNs are dropped. (a40/metrics.py)"""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def auc_ci(pos, neg, n_boot=1000, rng=None):
    """AUC with a 95% bootstrap interval (resample each side with replacement). (a40/metrics.py::auc_ci)"""
    rng = rng if rng is not None else np.random.default_rng(0)
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[~np.isnan(pos)]; neg = neg[~np.isnan(neg)]
    a = auc(pos, neg)
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a, lo, hi


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n. (a40/metrics.py)"""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def thr_tpr90(preds):
    """Threshold that flags 90% of the validation AI essays (10th percentile of their scores). (a40/metrics.py)"""
    v = preds[(preds.exam == "VAL") & (preds.corpus.isin(AI_VAL))].score.to_numpy(float)
    return float(np.quantile(v, 0.10))


def flag_rate(scores, thr=0.5):
    """Share of essays with score >= thr (NaN scores are ignored)."""
    s = np.asarray(scores, float); s = s[~np.isnan(s)]
    return float((s >= thr).mean()) if len(s) else np.nan


def by_corpus(preds):
    """dict corpus -> Series(score indexed by essay_id)."""
    return {c: g.set_index("essay_id").score for c, g in preds.groupby("corpus")}


def drop_train_exams(preds, train_ids):
    """Remove the FCE / W&I essays that arm C trained on (done for every arm, as in a40/followup_metrics.py)."""
    return preds[~preds.essay_id.isin(train_ids)]


# ---- one run -> one row ----------------------------------------------------------------------------------------
def run_row(preds, train_ids=None):
    """Headline numbers of one run (one model, one arm, one seed).
    Flag rates at 0.5 (`FP_*`, `TPR_*`) reproduce data/results/model_comparison_all.csv; `flag90_*` and the AUCs
    reproduce data/results/followup_runs*.csv (a40/followup_metrics.py)."""
    p = drop_train_exams(preds, train_ids) if train_ids else preds
    by = by_corpus(p)
    get = lambda cs: np.concatenate([by[c].to_numpy(float) for c in cs if c in by]) if any(c in by for c in cs) else np.array([])
    r = {}
    has_val = ((p.exam == "VAL") & p.corpus.isin(AI_VAL)).any()
    thr = thr_tpr90(p) if has_val else np.nan
    r["thr_tpr90"] = thr
    for col, c in FP_COLS.items():
        r[col] = flag_rate(by[c], 0.5) if c in by else np.nan
    for col, cs in TPR_COLS.items():
        r[col] = flag_rate(get(cs), 0.5)
    r["FP_nativeOOD_mean"] = float(np.mean([r["FP_CollegeEss"], r["FP_CS224N"], r["FP_LOCNESS"]]))
    r["flag90_TOEFL"] = flag_rate(by[TOEFL], thr) if has_val else np.nan
    r["flag90_Hewlett"] = flag_rate(by[HEWLETT], thr) if has_val else np.nan
    r["flag90_nativeOOD_mean"] = float(np.mean([flag_rate(by[c], thr) if c in by else np.nan for c in NATIVE_OOD])) if has_val else np.nan
    r["AUC_ArguGPTvsTOEFL"] = auc(get(["ArguGPT_TOEFL_AI"]), by[TOEFL])
    r["AUC_Liang352vsHewlett"] = auc(get(AI352), by[HEWLETT])
    r["bias_TOEFLvsHewlett"] = auc(by[TOEFL], by[HEWLETT])
    r["bias_TOEFLvsNativeOOD"] = auc(by[TOEFL], get(NATIVE_OOD))
    return r


# ---- precision / recall / F1 on the Liang test set (a40/prf1.py) ----------------------------------------------
def prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, (tp + tn) / (tp + fp + fn + tn)


def liang_prf(df):
    """df: columns corpus, pred (1 = flagged as AI), rows = Liang test-set essays. AI is the positive class.
    Human = TOEFL 91 + Hewlett 88 + CollegeEssay 70 + CS224N 145 (394); AI = 352 GPT-3 essays.
    f1_bal_<group>: F1 with 'balanced' precision TPR/(TPR+FPR_group), i.e. as if that human group and the AI essays
    were equally common, so the TOEFL and Hewlett groups can be compared. Verbatim logic of a40/prf1.py::metrics."""
    h = df[df.corpus.isin(E1_HUMAN)]; a = df[df.corpus.isin(E1_AI)]
    tp = int(a.pred.sum()); fn = len(a) - tp; fp = int(h.pred.sum()); tn = len(h) - fp
    p, r, f, acc = prf(tp, fp, fn, tn)
    out = dict(n_human=len(h), n_ai=len(a), accuracy=acc, precision=p, recall=r, f1=f, false_flag_rate=fp / max(1, len(h)))
    for g, key in ((TOEFL, "TOEFL"), (HEWLETT, "Hewlett")):
        hg = df[df.corpus == g]; ffr = hg.pred.mean() if len(hg) else np.nan
        pb = r / (r + ffr) if (r + ffr) > 0 else 0.0
        out[f"f1_bal_{key}"] = 2 * pb * r / (pb + r) if pb + r else 0.0; out[f"prec_bal_{key}"] = pb
    return out


def liang_prf_from_scores(preds, thr=0.5):
    p = preds[preds.corpus.isin(E1_HUMAN + E1_AI)].copy()
    p = p[~p.score.isna()]           # commercial detectors: essays without a score are skipped (as in a40/prf1.py)
    p["pred"] = (p.score >= thr).astype(int)
    return liang_prf(p)


# ---- paired comparison of two arms on the same essays ----------------------------------------------------------
def paired_flags(fx, fy, n_boot=1000, rng=None):
    """fx, fy: 0/1 Series indexed by essay_id (same essays). Returns rate difference x - y, 95% bootstrap CI over
    essays, and the exact McNemar p-value (binomial test on the discordant pairs). (a40/followup_metrics.py)"""
    rng = rng if rng is not None else np.random.default_rng(0)
    fy = fy.loc[fx.index]
    n_xy = int(((fx == 1) & (fy == 0)).sum()); n_yx = int(((fx == 0) & (fy == 1)).sum())
    p = binomtest(n_xy, n_xy + n_yx, 0.5).pvalue if n_xy + n_yx else 1.0
    x, y = fx.to_numpy(), fy.to_numpy()
    d = [x[i].mean() - y[i].mean() for i in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
    return dict(rate_x=x.mean(), rate_y=y.mean(), diff=x.mean() - y.mean(),
                lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), x_only=n_xy, y_only=n_yx, p=p)


# ---- comparison with archived numbers ---------------------------------------------------------------------------
def compare(label, got, expected, tol):
    """One pass/fail line. Returns dict for a summary table."""
    ok = (not (isinstance(got, float) and math.isnan(got))) and abs(got - expected) <= tol
    return dict(check=label, recomputed=got, archived=expected, tolerance=tol, result="PASS" if ok else "FAIL")


def compare_frames(new, old, cols, key, tol=1e-9):
    """Largest absolute difference per column between two tables joined on `key`."""
    m = new.merge(old, on=key, suffixes=("_new", "_old"))
    rows = []
    for c in cols:
        if f"{c}_new" in m and f"{c}_old" in m:
            d = (pd.to_numeric(m[f"{c}_new"], errors="coerce") - pd.to_numeric(m[f"{c}_old"], errors="coerce")).abs().max()
            rows.append(dict(column=c, rows_compared=len(m), max_abs_diff=d, result="PASS" if d <= tol else "FAIL"))
    return pd.DataFrame(rows)
