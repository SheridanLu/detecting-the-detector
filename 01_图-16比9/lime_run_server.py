# LIME word-level explanation of the unified_v4 LR detector (TF-IDF + LogisticRegression). CPU only.
# Retrains seed0 arm A (seed0_A_n4000) and seed0 arm C (followup seed0_C) exactly as train_lr.py / followup_ml.py,
# saves with joblib to lime_work/, verifies against archived preds, runs LimeTextExplainer on the two hook essays.
import sys, json, pathlib, re
sys.path.insert(0, "/data/int4096_unified_v4/lime_work/pylib")
sys.path.insert(0, "/data/int4096_unified_v4/scripts")
import numpy as np, pandas as pd, joblib
from common import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import lime, lime.lime_text as LT
from lime.lime_text import LimeTextExplainer

ROOT = pathlib.Path("/data/int4096_unified_v4"); W = ROOT / "lime_work"; S = ROOT / "splits"
SEED = 0
texts = json.load(open(W / "essays_raw.json"))
pool = pd.read_parquet(S / "pool.parquet").set_index("id"); pool["text"] = pool.text.map(clean)
man = json.load(open(S / "manifests.json"))
exams = pd.read_parquet(S / "exams.parquet"); val = pd.read_parquet(S / "val.parquet")
score_df = pd.concat([exams[["exam","corpus","essay_id","text"]],
    val.assign(exam="VAL", corpus=val.source, essay_id=val.id)[["exam","corpus","essay_id","text"]]], ignore_index=True)
score_df["text"] = score_df.text.map(clean)

def mk(): return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
                           ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])
models = {}
m = man["seed0_A_n4000"]; ids = m["human"] + m["ai"]
models["A"] = (pool.loc[ids, "text"].tolist(), [0]*len(m["human"]) + [1]*len(m["ai"]), ROOT / "preds/lr/seed0_A_n4000.parquet")
t = pd.read_parquet(ROOT / "followup/trainsets/seed0_C.parquet")
models["C"] = (t.text.tolist(), t.label.tolist(), ROOT / "followup/preds/seed0_C.parquet")

pipes = {}
for arm, (X, y, ref) in models.items():
    f = W / f"lr_seed0_arm{arm}.joblib"
    if f.exists(): pipes[arm] = joblib.load(f); print("loaded existing", f)
    else:
        clf = mk().fit(X, y); joblib.dump(clf, f); pipes[arm] = clf; print("trained+saved", f, "n_train", len(X), "n_ai", int(sum(y)))
    sc = pipes[arm].predict_proba(score_df.text.tolist())[:, 1]
    r = pd.read_parquet(ref); mm = score_df[["exam","corpus","essay_id"]].assign(new=sc).merge(r[["exam","corpus","essay_id","score"]], on=["exam","corpus","essay_id"])
    print(f"arm {arm}: reproduce check n={len(mm)} max|new-archived|={np.abs(mm.new-mm.score).max():.3e}")

res = []
expl = LimeTextExplainer(class_names=["human", "AI"], random_state=42)
for key, raw in texts.items():
    c = clean(raw)
    row = {"essay": key, "n_words": len(c.split())}
    for arm in ("A", "C"):
        row[f"p_ai_{arm}"] = float(pipes[arm].predict_proba([c])[0, 1])
    e = expl.explain_instance(c, pipes["A"].predict_proba, num_features=10, num_samples=5000, labels=(1,))
    row["lime_score_local_pred"] = float(e.local_pred[0]); row["lime_R2"] = float(e.score); row["intercept"] = float(e.intercept[1])
    row["weights_A"] = [[w, float(v)] for w, v in e.as_list(label=1)]
    # same explainer settings, arm C model, for the 'C' column (reported, not drawn unless wanted)
    e2 = LimeTextExplainer(class_names=["human","AI"], random_state=42).explain_instance(c, pipes["C"].predict_proba, num_features=10, num_samples=5000, labels=(1,))
    row["weights_C"] = [[w, float(v)] for w, v in e2.as_list(label=1)]
    row["lime_R2_C"] = float(e2.score)
    row["text_clean"] = c
    res.append(row)
    print(key, {k: v for k, v in row.items() if k not in ("text_clean",)})
json.dump({"lime_version": getattr(lime, "__version__", "0.2.0.1"), "random_state": 42, "num_samples": 5000, "num_features": 10, "seed": SEED, "results": res},
          open(W / "lime_results.json", "w"), indent=1, ensure_ascii=False)
print("DONE")
