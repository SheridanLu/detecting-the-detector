# int4096 unified_v4 — classical ML detector (TF-IDF + LogisticRegression), CPU.
# Same featurizer/model as the course notebook (no tuning). Trains on every manifest, scores exams + val.
import json, pathlib, time, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import clean
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = pathlib.Path("/data/int4096_unified_v4")
S = ROOT / "splits"
OUT = ROOT / "preds/lr"
OUT.mkdir(parents=True, exist_ok=True)
pool = pd.read_parquet(S / "pool.parquet").set_index("id")
pool["text"] = pool.text.map(clean)
val = pd.read_parquet(S / "val.parquet")
exams = pd.read_parquet(S / "exams.parquet")
manifests = json.load(open(S / "manifests.json"))
score_df = pd.concat([
    exams[["exam", "corpus", "essay_id", "text"]],
    val.assign(exam="VAL", corpus=val.source, essay_id=val.id)[["exam", "corpus", "essay_id", "text"]],
], ignore_index=True)
score_df["text"] = score_df.text.map(clean)

for name, m in manifests.items():
    f = OUT / f"{name}.parquet"
    if f.exists():
        continue
    t0 = time.time()
    ids = m["human"] + m["ai"]
    X = pool.loc[ids, "text"].tolist()
    y = [0] * len(m["human"]) + [1] * len(m["ai"])
    seed = int(name.split("_")[0].replace("seed", ""))
    clf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
                    ("lr", LogisticRegression(max_iter=2000, random_state=seed))])
    clf.fit(X, y)
    out = score_df[["exam", "corpus", "essay_id"]].copy()
    out["score"] = clf.predict_proba(score_df.text.tolist())[:, 1]
    out["model"] = "LR"; out["run"] = name
    out.to_parquet(f, index=False)
    print(f"{name} n_train={len(ids)} secs={time.time()-t0:.0f}", flush=True)
print("LR_DONE", len(list(OUT.glob("*.parquet"))), flush=True)
