# int4096 follow-up: arms C and D for classical ML (CPU).
import json, pathlib, random, sys, time, hashlib
import numpy as np, pandas as pd
sys.path.insert(0, "/data/int4096_unified_v4/scripts")
from common import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = pathlib.Path("/data/int4096_unified_v4"); S = ROOT / "splits"; D = ROOT / "data"
OUT = ROOT / "followup"; (OUT / "preds").mkdir(parents=True, exist_ok=True)
norm = lambda t: " ".join(str(t).split())
def keys(t): n = norm(t); return n, n[:200]

pool = pd.read_parquet(S / "pool.parquet").set_index("id"); pool["text"] = pool.text.map(clean)
exams = pd.read_parquet(S / "exams.parquet"); exams["text"] = exams.text.map(clean)
val = pd.read_parquet(S / "val.parquet"); val["text"] = val.text.map(clean)
man = json.load(open(S / "manifests.json"))

# --- fixed 50/50 split of FCE and each W&I level (seed 0): train half feeds arm C, other half stays an exam ---
rng = random.Random(0); train_ids = set()
for c in ["FCE", "WI_A", "WI_B", "WI_C"]:
    ids = sorted(exams[exams.corpus == c].essay_id); rng.shuffle(ids); train_ids |= set(ids[: len(ids) // 2])
json.dump(sorted(train_ids), open(OUT / "train_exam_ids.json", "w"))
ex_train = exams[exams.essay_id.isin(train_ids)].copy()
eval_frame = exams[~exams.essay_id.isin(train_ids)]

# --- HC-Var human `writing` essays (arm D control) ---
hv = pd.read_json(D / "hcvar/hc_var.jsonl", lines=True)
hv = hv[(hv.domain == "writing") & (hv.label == 0)].copy(); hv["text"] = hv.text.map(clean)

# --- overlap gate: nothing used for training may equal (full or first-200) any evaluation / validation text ---
ev_full, ev_200 = set(), set()
for t in list(eval_frame.text) + list(val.text):
    a, b = keys(t); ev_full.add(a); ev_200.add(b)
def drop_overlap(df):
    bad = df.text.map(lambda t: keys(t)[0] in ev_full or keys(t)[1] in ev_200)
    return df[~bad].drop_duplicates("text"), int(bad.sum())
ex_train, n_bad_c = drop_overlap(ex_train); hv, n_bad_d = drop_overlap(hv)
gate = {"exam_style_train_pool": int(len(ex_train)), "removed_overlap_C": n_bad_c,
        "hcvar_human_pool": int(len(hv)), "removed_overlap_D": n_bad_d}
for name, df in (("C", ex_train), ("D", hv)):
    ov = sum(keys(t)[0] in ev_full or keys(t)[1] in ev_200 for t in df.text); gate[f"final_overlap_{name}"] = ov
    assert ov == 0
assert len(ex_train) >= 2000 and len(hv) >= 2000, gate
json.dump(gate, open(OUT / "overlap_gate.json", "w"), indent=1); print("GATE", gate, flush=True)

score_df = pd.concat([exams[["exam", "corpus", "essay_id", "text"]],
                      val.assign(exam="VAL", corpus=val.source, essay_id=val.id)[["exam", "corpus", "essay_id", "text"]]], ignore_index=True)
for s in range(5):
    base = man[f"seed{s}_A_n4000"]; r = random.Random(7000 + s)
    nat = pool.loc[base["human"][:2000], "text"].tolist(); ai = pool.loc[base["ai"], "text"].tolist()
    arms = {"C": r.sample(list(ex_train.text), 2000), "D": r.sample(list(hv.text), 2000)}
    for arm, extra in arms.items():
        td = OUT / "trainsets"; td.mkdir(exist_ok=True)
        tf = td / f"seed{s}_{arm}.parquet"
        if not tf.exists():
            pd.DataFrame({"text": nat + extra + ai, "label": [0] * (len(nat) + len(extra)) + [1] * len(ai),
                          "part": ["native"] * len(nat) + [arm] * len(extra) + ["ai"] * len(ai)}).to_parquet(tf, index=False)
        f = OUT / "preds" / f"seed{s}_{arm}.parquet"
        if f.exists(): continue
        X = nat + extra + ai; y = [0] * (len(nat) + len(extra)) + [1] * len(ai)
        clf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
                        ("lr", LogisticRegression(max_iter=2000, random_state=s))]).fit(X, y)
        o = score_df[["exam", "corpus", "essay_id"]].copy(); o["score"] = clf.predict_proba(score_df.text.tolist())[:, 1]
        o["model"] = "LR"; o["run"] = f"seed{s}_{arm}"; o.to_parquet(f, index=False)
        print(f"seed{s} arm {arm} n_train={len(X)}", flush=True)
print("FOLLOWUP_TRAIN_DONE", flush=True)
