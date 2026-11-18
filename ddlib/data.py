"""Loading the data in data/. Every loader returns pandas DataFrames; nothing here trains a model.

Naming used throughout:
  arm A  human side = 4,000 native PERSUADE essays
  arm B  human side = 2,000 native + 1,000 ELLIPSE + 1,000 PERSUADE-ELL (school-age non-native)
  arm C  human side = 2,000 native + 2,000 exam-style non-native (half of FCE and of W&I A/B/C)
  arm D  human side = 2,000 native + 2,000 other native essays (HC-Var human 'writing'), the control arm
  All four arms use the same 4,000 AI essays (2,000 daigt + 2,000 HC-Var ChatGPT) for a given seed.
Runs are named like the original scripts: seed{s}_{A|B}_n{1000|2000|4000} and seed{s}_{C|D} (C/D always have 4,000
human + 4,000 AI essays).
"""
import json
import pathlib

import pandas as pd

from .paths import data_dir
from .textclean import clean

MODEL_DIRS = {"LR": "lr", "RoBERTa": "roberta", "Qwen27B": "qwen"}
SEEDS = {"LR": [0, 1, 2, 3, 4], "RoBERTa": [0, 1, 2, 3, 4], "Qwen27B": [0, 1]}
ARMS = ["A", "B", "C", "D"]


def _d(root=None):
    return data_dir(root)


def load_exams(root=None, cleaned=False):
    ex = pd.read_parquet(_d(root) / "splits" / "exams.parquet")
    if cleaned:
        ex = ex.copy(); ex["text"] = ex.text.map(clean)
    return ex


def load_val(root=None, cleaned=False):
    v = pd.read_parquet(_d(root) / "splits" / "val.parquet")
    if cleaned:
        v = v.copy(); v["text"] = v.text.map(clean)
    return v


def load_pool(root=None, cleaned=False):
    p = pd.read_parquet(_d(root) / "splits" / "pool.parquet")
    if cleaned:
        p = p.copy(); p["text"] = p.text.map(clean)
    return p


def load_manifests(root=None):
    return json.load(open(_d(root) / "splits" / "manifests.json"))


def load_split_summary(root=None):
    return json.load(open(_d(root) / "splits" / "split_summary.json"))


def train_exam_ids(root=None):
    """FCE / W&I essays that were moved into arm C's training data; they are removed from every evaluation."""
    return set(json.load(open(_d(root) / "followup" / "train_exam_ids.json")))


def load_trainset(run, root=None, pool_cleaned=None, manifests=None):
    """Training texts and labels of one run, exactly as a40/common.py::train_set() builds them.
    Arms A/B: manifest ids looked up in the pool, then clean(). Arms C/D: the saved training set (already cleaned).
    Returns (texts: list[str], labels: list[int])."""
    if run.endswith(("_C", "_D")):
        t = pd.read_parquet(_d(root) / "followup" / "trainsets" / f"{run}.parquet")
        return t.text.tolist(), t.label.astype(int).tolist()
    if pool_cleaned is None:
        pool_cleaned = load_pool(root, cleaned=True)
    pc = pool_cleaned.set_index("id") if "id" in pool_cleaned.columns else pool_cleaned
    m = (manifests or load_manifests(root))[run]
    ids = m["human"] + m["ai"]
    return pc.loc[ids, "text"].tolist(), [0] * len(m["human"]) + [1] * len(m["ai"])


def score_frame(root=None, exams_cleaned=None, val_cleaned=None):
    """Every text a detector is asked to score: all exam rows plus the shared validation set (exam='VAL').
    Same row order as the original scripts, so saved predictions line up row by row."""
    ex = exams_cleaned if exams_cleaned is not None else load_exams(root, cleaned=True)
    v = val_cleaned if val_cleaned is not None else load_val(root, cleaned=True)
    return pd.concat([ex[["exam", "corpus", "essay_id", "text"]],
                      v.assign(exam="VAL", corpus=v.source, essay_id=v.id)[["exam", "corpus", "essay_id", "text"]]],
                     ignore_index=True)


def run_name(arm, seed, n=4000):
    return f"seed{seed}_{arm}_n{n}" if arm in ("A", "B") else f"seed{seed}_{arm}"


def load_preds(model, arm, seed, n=4000, root=None):
    """Saved scores of one trained run. model in {'LR','RoBERTa','Qwen27B'}. Columns: exam, corpus, essay_id, score, model, run."""
    return pd.read_parquet(_d(root) / "preds" / MODEL_DIRS[model] / f"{run_name(arm, seed, n)}.parquet")


def load_opensource(name, root=None):
    """name in {'Binoculars', 'FastDetectGPT'}. 'score' is already mapped to 0-1 with the detector's official threshold at 0.5."""
    f = {"Binoculars": "binoculars/binoculars.parquet", "FastDetectGPT": "fastdetect/fastdetect.parquet"}[name]
    return pd.read_parquet(_d(root) / "preds" / f)


def load_commercial(root=None):
    """Long table of the 7 commercial detectors' scores on Liang et al.'s essays (exam E1)."""
    return pd.read_parquet(_d(root) / "commercial" / "liang2023_detector_scores.parquet")


def load_result(name, root=None):
    """Archived result tables from the original A40 run (data/results/<name>.csv)."""
    return pd.read_csv(_d(root) / "results" / f"{name}.csv")


def available_runs(model, root=None):
    """List of (arm, seed, n) for which saved predictions exist."""
    out = []
    for f in sorted((_d(root) / "preds" / MODEL_DIRS[model]).glob("seed*.parquet")):
        parts = f.stem.split("_")
        seed = int(parts[0][4:]); arm = parts[1]; n = int(parts[2][1:]) if len(parts) > 2 else 4000
        out.append((arm, seed, n))
    return out
