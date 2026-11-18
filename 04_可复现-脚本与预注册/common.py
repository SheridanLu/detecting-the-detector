# int4096 unified_v4 — shared helpers for GPU scripts.
import json, pathlib
import pandas as pd

import re as _re, unicodedata as _ud

# Formatting normalisation applied to EVERY text (train, validation, exams) so that detectors cannot use
# formatting artefacts (newlines, curly vs straight quotes, mojibake, homoglyphs) as a shortcut.
_CP1252 = {"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x96": "-", "\x97": "-", "\x85": "...", "\x82": ","}
_PUNCT = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "´": "'", "`": "'",
          "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", "―": "-", "−": "-"}
_HOMO = str.maketrans("аеорсхуіѕԁјԛԝАВЕКМНОРСТХ", "aeopcxyisdjqwABEKMHOPCTX")
_TL_LOWER = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "i",
             "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
             "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
             "і": "i", "ї": "i", "є": "e", "ѕ": "s", "ј": "j"}
_TRANSLIT = dict(_TL_LOWER)
_TRANSLIT.update({k.upper(): v.capitalize() for k, v in _TL_LOWER.items()})
def clean(t):
    t = str(t)
    t = t.replace("Â ", " ").replace("Â", "")
    for a, b in _CP1252.items(): t = t.replace(a, b)
    t = _ud.normalize("NFKC", t)
    for a, b in _PUNCT.items(): t = t.replace(a, b)
    t = t.translate(_HOMO)
    t = "".join(_TRANSLIT.get(ch, ch) for ch in t)
    t = _re.sub(r"[\x80-\x9f]", "", t)
    t = _re.sub("[\u0400-\u052f]", "", t)   # any Cyrillic left after transliteration (9 chars in total)
    return _re.sub(r"\s+", " ", t).strip()

ROOT = pathlib.Path("/data/int4096_unified_v4")
S = ROOT / "splits"
MODELS = ROOT / "models"
QWEN = ROOT / "qwen27b_bf16_from_fp8"


def score_frame():
    """All texts to be scored: every exam row plus the shared validation set (exam='VAL')."""
    exams = pd.read_parquet(S / "exams.parquet")
    val = pd.read_parquet(S / "val.parquet")
    extra = []
    glm = ROOT / "data/glm_modern_ai/glm_modern_ai.jsonl"
    if glm.exists():
        for i, line in enumerate(open(glm, encoding="utf-8")):
            j = json.loads(line)
            extra.append(dict(exam="E6", corpus="GLM_modern_AI", essay_id=f"GLM#{j.get('id', i)}", text=j["text"]))
    parts = [exams[["exam", "corpus", "essay_id", "text"]],
             val.assign(exam="VAL", corpus=val.source, essay_id=val.id)[["exam", "corpus", "essay_id", "text"]]]
    if extra:
        parts.append(pd.DataFrame(extra))
    out = pd.concat(parts, ignore_index=True)
    out["text"] = out.text.map(clean)
    return out


def train_set(run):
    if run.endswith(("_C", "_D")):
        t = pd.read_parquet(ROOT / "followup/trainsets" / f"{run}.parquet")
        return t.text.tolist(), t.label.tolist()
    pool = pd.read_parquet(S / "pool.parquet").set_index("id")
    m = json.load(open(S / "manifests.json"))[run]
    ids = m["human"] + m["ai"]
    return [clean(t) for t in pool.loc[ids, "text"].tolist()], [0] * len(m["human"]) + [1] * len(m["ai"])


def save_preds(df_scores, model, run, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_scores = df_scores.copy()
    df_scores["model"] = model
    df_scores["run"] = run
    df_scores.to_parquet(out_dir / f"{run}.parquet", index=False)
