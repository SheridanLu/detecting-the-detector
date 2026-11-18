# int4096 unified_v4 — zero-shot open-source detectors, official code + official thresholds, scoring only.
# Usage: score_opensource.py binoculars|fastdetect [--dry]
# Stored 'score' is a monotone transform where higher = more AI and the OFFICIAL threshold maps to 0.5,
# so AUCs are unchanged and rate@0.5 equals the official decision. Raw scores are kept in 'raw'.
import sys, math, time, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import ROOT, MODELS, score_frame, save_preds

which = sys.argv[1]; dry = "--dry" in sys.argv
REPOS = ROOT / "data/repos"
out_dir = ROOT / ("preds_dry" if dry else "preds") / which
if (out_dir / f"{which}.parquet").exists():
    print("SKIP exists"); sys.exit(0)
sf = score_frame()
if dry:
    sf = sf.groupby("exam").head(2)
texts = sf.text.tolist()
dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time(); raw = []

if which == "binoculars":
    sys.path.insert(0, str(REPOS / "Binoculars"))
    from binoculars.detector import Binoculars, BINOCULARS_FPR_THRESHOLD
    det = Binoculars(observer_name_or_path=str(MODELS / "falcon-7b"),
                     performer_name_or_path=str(MODELS / "falcon-7b-instruct"),
                     use_bfloat16=True, max_token_observed=512, mode="low-fpr")
    THR = BINOCULARS_FPR_THRESHOLD   # official low-FPR threshold; Binoculars score < THR => AI
    bs = 2 if dry else 16
    for i in range(0, len(texts), bs):
        r = det.compute_score(texts[i:i + bs])
        raw += list(r) if isinstance(r, (list, tuple)) else [r]
        if i % 1600 == 0: print(f"{i}/{len(texts)} {time.time()-t0:.0f}s", flush=True)
    score = [1 / (1 + math.exp(-40 * (THR - b))) for b in raw]   # b < THR -> score > 0.5
    model_name = "Binoculars"
elif which == "fastdetect":
    sys.path.insert(0, str(REPOS / "fast-detect-gpt/scripts"))
    from fast_detect_gpt import get_sampling_discrepancy_analytic
    from local_infer import compute_prob_norm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    P = {'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039}   # official falcon-7b_falcon-7b-instruct params
    tok = AutoTokenizer.from_pretrained(MODELS / "falcon-7b-instruct")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dt = torch.bfloat16
    m_score = AutoModelForCausalLM.from_pretrained(MODELS / "falcon-7b-instruct", dtype=dt).to(dev).eval()
    m_ref = AutoModelForCausalLM.from_pretrained(MODELS / "falcon-7b", dtype=dt).to(dev).eval()
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok(t, truncation=True, max_length=512, return_tensors="pt", return_token_type_ids=False).to(dev)
            labels = enc.input_ids[:, 1:]
            ls = m_score(**enc).logits[:, :-1]; lr = m_ref(**enc).logits[:, :-1]
            crit = float(get_sampling_discrepancy_analytic(lr, ls, labels))
            raw.append(crit)
            if i % 1600 == 0: print(f"{i}/{len(texts)} {time.time()-t0:.0f}s", flush=True)
    score = [float(compute_prob_norm(c, P['mu0'], P['sigma0'], P['mu1'], P['sigma1'])) for c in raw]
    model_name = "FastDetectGPT"
else:
    raise SystemExit("unknown detector")

out = sf[["exam", "corpus", "essay_id"]].copy(); out["raw"] = raw; out["score"] = score
save_preds(out, model_name, which, out_dir)
print(f"DONE {which} n={len(texts)} secs={time.time()-t0:.0f}", flush=True)
