"""Plumbing for notebook 01 (live demo of the zero-shot detectors). The two detector FORMULAS are deliberately NOT here:
they are written out in the notebook so that readers see them. This module only loads models, tokenises essays, picks
the demo sample and draws the word-level highlight. `torch` and `transformers` are imported inside the functions, so
`import ddlib.zeroshot` works on a machine without them (the replay part of notebook 01 needs neither).

Original methods (implemented from the papers, not copied from the code):
  Binoculars      Hans et al. 2024, "Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text"
                  (ICML 2024), official code github.com/ahans30/Binoculars
  Fast-DetectGPT  Bao et al. 2024, "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via
                  Conditional Probability Curvature" (ICLR 2024), official code github.com/baoguangsheng/fast-detect-gpt
"""
import bisect
import html
import re

import numpy as np

MAX_TOKENS = 512            # same truncation as the archived falcon-7b run (a40/score_opensource.py)

# human / AI groups of the Liang et al. 2023 set that the live demo draws from (names as in exams.parquet)
G_TOEFL = "TOEFL_real_91"
G_NATIVE = "CollegeEssay_real_70"      # native adults' essays; Hewlett is NOT used (85 of 88 essays contain @CAPS)
G_AI = ["CS224N_gpt3_145", "CS224N_gpt3PromptEng_145", "CollegeEssay_gpt3_31", "CollegeEssay_gpt3PromptEng_31"]


# ---- hardware ----------------------------------------------------------------------------------------------------
def pick_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_memory_gb():
    """Total memory of GPU 0 in GiB, or None if there is no GPU."""
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3


# ---- models ------------------------------------------------------------------------------------------------------
def load_lm(name, device, dtype="float32"):
    """Tokenizer + causal language model in eval mode, on `device`, weights in `dtype` ('float32', 'float16', 'bfloat16')."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()          # progress bars become notebook widgets that do not render everywhere
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass
    print(f"loading {name} ({dtype}, {device}); the first time this downloads the weights, which can take a few minutes ...", flush=True)
    dt = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dt)            # newer transformers
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt)      # older transformers
    return tok, model.to(device=device, dtype=dt).eval()


def load_pair(observer, performer, device, dtype="float32"):
    """Two models that share one tokenizer (required by both detectors: they compare the two models' opinions about
    the SAME next word). Returns (tokenizer, observer_model, performer_model)."""
    tok_o, m_o = load_lm(observer, device, dtype)
    tok_p, m_p = load_lm(performer, device, dtype)
    if tok_o.get_vocab() != tok_p.get_vocab():
        raise ValueError(f"{observer} and {performer} do not share a tokenizer; the detectors need a shared vocabulary.")
    return tok_o, m_o, m_p


def encode(tok, text, device, max_tokens=MAX_TOKENS):
    """Token ids of an essay, shape [1, L], truncated to `max_tokens`. No special token is added, so (as in the
    official code) the first token is only context and is not scored."""
    enc = tok(text, truncation=True, max_length=max_tokens, return_tensors="pt", return_token_type_ids=False)
    return enc["input_ids"].to(device)


def next_word_logits(model, ids):
    """Logits for tokens 1..L-1 given their left context: shape [L-1, vocab], float32. Raises if not finite."""
    import torch
    with torch.no_grad():
        lg = model(input_ids=ids).logits[0, :-1].float()
    if not torch.isfinite(lg).all():
        raise RuntimeError("non-finite logits (float16 overflow?). Set LIVE_DTYPE = 'float32' in the CONFIG cell.")
    return lg


def token_surprisal(tok, model, text, device, max_tokens=MAX_TOKENS):
    """Per-token surprisal (-log probability, nats) of an essay under one model, with a start token in front so that the
    first word also has a context. Returns (offsets [(start, end)...], nll array). Needs a fast tokenizer."""
    import torch
    enc = tok(text, truncation=True, max_length=max_tokens, return_offsets_mapping=True, return_token_type_ids=False)
    ids = torch.tensor([enc["input_ids"]], device=device)
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    inp = torch.cat([torch.tensor([[start]], device=device), ids], dim=1)
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=inp).logits[0, :-1].float(), dim=-1)
    nll = -lp.gather(1, ids[0].unsqueeze(1)).squeeze(1)
    return [tuple(o) for o in enc["offset_mapping"]], nll.cpu().numpy()


# ---- demo sample -------------------------------------------------------------------------------------------------
def pick_sample(exams, n_toefl, n_native, n_ai, seed=0):
    """Small, reproducible sample of the Liang et al. 2023 essays: human TOEFL (non-native), human native college essays,
    GPT-3 essays. `exams`: data/splits/exams.parquet. Returns a DataFrame with columns essay_id, corpus, group, text."""
    import pandas as pd
    e1 = exams[exams.exam == "E1"]
    parts = []
    for corp, n, grp in ((G_TOEFL, n_toefl, "TOEFL (non-native, human)"), (G_NATIVE, n_native, "college (native, human)")):
        g = e1[e1.corpus == corp]
        parts.append(g.sample(min(n, len(g)), random_state=seed).assign(group=grp))
    g = e1[e1.corpus.isin(G_AI)]
    parts.append(g.sample(min(n_ai, len(g)), random_state=seed).assign(group="GPT-3 (AI)"))
    out = pd.concat(parts, ignore_index=True)
    return out[["essay_id", "corpus", "group", "text"]]


# ---- word-level view ---------------------------------------------------------------------------------------------
def word_surprisal(text, offsets, nll):
    """Group token surprisals into whitespace-separated words. Returns a list of (word, bits) where bits = total surprisal
    of the word's tokens in bits (log2). A word that the model could guess easily has few bits."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    starts = [s for s, _ in spans]
    bits = np.zeros(len(spans))
    for (a, b), v in zip(offsets, nll):
        while a < b and text[a].isspace():       # some tokenizers include the leading space in the token
            a += 1
        i = max(0, bisect.bisect_right(starts, a) - 1)
        bits[i] += v / np.log(2)
    return [(text[s:e], float(x)) for (s, e), x in zip(spans, bits)]


# five steps of one hue: pale = easy to guess, dark = hard to guess
HIGHLIGHT_STEPS = [(1.0, "#eaf2fc", "<1 bit: very easy to guess"), (3.0, "#b9d4f3", "1-3 bits"),
                   (6.0, "#6fa3e3", "3-6 bits"), (10.0, "#2a6ac0", "6-10 bits"), (1e9, "#143f80", ">10 bits: very surprising")]


def _step(bits):
    for hi, col, _ in HIGHLIGHT_STEPS:
        if bits < hi:
            return col
    return HIGHLIGHT_STEPS[-1][1]


def highlight_html(words):
    """HTML string: every word on a background whose darkness grows with its surprisal, plus a legend."""
    spans = []
    for w, b in words:
        col = _step(b)
        fg = "#ffffff" if col in ("#2a6ac0", "#143f80") else "#0b0b0b"
        spans.append(f'<span title="{b:.1f} bits" style="background:{col};color:{fg};padding:1px 3px;margin:1px;'
                     f'border-radius:3px;display:inline-block">{html.escape(w)}</span>')
    legend = " ".join(f'<span style="background:{c};color:{"#fff" if c in ("#2a6ac0", "#143f80") else "#0b0b0b"};padding:1px 6px;'
                      f'border-radius:3px;margin-right:4px;font-size:90%">{html.escape(t)}</span>' for _, c, t in HIGHLIGHT_STEPS)
    return (f'<div style="line-height:2.0;font-family:Georgia,serif;font-size:105%;max-width:900px">{" ".join(spans)}</div>'
            f'<div style="margin-top:8px;font-family:sans-serif">{legend}</div>')
