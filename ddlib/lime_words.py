"""Helpers for notebook 06 (LIME word explanations). EXPLORATORY side analysis, not part of the presentation slides.

What is here
  make_detector / train_arm   the TF-IDF + logistic-regression detector, same settings as notebook 02 and a40/train_lr.py
  get_essays                  the two essays that are explained (cleaned with ddlib.clean, college essay cut to 98 words)
  explain                     one LIME run (official `lime` package, LimeTextExplainer) for the probability of "AI"
  top_overlap                 how many of the 10 words two explanations share
  draw_word_figure            the word-shaded figure (red = pushes toward AI, blue-grey = toward human)

LIME is imported only inside explain(), so `import ddlib` never needs the package.
LIME: Ribeiro, Singh, Guestrin (2016), "Why Should I Trust You?": Explaining the Predictions of Any Classifier, KDD.
Package: https://github.com/marcotcr/lime (version 0.2.0.1 used here).
"""
import re

import numpy as np

from . import data as D
from .textclean import clean

# the two essays: key -> (essay_id in data/splits/exams.parquet, card title, number of words kept or None = all)
ESSAYS = {
    "toefl84": ("TOEFL_real_91#84", "TOEFL essay, non-native writer", None),
    "college23_excerpt": ("CollegeEssay_real_70#23", "College application essay, native writer (first 98 words)", 98),
}
CLASS_NAMES = ["human", "AI"]


def make_detector(seed):
    """Same pipeline and settings as notebook 02 (copied from a40/train_lr.py): unigrams + bigrams, 20,000 features,
    min_df=2, LogisticRegression(max_iter=2000, random_state=seed), everything else default."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
                     ("lr", LogisticRegression(max_iter=2000, random_state=seed))])


def train_arm(arm, seed=0, pool_cleaned=None, manifests=None):
    """Train the detector of one arm (A: run seed{s}_A_n4000 from the pool; C: saved follow-up set seed{s}_C) exactly as
    notebook 02 does. Returns (fitted pipeline, number of training essays)."""
    run = D.run_name(arm, seed, 4000)
    X, y = D.load_trainset(run, pool_cleaned=pool_cleaned, manifests=manifests)
    return make_detector(seed).fit(X, y), len(X)


def get_essays(exams=None):
    """dict key -> dict(text=cleaned text used for LIME, raw_words=word count of the cleaned essay before any cut, title).
    The college essay is cut to its first 98 whitespace-separated words *after* cleaning."""
    exams = exams if exams is not None else D.load_exams()
    by_id = dict(zip(exams.essay_id, exams.text))
    out = {}
    for key, (eid, title, nmax) in ESSAYS.items():
        words = clean(by_id[eid]).split()
        kept = words if nmax is None else words[:nmax]
        out[key] = dict(essay_id=eid, title=title, text=" ".join(kept), n_words=len(kept), n_words_full=len(words))
    return out


def explain(pipe, text, random_state=42, num_samples=5000, num_features=10):
    """One LIME explanation of P(AI) for one text. A fresh LimeTextExplainer is created for every call, so the result
    depends only on (model, text, random_state, num_samples). Returns dict(p_ai, weights, r2, local_pred, intercept):
    weights = list of (word, weight) for the num_features strongest words, sorted by |weight|; r2 = explanation.score
    (how well LIME's small word model fits the detector around this essay, 1 = perfectly)."""
    from lime.lime_text import LimeTextExplainer
    expl = LimeTextExplainer(class_names=CLASS_NAMES, random_state=random_state)
    e = expl.explain_instance(text, pipe.predict_proba, num_features=num_features, num_samples=num_samples, labels=(1,))
    return dict(p_ai=float(pipe.predict_proba([text])[0, 1]),
                weights=[(str(w), float(v)) for w, v in e.as_list(label=1)],
                r2=float(e.score), local_pred=float(e.local_pred[0]), intercept=float(e.intercept[1]))


def top_overlap(w1, w2):
    """Number of words that appear in both weight lists (each a list of (word, weight))."""
    return len({w for w, _ in w1} & {w for w, _ in w2})


# ---------------------------------------------------------------------------------------------------------------
# Figure: adapted from the slide script make_fig24_lime.py (same colours and layout); takes data instead of a CSV.
# ---------------------------------------------------------------------------------------------------------------
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_AI, C_HUMAN = "#e34948", "#4f7a9e"          # red = pushes toward AI; blue-grey = pushes toward human
FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]   # first one that is installed is used (DejaVu Sans on Colab)


def draw_word_figure(cards, arm, title="Which words push our detector toward AI?", footer=None, card_h=350):
    """cards: list of dict(title, color, text, n_words, p_a, p_c, weights) where weights = {word: LIME weight} for the arm
    named in `arm` ("A" or "C"). Every word of the text that is one of the strongest words is shaded (red = pushes toward
    AI, blue-grey = toward human, stronger weight = darker) and followed by its weight in probability points.
    LIME gives one weight per word type, so every occurrence of a word is shaded alike. Returns the matplotlib Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import to_rgb
    arm_txt = {"A": "detector A (native essays only)",
               "C": "detector C (native + exam-style non-native)"}[arm]
    with plt.rc_context({"axes.unicode_minus": False}):
        fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1920); ax.set_ylim(1080, 0); ax.axis("off")
        rend = fig.canvas.get_renderer()

        def width(s, fs, bold=False):
            t = ax.text(0, 0, s, family=FONT, fontsize=fs, fontweight="bold" if bold else "normal")
            w = t.get_window_extent(rend).width; t.remove(); return w

        wmax = max(abs(v) for c in cards for v in c["weights"].values())
        ax.text(48, 30, title, family=FONT, fontsize=30, fontweight="bold", color=INK, va="top", ha="left")
        ax.text(48, 92, f"Each word is shaded by its LIME weight for P(AI) under {arm_txt}. The 10 strongest words per essay are shaded.",
                family=FONT, fontsize=15, color=INK2, va="top", ha="left")
        ky = 148
        ax.text(48, ky, "pushes toward human", family=FONT, fontsize=20, color=INK2, va="center", ha="left")
        kx = 48 + width("pushes toward human", 20) + 16; kw = 440
        for i in range(kw):
            s = (i / (kw - 1)) * 2 - 1
            ax.add_patch(Rectangle((kx + i, ky - 13), 1.5, 26, facecolor=(*to_rgb(C_AI if s > 0 else C_HUMAN), 0.70 * abs(s)), edgecolor="none"))
        ax.text(kx + kw + 16, ky, "pushes toward AI", family=FONT, fontsize=20, color=INK2, va="center", ha="left")
        ax.text(1872, ky, "0.5 = flagged as AI", family=FONT, fontsize=20, color=INK2, va="center", ha="right")

        def card(top, c, fs_try=(19.5, 19, 18.5, 18, 17.5, 17, 16)):
            wt = c["weights"]
            ax.text(48, top, f"{c['title']}  ·  {c['n_words']} words", family=FONT, fontsize=20, fontweight="bold", color=c["color"], va="center", ha="left")
            xr = 1872
            ax.text(xr, top, f"arm C  {c['p_c']:.2f}", family=FONT, fontsize=22, fontweight="bold", color=INK, va="center", ha="right")
            xr -= width(f"arm C  {c['p_c']:.2f}", 22, True) + 34
            ax.text(xr, top, f"arm A  {c['p_a']:.2f}", family=FONT, fontsize=22, fontweight="bold", color=INK, va="center", ha="right")
            xr -= width(f"arm A  {c['p_a']:.2f}", 22, True) + 34
            ax.text(xr, top, "P(AI):", family=FONT, fontsize=20, color=INK2, va="center", ha="right")
            y0 = top + 26; W = 1824; pad = 20
            # chunks = whitespace-delimited runs (never broken across lines); each is split into words and punctuation pieces
            chunks = [re.findall(r"\w+|[^\w\s]+", w) for w in c["text"].split()]
            found = {w: 0 for w in wt}
            for fs in fs_try:
                space = fs * 120 / 72 * 0.32; lh = fs * 120 / 72 * 1.7
                lay, x, line = [], 0.0, 0
                for ch in chunks:
                    ws = [width(t, fs) + (width(f" {wt[t]:+.2f}", fs * 0.62) + 9 if (t in wt) else 0) for t in ch]
                    need = sum(ws)
                    if x + need > W - 2 * pad and x > 0: x = 0.0; line += 1
                    lay.append((x, line, ch, ws)); x += need + space
                if (line + 1) * lh + 2 * pad - 8 <= card_h: break
            ax.add_patch(Rectangle((48, y0), W, card_h, facecolor="white", edgecolor=GRID, linewidth=1.5))
            for x, line, ch, ws in lay:
                xx = x
                for t, w0 in zip(ch, ws):
                    y = y0 + pad - 4 + line * lh
                    if t in wt:
                        v = wt[t]; found[t] += 1; col = C_AI if v > 0 else C_HUMAN; tw = width(t, fs)
                        ax.add_patch(Rectangle((48 + pad + xx - 2, y), tw + 4, lh * 0.86, facecolor=(*to_rgb(col), 0.18 + 0.62 * abs(v) / wmax), edgecolor="none", zorder=1))
                        ax.text(48 + pad + xx, y + lh * 0.45, t, family=FONT, fontsize=fs, color=INK, va="center", ha="left", zorder=3)
                        ax.text(48 + pad + xx + tw, y + lh * 0.45, f" {v:+.2f}".replace("-", "−"), family=FONT, fontsize=fs * 0.62, color=col, va="center", ha="left", zorder=3, fontweight="bold")
                    else:
                        ax.text(48 + pad + xx, y + lh * 0.45, t, family=FONT, fontsize=fs, color=INK, va="center", ha="left", zorder=3)
                    xx += w0
            miss = [w for w, n in found.items() if n == 0]
            assert not miss, f"LIME words not found in the displayed text: {miss}"
            return y0 + card_h

        top = 214
        for c in cards:
            top = card(top, c) + 34
        fig.text(0.025, 0.043, "LIME (Ribeiro et al., 2016), logistic regression, 1 seed, one essay each; illustrates, does not prove. "
                 "EXPLORATORY side analysis, not in the slides.", family=FONT, fontsize=14, color=INK2, va="bottom", ha="left")
        fig.text(0.025, 0.014, footer or "Shading = LIME weight in probability points (number after each word); detector seed 0; flag = P(AI) of 0.5 or more; "
                 "arm C = trained on native + exam-style non-native essays.", family=FONT, fontsize=11, color=MUTED, va="bottom", ha="left")
    return fig
