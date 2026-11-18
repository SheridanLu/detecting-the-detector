# make_fig24_lime.py -- INT4096 "Detecting the Detector": fig24, LIME word weights for our TF-IDF + logistic-regression detector.
# Usage:
#   python make_fig24_lime.py --export <lime_results.json> <tables/fig24_lime_words.csv>   # server LIME output -> table twin
#   python make_fig24_lime.py <tables/fig24_lime_words.csv> <fig24_lime_words.png>          # draw the 1920x1080 figure
# LIME itself (marcotcr/lime 0.2.0.1, LimeTextExplainer) ran on the server (/data/int4096_unified_v4/lime_work/lime_run.py);
# this script only draws what is saved in the CSV. Weights = arm A, class "AI", num_features=10, num_samples=5000, random_state=42.
import sys, re, json, textwrap
import pandas as pd

if len(sys.argv) == 4 and sys.argv[1] == "--export":
    res = json.load(open(sys.argv[2], encoding="utf-8"))["results"]
    rows = []
    for r in res:
        rows.append(dict(essay=r["essay"], row_type="essay", word="", lime_weight_armA="", p_ai_armA=r["p_ai_A"], p_ai_armC=r["p_ai_C"],
                         n_words=r["n_words"], lime_local_R2_armA=r["lime_R2"], text=r["text_clean"]))
        for w, v in r["weights_A"]:
            rows.append(dict(essay=r["essay"], row_type="word", word=str(w), lime_weight_armA=v, p_ai_armA="", p_ai_armC="", n_words="", lime_local_R2_armA="", text=""))
    pd.DataFrame(rows).to_csv(sys.argv[3], index=False); print("wrote", sys.argv[3]); sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb

CSV, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, MUTED, GRID, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#b4b2ab"
C_TOEFL, C_HUMAN = "#e34948", "#4f7a9e"          # red = pushes toward AI; blue-grey = pushes toward human
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "axes.unicode_minus": False})

df = pd.read_csv(CSV, keep_default_na=False)
ESS = {"toefl84": ("TOEFL essay, non-native writer", C_TOEFL), "college23_excerpt": ("College application essay, native writer (first 98 words)", INK2)}
info, wts = {}, {}
for k in ESS:
    e = df[(df.essay == k) & (df.row_type == "essay")].iloc[0]
    info[k] = dict(text=e.text, n=int(e.n_words), a=float(e.p_ai_armA), c=float(e.p_ai_armC), r2=float(e.lime_local_R2_armA))
    wts[k] = {r.word: float(r.lime_weight_armA) for r in df[(df.essay == k) & (df.row_type == "word")].itertuples()}
WMAX = max(abs(v) for d in wts.values() for v in d.values())

fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1920); ax.set_ylim(1080, 0); ax.axis("off")
rend = fig.canvas.get_renderer()
def width(s, fs, bold=False):
    t = ax.text(0, 0, s, fontsize=fs, fontweight="bold" if bold else "normal"); w = t.get_window_extent(rend).width; t.remove(); return w

ax.text(48, 30, "Which words push our detector toward AI?", fontsize=30, fontweight="bold", color=INK, va="top", ha="left")
ax.text(48, 92, "Each word is shaded by its LIME weight for P(AI) under detector A (trained on native essays only). Only the 10 strongest words per essay are shaded.",
        fontsize=16, color=INK2, va="top", ha="left")
# colour key
ky = 148
ax.text(48, ky, "pushes toward human", fontsize=20, color=INK2, va="center", ha="left")
kx = 48 + width("pushes toward human", 20) + 16; kw = 440
for i in range(kw):
    s = (i / (kw - 1)) * 2 - 1
    col = C_TOEFL if s > 0 else C_HUMAN
    ax.add_patch(Rectangle((kx + i, ky - 13), 1.5, 26, facecolor=(*to_rgb(col), 0.70 * abs(s)), edgecolor="none"))
ax.text(kx + kw + 16, ky, "pushes toward AI", fontsize=20, color=INK2, va="center", ha="left")
ax.text(1872, ky, "0.5 = flagged as AI", fontsize=20, color=INK2, va="center", ha="right")

def card(top, key, fs_try=(19.5, 19, 18.5, 18, 17.5, 17, 16)):
    title, col = ESS[key]; d = info[key]; wt = wts[key]; H_AVAIL = CARD_H
    ax.text(48, top, f"{title}  ·  {d['n']} words", fontsize=20, fontweight="bold", color=col, va="center", ha="left")
    xr = 1872
    ax.text(xr, top, f"arm C  {d['c']:.2f}", fontsize=22, fontweight="bold", color=INK, va="center", ha="right")
    xr -= width(f"arm C  {d['c']:.2f}", 22, True) + 34
    ax.text(xr, top, f"arm A  {d['a']:.2f}", fontsize=22, fontweight="bold", color=INK, va="center", ha="right")
    xr -= width(f"arm A  {d['a']:.2f}", 22, True) + 34
    ax.text(xr, top, "P(AI):", fontsize=20, color=INK2, va="center", ha="right")
    y0 = top + 26; W = 1824; pad = 20
    toks = re.findall(r"\w+|\W+", d["text"])
    # chunks = whitespace-delimited runs (never broken across lines)
    chunks, cur = [], []
    for t in toks:
        if t.isspace() or t.strip() == "":
            if cur: chunks.append(cur); cur = []
        else:
            parts = re.findall(r"\w+|[^\w\s]+", t); cur.extend(parts)
    if cur: chunks.append(cur)
    found = {w: 0 for w in wt}
    for fs in fs_try:
        space = fs * 120 / 72 * 0.32; lh = fs * 120 / 72 * 1.7
        lay, x, line = [], 0.0, 0
        for ch in chunks:
            ws = [width(t, fs) + (width(f" {wt[t]:+.2f}", fs * 0.62) + 9 if (t in wt) else 0) for t in ch]
            need = sum(ws)
            if x + need > W - 2 * pad and x > 0: x = 0.0; line += 1
            lay.append((x, line, ch, ws)); x += need + space
        if (line + 1) * lh + 2 * pad - 8 <= H_AVAIL: break
    ax.add_patch(Rectangle((48, y0), W, H_AVAIL, facecolor="white", edgecolor=GRID, linewidth=1.5))
    for x, line, ch, ws in lay:
        xx = x
        for t, w0 in zip(ch, ws):
            y = y0 + pad - 4 + line * lh
            if t in wt:
                v = wt[t]; found[t] += 1; c = C_TOEFL if v > 0 else C_HUMAN
                tw = width(t, fs)
                ax.add_patch(Rectangle((48 + pad + xx - 2, y), tw + 4, lh * 0.86, facecolor=(*to_rgb(c), 0.18 + 0.62 * abs(v) / WMAX), edgecolor="none", zorder=1))
                ax.text(48 + pad + xx, y + lh * 0.45, t, fontsize=fs, color=INK, va="center", ha="left", zorder=3)
                ax.text(48 + pad + xx + tw, y + lh * 0.45, f" {v:+.2f}".replace("-", "−"), fontsize=fs * 0.62, color=c, va="center", ha="left", zorder=3, fontweight="bold")
            else:
                ax.text(48 + pad + xx, y + lh * 0.45, t, fontsize=fs, color=INK, va="center", ha="left", zorder=3)
            xx += w0
    miss = [w for w, n in found.items() if n == 0]
    assert not miss, f"LIME words not found in displayed text: {miss}"
    return fs, y0 + H_AVAIL

CARD_H = 350
fs1, b1 = card(214, "toefl84")
fs2, b2 = card(b1 + 34, "college23_excerpt")
print("essay font sizes", fs1, fs2, "bottom of card 2 at px", b2)
fig.text(0.025, 0.043, "LIME (Ribeiro et al., 2016), logistic regression, 1 seed, one essay each; illustrates, does not prove", fontsize=14, color=INK2, va="bottom", ha="left")
fig.text(0.025, 0.014, "Shading = LIME weight in probability points (number after each word); detector seed 0, flag = P(AI) of 0.5 or more; arm C = trained on native + exam-style non-native essays. Data table: tables/fig24_lime_words.csv",
         fontsize=11, color=MUTED, va="bottom", ha="left")
fig.savefig(OUT, dpi=120, facecolor=SURF); print("saved", OUT)
