# make_figures_plainwords.py -- simplified "plain words" redraws of fig19/21/23 (per-model results), fig04 (unfamiliar
# writing), fig05 (the fix) and fig15 (four mistakes) for the INT4096 talk. Fewer numbers on the chart, one callout each.
# It reuses the palette, data loading and helpers of make_figures.py (same folder as the figures), so every number still
# comes from data/model_comparison_all.csv and the existing tables.
#
# Usage (needs matplotlib + pandas):  python3 make_figures_plainwords.py <01_图-16比9 dir> <output dir>
# Writes <name>.png into <output dir> and the plotted numbers into <output dir>/tables/<name>.csv.
import sys, pathlib, textwrap

FIGDIR, OUTDIR = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.argv = [sys.argv[0], str(FIGDIR), str(FIGDIR)]       # make_figures reads argv[1:3] at import; LIANG dir is unused here
sys.path.insert(0, str(FIGDIR))
import make_figures as mf                                  # noqa: E402
import pandas as pd                                         # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle   # noqa: E402

mf.ROOT = OUTDIR; mf.TBL = OUTDIR / "tables"; mf.TBL.mkdir(parents=True, exist_ok=True)
# Helvetica Neue ships as a .ttc here and matplotlib only reads its regular face (no bold, no arrow glyph); Arial has both.
mf.plt.rcParams["font.sans-serif"] = ["Arial", "Arial Unicode MS", "Helvetica Neue", "DejaVu Sans"]
CM = mf.CMP
INK, INK2, MUTED, GRID, BASE, GRAY, SURF = mf.INK, mf.INK2, mf.MUTED, mf.GRID, mf.BASE, mf.GRAY, mf.SURF
C_OS, C_TOEFL = mf.C_OS, mf.C_TOEFL
LEG_FS, TICK_FS, VAL_FS, CALL_FS = 21, 21, 25, 26
NATIVE_NEW = "Native essays, new style"
ARM_LABELS = ["A\nnative only", "B\n+ school-age", "C\n+ exam-style", "D (control)\n+ other native"]


def _axes_clean(ax, ymax=112):
    ax.set_facecolor(SURF); ax.set_ylim(0, ymax)
    ax.set_yticks([0, 50, 100]); ax.set_yticklabels(["0%", "50%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(axis="y", colors=INK2, labelsize=TICK_FS - 1, length=0); ax.tick_params(axis="x", length=0, pad=8)


def _callout(ax, x0, y0, x1, y1, label, color=INK):
    """Curved arrow from (x0,y0) to (x1,y1) with a short bold label at its middle."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
                                 mutation_scale=26, color=color, linewidth=2.6, zorder=7))
    ax.text((x0 + x1) / 2, max(y0, y1) + 4, label, ha="center", va="bottom", fontsize=CALL_FS, fontweight="bold",
            color=color, path_effects=mf.HALO, zorder=8)


def results_plain(name, title, model, seeds_note):
    fig = mf.new_fig(title, "Share of real human essays the detector wrongly calls AI, by which essays it learned from")
    mf.swatch_legend(fig, [("Native essays, like training", GRAY, "bar"), (NATIVE_NEW, C_OS, "bar"),
                           ("TOEFL (non-native exam essays)", C_TOEFL, "bar")], y=0.835, fs=LEG_FS)
    ax = fig.add_axes([0.09, 0.42, 0.88, 0.36]); out = []
    bw = 0.25
    for i, a in enumerate("ABCD"):
        h, n, t = (CM.loc[(model, a), k] * 100 for k in ("FP_Hewlett", "FP_nativeOOD_mean", "FP_TOEFL"))
        for v, c, off in ((h, GRAY, -bw), (n, C_OS, 0), (t, C_TOEFL, bw)):
            ax.bar(i + off * 1.1, v, width=bw, color=c, zorder=3)
        out.append(dict(arm=a, native_like_training=h, native_new_style=n, toefl=t))
    # only the numbers the speaker says: TOEFL in A and C, and the new-style native bar in C
    tA, tC, nC = out[0]["toefl"], out[2]["toefl"], out[2]["native_new_style"]
    ax.text(0 + bw * 1.1, tA + 2, f"{tA:.0f}%", ha="center", va="bottom", fontsize=VAL_FS, fontweight="bold", color=INK, zorder=6)
    ax.text(2 + bw * 1.1, tC + 2, f"{tC:.0f}%", ha="center", va="bottom", fontsize=VAL_FS, fontweight="bold", color=INK, zorder=6)
    ax.text(2, nC + 2, f"{nC:.0f}%", ha="center", va="bottom", fontsize=VAL_FS - 3, color=INK2, zorder=6)
    top = max(tA, 60)
    ax.annotate(f"{tA:.0f}% → {tC:.0f}%", xy=(2 + bw * 1.1, tC + 15), xytext=(2 + bw * 1.1, max(nC + 30, tC + 48, 64)),
                fontsize=CALL_FS + 3, fontweight="bold", color=INK, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.6, mutation_scale=26))
    ax.set_xticks(range(4)); ax.set_xticklabels(ARM_LABELS, fontsize=TICK_FS - 1, color=INK)
    _axes_clean(ax, ymax=max(118, top + 26))
    mf.footer(fig, name, f"Threshold 0.5, {seeds_note}. '{NATIVE_NEW}' = mean of CollegeEssay, CS224N and LOCNESS. "
                         "Precision/recall/F1 are on the backup slides.")
    mf.save(fig, name, pd.DataFrame(out))


def fig04_plain():
    fig = mf.new_fig("Detectors also flag native writers when the style is new to them",
                     "Trained on US school essays only (arm A). Share of real human essays wrongly called AI")
    mf.swatch_legend(fig, [("Native essays, like training", GRAY, "bar"), (NATIVE_NEW, C_OS, "bar"),
                           ("TOEFL (non-native exam essays)", C_TOEFL, "bar")], y=0.835, fs=LEG_FS)
    ax = fig.add_axes([0.09, 0.36, 0.88, 0.34]); out = []
    bw = 0.24
    names = ["Logistic regression\n(5 seeds)", "RoBERTa\n(5 seeds)", "Qwen 27B + LoRA\n(2 seeds)"]
    for i, (m, _nm, _c) in enumerate(mf.MODELS):
        h, n, t = (CM.loc[(m, "A"), k] * 100 for k in ("FP_Hewlett", "FP_nativeOOD_mean", "FP_TOEFL"))
        ax.bar(i - bw * 1.1, h, width=bw, color=GRAY, zorder=3)
        ax.bar(i, n, width=bw, color=C_OS, zorder=3)
        ax.bar(i + bw * 1.1, t, width=bw, color=C_TOEFL, zorder=3)
        ax.text(i, n + 2, f"{n:.0f}%", ha="center", va="bottom", fontsize=VAL_FS, fontweight="bold", color=INK, zorder=6)
        ax.text(i + bw * 1.1, t + 2, f"{t:.0f}%", ha="center", va="bottom", fontsize=VAL_FS - 4, color=MUTED, zorder=6)
        out.append(dict(model=m, native_like_training=h, native_new_style=n, toefl=t))
    # callout on the purple bars
    ax.text(1.0, 133, "Native writers, flagged too", fontsize=CALL_FS + 2, fontweight="bold", color=C_OS, ha="center", va="bottom")
    for i, start in ((0, 0.55), (1, 1.0), (2, 1.45)):
        ax.annotate("", xy=(i, out[i]["native_new_style"] + 15), xytext=(start, 131),
                    arrowprops=dict(arrowstyle="-|>", color=C_OS, lw=2.4, mutation_scale=24))
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=TICK_FS, color=INK)
    _axes_clean(ax, ymax=150)
    mf.footer(fig, "fig04_shift", f"Threshold 0.5. '{NATIVE_NEW}' = mean of CollegeEssay, CS224N and LOCNESS. For Qwen (2 seeds) "
                                  "the average is below TOEFL, but CS224N alone is flagged more than TOEFL (61% vs 40%).")
    mf.save(fig, "fig04_shift", pd.DataFrame(out))


def fig05_plain():
    fig = mf.new_fig("In all three models, the right training essays cut TOEFL false flags",
                     "Share of real TOEFL essays wrongly called AI, by which essays the detector learned from")
    mf.swatch_legend(fig, [("Logistic regression", mf.C_LR, "line"), ("RoBERTa", mf.C_RB, "line"),
                           ("Qwen 27B + LoRA (2 seeds)", mf.C_QW, "line")], y=0.835, fs=LEG_FS)
    ax = fig.add_axes([0.09, 0.42, 0.88, 0.36]); out = []
    xd = {"LR": 3.05, "RoBERTa": 3.25, "Qwen27B": 3.45}
    endpos = {"LR": 14, "RoBERTa": 30, "Qwen27B": -2}
    for m, name, col in mf.MODELS:
        y = [CM.loc[(m, a), "FP_TOEFL"] * 100 for a in "ABC"]
        ax.plot([0, 1, 2], y, color=col, linewidth=3.2, marker="o", markersize=12, markerfacecolor=col,
                markeredgecolor=SURF, markeredgewidth=2, zorder=4)
        d = CM.loc[(m, "D"), "FP_TOEFL"] * 100
        ax.plot([xd[m]], [d], marker="o", markersize=12, markerfacecolor=col, markeredgecolor=SURF, alpha=0.30,
                linestyle="none", zorder=3)
        ax.text(2.12, endpos[m], f"{y[0]:.0f}% → {y[2]:.0f}%", fontsize=CALL_FS, fontweight="bold", color=col,
                ha="left", va="center", path_effects=mf.HALO, zorder=6)
        out += [dict(model=m, arm=a, toefl_flagged_pct=CM.loc[(m, a), "FP_TOEFL"] * 100) for a in "ABCD"]
    ax.text(3.25, 103, "D (control):\nhelps much less", fontsize=TICK_FS, color=MUTED, ha="center", va="bottom", linespacing=1.15)
    ax.axvline(2.78, color=GRID, linewidth=1)
    ax.set_xlim(-0.35, 3.75)
    ax.set_xticks([0, 1, 2, 3.25]); ax.set_xticklabels(ARM_LABELS, fontsize=TICK_FS - 1, color=INK)
    _axes_clean(ax, ymax=125); ax.set_ylim(-8, 125)
    mf.footer(fig, "fig05_trend", "Threshold 0.5; means over seeds (Qwen: 2 seeds, so arm A varies a lot). Faded dots = arm D. "
                                  "Error bars, commercial range and open-source lines are on the backup version of this chart.")
    mf.save(fig, "fig05_trend", pd.DataFrame(out))


def fig15_detail():
    rows = [
        ("1", "Our first model looked strong", "0.890 vs 0.888", "AUC: our model vs just guessing the source",
         "A high score can just mean the model recognises the source.",
         "Stopped using it; tested on seven new data sources."),
        ("2", "RoBERTa learned a formatting shortcut", "0.04 to 0.91", "AUC, same prompts: before vs after cleaning",
         "It learned the layout of the text, not the writing.",
         "Made every text look the same; added a check."),
        ("3", "Qwen was tested on essays it had trained on", "0% to 40%", "TOEFL essays flagged (2 seeds: 56% and 23%)",
         "Testing on training essays makes a model look far better than it is.",
         "Kept TOEFL out of training; re-tested."),
        ("4", "Our native comparison was unfair", "2% vs 80-100%", "native flagged: familiar vs new style (logistic regression, RoBERTa)",
         "Testing only on familiar native essays hid the real problem.",
         "Added arms C and D and more kinds of native writing."),
    ]
    fig = mf.new_fig("Four mistakes we made, and what each one taught us",
                     "We kept the failed runs and re-ran everything; the numbers are from the final runs")
    ax = fig.add_axes([0.025, 0.22, 0.95, 0.60]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    cols = ((4.5, "What went wrong"), (25, "The number"), (48.5, "What it means"), (77, "What we changed"))
    for x, hd in cols:
        ax.text(x, 99.5, hd, fontsize=17, color=MUTED, fontweight="bold", va="top")
    rh, gap = 21.5, 1.5
    for k, (num, wrong, big, cap, means, fix) in enumerate(rows):
        y0 = 91 - (k + 1) * rh - k * gap
        ax.add_patch(Rectangle((0, y0), 100, rh, facecolor="white", edgecolor=GRID, linewidth=1.5))
        ax.add_patch(Rectangle((0, y0), 0.6, rh, facecolor=INK2, linewidth=0))
        ax.text(1.6, y0 + rh / 2, num, fontsize=30, color=MUTED, fontweight="bold", va="center")
        ax.text(4.5, y0 + rh / 2, "\n".join(textwrap.wrap(wrong, 20)), fontsize=20, color=INK, fontweight="bold", va="center", linespacing=1.2)
        ax.text(25, y0 + rh * 0.76, big, fontsize=30, color=INK, fontweight="bold", va="center")
        ax.text(25, y0 + rh * 0.54, "\n".join(textwrap.wrap(cap, 37)), fontsize=14.5, color=INK2, va="top", linespacing=1.15)
        ax.text(48.5, y0 + rh / 2, "\n".join(textwrap.wrap(means, 30)), fontsize=19, color=mf.C_LR, fontweight="bold", va="center", linespacing=1.2)
        ax.text(77, y0 + rh / 2, "\n".join(textwrap.wrap(fix, 28)), fontsize=17, color=INK, va="center", linespacing=1.2)
    mf.footer(fig, "fig15_pitfalls_detail", "Sources: the archived first runs and the final runs (arm A and B, threshold 0.5). 0.888 was estimated from the per-corpus flag rates of the first analysis.")
    mf.save(fig, "fig15_pitfalls_detail", pd.DataFrame([dict(pitfall=n, what_went_wrong=w, number=b, caption=c, what_it_means=mn, change=f)
                                                for n, w, b, c, mn, f in rows]))


def fig15_compact():
    rows = [
        ("1", "First model looked strong", "0.890 vs 0.888", "AUC: our model vs guessing the source",
         "A high score can just mean the model spots where an essay came from."),
        ("2", "RoBERTa learned a format shortcut", "0.04 to 0.91", "AUC on same-prompt essays, before vs after cleaning",
         "It learned how the text was laid out instead of how it was written."),
        ("3", "Qwen tested on its own training essays", "0% to 40%", "TOEFL essays flagged (2 runs: 56% and 23%)",
         "Testing on training essays makes a model look far better than it is."),
        ("4", "Unfair native comparison", "2% vs 80-100%", "native flagged, familiar vs new style; logistic regression and RoBERTa",
         "Testing only on familiar native essays hid the problem."),
    ]
    fig = mf.new_fig("Four mistakes we made, and what each one taught us",
                     "We kept the failed runs and re-ran everything; the numbers are from the final runs")
    ax = fig.add_axes([0.025, 0.30, 0.95, 0.54]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    for x, hd in ((4.5, "What went wrong"), (27, "The number"), (56, "What it means")):
        ax.text(x, 99.5, hd, fontsize=20, color=MUTED, fontweight="bold", va="top")
    rh, gap = 22.0, 1.4
    for k, (num, wrong, big, cap, means) in enumerate(rows):
        y0 = 92 - (k + 1) * rh - k * gap
        ax.add_patch(Rectangle((0, y0), 100, rh, facecolor="white", edgecolor=GRID, linewidth=1.5))
        ax.add_patch(Rectangle((0, y0), 0.6, rh, facecolor=INK2, linewidth=0))
        ax.text(1.6, y0 + rh / 2, num, fontsize=32, color=MUTED, fontweight="bold", va="center")
        ax.text(4.5, y0 + rh / 2, "\n".join(textwrap.wrap(wrong, 20)), fontsize=24, color=INK, fontweight="bold", va="center", linespacing=1.15)
        ax.text(27, y0 + rh * 0.76, big, fontsize=34, color=INK, fontweight="bold", va="center")
        ax.text(27, y0 + rh * 0.52, "\n".join(textwrap.wrap(cap, 37)), fontsize=16, color=INK2, va="top", linespacing=1.1)
        ax.text(56, y0 + rh / 2, "\n".join(textwrap.wrap(means, 40)), fontsize=23, color=mf.C_LR, fontweight="bold", va="center", linespacing=1.2)
    mf.footer(fig, "fig15_pitfalls", "Sources: the archived first runs and the final runs (arm A and B, threshold 0.5). 0.888 was estimated from the per-corpus flag rates of the first analysis.")
    mf.save(fig, "fig15_pitfalls", pd.DataFrame([dict(pitfall=n, what_went_wrong=w, number=b, caption=c, what_it_means=mn)
                                                for n, w, b, c, mn in rows]))


def fig03_plain():
    t = pd.read_csv(FIGDIR / "tables" / "fig03_opensource_before_plainwords.csv")
    fig = mf.new_fig("Open-source detectors from 2024 flag far fewer TOEFL essays",
                     "Share of 91 real TOEFL essays called AI by each detector (same Liang benchmark)")
    mf.swatch_legend(fig, [("Commercial detector, 2023", GRAY, "bar"), ("Open-source detector, 2024", C_OS, "bar")], y=0.835, fs=LEG_FS)
    ax = fig.add_axes([0.09, 0.40, 0.88, 0.37]); out = []
    vals = list(t.toefl_flagged_pct); lo, hi = min(v for v, o in zip(vals, t.open_source) if not o), max(vals)
    for i, r in t.reset_index(drop=True).iterrows():
        x = i + (0.6 if r.open_source else 0)
        ax.bar(x, r.toefl_flagged_pct, width=0.62, color=C_OS if r.open_source else GRAY, zorder=3)
        if r.open_source or r.toefl_flagged_pct in (lo, hi):
            ax.text(x, r.toefl_flagged_pct + 2, f"{r.toefl_flagged_pct:.0f}%", ha="center", va="bottom", fontsize=VAL_FS,
                    fontweight="bold", color=INK, zorder=6)
        out.append(dict(detector=r.detector, x=x, toefl_flagged_pct=r.toefl_flagged_pct, open_source=bool(r.open_source)))
    ax.set_xticks([o["x"] for o in out]); ax.set_xticklabels([{"Fast-DetectGPT": "Fast-\nDetectGPT", "OriginalityAI": "Originality\nAI"}.get(o["detector"], o["detector"]) for o in out], fontsize=TICK_FS - 2)
    for lab in ax.get_xticklabels():
        lab.set_text(lab.get_text())
    _axes_clean(ax, ymax=100)
    ax.set_yticks([0, 25, 50, 75]); ax.set_yticklabels(["0%", "25%", "50%", "75%"])
    mf.footer(fig, "fig03_opensource", "Threshold 0.5 for commercial detectors; official thresholds for Binoculars and Fast-DetectGPT, no training on our data. Bias and AI-detection scores are on the backup slide.")
    mf.save(fig, "fig03_opensource", pd.DataFrame(out))


def fig14_plain():
    steps = [("1", "Collect", "8 sources"), ("2", "Clean", "same rules"), ("3", "Deduplicate", "0 overlap"),
             ("4", "Split", "10,171 test essays"), ("5", "Train", "3 models x 4 sets"), ("6", "Evaluate", "threshold 0.5")]
    fig = mf.new_fig("How we built the test: from raw essays to a test with no overlap",
                     "Every number in this talk comes from this pipeline; the same rules were applied to all models")
    ax = fig.add_axes([0.045, 0.50, 0.91, 0.30]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    w, gap = 14.6, 2.48
    for k, (num, name, big) in enumerate(steps):
        x0 = k * (w + gap)
        ax.add_patch(Rectangle((x0, 4), w, 92, facecolor="white", edgecolor=GRID, linewidth=1.8))
        ax.add_patch(Rectangle((x0, 90), w, 6, facecolor=INK2, linewidth=0))
        ax.text(x0 + 1.2, 74, f"{num}  {name}", fontsize=20, fontweight="bold", color=INK2, va="center")
        ax.text(x0 + w / 2, 38, "\n".join(textwrap.wrap(big, 11)), fontsize=27, fontweight="bold", color=INK, ha="center", va="center", linespacing=1.1)
        if k < len(steps) - 1:
            ax.annotate("", xy=(x0 + w + gap - 0.3, 50), xytext=(x0 + w + 0.3, 50), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=2))
    mf.footer(fig, "fig14_pipeline", "Test essays: 13,672 in total, minus the half of the FCE and W&I essays that set C trains on (10,171 left). Details of each step: backup slide.")
    mf.save(fig, "fig14_pipeline", pd.DataFrame([dict(step=n, name=nm, headline=b) for n, nm, b in steps]))


if __name__ == "__main__":
    results_plain("fig19_ml_results", "Learning from exam-style essays stops most false flags on TOEFL essays", "LR", "mean of 5 seeds")
    results_plain("fig21_dl_results", "RoBERTa improves with exam-style essays too, but still flags 1 in 4", "RoBERTa",
                  "mean of 5 seeds; arm C varies by 8 points")
    results_plain("fig23_lora_results", "The biggest model starts lower, and exam-style essays help it too", "Qwen27B",
                  "mean of only 2 seeds; arm A was 56% in one run and 23% in the other")
    fig04_plain(); fig05_plain(); fig15_compact(); fig15_detail(); fig03_plain(); fig14_plain()
