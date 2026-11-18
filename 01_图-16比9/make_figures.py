# make_figures.py -- INT4096 "Detecting the Detector": slide figures (16:9, 1920x1080) generated from the run's CSVs.
# Usage: python make_figures.py <figures_dir> <Liang Data_and_Results dir>
# Every figure also writes its plotted numbers to tables/<name>.csv (table twin).
import sys, json, pathlib, textwrap
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

ROOT = pathlib.Path(sys.argv[1]); LIANG = pathlib.Path(sys.argv[2])
DATA = ROOT / "data"; TBL = ROOT / "tables"; TBL.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "axes.unicode_minus": False})

# ---- validated palette (light surface), colour follows the entity ----
SURF, INK, INK2, MUTED, GRID, BASE, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#b4b2ab"
C_LR, C_RB, C_QW = "#2a78d6", "#eb6834", "#1baf7a"      # logistic regression / RoBERTa / Qwen (slots 1-3)
C_OS, C_TOEFL = "#6250d6", "#e34948"                    # open-source detectors (slot 7) / TOEFL non-native essays (slot 8)
MODELS = [("LR", "Logistic regression", C_LR), ("RoBERTa", "RoBERTa", C_RB), ("Qwen27B", "Qwen 27B + LoRA", C_QW)]
SRC = "Source: unified_v4 run, 16-17 Nov 2026  |  threshold 0.5 unless stated  |  data table: tables/{}.csv"
HALO = [pe.withStroke(linewidth=4, foreground=SURF)]

CMP = pd.read_csv(DATA / "model_comparison_all.csv").set_index(["model", "arm"])
RUNS = pd.read_csv(DATA / "metrics_runs.csv")
AGG = pd.read_csv(DATA / "metrics_agg.csv")
LEN = pd.read_csv(DATA / "length_control.csv")
DETS = ["Crossplag", "GPTZero", "HFOpenAI", "OriginalityAI", "Quil", "Sapling", "ZeroGPT"]


def new_fig(title, subtitle=None):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    fig.text(0.04, 0.955, title, fontsize=27, fontweight="bold", color=INK, va="top", ha="left")
    if subtitle:
        fig.text(0.04, 0.885, subtitle, fontsize=15.5, color=INK2, va="top", ha="left")
    return fig


def footer(fig, name, extra=""):
    lines = textwrap.wrap(extra, 168) if extra else []
    fig.text(0.04, 0.022, SRC.format(name), fontsize=11, color=MUTED, va="bottom", ha="left")
    for i, ln in enumerate(lines):
        fig.text(0.04, 0.022 + 0.026 * (len(lines) - i), ln, fontsize=12, color=INK2, va="bottom", ha="left")


def style_hbar(ax, xmax, pct=True):
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASE); ax.spines["left"].set_linewidth(1)
    ax.set_facecolor(SURF); ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.tick_params(axis="both", colors=INK2, labelsize=13, length=0)
    ax.set_xlim(0, xmax)


def save(fig, name, df=None):
    fig.savefig(ROOT / f"{name}.png", dpi=120, facecolor=SURF)
    plt.close(fig)
    if df is not None:
        df.to_csv(TBL / f"{name}.csv", index=False)
    print("saved", name)


def swatch_legend(fig, items, x=0.04, y=0.835, fs=14):
    """items: list of (label, color, kind) kind in {'bar','line','hollow'}; one row of key + label (text stays in ink)."""
    handles = []
    for label, col, kind in items:
        if kind == "bar":
            handles.append(Patch(facecolor=col, edgecolor="none", label=label))
        elif kind == "line":
            handles.append(Line2D([0], [0], color=col, lw=2, marker="o", markersize=9, markerfacecolor=col, markeredgecolor=SURF, label=label))
        elif kind == "hollow":
            handles.append(Line2D([0], [0], color="none", marker="o", markersize=9, markerfacecolor=SURF, markeredgecolor=col, markeredgewidth=2, label=label))
        elif kind == "band":
            handles.append(Patch(facecolor=col, edgecolor="none", alpha=0.35, label=label))
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x, y), ncol=len(handles), frameon=False,
                     fontsize=fs, labelcolor=INK2, handlelength=1.6, columnspacing=1.8, handletextpad=0.6)
    return leg


# =====================================================================================================
# 1  Hook: one real essay, 7 commercial detectors
# =====================================================================================================
def load_liang(corpus):
    base = LIANG / "Human_Data" / corpus
    docs = [d["document"] for d in json.loads((base / "data.json").read_text(encoding="utf-8"))]
    sc = {}
    for d in DETS:
        arr = json.loads((base / f"{d}.json").read_text(encoding="utf-8"))
        assert len(arr) == len(docs) and all(a["document"] == b for a, b in zip(arr, docs)), f"order mismatch {corpus} {d}"
        sc[d] = [float(a["score"]) for a in arr]
    return docs, sc


def essay_card(fig, rect, text, edge=GRID):
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=edge, linewidth=1.5, transform=ax.transAxes))
    px_w = rect[2] * 1920; px_h = rect[3] * 1080
    for fs in (15, 14, 13.5, 13, 12.5, 12):
        chars = int(0.92 * px_w / (0.47 * fs * 120 / 72))
        lines = textwrap.wrap(text, chars)
        need = len(lines) * fs * 1.2 * 1.5 * 120 / 72 + 36
        if need <= px_h: break
    ax.text(0.018, 0.93, "\n".join(lines), va="top", ha="left", fontsize=fs, color=INK, linespacing=1.5)
    return fs


def score_bars(fig, rect, scores, color, title, title_color):
    ax = fig.add_axes(rect)
    ys = np.arange(len(DETS))[::-1]
    for y, d in zip(ys, DETS):
        ax.barh(y, scores[d], height=0.55, color=color)
        ax.text(scores[d] + 0.02, y, f"{scores[d]:.2f}", va="center", ha="left", fontsize=13, color=INK2, path_effects=HALO)
    ax.set_ylim(-0.6, len(DETS) + 0.5)
    ax.axvline(0.5, ymin=0, ymax=(len(DETS) - 0.45 + 0.6) / (len(DETS) + 1.1), color=INK2, linewidth=1.5)
    ax.text(0.5, len(DETS) - 0.35, "0.5 = flagged as AI", ha="center", va="bottom", fontsize=11.5, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels(DETS); style_hbar(ax, 1.12)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_title(title, loc="left", fontsize=17, fontweight="bold", color=title_color, pad=26)
    return ax


def fig_hook():
    docs, sc = load_liang("TOEFL_real_91"); hdocs, hsc = load_liang("HewlettStudentEssay_real_88")
    t_idx = 84
    assert all(sc[d][t_idx] >= 0.5 for d in DETS)
    n_all7 = sum(all(sc[d][i] >= 0.5 for d in DETS) for i in range(len(docs)))
    n_none = sum(all(hsc[d][i] < 0.5 for d in DETS) for i in range(len(hdocs)))
    h_idx = 26   # 129-word native essay, scored 0 of 7; '@CAPS1' is the dataset's anonymisation placeholder
    assert all(hsc[d][h_idx] < 0.5 for d in DETS)
    fig = new_fig("A student wrote this essay. Seven AI detectors said: AI.",
                  "Real essays from the Liang et al. (2023) benchmark, scored by seven commercial detectors")
    fig.text(0.04, 0.822, f"TOEFL essay by a non-native English writer  ·  {len(docs[t_idx].split())} words", fontsize=14, color=INK2, fontweight="bold")
    essay_card(fig, [0.04, 0.505, 0.55, 0.29], docs[t_idx])
    score_bars(fig, [0.71, 0.525, 0.25, 0.26], {d: sc[d][t_idx] for d in DETS}, C_TOEFL, "Flagged as AI by 7 of 7", C_TOEFL)
    fig.text(0.04, 0.452, f"US school essay by a native English writer  ·  {len(hdocs[h_idx].split())} words", fontsize=14, color=INK2, fontweight="bold")
    essay_card(fig, [0.04, 0.135, 0.55, 0.30], hdocs[h_idx])
    score_bars(fig, [0.71, 0.155, 0.25, 0.26], {d: hsc[d][h_idx] for d in DETS}, GRAY, "Flagged as AI by 0 of 7", INK2)
    footer(fig, "fig01_hook",
           f"Illustrative pair chosen from the data, not a random sample: {n_all7} of {len(docs)} TOEFL essays were flagged by all seven detectors, {n_none} of {len(hdocs)} native essays by none. Essays are shown verbatim ('@CAPS1' = anonymised name in the dataset). Scores archived from Liang et al. (2023); flag = score of 0.5 or more.")
    rows = [dict(essay="TOEFL idx %d" % t_idx, detector=d, score=sc[d][t_idx]) for d in DETS] + \
           [dict(essay="Hewlett idx %d" % h_idx, detector=d, score=hsc[d][h_idx]) for d in DETS]
    save(fig, "fig01_hook", pd.DataFrame(rows))


# =====================================================================================================
# 2  Commercial detectors: flag TOEFL, cannot spot AI
# =====================================================================================================
def fig_commercial():
    c = RUNS[RUNS.model == "commercial"].sort_values("rate_t05_TOEFL_real_91", ascending=False)
    fig = new_fig("2023 commercial detectors flag TOEFL essays, and barely spot AI",
                  "Left: share of real human essays flagged as AI. Right: how well each detector tells AI essays from human essays (AUC)")
    swatch_legend(fig, [("TOEFL essays (non-native)", C_TOEFL, "bar"), ("Hewlett essays (native)", GRAY, "bar")], y=0.835)
    axL = fig.add_axes([0.16, 0.17, 0.46, 0.54]); axR = fig.add_axes([0.70, 0.17, 0.26, 0.54], sharey=axL)
    ys = np.arange(len(c))[::-1]
    for y, (_, r) in zip(ys, c.iterrows()):
        t, h = r.rate_t05_TOEFL_real_91 * 100, r.rate_t05_HewlettStudentEssay_real_88 * 100
        axL.barh(y + 0.17, t, height=0.3, color=C_TOEFL); axL.barh(y - 0.17, h, height=0.3, color=GRAY)
        axL.text(t + 1.2, y + 0.17, f"{t:.0f}%", va="center", fontsize=13, color=INK2, path_effects=HALO)
        axL.text(h + 1.2, y - 0.17, f"{h:.0f}%", va="center", fontsize=13, color=INK2, path_effects=HALO)
        a = r.det_AI352_vs_LiangHuman2
        axR.barh(y, a, height=0.34, color=GRAY); axR.text(a + 0.02, y, f"{a:.2f}", va="center", fontsize=13, color=INK2, path_effects=HALO, zorder=5)
    axL.set_yticks(ys); axL.set_yticklabels(list(c.detector)); style_hbar(axL, 88); axL.set_xticks([0, 25, 50, 75])
    axL.set_xticklabels(["0%", "25%", "50%", "75%"]); axL.set_ylim(-0.6, len(c) + 0.35)
    style_hbar(axR, 1.0); axR.set_xticks([0, 0.5, 1.0]); axR.tick_params(axis="y", labelleft=False)
    axR.axvline(0.5, ymin=0, ymax=(len(c) - 0.5 + 0.6) / (len(c) + 0.95), color=INK2, linewidth=1.5); axR.text(0.5, len(c) - 0.4, "coin flip (0.5)", ha="center", va="bottom", fontsize=12, color=INK2)
    axL.set_title("Real human essays flagged as AI", loc="left", fontsize=17, fontweight="bold", color=INK, pad=14)
    axR.set_title("Catching AI essays (AUC)", loc="left", fontsize=17, fontweight="bold", color=INK, pad=14)
    footer(fig, "fig02_commercial", "AUC: Liang's 352 AI essays vs the 179 real TOEFL + Hewlett essays; 1.0 = perfect, 0.5 = chance.")
    save(fig, "fig02_commercial", c[["detector", "rate_t05_TOEFL_real_91", "rate_t05_HewlettStudentEssay_real_88", "det_AI352_vs_LiangHuman2", "bias_TOEFL_vs_Hewlett"]])


# =====================================================================================================
# 3  Open-source vs commercial
# =====================================================================================================
def fig_opensource():
    com = RUNS[RUNS.model == "commercial"].sort_values("rate_t05_TOEFL_real_91", ascending=False)
    osr = RUNS[RUNS.model.isin(["Binoculars", "FastDetectGPT"])].copy()
    osr["detector"] = osr.model.map({"Binoculars": "Binoculars", "FastDetectGPT": "Fast-DetectGPT"})
    rows = list(zip(com.detector, com.rate_t05_TOEFL_real_91 * 100, com.bias_TOEFL_vs_Hewlett, com.det_AI352_vs_LiangHuman2, [False] * len(com))) + \
           [(None, None, None, None, None)] + \
           list(zip(osr.detector, osr.rate_t05_TOEFL_real_91 * 100, osr.bias_TOEFL_vs_Hewlett, osr.det_AI352_vs_LiangHuman2, [True] * len(osr)))
    fig = new_fig("Open-source detectors from 2024 are far better, but still flag TOEFL more",
                  "Seven 2023 commercial detectors (grey) against two 2024 open-source detectors (purple), same Liang benchmark")
    swatch_legend(fig, [("Commercial, 2023", GRAY, "line"), ("Open-source, 2024 (official thresholds)", C_OS, "line")], y=0.835)
    specs = [("Real TOEFL essays flagged as AI", 1, (0, 100), [0, 25, 50, 75], lambda v: f"{v:.0f}%", None),
             ("Bias: TOEFL vs Hewlett (AUC)", 2, (0.45, 1.10), [0.5, 0.75, 1.0], lambda v: f"{v:.2f}", (0.5, "no bias (0.5)")),
             ("Catching AI essays (AUC)", 3, (0.3, 1.10), [0.5, 0.75, 1.0], lambda v: f"{v:.2f}", (0.5, "coin flip (0.5)"))]
    lefts = [0.17, 0.465, 0.74]; widths = [0.27, 0.24, 0.22]
    ny = len(rows); axes = []
    for k, (title, col, xlim, ticks, fmt, ref) in enumerate(specs):
        ax = fig.add_axes([lefts[k], 0.17, widths[k], 0.53], sharey=axes[0] if axes else None); axes.append(ax)
        vals = [r[col] for r in rows if r[0] is not None and not r[4]]
        lo, hi = min(vals), max(vals)
        for i, r in enumerate(rows):
            if r[0] is None: continue
            y = ny - 1 - i; v = r[col]; is_os = r[4]
            ax.plot([xlim[0], v], [y, y], color=GRID, linewidth=1, zorder=1)
            ax.scatter([v], [y], s=150, color=C_OS if is_os else GRAY, edgecolor=SURF, linewidth=2, zorder=3)
            if is_os or v in (lo, hi):
                ax.annotate(fmt(v), (v, y), xytext=(12, 0), textcoords="offset points", ha="left", va="center", fontsize=13, color=INK2, path_effects=HALO, zorder=5)
        if ref:
            ax.axvline(ref[0], ymin=0, ymax=(ny - 0.5 + 0.6) / (ny + 0.95), color=INK2, linewidth=1.5, zorder=2); ax.text(ref[0], ny - 0.4, ref[1], ha="center", va="bottom", fontsize=12, color=INK2)
        ax.set_xlim(*xlim); ax.set_xticks(ticks); ax.set_xticklabels([fmt(t) for t in ticks])
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASE); ax.set_facecolor(SURF); ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
        ax.tick_params(axis="both", colors=INK2, labelsize=13, length=0)
        ax.set_title(title, loc="left", fontsize=16, fontweight="bold", color=INK, pad=14)
        ax.set_ylim(-0.6, ny + 0.35)
        if k == 0:
            ax.set_yticks([ny - 1 - i for i, r in enumerate(rows) if r[0] is not None]); ax.set_yticklabels([r[0] for r in rows if r[0] is not None])
        else:
            ax.tick_params(axis="y", labelleft=False)
    footer(fig, "fig03_opensource", "Gains are in far fewer false flags and much better AI detection. On bias (AUC) the open-source detectors are no better than the best commercial ones. Binoculars and Fast-DetectGPT use official thresholds and no training on our data; the lowest and highest commercial values are labelled.")
    out = pd.DataFrame([dict(detector=r[0], toefl_flagged_pct=r[1], bias_auc=r[2], detection_auc=r[3], open_source=r[4]) for r in rows if r[0] is not None])
    save(fig, "fig03_opensource", out)


# =====================================================================================================
# 4  Distribution shift: self-trained detector (arm A) flags unfamiliar native writing as much as TOEFL
# =====================================================================================================
def fig_shift():
    rows = [("Hewlett\n(same type as training)", "FP_Hewlett", GRAY), ("CollegeEssay\n(native, new style)", "FP_CollegeEss", C_OS),
            ("CS224N\n(native, new style)", "FP_CS224N", C_OS), ("LOCNESS\n(native, new style)", "FP_LOCNESS", C_OS),
            ("TOEFL\n(non-native exam essays)", "FP_TOEFL", C_TOEFL)]
    fig = new_fig("Self-trained detectors also flag unfamiliar native writing",
                  "Arm A: trained only on US school essays. Share of real human essays flagged as AI at threshold 0.5")
    swatch_legend(fig, [("Native, same type as training data", GRAY, "bar"), ("Native, unfamiliar type", C_OS, "bar"), ("Non-native (TOEFL exam essays)", C_TOEFL, "bar")], y=0.835)
    axes = []; out = []
    for k, (m, name, col) in enumerate(MODELS):
        ax = fig.add_axes([0.285 + k * 0.235, 0.17, 0.19, 0.57], sharey=axes[0] if axes else None); axes.append(ax)
        n = len(rows)
        for i, (lab, key, c) in enumerate(rows):
            v = CMP.loc[(m, "A"), key] * 100; y = n - 1 - i
            ax.barh(y, v, height=0.52, color=c); ax.text(v + 2, y, f"{v:.0f}%", va="center", fontsize=14, color=INK2, fontweight="bold" if key == "FP_TOEFL" else "normal")
            out.append(dict(model=m, corpus=key, flagged_pct=v))
        style_hbar(ax, 112); ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0%", "50%", "100%"]); ax.set_ylim(-0.6, n - 0.4)
        ax.set_title(name + (" (2 seeds)" if m == "Qwen27B" else " (5 seeds)"), loc="left", fontsize=17, fontweight="bold", color=INK, pad=14)
        if k == 0:
            ax.set_yticks([n - 1 - i for i in range(n)]); ax.set_yticklabels([r[0] for r in rows], fontsize=13)
        else:
            ax.tick_params(axis="y", labelleft=False)
    footer(fig, "fig04_shift", "Logistic regression and RoBERTa flag native essays of a new style (CollegeEssay, CS224N) about as often as TOEFL essays; for Qwen, CS224N is flagged more than TOEFL and CollegeEssay less. LOCNESS (only 50 essays) is flagged much less by RoBERTa and Qwen.")
    save(fig, "fig04_shift", pd.DataFrame(out))


# =====================================================================================================
# 5  Trend: A -> B -> C (+ control D)
# =====================================================================================================
def fig_trend():
    com = RUNS[RUNS.model == "commercial"].rate_t05_TOEFL_real_91 * 100
    bino = RUNS[RUNS.model == "Binoculars"].rate_t05_TOEFL_real_91.iloc[0] * 100
    fast = RUNS[RUNS.model == "FastDetectGPT"].rate_t05_TOEFL_real_91.iloc[0] * 100
    fig = new_fig("Training on the right kind of essays cuts TOEFL false flags, in all three models",
                  "Share of real TOEFL essays flagged as AI, by what the detector learned from (threshold 0.5)")
    swatch_legend(fig, [("Logistic regression", C_LR, "line"), ("RoBERTa", C_RB, "line"), ("Qwen 27B (2 seeds)", C_QW, "line"),
                        ("Arm D: control", INK2, "hollow"), ("Commercial detectors", GRAY, "band")], y=0.84, fs=13.5)
    ax = fig.add_axes([0.09, 0.19, 0.80, 0.58]); ax.set_facecolor(SURF)
    xs = {"A": 0, "B": 1, "C": 2}; xd = {"LR": 3.1, "RoBERTa": 3.3, "Qwen27B": 3.5}
    ax.axhspan(com.min(), com.max(), color=GRAY, alpha=0.30, linewidth=0, zorder=0)
    ax.text(2.12, (com.min() + com.max()) / 2, f"Commercial detectors,\n2023: {com.min():.0f}-{com.max():.0f}%", fontsize=12.5, color=INK2, va="center", ha="left", linespacing=1.3)
    ax.plot([-0.35, 3.85], [bino, bino], color=C_OS, linewidth=1.5, zorder=1); ax.plot([-0.35, 3.85], [fast, fast], color=C_OS, linewidth=1.5, zorder=1)
    ax.text(3.85, bino - 1.0, f"Binoculars {bino:.0f}%", fontsize=12.5, color=INK2, va="top", ha="right")
    ax.text(2.82, fast + 1.0, f"Fast-DetectGPT {fast:.0f}%", fontsize=12.5, color=INK2, va="bottom", ha="left")
    out = []
    for m, name, col in MODELS:
        y = [CMP.loc[(m, a_), "FP_TOEFL"] * 100 for a_ in "ABC"]; sd = [CMP.loc[(m, a_), "TOEFL_sd"] * 100 for a_ in "ABC"]
        ax.errorbar([0, 1, 2], y, yerr=sd, fmt="none", ecolor=col, elinewidth=1.3, capsize=4, alpha=0.7, zorder=2)
        ax.plot([0, 1, 2], y, color=col, linewidth=2, marker="o", markersize=10, markerfacecolor=col, markeredgecolor=SURF, markeredgewidth=2, zorder=4)
        d = CMP.loc[(m, "D"), "FP_TOEFL"] * 100; dsd = CMP.loc[(m, "D"), "TOEFL_sd"] * 100
        ax.errorbar([xd[m]], [d], yerr=[dsd], fmt="none", ecolor=col, elinewidth=1.3, capsize=4, alpha=0.7, zorder=2)
        ax.plot([xd[m]], [d], marker="o", markersize=10, markerfacecolor=SURF, markeredgecolor=col, markeredgewidth=2.2, linestyle="none", zorder=4)
        # label placement per point: (x offset, y offset, ha, va)
        place = {("LR", 0): (0, 5, "center", "bottom"), ("RoBERTa", 0): (0, 5, "center", "bottom"), ("Qwen27B", 0): (-0.09, 0, "right", "center"),
                 ("LR", 1): (0, 5, "center", "bottom"), ("RoBERTa", 1): (0, 5, "center", "bottom"), ("Qwen27B", 1): (0, -5, "center", "top"),
                 ("LR", 2): (0.12, 3.5, "left", "bottom"), ("RoBERTa", 2): (0.12, 0, "left", "center"), ("Qwen27B", 2): (0.12, -3.5, "left", "top")}
        for k, (xx, v) in enumerate(zip([0, 1, 2], y)):
            xo, yo, ha, va = place[(m, k)]
            ax.text(xx + xo, v + yo, f"{v:.0f}%", fontsize=14, color=INK2, ha=ha, va=va, fontweight="bold", path_effects=HALO, zorder=6)
        ax.text(xd[m] + 0.09, d, f"{d:.0f}%", fontsize=14, color=INK2, ha="left", va="center", fontweight="bold", path_effects=HALO, zorder=6)
        out += [dict(model=m, arm=a_, toefl_flagged_pct=CMP.loc[(m, a_), "FP_TOEFL"] * 100, sd_pct=CMP.loc[(m, a_), "TOEFL_sd"] * 100) for a_ in "ABCD"]
    ax.axvline(2.75, color=GRID, linewidth=1)
    ax.set_xlim(-0.4, 3.95); ax.set_ylim(-4, 108)
    ax.set_xticks([0, 1, 2, 3.3])
    ax.set_xticklabels(["A\nnative essays only", "B\n+ school-age\nnon-native essays", "C\n+ exam-style\nnon-native essays", "D (control)\n+ more native\nessay types"], fontsize=13.5, color=INK)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(BASE)
    ax.tick_params(axis="y", colors=INK2, labelsize=13, length=0); ax.tick_params(axis="x", length=0, pad=8)
    footer(fig, "fig05_trend", "Error bars: SD across seeds (Qwen: 2 seeds, so arm A varies a lot). Arm C was trained on essays like TOEFL; it is not a general fix. Arm D adds other native essay types and barely helps.")
    save(fig, "fig05_trend", pd.DataFrame(out))


# =====================================================================================================
# 6  Scoreboard table
# =====================================================================================================
def fig_scoreboard():
    g = lambda m, a, k: CMP.loc[(m, a), k]
    spec = [("Catches AI on the same prompts\n(AUC, arm A; higher = better)", [g(m, "A", "AUC_ArguGPTvsTOEFL") for m, _, _ in MODELS], lambda v: f"{v:.2f}", "max", 0.02),
            ("Unfamiliar native essays wrongly flagged\n(arm A; lower = better)", [g(m, "A", "FP_nativeOOD_mean") * 100 for m, _, _ in MODELS], lambda v: f"{v:.0f}%", "min", 3),
            ("TOEFL essays wrongly flagged,\nafter adding exam-style data (arm C)", [g(m, "C", "FP_TOEFL") * 100 for m, _, _ in MODELS], lambda v: f"{v:.0f}%", "min", 3.5),
            ("Run-to-run variation of that rate\n(SD across seeds, arm C)", [g(m, "C", "TOEFL_sd") * 100 for m, _, _ in MODELS], lambda v: f"±{v:.0f}", "min", 1.5)]
    text_rows = [("Training + scoring time", ["seconds", "about 4 min", "about 4 h"]),
                 ("Hardware used", ["laptop CPU", "1 GPU (A40)", "2 GPUs (A40)"]),
                 ("Shows which words push the score?", ["Yes", "No", "No"])]
    fig = new_fig("Bigger models detect better, but the right training data matters more than size",
                  "Same data, same threshold (0.5), same exams; logistic regression and RoBERTa: 5 seeds, Qwen: 2 seeds")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.74]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    colx = [0.0, 0.42, 0.62, 0.82]; n = len(spec) + len(text_rows) + 1; rh = 1.0 / n
    for j, (m, name, col) in enumerate(MODELS):
        ax.add_patch(Rectangle((colx[j + 1] + 0.012, 1 - rh + 0.02), 0.17, 0.012, color=col, linewidth=0))
        ax.text(colx[j + 1] + 0.012, 1 - rh * 0.42, name + ("\n(classical ML)" if m == "LR" else "\n(deep learning)" if m == "RoBERTa" else "\n(large language model)"),
                fontsize=15, fontweight="bold", color=INK, va="center", ha="left", linespacing=1.3)
    rows_out = []
    for i, (lab, vals, fmt, best, tol) in enumerate(spec):
        y = 1 - rh * (i + 1.5)
        ax.plot([0, 1], [y + rh / 2, y + rh / 2], color=GRID, linewidth=1)
        ax.text(0.0, y, lab, fontsize=14, color=INK2, va="center", ha="left", linespacing=1.3)
        b = max(vals) if best == "max" else min(vals)
        for j, v in enumerate(vals):
            win = abs(v - b) <= tol
            ax.text(colx[j + 1] + 0.012, y, fmt(v), fontsize=21 if win else 19, fontweight="bold" if win else "normal", color=INK if win else INK2, va="center", ha="left")
        rows_out.append(dict(metric=lab.replace("\n", " "), **{m: v for (m, _, _), v in zip(MODELS, vals)}))
    for k, (lab, vals) in enumerate(text_rows):
        i = len(spec) + k; y = 1 - rh * (i + 1.5)
        ax.plot([0, 1], [y + rh / 2, y + rh / 2], color=GRID, linewidth=1)
        ax.text(0.0, y, lab, fontsize=14, color=INK2, va="center", ha="left")
        for j, v in enumerate(vals):
            ax.text(colx[j + 1] + 0.012, y, v, fontsize=17, color=INK2, va="center", ha="left")
        rows_out.append(dict(metric=lab, **{m: v for (m, _, _), v in zip(MODELS, vals)}))
    footer(fig, "fig06_scoreboard", "Bold = best in row (ties within the run-to-run noise are both bold). Logistic regression with the right data (8% in arm C) beats Qwen without it (40% in arm A). Times from the logs: Qwen 2.9 h training + 74 min scoring on 2 GPUs.")
    save(fig, "fig06_scoreboard", pd.DataFrame(rows_out))


# =====================================================================================================
# 7  What arm C does not fix (A -> C dumbbell)
# =====================================================================================================
def fig_notfixed():
    rows = [("TOEFL\n(like the added data)", "FP_TOEFL"), ("CollegeEssay\n(native, new style)", "FP_CollegeEss"),
            ("CS224N\n(native, new style)", "FP_CS224N"), ("LOCNESS\n(native, new style)", "FP_LOCNESS")]
    fig = new_fig("Exam-style data fixes TOEFL and helps elsewhere, but not everywhere",
                  "Share of real human essays flagged as AI: arm A (native only) vs arm C (+ exam-style non-native essays)")
    swatch_legend(fig, [("Arm A: native essays only", GRAY, "line"), ("Arm C: + exam-style non-native (colour = model)", INK2, "line")], y=0.835)
    axes = []; out = []
    for k, (m, name, col) in enumerate(MODELS):
        ax = fig.add_axes([0.285 + k * 0.235, 0.17, 0.19, 0.57], sharey=axes[0] if axes else None); axes.append(ax)
        n = len(rows)
        for i, (lab, key) in enumerate(rows):
            a, c = CMP.loc[(m, "A"), key] * 100, CMP.loc[(m, "C"), key] * 100; y = n - 1 - i
            ax.plot([a, c], [y, y], color=BASE, linewidth=3, zorder=1, solid_capstyle="round")
            ax.scatter([a], [y], s=170, color=GRAY, edgecolor=SURF, linewidth=2, zorder=3); ax.scatter([c], [y], s=170, color=col, edgecolor=SURF, linewidth=2, zorder=4)
            da, dc = (14, -14) if c < a else (-14, 14)
            ax.annotate(f"{a:.0f}", (a, y), xytext=(da, -1), textcoords="offset points", ha="center" if abs(a - c) > 14 else ("left" if da > 0 else "right"), va="center", fontsize=13.5, color=INK2)
            ax.annotate(f"{c:.0f}", (c, y), xytext=(dc, -1), textcoords="offset points", ha="center" if abs(a - c) > 14 else ("left" if dc > 0 else "right"), va="center", fontsize=13.5, color=INK, fontweight="bold")
            out.append(dict(model=m, corpus=key, arm_A_pct=a, arm_C_pct=c))
        style_hbar(ax, 112); ax.set_xlim(-14, 112); ax.set_xticks([0, 50, 100]); ax.set_xticklabels(["0%", "50%", "100%"]); ax.set_ylim(-0.6, n - 0.4)
        ax.set_title(name + (" (2 seeds)" if m == "Qwen27B" else " (5 seeds)"), loc="left", fontsize=17, fontweight="bold", color=INK, pad=14)
        if k == 0:
            ax.set_yticks([n - 1 - i for i in range(n)]); ax.set_yticklabels([r[0] for r in rows], fontsize=13)
        else:
            ax.tick_params(axis="y", labelleft=False)
    footer(fig, "fig07_notfixed", "Numbers are % flagged (grey = arm A, bold = arm C). Arm C also lowers CollegeEssay and LOCNESS, but CS224N is still flagged 97% / 70% / 23%: fixing the data for one kind of writing does not make a detector fair in general.")
    save(fig, "fig07_notfixed", pd.DataFrame(out))


# =====================================================================================================
# 8  Backup: more data does not fix it (learning curve)
# =====================================================================================================
def fig_learning():
    fig = new_fig("Backup: more training data barely helps; the kind of data does",
                  "Real TOEFL essays flagged as AI at threshold 0.5, by training-set size (essays per class; 5 seeds)")
    swatch_legend(fig, [("Arm A: native only", GRAY, "line"), ("Arm B: + school-age non-native (colour = model)", INK2, "line")], y=0.835)
    out = []
    for k, (m, name, col) in enumerate(MODELS[:2]):
        ax = fig.add_axes([0.08 + k * 0.46, 0.19, 0.38, 0.55]); ax.set_facecolor(SURF)
        for arm, c in (("A", GRAY), ("B", col)):
            d = AGG[(AGG.model == m) & (AGG.arm == arm)].sort_values("n")
            xs = d.n.astype(int).to_numpy(); ys = d["rate_t05_TOEFL_real_91__mean"].to_numpy() * 100; sd = d["rate_t05_TOEFL_real_91__std"].to_numpy() * 100
            ax.errorbar(xs, ys, yerr=sd, fmt="none", ecolor=c, elinewidth=1.2, capsize=4, alpha=0.7)
            ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=10, markerfacecolor=c, markeredgecolor=SURF, markeredgewidth=2)
            for x, y in zip(xs, ys):
                up = arm == "A"
                ax.text(x, y + (5 if up else -5), f"{y:.0f}%", ha="center", va="bottom" if up else "top", fontsize=13.5, color=INK2, path_effects=HALO)
            out += [dict(model=m, arm=arm, n=int(x), toefl_flagged_pct=float(y)) for x, y in zip(xs, ys)]
        ax.set_xscale("log"); ax.set_xticks([1000, 2000, 4000]); ax.set_xticklabels(["1,000", "2,000", "4,000"]); ax.minorticks_off()
        ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASE); ax.tick_params(axis="both", colors=INK2, labelsize=13, length=0)
        ax.set_title(name, loc="left", fontsize=17, fontweight="bold", color=INK, pad=12)
        ax.set_xlabel("training essays per class", fontsize=13, color=INK2, labelpad=8)
    footer(fig, "fig08_learning_backup", "Quadrupling the data does not lower arm A (it even rises for logistic regression, 70% to 80%); changing what the data contains moves the rate far more.")
    save(fig, "fig08_learning_backup", pd.DataFrame(out))


# =====================================================================================================
# 9  Backup: length control
# =====================================================================================================
def fig_length():
    L = LEN.copy(); L = L[L.run.str.contains("n4000") | L.model.isin(["Binoculars", "FastDetectGPT"])]
    g = L.groupby("model")[["bias_all_pairs", "bias_length_matched"]].mean()
    names = {"Binoculars": "Binoculars", "FastDetectGPT": "Fast-DetectGPT", "LR": "Logistic regression (arms A, B)", "RoBERTa": "RoBERTa (arms A, B)", "Qwen27B": "Qwen 27B (arms A, B)"}
    cols = {"Binoculars": C_OS, "FastDetectGPT": C_OS, "LR": C_LR, "RoBERTa": C_RB, "Qwen27B": C_QW}
    order = ["Binoculars", "FastDetectGPT", "LR", "RoBERTa", "Qwen27B"]
    fig = new_fig("Backup: essay length explains only a small part of the TOEFL bias",
                  "Bias AUC (TOEFL vs Hewlett; 0.5 = none) over all pairs vs over length-matched pairs only")
    swatch_legend(fig, [("All pairs", GRAY, "line"), ("Length-matched pairs (within 30% of word count; colour = detector)", INK2, "line")], y=0.835)
    ax = fig.add_axes([0.25, 0.17, 0.68, 0.57]); ax.set_facecolor(SURF); n = len(order); out = []
    for i, m in enumerate(order):
        y = n - 1 - i; a, b = g.loc[m, "bias_all_pairs"], g.loc[m, "bias_length_matched"]
        ax.plot([a, b], [y, y], color=BASE, linewidth=3, zorder=1)
        ax.scatter([a], [y], s=170, color=GRAY, edgecolor=SURF, linewidth=2, zorder=3); ax.scatter([b], [y], s=170, color=cols[m], edgecolor=SURF, linewidth=2, zorder=4)
        ax.annotate(f"{a:.2f}", (a, y), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=13, color=INK2)
        ax.annotate(f"{b:.2f}", (b, y), xytext=(0, -24), textcoords="offset points", ha="center", fontsize=13.5, color=INK, fontweight="bold")
        out.append(dict(detector=m, bias_all_pairs=a, bias_length_matched=b))
    ax.axvline(0.5, ymin=0, ymax=0.93, color=INK2, linewidth=1.5); ax.text(0.5, n - 0.25, "no bias (0.5)", ha="center", va="bottom", fontsize=12, color=INK2)
    ax.set_xlim(0.45, 1.05); ax.set_ylim(-0.7, n - 0.1); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([n - 1 - i for i in range(n)]); ax.set_yticklabels([names[m] for m in order], fontsize=14)
    ax.xaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE); ax.tick_params(axis="both", colors=INK2, labelsize=13, length=0)
    footer(fig, "fig09_length_backup", "TOEFL essays are about 104 words, Hewlett about 378. The self-trained models do not drop when length is matched; the open-source detectors drop by 0.06-0.08.")
    save(fig, "fig09_length_backup", pd.DataFrame(out))



# =====================================================================================================
# 10/11  Precision / recall / F1 table (slide version + full version)
# =====================================================================================================
def _prf_rows():
    P = pd.read_csv(DATA / "prf1_summary.csv")
    P["run"] = P["run"].fillna("")
    return P


def _fmt(v, sd=None, digits=2):
    return f"{v:.{digits}f}" + (f" ±{sd:.2f}" if sd is not None and not np.isnan(sd) else "")


def table_figure(name, title, subtitle, rows, groups, fs=15, rowh=None, note=""):
    """rows: list of dict(label, color, vals[list[str]], bold[list[bool]], group). groups: {group_key: header text}."""
    cols = ["Accuracy", "Precision", "Recall", "F1", "F1 on TOEFL\nessays", "F1 on Hewlett\nessays", "Gap\n(Hewlett - TOEFL)"]
    fig = new_fig(title, subtitle)
    ax = fig.add_axes([0.04, 0.115, 0.92, 0.72]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nrow = len(rows) + len(groups) + 1.6
    rh = rowh or 1.0 / nrow
    colx = [0.30, 0.41, 0.52, 0.62, 0.72, 0.82, 0.93]
    y = 1 - rh * 0.6
    for c, x in zip(cols, colx):
        ax.text(x, y, c, fontsize=fs - 1.5, color=INK2, ha="center", va="center", fontweight="bold", linespacing=1.15)
    y -= rh * 0.75; ax.plot([0, 1], [y, y], color=BASE, linewidth=1.2)
    last_group = None
    for r in rows:
        if r["group"] != last_group:
            y -= rh * 0.62
            ax.text(0.0, y, groups[r["group"]], fontsize=fs - 1, color=INK2, ha="left", va="center", fontweight="bold")
            y -= rh * 0.38; last_group = r["group"]
        y -= rh * 0.62
        ax.add_patch(Rectangle((0.0, y - rh * 0.17), 0.007, rh * 0.34, color=r["color"], linewidth=0))
        ax.text(0.016, y, r["label"], fontsize=fs, color=INK, ha="left", va="center")
        for v, b, x in zip(r["vals"], r["bold"], colx):
            ax.text(x, y, v, fontsize=fs + (1 if b else 0), color=INK if b else INK2, ha="center", va="center", fontweight="bold" if b else "normal")
        y -= rh * 0.38; ax.plot([0, 1], [y, y], color=GRID, linewidth=1)
    footer(fig, name, note)
    return fig


def fig_prf1():
    P = _prf_rows()
    m = lambda model, arm, key: P[(P.model == model) & (P.arm.astype(str) == arm)][f"{key}__mean"]
    sdv = lambda model, arm, key: P[(P.model == model) & (P.arm.astype(str) == arm)][f"{key}__std"]
    keys = ["accuracy", "precision", "recall", "f1", "f1_bal_TOEFL", "f1_bal_Hewlett"]
    com = P[P.model == "commercial"]

    def row_vals(model, arm, sd=False):
        v = [float(m(model, arm, k).iloc[0]) for k in keys]
        s = [float(sdv(model, arm, k).iloc[0]) if sd else None for k in keys]
        gap = v[5] - v[4]
        return v, s, gap

    # ---------- slide version ----------
    rows, tab = [], []
    cv = {k: (com[f"{k}__mean"].min(), com[f"{k}__mean"].max()) for k in keys}
    gaps = (com["f1_bal_Hewlett__mean"] - com["f1_bal_TOEFL__mean"])
    rows.append(dict(label="7 commercial detectors, 2023 (range)", color=GRAY, group="ext",
                     vals=[f"{cv[k][0]:.2f}-{cv[k][1]:.2f}" for k in keys] + [f"{gaps.min():.2f}-{gaps.max():.2f}"], bold=[False] * 7))
    ext = {}
    for lab, mod in (("Binoculars", "Binoculars"), ("Fast-DetectGPT", "FastDetectGPT")):
        v, _, g = row_vals(mod, "-"); ext[lab] = (v, g)
        rows.append(dict(label=lab, color=C_OS, group="ext", vals=[_fmt(x) for x in v] + [_fmt(g)], bold=[False] * 7))
    own = [("Logistic regression", "LR", C_LR), ("RoBERTa", "RoBERTa", C_RB), ("Qwen 27B + LoRA", "Qwen27B", C_QW)]
    allv = {}
    for lab, mod, col in own:
        for arm, aname in (("A", "A: native only"), ("C", "C: + exam-style")):
            v, _, g = row_vals(mod, arm); allv[(mod, arm)] = (v, g)
    best = [max(allv[k][0][i] for k in allv) for i in range(6)]
    for lab, mod, col in own:
        for arm, aname in (("A", "A: native only"), ("C", "C: + exam-style")):
            v, g = allv[(mod, arm)]
            rows.append(dict(label=f"{lab}, {aname}", color=col, group="own", vals=[_fmt(x) for x in v] + [_fmt(g)],
                             bold=[round(v[i], 2) == round(best[i], 2) for i in range(6)] + [False]))
        tab += [dict(model=mod, arm=a, **{k: allv[(mod, a)][0][i] for i, k in enumerate(keys)}, gap=allv[(mod, a)][1]) for a in "AC"]
    fig = table_figure("fig10_prf1", "Precision, recall and F1: the right training data lifts every model",
                       "Liang test set: 394 real human essays and 352 AI essays; AI is the positive class; threshold 0.5",
                       rows, {"ext": "Existing detectors (not trained on our data)", "own": "Trained by us: arm A (native essays only) vs arm C (+ exam-style non-native essays)"},
                       fs=15, note="Precision: of essays flagged AI, the share that are AI. Recall: of AI essays, the share caught. F1 combines both. F1 on TOEFL / Hewlett counts AI and that human group equally. Quil and Sapling have no scores for 145 CS224N human essays (tested on 249). Bold = best of the six models; means over seeds (5; Qwen 2).")
    save(fig, "fig10_prf1", pd.DataFrame(tab + [dict(model="commercial_range", arm="-", **{k: f"{cv[k][0]:.3f}-{cv[k][1]:.3f}" for k in keys})]))

    # ---------- full version (every detector, every arm, mean +- SD) ----------
    rows = []
    for det in com.sort_values("f1__mean", ascending=False).itertuples():
        v = [getattr(det, f"{k}__mean") for k in keys]
        rows.append(dict(label=det.run, color=GRAY, group="com", vals=[_fmt(x) for x in v] + [_fmt(v[5] - v[4])], bold=[False] * 7))
    for lab, mod in (("Binoculars", "Binoculars"), ("Fast-DetectGPT", "FastDetectGPT")):
        v, _, g = row_vals(mod, "-"); rows.append(dict(label=lab, color=C_OS, group="ext2", vals=[_fmt(x) for x in v] + [_fmt(g)], bold=[False] * 7))
    names = {"A": "A native only", "B": "B + school-age ELL", "C": "C + exam-style", "D": "D + more native (control)"}
    for lab, mod, col in own:
        for arm in "ABCD":
            v, sd, g = row_vals(mod, arm, sd=True)
            n = int(P[(P.model == mod) & (P.arm.astype(str) == arm)]["n_seeds"].iloc[0])
            rows.append(dict(label=f"{lab}, {names[arm]}", color=col, group=mod, vals=[_fmt(x, s) for x, s in zip(v, sd)] + [_fmt(g)], bold=[False] * 7))
    groups = {"com": "Commercial detectors, 2023", "ext2": "Open-source detectors, 2024", "LR": "Logistic regression (5 seeds)", "RoBERTa": "RoBERTa (5 seeds)", "Qwen27B": "Qwen 27B + LoRA (2 seeds)"}
    fig = table_figure("fig11_prf1_full", "Backup: accuracy, precision, recall and F1 for every detector and arm",
                       "Liang test set: 394 real human essays and 352 AI essays; AI is the positive class; threshold 0.5; mean ± SD over seeds",
                       rows, groups, fs=11.5, note="Quil and Sapling have no scores for 145 CS224N human essays (tested on 249). Open-source detectors use their official thresholds.")
    save(fig, "fig11_prf1_full", P[["model", "arm", "run", "n_seeds"] + [c for c in P.columns if c.endswith("__mean") or c.endswith("__std")]])


# =====================================================================================================
# 12/13  Which words and sentences make an essay look like AI? (predictability heat map)
# =====================================================================================================
import re as _re
import statistics as _st
from matplotlib.colors import to_rgb as _rgb


def _words_with_nll(item):
    """Group falcon-7b token surprisals (nats) into words and sentences.
    Returns (words, sent_ppl): words = [(text, nll_sum, sentence_id, n_tokens, kind)] with a 'tag' entry after each sentence."""
    text = item["text"]; spans = [(m.start(), m.end(), m.group()) for m in _re.finditer(r"\S+", text)]
    nll = [0.0] * len(spans); cnt = [0] * len(spans)
    for t in item["tokens"]:
        a = t["s"]
        while a < len(text) and text[a].isspace():
            a += 1
        for i, (ws, we, _) in enumerate(spans):
            if ws <= a < we:
                nll[i] += t["nll"]; cnt[i] += 1; break
    words, sid = [], 0
    for (ws, we, w), v, c in zip(spans, nll, cnt):
        words.append([w, v, sid, c, "word"])
        if _re.search(r"[.!?][\"')]*$", w):
            sid += 1
    sent = {}
    for w, v, i, c, _ in words:
        n, k = sent.get(i, (0.0, 0)); sent[i] = (n + v, k + c)
    ppl = {i: float(np.exp(n / max(k, 1))) for i, (n, k) in sent.items()}
    out, last = [], None
    for idx, (w, v, i, c, kind) in enumerate(words):
        out.append((w, v, i, c, "word"))
        if idx + 1 == len(words) or words[idx + 1][2] != i:
            out.append((f"ppl {ppl[i]:.0f}", 0.0, i, 0, "tag"))
    return out, ppl


def _layout(fig, words, fs, width_px, space_px):
    r = fig.canvas.get_renderer(); probe = fig.text(0, 0, "x", fontsize=fs)
    def wd(w, kind):
        probe.set_fontsize(fs * (0.72 if kind == "tag" else 1.0)); probe.set_text(w); return probe.get_window_extent(r).width
    pos, x, line, n = [], 0.0, 0, len(words)
    for k, (w, v, sid, c, kind) in enumerate(words):
        wpx = wd(w, kind); need = wpx
        if kind == "word" and k + 1 < n and words[k + 1][4] == "tag":
            need = wpx + space_px + wd(words[k + 1][0], "tag")
        if x + need > width_px and x > 0:
            x = 0.0; line += 1
        pos.append((x, line, wpx)); x += wpx + space_px
    probe.remove()
    return pos, line + 1


def _heat_card(fig, rect, item, title, verdict, verdict_col, ppl_text, fonts=(15, 14.5, 14, 13.5, 13, 12.5, 12, 11.5, 11)):
    W, H = rect[2] * 1920, rect[3] * 1080
    ax = fig.add_axes(rect); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="white", edgecolor=GRID, linewidth=1.5))
    words, ppl = _words_with_nll(item)
    pad = 18
    for fs in fonts:
        space = fs * 120 / 72 * 0.30; lh = fs * 120 / 72 * 1.75
        pos, nl = _layout(fig, words, fs, W - 2 * pad, space)
        if nl * lh + 2 * pad - 10 <= H: break
    red = _rgb(C_TOEFL); low = min(ppl, key=ppl.get)
    for (w, v, sid, c, kind), (x, line, wpx) in zip(words, pos):
        y = pad - 6 + line * lh
        if kind == "word":
            al = float(np.clip((5.0 - v) / 4.3, 0, 1)) * 0.62
            ax.add_patch(Rectangle((pad + x - 1, y), wpx + space + 1, lh * 0.88, facecolor=(*red, al), edgecolor="none", zorder=1))
            ax.text(pad + x, y + lh * 0.44, w, fontsize=fs, color=INK, ha="left", va="center", zorder=3)
        else:
            hot = (sid == low)
            ax.text(pad + x, y + lh * 0.46, w, fontsize=fs * 0.72, color=C_TOEFL if hot else MUTED, ha="left", va="center", zorder=3, fontweight="bold" if hot else "normal")
    ax.text(0, -34, title, fontsize=15.5, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0, -9, verdict, fontsize=14, fontweight="bold", color=verdict_col, ha="left", va="bottom")
    ax.text(W, -9, ppl_text, fontsize=12.5, color=INK2, ha="right", va="bottom")
    return fs


def _heat_key(fig, x=0.04, y=0.842):
    kax = fig.add_axes([x + 0.095, y, 0.13, 0.014])
    kax.imshow(np.linspace(0, 1, 100)[None, :], aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list("k", [(*_rgb(C_TOEFL), 0.62), (*_rgb(C_TOEFL), 0.0)]))
    kax.set_xticks([]); kax.set_yticks([]); [sp.set_visible(False) for sp in kax.spines.values()]
    fig.text(x + 0.09, y + 0.007, "very predictable", fontsize=12, color=INK2, va="center", ha="right")
    fig.text(x + 0.231, y + 0.007, "surprising", fontsize=12, color=INK2, va="center", ha="left")
    fig.text(x + 0.30, y + 0.007, "ppl = sentence perplexity (low = predictable); the most predictable sentence is in red", fontsize=12, color=INK2, va="center", ha="left")


def _excerpt(item, max_words=120):
    text = item["text"]; best = None
    for m in _re.finditer(r"[.!?][\"'”’)]*(?=\s|$)", text):
        if len(text[:m.end()].split()) <= max_words: best = m.end()
    cut = best or len(text)
    return {"text": text[:cut], "tokens": [t for t in item["tokens"] if t["e"] <= cut]}, len(text[:cut].split()), len(text.split())


def _ppl(item):
    return float(np.exp(np.mean([t["nll"] for t in item["tokens"]])))


def _hot_inputs():
    surp = json.load(open(DATA / "hot_surprisal.json", encoding="utf-8"))
    def load(sub, c):
        b = LIANG / sub / c
        return [d["document"] for d in json.loads((b / "data.json").read_text(encoding="utf-8"))], {d: json.loads((b / f"{d}.json").read_text(encoding="utf-8")) for d in DETS}
    td, ts = load("Human_Data", "TOEFL_real_91"); cd, cs = load("Human_Data", "CollegeEssay_real_70"); pdc, ps = load("Human_Data", "TOEFL_gpt4polished_91")
    ci = 23
    assert td[84] == surp["toefl84"]["text"] and cd[ci] == surp["college"]["text"] and pdc[84] == surp["polished84"]["text"]
    w_ = lambda t: set(_re.findall(r"[a-z']+", t.lower())); a_ = w_(td[84]); b_ = w_(pdc[84])
    assert len(a_ & b_) / len(a_ | b_) > 0.25, "polished essay is not a rewrite of the TOEFL essay"
    n_flag = lambda sc, i: sum(float(sc[d][i]["score"]) >= 0.5 for d in DETS)
    flags = {"toefl84": n_flag(ts, 84), "college": n_flag(cs, ci), "polished84": n_flag(ps, 84)}
    ppl = {k: _ppl(v) for k, v in surp.items() if k in flags}
    return surp, ts, cs, flags, ppl


def fig_hotspots():
    surp, ts, cs, flags, ppl = _hot_inputs()
    ex_c, n_shown, n_all = _excerpt(surp["college"]); ppl["college"] = _ppl(ex_c)
    tp = [x["perplexity"] for x in ts["GPTZero"]]; cp = [x["perplexity"] for x in cs["GPTZero"]]
    tb = [x["overall_burstiness"] for x in ts["GPTZero"]]; cb = [x["overall_burstiness"] for x in cs["GPTZero"]]
    ft = [r["falcon_ppl"] for r in json.load(open(DATA / "all_ppl.json"))["TOEFL_real_91"]]
    fc = [r["falcon_ppl"] for r in json.load(open(DATA / "all_ppl_native.json"))["CollegeEssay_real_70"]]
    share = float(np.mean([v < _st.median(fc) for v in ft]))
    fig = new_fig("Which words make an essay look like AI? The ones a language model can predict",
                  "Perplexity-based detectors score how predictable the wording is. Shading: the more predictable the word, the darker (same scale on both essays)")
    _heat_key(fig, y=0.838)
    _heat_card(fig, [0.04, 0.455, 0.60, 0.315], surp["toefl84"], "TOEFL essay, non-native writer", f"Flagged as AI by {flags['toefl84']} of 7 detectors", C_TOEFL, f"language-model perplexity {ppl['toefl84']:.1f}")
    _heat_card(fig, [0.04, 0.105, 0.60, 0.285], ex_c, f"College application essay, native writer (first {n_shown} of {n_all} words)", f"Flagged as AI by {flags['college']} of 7 detectors", INK2, f"language-model perplexity {ppl['college']:.1f}")
    tiles = (("Perplexity", _st.median(tp), _st.median(cp), "lower = more predictable", f"falcon-7b agrees: {share:.0%} of TOEFL essays are more predictable than the median native essay"),
             ("Burstiness", _st.median(tb), _st.median(cb), "lower = more uniform sentences", "TOEFL sentences are more alike in length and wording"))
    for k, (name, tv, hv, note, extra) in enumerate(tiles):
        y0 = [0.455, 0.105][k]; h = [0.315, 0.285][k]
        tile = fig.add_axes([0.67, y0, 0.29, h]); tile.axis("off"); tile.set_xlim(0, 1); tile.set_ylim(0, 1)
        tile.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID, linewidth=1.5))
        tile.text(0.06, 0.90, f"{name}: what GPTZero itself reported", fontsize=14, fontweight="bold", color=INK, va="top")
        tile.text(0.06, 0.77, f"median over all essays; {note}", fontsize=11.5, color=INK2, va="top")
        for j, (val, lab, col) in enumerate(((tv, "91 TOEFL essays", C_TOEFL), (hv, "70 native college essays", INK2))):
            xc = 0.27 + j * 0.46
            tile.text(xc, 0.50, f"{val:.0f}", fontsize=44, color=col, ha="center", va="center", fontweight="bold")
            tile.text(xc, 0.27, lab, fontsize=11.5, color=INK2, ha="center", va="center")
        tile.text(0.06, 0.14, textwrap.fill(extra, 50), fontsize=11, color=INK2, va="center", linespacing=1.3)
    footer(fig, "fig12_hotspots", "Shading: falcon-7b word probabilities (the model behind Binoculars and Fast-DetectGPT). Illustrative pair (the native essay sits at the median GPTZero perplexity); medians use all essays. Hewlett essays are not used: 85 of 88 contain placeholders (@CAPS1) that inflate perplexity.")
    save(fig, "fig12_hotspots", pd.DataFrame([dict(item=k, flagged_by_of_7=flags[k], falcon_perplexity=ppl[k]) for k in flags] +
                                              [dict(item="GPTZero median perplexity TOEFL", value=_st.median(tp)), dict(item="GPTZero median perplexity native college", value=_st.median(cp)),
                                               dict(item="GPTZero median burstiness TOEFL", value=_st.median(tb)), dict(item="GPTZero median burstiness native college", value=_st.median(cb)),
                                               dict(item="share of TOEFL essays below median native falcon perplexity", value=share)]))


def fig_polish():
    surp, ts, cs, flags, ppl = _hot_inputs()
    fig = new_fig("The same essay, polished by GPT-4: from 7 of 7 detectors to 0 of 7",
                  "GPT-4 replaced common phrases with rarer ones. Shading: the more predictable the word, the darker (same scale on both)")
    _heat_key(fig, y=0.838)
    fonts = (16, 15.5, 15, 14.5, 14, 13.5, 13)
    _heat_card(fig, [0.04, 0.285, 0.46, 0.485], surp["toefl84"], "Original, written by a non-native student", f"Flagged as AI by {flags['toefl84']} of 7 detectors", C_TOEFL, f"language-model perplexity {ppl['toefl84']:.1f}", fonts=fonts)
    _heat_card(fig, [0.52, 0.285, 0.44, 0.485], surp["polished84"], "After GPT-4 polishing", f"Flagged as AI by {flags['polished84']} of 7 detectors", INK2, f"language-model perplexity {ppl['polished84']:.1f}", fonts=fonts)
    changed = 'What changed:  "important" became "crucial";  "would be going to" became "attending";  "always been looking forward to since I was a little kid" became "a lifelong dream since my childhood".'
    for i, ln in enumerate(textwrap.wrap(changed, 150)):
        fig.text(0.04, 0.225 - 0.034 * i, ln, fontsize=14.5, color=INK, va="center")
    fig.text(0.04, 0.150, "The detectors react to how plain the wording is, not to who wrote it: fancier words pass, simple correct English gets flagged.", fontsize=14.5, color=INK2, va="center", fontweight="bold")
    footer(fig, "fig13_polish", "Same essay as fig12; the polished text comes from Liang et al. (2023) (TOEFL_gpt4polished_91, matched to the original by word overlap). Shading from falcon-7b word probabilities.")
    save(fig, "fig13_polish", pd.DataFrame([dict(item=k, flagged_by_of_7=flags[k], falcon_perplexity=ppl[k]) for k in ("toefl84", "polished84")]))


# =====================================================================================================
# 14  Pipeline: how the experiment was built
# =====================================================================================================
def fig_pipeline():
    fig = new_fig("How the experiment was built: from raw essays to a test with no overlap",
                  "Every number in this talk comes from this pipeline; the same rules were applied to all models")
    ax = fig.add_axes([0.04, 0.215, 0.92, 0.62]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    steps = [
        ("1", "Collect", "8 sources", ["Liang et al. 925 essays", "+ 7 new: PERSUADE 2.0, ELLIPSE, daigt_v4, HC-Var, W&I + LOCNESS, FCE, ArguGPT"]),
        ("2", "Clean", "same rules", ["Line breaks, quote marks, garbled characters and look-alike letters made identical in every text", "Check: no formatting difference left between groups"]),
        ("3", "Remove duplicates", "0 overlap", ["Full text and first 200 characters compared", "452 PERSUADE essays that are also in ELLIPSE removed", "Check: no essay in both training and test"]),
        ("4", "Split", "10,171 test essays", ["Train: 4,000 human + 4,000 AI essays per arm", "Validation: 600 essays", "Test: never used for training"]),
        ("5", "Train", "3 models x 4 arms", ["Logistic regression (5 seeds), RoBERTa (5), Qwen 27B + LoRA (2)", "Same data and settings within a model family"]),
        ("6", "Evaluate", "threshold 0.5", ["False-flag rate, bias AUC, precision, recall, F1", "Main measures written down and time-stamped before the GPU runs"]),
    ]
    bw, bh, gx = 30.0, 46.5, 5.0
    for k, (num, head, big, lines) in enumerate(steps):
        col, row = k % 3, k // 3
        x0 = col * (bw + gx); y0 = (1 - row) * (bh + 4.0) + 0.5
        ax.add_patch(Rectangle((x0, y0), bw, bh, facecolor="white", edgecolor=GRID, linewidth=1.5))
        ax.add_patch(Rectangle((x0, y0 + bh - 1.2), bw, 1.2, facecolor=INK2, linewidth=0))
        ax.text(x0 + 1.6, y0 + bh - 6.0, num, fontsize=20, fontweight="bold", color=MUTED, va="center")
        ax.text(x0 + 4.6, y0 + bh - 6.0, head, fontsize=18, fontweight="bold", color=INK, va="center")
        ax.text(x0 + 1.6, y0 + bh - 14.5, big, fontsize=23, fontweight="bold", color=INK, va="center")
        yy = y0 + bh - 21.5
        for ln in lines:
            wrapped = textwrap.wrap(ln, 50)
            ax.text(x0 + 1.6, yy, "\n".join(wrapped), fontsize=11.8, color=INK2, va="top", linespacing=1.28)
            yy -= 4.6 * len(wrapped) + 1.2
        if col < 2:
            ax.annotate("", xy=(x0 + bw + gx - 0.5, y0 + bh / 2), xytext=(x0 + bw + 0.4, y0 + bh / 2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8))
    fig.text(0.04, 0.155, "Four training sets (arms):  A native only  |  B + school-age non-native  |  C + exam-style non-native  |  D + other native types (control)",
             fontsize=13.5, color=INK, va="center", fontweight="bold")
    footer(fig, "fig14_pipeline", "Test essays: 13,672 in total, minus the half of the FCE and W&I essays that arm C trains on (10,171 left). The checks behind steps 2 and 3 are in the reproducibility folder of the package.")
    save(fig, "fig14_pipeline", pd.DataFrame([dict(step=n, name=h, headline=b) for n, h, b, _ in steps]))


# =====================================================================================================
# 15  Four pitfalls we found and fixed
# =====================================================================================================
def fig_pitfalls():
    rows = [
        ("1", "Our first model looked strong", "0.890 vs 0.888", "AUC of our model vs just guessing which corpus an essay came from",
         "We stopped using it as evidence and tested detectors on seven new data sources."),
        ("2", "RoBERTa learned a formatting shortcut: no paragraph breaks and curly quotes meant AI", "0.04 to 0.91", "AUC for AI vs human essays on the same prompts, before and after cleaning",
         "We made every text look the same and added a check that no formatting difference is left."),
        ("3", "Qwen was tested on essays it had trained on", "0% to 40%", "TOEFL essays flagged as AI (2 seeds: 56% and 23%)",
         "We kept TOEFL out of training, retrained and re-tested."),
        ("4", "Our native comparison was unfair: it looked like the training essays", "2% vs 80-100%", "native essays flagged: training-like vs new style (logistic regression, RoBERTa)",
         "We added arms C and D and tested several kinds of native writing."),
    ]
    fig = new_fig("Four mistakes we found, and what we changed",
                  "We kept the failed runs and re-ran everything; the numbers on the right are from the final runs")
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.72]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(3, 98.5, "What went wrong", fontsize=14, color=MUTED, fontweight="bold", va="top")
    ax.text(41, 98.5, "The number", fontsize=14, color=MUTED, fontweight="bold", va="top")
    ax.text(70, 98.5, "What we changed", fontsize=14, color=MUTED, fontweight="bold", va="top")
    rh, gap = 21.0, 2.6
    for k, (num, wrong, big, cap, fix) in enumerate(rows):
        y0 = 92 - (k + 1) * rh - k * gap + 2
        ax.add_patch(Rectangle((0, y0), 100, rh, facecolor="white", edgecolor=GRID, linewidth=1.5))
        ax.add_patch(Rectangle((0, y0), 0.9, rh, facecolor=INK2, linewidth=0))
        ax.text(2.8, y0 + rh / 2, num, fontsize=30, color=MUTED, fontweight="bold", va="center")
        ax.text(7.0, y0 + rh / 2, "\n".join(textwrap.wrap(wrong, 30)), fontsize=15.5, color=INK, fontweight="bold", va="center", linespacing=1.3)
        ax.text(41, y0 + rh * 0.70, big, fontsize=28, color=INK, fontweight="bold", va="center")
        ax.text(41, y0 + rh * 0.34, "\n".join(textwrap.wrap(cap, 44)), fontsize=11.5, color=INK2, va="center", linespacing=1.25)
        ax.text(70, y0 + rh / 2, "\n".join(textwrap.wrap(fix, 36)), fontsize=14, color=INK, va="center", linespacing=1.35)
    footer(fig, "fig15_pitfalls", "Sources: the archived first runs and the final runs (arm A and B, threshold 0.5). 0.888 was estimated from the per-corpus flag rates of the first analysis.")
    save(fig, "fig15_pitfalls", pd.DataFrame([dict(pitfall=n, what_went_wrong=w, number=b, caption=c, change=f) for n, w, b, c, f in rows]))


# =====================================================================================================
# 16  How we used ML, deep learning and LoRA (method slide; the brief asks to explain models and why)
# =====================================================================================================
def fig_models():
    cards = [
        ("Logistic regression", C_LR, "CLASSICAL ML  |  course method",
         "Counts words and word pairs (TF-IDF, 20,000 features) and gives each one a weight",
         "8,000 essays per arm (4,000 human + 4,000 AI). 5 random seeds",
         "Simple and fast, and it shows which words push the score up",
         "Seconds on a laptop CPU"),
        ("RoBERTa", C_RB, "DEEP LEARNING  |  extension",
         "Reads the whole sentence in context. All 125 million parameters are updated",
         "The same 8,000 essays. 2 passes, up to 512 words each. 5 seeds",
         "The standard deep-learning text classifier: shows what context adds",
         "About 4 minutes on one GPU"),
        ("Qwen 27B + LoRA", C_QW, "LARGE LANGUAGE MODEL  |  extension",
         "A 27-billion-parameter model that has read huge amounts of text. We keep it frozen",
         "Only a small add-on is trained (LoRA): 125 million numbers, 0.48% of the model. The same 8,000 essays, 1 pass, 2 seeds",
         "LoRA is what makes a 27B model trainable on two 46 GB GPUs",
         "About 4 hours on two GPUs (2.9 h training + 74 min scoring)"),
    ]
    rows = [("How it reads an essay", 3), ("What we trained", 4), ("Why we chose it", 3), ("Cost", 2)]
    fig = new_fig("How we used machine learning, deep learning and LoRA",
                  "Same input and output for all three: an essay goes in, the probability that it is AI comes out (flagged at 0.5)")
    ax = fig.add_axes([0.04, 0.215, 0.92, 0.625]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    cw, gap = 31.0, 3.5
    heights = [22, 27, 19, 15]
    for k, (name, col, tag, *vals) in enumerate(cards):
        x0 = k * (cw + gap)
        ax.add_patch(Rectangle((x0, 0), cw, 100, facecolor="white", edgecolor=GRID, linewidth=1.5))
        ax.add_patch(Rectangle((x0, 91.5), cw, 8.5, facecolor=col, linewidth=0))
        ax.text(x0 + 1.6, 95.75, name, fontsize=18, fontweight="bold", color="white", va="center")
        ax.text(x0 + 1.6, 87.6, tag, fontsize=11.5, color=INK2, va="center", fontweight="bold")
        y = 83.0
        for (lab, _), v, h in zip(rows, vals, heights):
            ax.text(x0 + 1.6, y, lab.upper(), fontsize=10.5, color=MUTED, va="top", fontweight="bold")
            ax.text(x0 + 1.6, y - 3.4, "\n".join(textwrap.wrap(v, 38)), fontsize=13.2, color=INK, va="top", linespacing=1.28)
            y -= h
            if lab != "Cost":
                ax.plot([x0 + 1.6, x0 + cw - 1.6], [y + 1.6, y + 1.6], color=GRID, linewidth=1)
    fig.text(0.04, 0.182, "LoRA in one line:  freeze the big model, add a small trainable layer next to it, and train only that layer.",
             fontsize=14, color=INK, va="center", fontweight="bold")
    fig.text(0.04, 0.143, "Tuning: none, on purpose.  Settings were fixed in advance, so the training essays are the only thing that differs between arms.",
             fontsize=14, color=INK, va="center", fontweight="bold")
    footer(fig, "fig16_models", "All settings are listed in the backup slide. The 600-essay validation set only sets a second threshold (catch 90% of AI); the main results use 0.5. Deep learning is an extension the instructor approved verbally.")
    save(fig, "fig16_models", pd.DataFrame([dict(model=c[0], family=c[2], reads=c[3], trained=c[4], why=c[5], cost=c[6]) for c in cards]))


# =====================================================================================================
# 17  Backup: every setting, and what we did not tune
# =====================================================================================================
def fig_settings():
    cols = [("Logistic regression", C_LR), ("RoBERTa", C_RB), ("Qwen 27B + LoRA", C_QW)]
    rows = [
        ("What it is", ["TF-IDF of words and word pairs (20,000 features, rare words dropped) + logistic regression", "roberta-base, all 125 million parameters trained", "Qwen3.8-27B, frozen; LoRA add-on on all linear layers (rank 16, alpha 32, dropout 0.05)"]),
        ("Training settings", ["Same as the course notebook; up to 2,000 solver steps", "Learning rate 2e-5, batch 16, 2 passes, up to 512 words", "Learning rate 1e-4, batch 8 x 2, 1 pass (500 steps), up to 512 words"]),
        ("Where the settings came from", ["Course notebook defaults", "Common values for fine-tuning; passes cut from 4 to 2 when training grew from 220 to 8,000 essays", "Common LoRA values; passes cut from 3 to 1 when training grew from 267 to 8,000 essays"]),
        ("Tuned on validation data?", ["No", "No", "No"]),
        ("Random seeds", ["5", "5", "2 (too slow for more)"]),
    ]
    fig = new_fig("Backup: every setting, and what we did not tune",
                  "Why not tune: it keeps the four training sets (A to D) as the only difference; a search per arm could hide or create differences")
    ax = fig.add_axes([0.04, 0.215, 0.92, 0.62]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    lx, cx, cw = 0, 22.0, 25.5
    for k, (name, col) in enumerate(cols):
        x0 = cx + k * (cw + 1.2)
        ax.add_patch(Rectangle((x0, 91.5), cw, 8.5, facecolor=col, linewidth=0))
        ax.text(x0 + 1.4, 95.75, name, fontsize=15.5, fontweight="bold", color="white", va="center")
    hs = [20, 20, 22, 9, 10]; y = 90.0
    for (lab, vals), h in zip(rows, hs):
        ax.plot([0, 100], [y, y], color=GRID, linewidth=1)
        ax.text(lx + 0.6, y - h / 2 + 0.5, lab, fontsize=13.5, fontweight="bold", color=INK, va="center")
        for k, v in enumerate(vals):
            x0 = cx + k * (cw + 1.2)
            bold = (lab == "Tuned on validation data?")
            ax.text(x0 + 1.0, y - h / 2 + 0.5, "\n".join(textwrap.wrap(v, 33)), fontsize=18 if bold else 12.8, color=INK, va="center", linespacing=1.3, fontweight="bold" if bold else "normal")
        y -= h
    ax.plot([0, 100], [y, y], color=GRID, linewidth=1)
    fig.text(0.04, 0.165, "What the 600-essay validation set was used for:  a second threshold that catches 90% of AI essays. The main results use 0.5.", fontsize=14, color=INK, va="center", fontweight="bold")
    fig.text(0.04, 0.128, "What this costs:  results could change with tuning. Seeds show run-to-run spread only, not the effect of other settings.", fontsize=14, color=INK, va="center", fontweight="bold")
    footer(fig, "fig17_settings", "Qwen weights were converted from FP8 to bf16 before training (about 2.7% weight error, checked against an original bf16 shard). Logistic regression uses default regularisation.")
    save(fig, "fig17_settings", pd.DataFrame([dict(row=lab, **{c[0]: v for c, v in zip(cols, vals)}) for lab, vals in rows]))


# =====================================================================================================
# 18-23  One method slide and one results slide per technique (ML, DL, LoRA)
# =====================================================================================================
def _flow_box(ax, x, y, w, h, title, sub, state, col, big=False):
    trained = state == "trained"
    face = (*_rgb(col), 0.14) if trained else "white"
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=col if trained else BASE, linewidth=2.4 if trained else 1.5))
    ax.text(x + w / 2, y + h * 0.66, "\n".join(textwrap.wrap(title, 17)), fontsize=14.5, fontweight="bold", color=INK, ha="center", va="center", linespacing=1.2)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, "\n".join(textwrap.wrap(sub, 20)), fontsize=11.5, color=INK2, ha="center", va="center", linespacing=1.2)
    tag = {"trained": "TRAINED", "frozen": "FROZEN", "fixed": ""}[state]
    if tag:
        ax.text(x + w / 2, y - 5.5, tag, fontsize=11.5, fontweight="bold", color=col if trained else MUTED, ha="center", va="center")


def method_figure(name, title, sub, col, boxes, cells, note, df_rows):
    fig = new_fig(title, sub)
    ax = fig.add_axes([0.04, 0.50, 0.92, 0.32]); ax.set_xlim(0, 100); ax.set_ylim(-12, 100); ax.axis("off")
    n = len(boxes); gap = 4.4; w = (100 - gap * (n - 1)) / n; h = 62; y = 18
    for i, (t, sb, st) in enumerate(boxes):
        x = i * (w + gap); _flow_box(ax, x, y, w, h, t, sb, st, col)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - 0.4, y + h / 2), xytext=(x + w + 0.4, y + h / 2), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8))
    ax2 = fig.add_axes([0.04, 0.17, 0.92, 0.29]); ax2.set_xlim(0, 100); ax2.set_ylim(0, 100); ax2.axis("off")
    cw = 31.0; g2 = 3.5
    for k, (head, text) in enumerate(cells):
        x0 = k * (cw + g2)
        ax2.add_patch(Rectangle((x0, 0), cw, 100, facecolor="white", edgecolor=GRID, linewidth=1.5))
        ax2.add_patch(Rectangle((x0, 86), cw, 14, facecolor=col, linewidth=0))
        ax2.text(x0 + 1.6, 93, head, fontsize=15, fontweight="bold", color="white", va="center")
        ax2.text(x0 + 1.6, 80, "\n".join(textwrap.wrap(text, 40)), fontsize=14.5, color=INK, va="top", linespacing=1.32)
    footer(fig, name, note)
    save(fig, name, pd.DataFrame(df_rows))


def results_figure(name, title, sub, col, model, seeds_note, extra):
    CM = pd.read_csv(DATA / "model_comparison_all.csv").set_index(["model", "arm"]); PF = pd.read_csv(DATA / "prf1_summary.csv")
    arms = ["A", "B", "C", "D"]; names = ["A\\nnative only", "B\\n+ school-age\\nnon-native", "C\\n+ exam-style\\nnon-native", "D (control)\\n+ other native"]
    fig = new_fig(title, sub)
    swatch_legend(fig, [("Native, like the training essays", GRAY, "bar"), ("Native, unfamiliar kind", C_OS, "bar"), ("TOEFL (non-native exam essays)", C_TOEFL, "bar")], y=0.835, fs=13)
    ax = fig.add_axes([0.07, 0.215, 0.55, 0.565]); ax.set_facecolor(SURF); out = []
    bw = 0.26
    for i, a in enumerate(arms):
        vals = [(CM.loc[(model, a), "FP_Hewlett"], GRAY, -bw), (CM.loc[(model, a), "FP_nativeOOD_mean"], C_OS, 0), (CM.loc[(model, a), "FP_TOEFL"], C_TOEFL, bw)]
        for v, c, off in vals:
            v *= 100
            ax.bar(i + off * 1.08, v, width=bw * 1.0, color=c, zorder=3)
            ax.text(i + off * 1.08, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=12.5, color=INK2, path_effects=HALO, zorder=5)
        out.append(dict(arm=a, hewlett=vals[0][0] * 100, native_unfamiliar=vals[1][0] * 100, toefl=vals[2][0] * 100))
    ax.set_xticks(range(4)); ax.set_xticklabels([n.replace("\\n", "\n") for n in names], fontsize=12.5, color=INK)
    ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASE); ax.tick_params(axis="y", colors=INK2, labelsize=13, length=0); ax.tick_params(axis="x", length=0, pad=8)
    # right panel: precision / recall / F1 on the Liang test set
    rx = fig.add_axes([0.66, 0.215, 0.30, 0.565]); rx.axis("off"); rx.set_xlim(0, 100); rx.set_ylim(0, 100)
    rx.add_patch(Rectangle((0, 0), 100, 100, facecolor="white", edgecolor=GRID, linewidth=1.5)); rx.add_patch(Rectangle((0, 94), 100, 6, facecolor=col, linewidth=0))
    rx.text(4, 97, "Catching AI essays (Liang test set)", fontsize=13.5, fontweight="bold", color="white", va="center")
    rx.text(4, 88, "394 human + 352 AI essays", fontsize=11.5, color=INK2, va="center")
    xs = [34, 56, 78]
    for x, hd in zip(xs, ("Precision", "Recall", "F1")):
        rx.text(x, 79, hd, fontsize=12.5, fontweight="bold", color=INK2, ha="center", va="center")
    rx.plot([2, 98], [74, 74], color=BASE, linewidth=1.2)
    f1s = [float(PF[(PF.model == model) & (PF.arm.astype(str) == a)]["f1__mean"].iloc[0]) for a in arms]
    for r, a in enumerate(arms):
        y = 64 - r * 15.5
        row = PF[(PF.model == model) & (PF.arm.astype(str) == a)].iloc[0]
        rx.text(4, y, a, fontsize=16, fontweight="bold", color=INK, va="center")
        vals = [row["precision__mean"], row["recall__mean"], row["f1__mean"]]
        for x, v, k in zip(xs, vals, range(3)):
            best = (k == 2 and round(v, 2) == round(max(f1s), 2))
            rx.text(x, y, f"{v:.2f}", fontsize=17 if best else 15, fontweight="bold" if best else "normal", color=INK if best else INK2, ha="center", va="center")
        rx.plot([2, 98], [y - 7.7, y - 7.7], color=GRID, linewidth=1)
        out[r].update(precision=float(row["precision__mean"]), recall=float(row["recall__mean"]), f1=float(row["f1__mean"]))
    rx.text(4, 3.5, "Bold F1 = best arm.  A-D as on the left.", fontsize=11, color=INK2, va="center")
    a_, c_ = CM.loc[(model, "A"), "FP_nativeOOD_mean"] * 100, CM.loc[(model, "C"), "FP_nativeOOD_mean"] * 100
    footer(fig, name, f"Native essays of an unfamiliar kind: {a_:.0f}% flagged in arm A, {c_:.0f}% in arm C (mean of CollegeEssay, CS224N and LOCNESS). " + extra + " " + seeds_note)
    save(fig, name, pd.DataFrame(out))


def fig_methods():
    # ---------------- ML ----------------
    method_figure("fig18_ml_method", "Machine learning: logistic regression on word counts",
                  "The classical method from the course: simple, fast, and it shows which words push the score up", C_LR,
                  [("Essay", "plain text", "fixed"), ("Count words and word pairs", "TF-IDF, 20,000 features, rare words dropped", "fixed"),
                   ("Logistic regression", "one weight per feature", "trained"), ("Probability that it is AI", "between 0 and 1", "fixed"), ("Flag as AI", "if 0.5 or more", "fixed")],
                  [("What we trained", "8,000 essays per arm: 4,000 human and 4,000 AI. Four versions of the human half (A to D). 5 random seeds. Settings as in the course notebook."),
                   ("Why we chose it", "It is the method taught in the course, trains in seconds on a laptop, and we can read which words raise the score. It is our main model."),
                   ("What it cannot do", "It only counts words. It does not understand sentences, so unfamiliar topics and styles easily confuse it.")],
                  "Logistic regression with default regularisation. Settings were not tuned (see the backup slide).",
                  [dict(step="essay"), dict(step="TF-IDF 20,000 features"), dict(step="logistic regression (trained)"), dict(step="probability"), dict(step="flag at 0.5")])
    # ---------------- DL ----------------
    method_figure("fig20_dl_method", "Deep learning: RoBERTa reads the sentence in context",
                  "A pre-trained language model that we fine-tune completely: every one of its 125 million parameters is updated", C_RB,
                  [("Essay", "plain text", "fixed"), ("Split into word pieces", "up to 512 pieces", "fixed"),
                   ("12 transformer layers", "all 125 million parameters", "trained"), ("Classification head", "AI or human", "trained"), ("Flag as AI", "if 0.5 or more", "fixed")],
                  [("What we trained", "The same 8,000 essays per arm. 2 passes over the data, learning rate 2e-5, batch of 16. 5 random seeds. About 4 minutes on one GPU."),
                   ("Why we chose it", "It is the standard deep-learning text classifier, so it shows what reading context adds on top of word counts. It is a deep-learning extension."),
                   ("What it cannot do", "It cannot explain its score, and its results change more from run to run than logistic regression.")],
                  "roberta-base; learning rate 2e-5, batch 16, 512 tokens, 2 epochs; settings not tuned. Deep learning is an extension the instructor approved verbally.",
                  [dict(step="essay"), dict(step="word pieces"), dict(step="12 transformer layers (trained)"), dict(step="classification head (trained)"), dict(step="flag at 0.5")])
    # ---------------- LoRA (frozen block + small trained add-on) ----------------
    fig = new_fig("Large language model + LoRA: train a small add-on, keep the 27B model frozen",
                  "A 27-billion-parameter model cannot be fully retrained on two 46 GB GPUs, so LoRA trains only about 0.5% of it")
    ax = fig.add_axes([0.04, 0.50, 0.92, 0.32]); ax.set_xlim(0, 100); ax.set_ylim(-12, 100); ax.axis("off")
    _flow_box(ax, 0, 18, 15, 62, "Essay", "plain text", "fixed", C_QW)
    ax.annotate("", xy=(19.2, 49), xytext=(15.4, 49), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8))
    ax.add_patch(Rectangle((19.6, 18), 38, 62, facecolor="white", edgecolor=BASE, linewidth=1.5))
    ax.text(38.6, 66, "Qwen3.8-27B", fontsize=15.5, fontweight="bold", color=INK, ha="center", va="center")
    ax.text(38.6, 56, "64 layers, 27 billion parameters", fontsize=12, color=INK2, ha="center", va="center")
    ax.text(38.6, 11.2, "FROZEN", fontsize=11.5, fontweight="bold", color=MUTED, ha="center", va="center")
    for k, xx in enumerate((23.0, 33.0, 43.0, 53.0)):
        ax.add_patch(Rectangle((xx, 24), 3.2, 24, facecolor=BASE, edgecolor="none"))
        ax.add_patch(Rectangle((xx + 3.6, 28), 1.9, 16, facecolor=(*_rgb(C_QW), 0.9), edgecolor=SURF, linewidth=1.5))
    ax.text(40.9, 36.5, "...", fontsize=18, color=MUTED, ha="center", va="center")
    ax.add_patch(Rectangle((60.2, 50), 13.5, 34, facecolor=(*_rgb(C_QW), 0.14), edgecolor=C_QW, linewidth=2.4))
    ax.text(66.95, 76, "LoRA add-on", fontsize=13.5, fontweight="bold", color=INK, ha="center", va="center"); ax.text(66.95, 66.2, "125 million numbers\n0.48% of the model", fontsize=10.8, color=INK2, ha="center", va="center", linespacing=1.2)
    ax.text(66.95, 53.6, "TRAINED", fontsize=11, fontweight="bold", color=C_QW, ha="center", va="center")
    ax.text(59.0, 66, "+", fontsize=20, fontweight="bold", color=C_QW, ha="center", va="center")
    ax.add_patch(Rectangle((60.2, 8), 13.5, 34, facecolor=(*_rgb(C_QW), 0.14), edgecolor=C_QW, linewidth=2.4))
    ax.text(66.95, 33, "Classification head", fontsize=12.5, fontweight="bold", color=INK, ha="center", va="center"); ax.text(66.95, 22.5, "AI or human", fontsize=10.8, color=INK2, ha="center", va="center")
    ax.text(66.95, 13.2, "TRAINED", fontsize=11, fontweight="bold", color=C_QW, ha="center", va="center")
    ax.annotate("", xy=(77.0, 49), xytext=(73.9, 49), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8))
    _flow_box(ax, 77.4, 18, 10.5, 62, "Probability", "that it is AI", "fixed", C_QW)
    ax.annotate("", xy=(91.4, 49), xytext=(88.2, 49), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.8))
    _flow_box(ax, 91.8, 18, 8.2, 62, "Flag", "at 0.5", "fixed", C_QW)
    ax2 = fig.add_axes([0.04, 0.17, 0.92, 0.29]); ax2.set_xlim(0, 100); ax2.set_ylim(0, 100); ax2.axis("off")
    cells = [("What we trained", "The same 8,000 essays per arm, 1 pass (500 steps), learning rate 1e-4, LoRA rank 16. Only 2 random seeds, because one run takes about 4 hours on two GPUs."),
             ("Why we chose it", "A large language model has read huge amounts of text, so it can recognise subtle patterns. LoRA is what makes training it possible on our hardware. It is a deep-learning extension."),
             ("What it cannot do", "It is slow and expensive, it cannot explain its score, and with only 2 seeds its results are less certain.")]
    for k, (head, text) in enumerate(cells):
        x0 = k * 34.5
        ax2.add_patch(Rectangle((x0, 0), 31, 100, facecolor="white", edgecolor=GRID, linewidth=1.5)); ax2.add_patch(Rectangle((x0, 86), 31, 14, facecolor=C_QW, linewidth=0))
        ax2.text(x0 + 1.6, 93, head, fontsize=15, fontweight="bold", color="white", va="center")
        ax2.text(x0 + 1.6, 80, "\n".join(textwrap.wrap(text, 40)), fontsize=14.5, color=INK, va="top", linespacing=1.32)
    footer(fig, "fig22_lora_method", "Qwen3.8-27B (weights converted from FP8 to bf16), LoRA rank 16, alpha 32, dropout 0.05, all linear layers, learning rate 1e-4, batch 8 x 2; settings not tuned. Instructor approved deep learning verbally.")
    save(fig, "fig22_lora_method", pd.DataFrame([dict(step=x) for x in ("essay", "Qwen 27B frozen", "LoRA add-on trained", "classification head trained", "probability", "flag at 0.5")]))
    # ---------------- results ----------------
    results_figure("fig19_ml_results", "Logistic regression: exam-style essays cut TOEFL false flags from 80% to 8%",
                   "Share of real human essays flagged as AI, by what the human half of the training data contained (threshold 0.5, mean of 5 seeds)", C_LR, "LR",
                   "5 seeds.", "Arm D adds other native essays and does not help TOEFL (47%).")
    results_figure("fig21_dl_results", "RoBERTa: exam-style essays cut TOEFL false flags from 99% to 26%",
                   "Share of real human essays flagged as AI, by what the human half of the training data contained (threshold 0.5, mean of 5 seeds; arm C varies by 8 points)", C_RB, "RoBERTa",
                   "5 seeds.", "Adding school-age non-native essays (B) barely helps; exam-style essays (C) do.")
    results_figure("fig23_lora_results", "Qwen 27B + LoRA: best at catching AI; TOEFL false flags fall from 40% to 5%",
                   "Share of real human essays flagged as AI, by what the human half of the training data contained (threshold 0.5, mean of only 2 seeds)", C_QW, "Qwen27B",
                   "Only 2 seeds: in arm A the TOEFL rate was 56% in one run and 23% in the other.", "The effect size is indicative; the direction matches the other two models.")


if __name__ == "__main__":
    only = sys.argv[3:] or None
    for name, fn in [("hook", fig_hook), ("commercial", fig_commercial), ("opensource", fig_opensource), ("shift", fig_shift), ("trend", fig_trend),
                     ("scoreboard", fig_scoreboard), ("notfixed", fig_notfixed), ("learning", fig_learning), ("length", fig_length), ("prf1", fig_prf1), ("hotspots", fig_hotspots), ("polish", fig_polish), ("pipeline", fig_pipeline), ("pitfalls", fig_pitfalls), ("models", fig_models), ("settings", fig_settings), ("methods", fig_methods)]:
        if only is None or name in only:
            fn()
