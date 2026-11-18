"""Plotting helpers (matplotlib only). Simplified from the slide script make_figures.py: same colours and layout
ideas, smaller figures that read well inside a notebook. Every function returns the matplotlib Figure."""
import matplotlib.pyplot as plt
import numpy as np

# colour follows the entity (same palette as the slides)
SURF, INK, INK2, MUTED, GRID, BASE, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#b4b2ab"
C_LR, C_RB, C_QW = "#2a78d6", "#eb6834", "#1baf7a"
C_OS, C_TOEFL = "#6250d6", "#e34948"
MODELS = [("LR", "Logistic regression", C_LR), ("RoBERTa", "RoBERTa", C_RB), ("Qwen27B", "Qwen 27B + LoRA", C_QW)]
ARM_LABELS = {"A": "A\nnative only", "B": "B\n+ school-age\nnon-native", "C": "C\n+ exam-style\nnon-native",
              "D": "D (control)\n+ other native"}


def setup():
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 12, "axes.titlesize": 14,
                         "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURF,
                         "figure.facecolor": SURF, "axes.edgecolor": BASE, "axes.unicode_minus": False})


def _pct_axis(ax, top=105):
    ax.set_ylim(0, top); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1); ax.set_axisbelow(True)


def flags_by_arm(cm, models=MODELS, arms="ABCD", seeds_note=None):
    """Three panels (one per model). For each training arm: share of human essays flagged as AI at threshold 0.5,
    for native essays like the training data (Hewlett), unfamiliar native essays (mean of CollegeEssay, CS224N,
    LOCNESS) and TOEFL non-native essays. `cm`: table indexed by (model, arm) with FP_Hewlett, FP_nativeOOD_mean, FP_TOEFL."""
    setup()
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5.2), sharey=True)
    groups = [("FP_Hewlett", "Native, like training data (Hewlett)", GRAY),
              ("FP_nativeOOD_mean", "Native, unfamiliar kind", C_OS),
              ("FP_TOEFL", "Non-native (TOEFL)", C_TOEFL)]
    bw = 0.26
    for ax, (m, name, col) in zip(np.atleast_1d(axes), models):
        for i, a in enumerate(arms):
            for k, (key, lab, c) in enumerate(groups):
                v = cm.loc[(m, a), key] * 100
                ax.bar(i + (k - 1) * bw, v, width=bw * 0.95, color=c, label=lab if i == 0 else None, zorder=3)
                ax.text(i + (k - 1) * bw, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels([ARM_LABELS[a] for a in arms], fontsize=10)
        _pct_axis(ax, 112); ax.set_title(name, loc="left", fontweight="bold", color=col)
    axes[0].set_ylabel("Human essays flagged as AI")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    if seeds_note:
        fig.text(0.01, -0.02, seeds_note, fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def trend(cm, commercial_toefl=None, binoculars=None, fastdetect=None, models=MODELS):
    """TOEFL false-flag rate as training data changes A -> B -> C, with the control arm D drawn separately (hollow
    markers). Error bars = SD across seeds. Grey band = range of the 7 commercial detectors (2023); purple lines =
    open-source zero-shot detectors. All at threshold 0.5."""
    setup()
    fig, ax = plt.subplots(figsize=(12, 6))
    if commercial_toefl is not None and len(commercial_toefl):
        lo, hi = min(commercial_toefl) * 100, max(commercial_toefl) * 100
        ax.axhspan(lo, hi, color=GRAY, alpha=0.3, linewidth=0, zorder=0)
        ax.text(2.1, (lo + hi) / 2, f"7 commercial detectors\n(2023): {lo:.0f}-{hi:.0f}%", fontsize=10.5, color=INK2, va="center")
    for v, lab, dy in ((binoculars, "Binoculars", -1.2), (fastdetect, "Fast-DetectGPT", 1.2)):
        if v is not None:
            ax.axhline(v * 100, color=C_OS, linewidth=1.3, zorder=1)
            ax.text(2.72, v * 100 + dy, f"{lab} {v * 100:.0f}%", fontsize=10.5, color=C_OS, ha="right",
                    va="bottom" if dy > 0 else "top")
    xd = {"LR": 3.1, "RoBERTa": 3.3, "Qwen27B": 3.5}
    for m, name, col in models:
        y = [cm.loc[(m, a), "FP_TOEFL"] * 100 for a in "ABC"]
        sd = [cm.loc[(m, a), "TOEFL_sd"] * 100 for a in "ABC"]
        ax.errorbar([0, 1, 2], y, yerr=sd, color=col, linewidth=2, marker="o", markersize=8, capsize=4, label=name, zorder=4)
        d, dsd = cm.loc[(m, "D"), "FP_TOEFL"] * 100, cm.loc[(m, "D"), "TOEFL_sd"] * 100
        ax.errorbar([xd[m]], [d], yerr=[dsd], color=col, marker="o", markersize=8, markerfacecolor=SURF, capsize=4, zorder=4)
        for x, v in zip([0, 1, 2, xd[m]], y + [d]):
            ax.text(x + 0.06, v + 1.5, f"{v:.0f}%", fontsize=10, color=col, fontweight="bold")
    ax.axvline(2.75, color=GRID, linewidth=1)
    ax.set_xticks([0, 1, 2, 3.3]); ax.set_xticklabels([ARM_LABELS[a] for a in "ABCD"])
    ax.set_xlim(-0.4, 3.95); _pct_axis(ax, 108)
    ax.set_ylabel("Real TOEFL essays flagged as AI")
    ax.set_title("TOEFL false flags by training set (threshold 0.5; hollow = control arm D)", loc="left")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    return fig


def table_figure(df, title, col_width=None, highlight_col=None):
    """Render a small DataFrame as a figure (for a quick visual of precision / recall / F1)."""
    setup()
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(min(16, 2.6 + 1.35 * n_cols), 0.5 + 0.32 * (n_rows + 1)))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=list(df.columns), rowLabels=list(df.index), cellLoc="center",
                 bbox=[0.22, 0, 0.78, 1])
    t.auto_set_font_size(False); t.set_fontsize(10.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK2); cell.set_facecolor("#f1f0ec")
        if highlight_col is not None and c == list(df.columns).index(highlight_col) and r > 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def bars(labels, values, title, color=C_LR, pct=True, xlabel=None, highlight=None):
    """Horizontal bar chart (used for commercial detectors and top LR words)."""
    setup()
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(labels) + 1.4))
    y = np.arange(len(labels))[::-1]
    cols = [highlight.get(l, color) if highlight else color for l in labels]
    ax.barh(y, [v * (100 if pct else 1) for v in values], color=cols, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    for yy, v in zip(y, values):
        ax.text(v * (100 if pct else 1), yy, f" {v * 100:.0f}%" if pct else f" {v:.2f}", va="center", fontsize=10, color=INK2)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    if pct:
        ax.set_xlim(0, 110)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig
