"""Figure 12: title-only routing as a horizontal bar chart.

BERTopic-style grouped horizontal bars: accuracy and macro-F1 for the routing
baselines and FC-NTM. Error bars are seed SD where available.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "work/scipilot_deps"))
sys.path.insert(0, str(Path.home() / ".codex/skills/scipilot-figure-skill/scripts"))
from visual_qa import audit_layout
from export_figure import export_figure

WIDTH = 372 / 72.27
GRID = "#E4E7EB"
METHODS = ["LDA, title only", "BTM, title only", "Concatenated control",
           "FC-NTM, title only", "FC-NTM, title+abstract", "TF–IDF supervised"]
ACC = np.array([0.532, 0.600, 0.653, 0.664, 0.719, 0.810])
ACC_SD = np.array([0.039, 0.013, 0.012, 0.004, 0.016, 0.0])
F1 = np.array([0.458, 0.529, 0.614, 0.624, 0.676, 0.805])
F1_SD = np.array([0.057, 0.043, 0.034, 0.013, 0.032, 0.0])
Y = np.arange(len(METHODS))[::-1]


def main():
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.65, "xtick.direction": "out", "ytick.direction": "out",
        "figure.constrained_layout.use": False,
    })
    fig, ax = plt.subplots(figsize=(WIDTH, 88 / 25.4))
    fig.subplots_adjust(left=0.33, right=0.97, bottom=0.27, top=0.93)
    ax.barh(Y + 0.17, ACC, height=0.34, color="#3A3F45", edgecolor="#1C1C1C",
            linewidth=.6, zorder=3, label="Accuracy")
    ax.barh(Y - 0.17, F1, height=0.34, color="#B9C4CC", edgecolor="#7A8086",
            linewidth=.6, hatch="...", zorder=3, label="Macro-F1")
    ax.errorbar(ACC, Y + 0.17, xerr=ACC_SD, fmt="none", ecolor="#1C1C1C",
                elinewidth=.8, capsize=2.2, zorder=5)
    ax.errorbar(F1, Y - 0.17, xerr=F1_SD, fmt="none", ecolor="#5E666D",
                elinewidth=.8, capsize=2.2, zorder=5)
    ax.set_yticks(Y)
    ax.set_yticklabels(METHODS)
    ax.set_xlim(0.40, 0.90)
    ax.set_xlabel("Score", fontsize=9.0)
    ax.grid(axis="x", color=GRID, lw=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=8.0, bbox_to_anchor=(.5, .01), handlelength=1.8)
    issues = audit_layout(fig)
    print("layout issues:", issues)
    assert not any(i[0] == "FAIL" for i in issues), issues
    fig.savefig(ROOT / "Fig12_preview.png", dpi=200)
    export_figure(fig, str(ROOT / "Fig12"), formats=["pdf", "svg", "png"],
                  dpi=600, size_inches=tuple(fig.get_size_inches()), tight=False)
    print("saved Fig12 routing")


if __name__ == "__main__":
    main()
