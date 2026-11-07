"""Figure 10: partial abstract masking (six models, civil corpus).

Two-panel line chart with seed-level mean +/- SD. Panel (a) is mean posterior
JS (lower is better); panel (b) is top-1 topic agreement. The data were
produced by occlusion_robustness.py with the abstract field masked at 10%-100%.
"""
from pathlib import Path
import sys
import csv
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
INK = "#1C1C1C"
GRID = "#E4E7EB"
STYLE = {
    "FC-NTM": dict(color="#111111", marker="o", lw=2.1, ms=4.0, zorder=10),
    "BTM": dict(color="#0072B2", marker="s", lw=1.15, ms=3.3, zorder=4),
    "ETM": dict(color="#D55E00", marker="^", lw=1.15, ms=3.5, zorder=4),
    "NMF": dict(color="#009E73", marker="v", lw=1.15, ms=3.4, zorder=4),
    "LDA": dict(color="#CC79A7", marker="D", lw=1.15, ms=3.2, zorder=4),
    "ProdLDA": dict(color="#E69F00", marker="P", lw=1.15, ms=3.4, zorder=4),
}
ORDER = ["FC-NTM", "BTM", "ETM", "NMF", "LDA", "ProdLDA"]


def main():
    rows = list(csv.DictReader(
        (ROOT.parent / "figure_preview/data/partial_mask_curves.csv").open(encoding="utf-8")))
    data = {}
    for r in rows:
        data.setdefault(r["model"], []).append(r)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 9.0, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.65, "xtick.direction": "out", "ytick.direction": "out",
        "figure.constrained_layout.use": False,
    })
    fig, axs = plt.subplots(2, 1, figsize=(WIDTH, 108 / 25.4), sharex=True)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.235, top=0.955, hspace=0.28)
    panels = [
        ("mean_js", "sd_js", r"Mean posterior JS", (0.0, 0.205)),
        ("mean_agree", "sd_agree", "Top-1 agreement", (0.55, 1.0)),
    ]
    handles = {}
    for ax, (mean_key, sd_key, ylabel, ylim) in zip(axs, panels):
        for model in ORDER:
            g = sorted(data[model], key=lambda r: float(r["rate"]))
            x = np.array([float(r["rate"]) * 100 for r in g])
            mu = np.array([float(r[mean_key]) for r in g])
            sd = np.array([float(r[sd_key]) for r in g])
            st = STYLE[model]
            ax.fill_between(x, mu - sd, mu + sd, color=st["color"], alpha=.13, lw=0)
            line, = ax.plot(x, mu, color=st["color"], marker=st["marker"],
                            lw=st["lw"], ms=st["ms"], zorder=st["zorder"],
                            label=model)
            if mean_key == "mean_js":
                handles[model] = line
        ax.set_xlim(8, 102)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(10, 101, 10))
        ax.set_ylabel(ylabel, fontsize=9.0)
        ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].text(0.5, 1.03, "(a)", transform=axs[0].transAxes, fontsize=9.0,
                fontweight="bold", ha="center", va="bottom")
    axs[1].text(0.5, 1.03, "(b)", transform=axs[1].transAxes, fontsize=9.0,
                fontweight="bold", ha="center", va="bottom")
    axs[1].set_xlabel("Masked abstract tokens (%)", fontsize=9.0)
    fig.legend([handles[m] for m in ORDER], ORDER, loc="lower center",
               ncol=3, frameon=False, columnspacing=1.2, handlelength=2.2,
               bbox_to_anchor=(.5, .015))
    issues = audit_layout(fig)
    print("layout issues:", issues)
    assert not any(i[0] == "FAIL" for i in issues), issues
    (ROOT / "Fig10_preview.png").parent.mkdir(exist_ok=True)
    fig.savefig(ROOT / "Fig10_preview.png", dpi=200)
    export_figure(fig, str(ROOT / "Fig10"), formats=["pdf", "svg", "png"],
                  dpi=600, size_inches=tuple(fig.get_size_inches()), tight=False)
    print("saved Fig10")


if __name__ == "__main__":
    main()
