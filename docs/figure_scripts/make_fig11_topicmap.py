"""Figure 11: BERTopic-style topic-mass bar chart (30 GitHub topics).

Bars are topic mass; tick labels carry the topic ID and the first three
summary-side words; colour is the manual topic label. This is the dense
bar-chart form requested for the final manuscript.
"""
from pathlib import Path
import re
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
TABLE = Path(r"C:\Users\20102\Desktop\FCNTM\FCNTM完整代码\outputs\topic_table_github_seed11.tex")
COLORS = {"redis": "#0072B2", "typescript": "#D55E00",
          "duckdb": "#009E73", "moby": "#CC79A7"}


def parse_topics():
    topics = []
    pat = re.compile(r"^(T\d+)\s+\[([^,]+),\s*mass=([0-9.]+)\]\s*&\s*(.*?)\s*&\s*(.*?)\s*\\\\")
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if m:
            tid, label, mass, summary, background = m.groups()
            topics.append(dict(tid=tid, label=label.strip(),
                               mass=float(mass), words=summary.split()[:10]))
    return topics


def main():
    topics = parse_topics()
    assert len(topics) == 30, len(topics)
    topics = sorted(topics, key=lambda t: t["mass"])
    masses = np.array([t["mass"] for t in topics])
    labels = [t["label"] for t in topics]
    tick = [f"{t['tid']}  " + ", ".join(t["words"][:3]) for t in topics]
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.0, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 6.9,
        "legend.fontsize": 7.5, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "figure.constrained_layout.use": False,
    })
    fig, ax = plt.subplots(figsize=(WIDTH, 126 / 25.4))
    fig.subplots_adjust(left=0.38, right=0.97, bottom=0.13, top=0.94)
    ax.barh(np.arange(30), masses, color=[COLORS[x] for x in labels],
            height=.78, edgecolor="white", lw=.3)
    ax.set_yticks(np.arange(30))
    ax.set_yticklabels(tick, fontsize=6.9)
    ax.set_xlim(0, 0.075)
    ax.set_xlabel("Topic mass", fontsize=8.5)
    ax.grid(axis="x", color="#E4E7EB", lw=.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=c, label=k)
               for k, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.68, .005), handlelength=1.0,
               columnspacing=.9, handletextpad=.4)
    issues = audit_layout(fig)
    print("layout issues:", issues)
    assert not any(i[0] == "FAIL" for i in issues), issues
    fig.savefig(ROOT / "Fig11_preview.png", dpi=200)
    export_figure(fig, str(ROOT / "Fig11"), formats=["pdf", "svg", "png"],
                  dpi=600, size_inches=tuple(fig.get_size_inches()), tight=False)
    print("saved Fig11 topic bar", len(topics))


if __name__ == "__main__":
    main()
