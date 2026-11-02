# -*- coding: utf-8 -*-
"""Append the tables and figures that leave the 30-page ASCE manuscript.

Tables tab:agnews ... tab:archival become S14-S20, figures fig:main,
fig:cond and fig:occl-cross become Figures S1-S3 of the supplement.
"""

from __future__ import annotations

import io
import re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

TABLES = ["tab:agnews", "tab:github", "tab:fair", "tab:occl-robust",
          "tab:occl-cross", "tab:concat", "tab:archival"]
FIGURES = ["fig:main", "fig:cond", "fig:occl-cross"]

SUBSECTIONS = [
    "Ablation: fusion reconstruction and gating",
    "Ablation: core components",
    "Sensitivity to the fusion weight",
    "Field-restricted read-out",
]

HEADER = r"""\section{Extended results}

Tables S14--S20 and Figures S1--S3 collect the cross-corpus, occlusion,
read-out and application results that the shortened manuscript cites to this
supplement. They use the same protocol, vocabulary and read-outs as the
main-text tables.

"""


def blocks(text: str, env: str) -> list:
    return re.findall(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), text, re.S)


def fix_refs(block: str) -> str:
    """Section cross-references do not exist in the supplement."""
    block = block.replace(r"Section~\ref{sec:protocol}", "the main text")
    block = re.sub(r"Section~\\ref\{[^}]*\}", "the main text", block)
    return block


def main() -> None:
    main_tex = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()
    supp = io.open(PAPER + r"\supplementary_en.tex", encoding="utf-8").read()
    if "Ablation and read-out details" in supp:
        cut = supp.index("\\section{Ablation and read-out details}")
        supp = supp[:cut].rstrip() + "\n\n\\end{document}\n"
        print("removed a previously appended details section")
    if "Extended results" in supp:
        cut = supp.index("\\section{Extended results}")
        supp = supp[:cut].rstrip() + "\n\n\\end{document}\n"
        print("removed a previously appended section")
    pieces = []
    for label in TABLES:
        hits = [b for b in blocks(main_tex, "table")
                if ("\\label{%s}" % label) in b]
        assert len(hits) == 1, (label, len(hits))
        pieces.append(fix_refs(hits[0]))
    for label in FIGURES:
        hits = [b for b in blocks(main_tex, "figure")
                if ("\\label{%s}" % label) in b]
        assert len(hits) == 1, (label, len(hits))
        pieces.append(fix_refs(hits[0]))
    detail = [r"\section{Ablation and read-out details}", "",
              "The paragraphs below accompany Tables S9--S12 and describe the ablations,",
              "the fusion-weight sensitivity and the field-restricted read-out in full.", ""]
    for title in SUBSECTIONS:
        pat = re.compile(r"\\subsection\{%s\}.*?(?=\\subsection\{|\\section\{)"
                         % re.escape(title), re.S)
        m = pat.search(main_tex)
        assert m, title
        block = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "",
                       m.group(0), flags=re.S).strip()
        detail.append(fix_refs(block))
        detail.append("")
    supp = supp.replace("\\end{document}",
                        HEADER + "\n\n".join(pieces) + "\n\n" +
                        "\n".join(detail) + "\n\\end{document}")
    io.open(PAPER + r"\supplementary_en.tex", "w", encoding="utf-8").write(supp)
    print("supplement updated:", len(TABLES), "tables and", len(FIGURES),
          "figures moved")


if __name__ == "__main__":
    main()
