# -*- coding: utf-8 -*-
"""Append the six moved tables to ``supplementary_en.tex`` as Tables S8--S13.

The ASCE manuscript (``main_asce.tex``) drops these tables and points to the
supplement instead; the analysis version ``main_en.tex`` keeps them.  The
tables are copied verbatim, except that references to main-text tables are
rewritten to plain text because the supplement is a separate document.
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(r"C:\Users\Administrator\Downloads\FCNTM完整论文")

ORDER = ("tab:conditional", "tab:fieldview-baselines", "tab:fusion",
         "tab:core-ablation", "tab:weight", "tab:occlusion")

def parse_aux_numbers() -> dict:
    text = (PAPER / "main_en.aux").read_text(encoding="utf-8",
                                             errors="replace")
    return {k: v for k, v, _ in re.findall(
        r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", text)}


def rewrite_refs(block: str, nums: dict) -> str:
    def phrase(label: str) -> str:
        num = nums.get(label, "?")
        if label.startswith("tab:"):
            return "Table %s of the main text" % num
        if label.startswith("fig:"):
            return "Figure %s of the main text" % num
        if label.startswith("eq:"):
            return "Eq. (%s) of the main text" % num
        return "Section %s of the main text" % num

    for word in ("Table", "Figure", "Section"):
        block = re.sub(r"%s~\\ref\{([^}]*)\}" % word,
                       lambda m: "%s %s of the main text"
                       % (word, nums.get(m.group(1), "?")), block)
    block = re.sub(r"\\eqref\{([^}]*)\}",
                   lambda m: "(%s)" % nums.get(m.group(1), "?"), block)
    block = re.sub(r"\\ref\{([^}]*)\}",
                   lambda m: phrase(m.group(1)), block)
    return block

HEADER = r"""\section{Ablations and secondary comparisons}

Tables S8--S13 collect the ablations and secondary comparisons that the main manuscript cites to
this supplement: the field-redundancy measures, the field-restricted read-out comparison, the
three-step fusion ablation, the core-component ablation, the sensitivity to the
fusion-reconstruction weight, and the whole-field occlusion diagnostic. All tables use the same
protocol, vocabulary and read-out definitions as the main-text tables.

"""


def main() -> None:
    main_tex = (PAPER / "main_en.tex").read_text(encoding="utf-8")
    supp = (PAPER / "supplementary_en.tex").read_text(encoding="utf-8")
    if "Ablations and secondary comparisons" in supp:
        # remove a previously appended copy so the script is re-runnable
        cut = supp.index("\\section{Ablations and secondary comparisons}")
        supp = supp[:cut].rstrip() + "\n\n\\end{document}\n"
        print("removed previously appended section")
    table_blocks = re.findall(r"\\begin\{table\}.*?\\end\{table\}", main_tex,
                              re.S)
    blocks = []
    nums = parse_aux_numbers()
    for label in ORDER:
        matches = [b for b in table_blocks
                   if ("\\label{%s}" % label) in b]
        if len(matches) != 1:
            raise SystemExit("expected exactly one table for %s, found %d"
                             % (label, len(matches)))
        block = matches[0]
        cites = re.findall(r"\\cite\{[^}]*\}", block)
        block = rewrite_refs(block, nums)
        if cites:
            print("cite found in", label, cites)
        blocks.append(block)
        print("moved", label, "(%.0f chars)" % len(block))
    joined = HEADER + "\n\n".join(blocks) + "\n\n"
    supp = supp.replace("\\end{document}", joined + "\\end{document}")
    (PAPER / "supplementary_en.tex").write_text(supp, encoding="utf-8")
    print("supplementary_en.tex updated")


if __name__ == "__main__":
    main()
