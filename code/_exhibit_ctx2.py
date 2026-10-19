# -*- coding: utf-8 -*-
"""Print the last paragraph before each exhibit environment."""
import io
import re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
s = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()

LABELS = ["tab:fair", "tab:occl-robust", "tab:occl-cross", "tab:concat",
          "tab:archival", "fig:main", "fig:cond", "fig:occlusion",
          "fig:occl-cross"]

for label in LABELS:
    i = s.find("\\label{%s}" % label)
    # start of the enclosing environment
    kind = "table" if label.startswith("tab") else "figure"
    j = s.rfind("\\begin{%s}" % kind, 0, i)
    before = s[:j]
    k = before.rstrip().rfind("\n\n")
    para = before[k:].strip()
    print("=" * 18, label)
    print(para[-700:])
    print()
