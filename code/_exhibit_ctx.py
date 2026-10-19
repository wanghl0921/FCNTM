# -*- coding: utf-8 -*-
"""Print the discussion text that precedes each uncited exhibit."""
import io
import re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
s = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()

LABELS = ["tab:fair", "tab:occl-robust", "tab:occl-cross", "tab:concat",
          "tab:archival", "fig:main", "fig:cond", "fig:occlusion",
          "fig:occl-cross"]

for label in LABELS:
    i = s.find("\\label{%s}" % label)
    print("=" * 20, label, "at", i)
    print(" ".join(s[max(0, i - 900):i - 40].split())[-820:])
    print()
