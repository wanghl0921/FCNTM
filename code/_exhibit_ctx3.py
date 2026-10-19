# -*- coding: utf-8 -*-
import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
s = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()

for anchor in ("\\subsection{Read-out-matched comparison}",
               "which would inflate the NLL",
               "\\subsection{Cross-corpus comparison}",
               "on all three corpora (seeds 11/23/37).",
               "with the full fields as an upper reference.",
               "including the corpora where the advantage is not observed.",
               "\\label{tab:conditional}"):
    i = s.find(anchor)
    print("=" * 16, anchor[:40], "at", i)
    print(repr(s[max(0, i - 260):i + 120]))
    print()
