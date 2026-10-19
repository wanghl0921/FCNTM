# -*- coding: utf-8 -*-
import io
import re

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

for aux in ("main_en.aux", "main_asce.aux", "main.aux"):
    s = io.open(PAPER + "\\" + aux, encoding="utf-8", errors="replace").read()
    labs = re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", s)
    eqs = [(k, v) for k, v, _ in labs if k.startswith("eq:")]
    tabs = [(k, v) for k, v, _ in labs if k.startswith("tab:")]
    figs = [(k, v) for k, v, _ in labs if k.startswith("fig:")]
    print(aux)
    print("  eq:", eqs)
    print("  tab:", tabs)
    print("  fig:", figs)

doc = pymupdf.open(PAPER + r"\main_en.pdf")
t = "".join(doc[p].get_text() for p in range(doc.page_count))
for pat in (r"Table \d+[:.]", r"Figure \d+[:.]", r"Fig\. \d+[:.]"):
    ms = sorted(set(re.findall(pat, t)))
    print(pat, "->", ms[:8])
i = t.find("(1)")
print("first (1) ctx:", repr(t[max(0, i - 60):i + 5]))
