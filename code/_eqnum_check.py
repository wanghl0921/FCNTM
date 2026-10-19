# -*- coding: utf-8 -*-
import io
import re

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

for aux in ("main_en.aux", "main_asce.aux"):
    s = io.open(PAPER + "\\" + aux, encoding="utf-8", errors="replace").read()
    labs = re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}\}", s)
    eqs = [(k, v) for k, v, _ in labs if k.startswith("eq:")]
    print(aux, "eq labels:", eqs)

doc = pymupdf.open(PAPER + r"\main_asce.pdf")
for pno in range(3, 9):
    t = doc[pno].get_text()
    for m in re.finditer(r"\(\d+\)", t):
        ctx = t[max(0, m.start() - 46):m.start() + 8].replace("\n", " ")
        if "Eq" in ctx or "=" in ctx or "\\" in ctx:
            print("asce p%d: %r" % (pno + 1, ctx))

doc2 = pymupdf.open(PAPER + r"\main_en.pdf")
for pno in range(3, 9):
    t = doc2[pno].get_text()
    for m in re.finditer(r"\(\d+\)", t):
        ctx = t[max(0, m.start() - 46):m.start() + 8].replace("\n", " ")
        if "Eq" in ctx or "=" in ctx:
            print("en p%d: %r" % (pno + 1, ctx))
