# -*- coding: utf-8 -*-
import re

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
doc = pymupdf.open(PAPER + r"\main_asce.pdf")
t = "".join(p.get_text() for p in doc)
print("pages:", doc.page_count)
tables = sorted(set(re.findall(r"TABLE (\d+)\.", t)), key=int)
figs = sorted(set(re.findall(r"Fig(?:ure)?\.? (\d+)\.", t)), key=int)
print("TABLE captions:", tables)
print("Fig captions:", figs)
print("in-text 'Table' refs:", len(re.findall(r"Table \d+", t)))
print("supplement pointer:", "Tables S8" in t)
print("moved tables leaked:",
      [k for k in ("Whole-field occlusion at inference time",
                   "Three-step fusion ablation",
                   "Core-component ablation",
                   "Sensitivity to the fusion-reconstruction weight",
                   "Field-restricted read-out",
                   "Field redundancy") if k in t])
print("question marks:", t.count("??"))
print("Practical Applications:", "PRACTICAL APPLICATIONS" in t)
print("new fig captions:",
      "Overview of FC-NTM" in t, "|", "Per-seed paired comparison" in t)
for pno in range(20, doc.page_count):
    head = " ".join(doc[pno].get_text().split())[:80]
    print("p%-3d %s" % (pno + 1, head))
