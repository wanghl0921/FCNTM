# -*- coding: utf-8 -*-
import re

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
doc = pymupdf.open(PAPER + r"\main_asce.pdf")
for pno in range(20, doc.page_count):
    txt = " ".join(doc[pno].get_text().split())
    print("p%-3d %s" % (pno + 1, txt[:110]))
print()
whole = "".join(doc[p].get_text() for p in range(doc.page_count))
print("counts: 'Table' ->", whole.count("Table"), "| 'Fig.' ->",
      whole.count("Fig."), "| 'Figure' ->", whole.count("Figure"))
