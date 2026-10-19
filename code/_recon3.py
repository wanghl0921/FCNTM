# -*- coding: utf-8 -*-
import io
import re

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

print("== all Table/Figure mentions in main_en.tex ==")
s = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()
body = s[s.index(r"\section{Introduction}"):s.index(r"\begin{thebibliography}")]
for m in re.finditer(r"Tables?\s*(~?\s*\\ref\{[^}]*\}|\d+(?:[--–]\d+)?)", body):
    st = max(0, m.start() - 90)
    print("  ...%s" % " ".join(body[st:m.end() + 25].split())[-150:])

print()
print("== supplement table captions ==")
t = io.open(PAPER + r"\supplementary_en.tex", encoding="utf-8").read()
for m in re.finditer(r"\\caption\{([^}]{0,80})", t):
    print("   ", m.group(1))

print()
print("== ASCE pdf page composition ==")
doc = pymupdf.open(PAPER + r"\main_asce.pdf")
tbl_pages, fig_pages, ref_pages = set(), set(), set()
for pno in range(doc.page_count):
    txt = doc[pno].get_text()
    if re.search(r"Table \d+:", txt):
        tbl_pages.add(pno + 1)
    if re.search(r"Figure \d+:", txt):
        fig_pages.add(pno + 1)
    if "REFERENCES" in txt:
        ref_pages.add(pno + 1)
print("   pages:", doc.page_count)
print("   pages with table captions:", sorted(tbl_pages))
print("   pages with figure captions:", sorted(fig_pages))
print("   references start:", sorted(ref_pages))
print("   text-only pages:", doc.page_count - len(tbl_pages | fig_pages))
