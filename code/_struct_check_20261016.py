# -*- coding: utf-8 -*-
"""Batch 6: structural checks before docx build and ASCE fix."""
import io, os, re
import pymupdf

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")

for name in ["main_en.tex", "main_asce.tex"]:
    s = io.open(os.path.join(PAPER, name), encoding="utf-8").read()
    print("=" * 30, name)
    for pat, label in [
        (r"\\ref\{[^}]*\}", "\\ref"),
        (r"\\eqref\{[^}]*\}", "\\eqref"),
        (r"\\label\{[^}]*\}", "\\label"),
        (r"Section\s*\d", "hardcoded 'Section N'"),
        (r"\\shortstack", "\\shortstack"),
        (r"\\cmidrule", "\\cmidrule"),
        (r"\\multicolumn", "\\multicolumn"),
        (r"\\resizebox", "\\resizebox"),
        (r"\\footnote", "\\footnote"),
        (r"\\citep", "\\citep"),
        (r"\\citet", "\\citet"),
        (r"\\citeauthor", "\\citeauthor"),
    ]:
        n = len(re.findall(pat, s))
        if n:
            print("  %-24s %d" % (label, n))
    refs = re.findall(r"\\(?:eq)?ref\{([^}]*)\}", s)
    if refs:
        print("  refs:", sorted(set(refs))[:30])
    secs = re.findall(r"Section\s*\d[\.\d]*", s)
    if secs:
        print("  hardcoded sections:", sorted(set(secs)))

print()
print("#" * 20, "multi-key citation rendering in current main_en.pdf text")
t = io.open(os.path.join(PAPER, "_main_en.txt"), encoding="utf-8", errors="replace").read()
cands = re.findall(r"\[[\d,\s]+\]", t)
from collections import Counter
multi = [c for c in cands if "," in c]
print("first multi brackets:", multi[:8])
print("bracket examples:", list(Counter(cands).keys())[:15])

print()
print("#" * 20, "where are figures in main_asce.pdf")
doc = pymupdf.open(os.path.join(PAPER, "main_asce.pdf"))
for pno in range(doc.page_count):
    tx = doc[pno].get_text()
    if "Fig. 1." in tx and pno > 5:
        print("first 'Fig. 1.' caption at page", pno + 1, "of", doc.page_count)
        break
for pno in range(doc.page_count):
    tx = doc[pno].get_text()
    if "Table 1." in tx and pno > 5:
        print("first 'Table 1.' caption at page", pno + 1)
        break
for pno in range(doc.page_count - 1, 0, -1):
    tx = doc[pno].get_text()
    if "REFERENCES" in tx.upper():
        print("'REFERENCES' page(s): include", pno + 1)
        break
doc.close()

print()
print("#" * 20, "main_en.log: check for other error kinds")
t = io.open(os.path.join(PAPER, "main_en.log"), encoding="utf-8", errors="replace").read()
errs = re.findall(r"^! .*$", t, re.M)
from collections import Counter
print(Counter(e[:100] for e in errs))
warns = re.findall(r"LaTeX Warning: [^\n]*", t)
print("warnings:", len(warns))
for w in warns[:12]:
    print("   ", w[:150])

print()
print("#" * 20, "main_asce.log errors")
t = io.open(os.path.join(PAPER, "main_asce.log"), encoding="utf-8", errors="replace").read()
errs = re.findall(r"^! .*$", t, re.M)
print(Counter(e[:100] for e in errs))
warns = re.findall(r"LaTeX Warning: [^\n]*", t)
print("warnings:", len(warns))
for w in warns[:12]:
    print("   ", w[:150])
