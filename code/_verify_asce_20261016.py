# -*- coding: utf-8 -*-
"""Batch 5: ASCE guidance wording, main_asce render details, Chinese citations, logs."""
import io, os, re
import pymupdf

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")
EXT = os.path.join(D, "_asce_template", "_extract", "ascelike-new-main")

print("#" * 22, "A. official preview: sections / practical / length wording")
pv = io.open(os.path.join(D, "_asce_template", "asce_preview.txt"), encoding="utf-8").read()
for key in ["should not be numbered", "numbered", "Practical Applications", "maximum length",
            "double-spaced manuscript pages", "author-date", "Data Availability"]:
    for m in re.finditer(re.escape(key), pv):
        i = m.start()
        seg = " ".join(pv[max(0, i-160):i+240].split())
        print("  [%s] ...%s..." % (key, seg[:340]))
        break
    else:
        print("  [%s] NOT FOUND" % key)

print()
print("#" * 22, "B. class file top docs (options)")
cls = io.open(os.path.join(PAPER, "ascelike-new.cls"), encoding="utf-8", errors="replace").read()
print("\n".join(cls.splitlines()[:80]))

print()
print("#" * 22, "C. main_asce.pdf: section headings + line numbers + cites")
doc = pymupdf.open(os.path.join(PAPER, "main_asce.pdf"))
t = doc[4].get_text()
print("---- page 5 text (first 900 chars):")
print(t[:900])
doc.close()

print()
print("#" * 22, "D. example pdf in-text citation style")
pex = os.path.join(D, "_asce_template", "_extract", "ascelike-new-example.pdf")
if os.path.exists(pex):
    doc = pymupdf.open(pex)
    txt = "".join(doc.p[i].get_text() for i in range(min(4, doc.page_count)))
    for m in re.finditer(r"\([A-Z][A-Za-z]+ et al\.?,? \d{4}\)|\([A-Z][A-Za-z]+ \d{4}\)", txt):
        print("   ", m.group(0))
    doc.close()

print()
print("#" * 22, "E. main_en.log error lines")
t = io.open(os.path.join(PAPER, "main_en.log"), encoding="utf-8", errors="replace").read()
for l in t.splitlines():
    if "Unicode character" in l or l.startswith("!") or "Error" in l:
        print("   ", l.strip()[:160])

print()
print("#" * 22, "F. fixed-phrase contexts in main_en.tex")
en = io.open(os.path.join(PAPER, "main_en.tex"), encoding="utf-8").read()
for needle in ["therefore scoped", "BTM keeps a small lead", "keeps its lead under the native"]:
    i = en.find(needle)
    if i >= 0:
        print("  [%s]" % needle, repr(en[max(0, i-120):i+180])[:400])
    else:
        print("  [%s] NOT FOUND" % needle)

print()
print("#" * 22, "G. ASCE abstract current text (full)")
t = io.open(os.path.join(PAPER, "main_asce.tex"), encoding="utf-8").read()
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S)
print(m.group(1).strip() if m else "not found")
