# -*- coding: utf-8 -*-
"""Batch 8: official example analysis, asce section refs, preview guidance, github script rest."""
import io, os, re

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")
EXT = os.path.join(D, "_asce_template", "_extract", "ascelike-new-main")
CODE = os.path.join(D, "FCNTM完整代码")

print("#" * 24, "1. official example headings & numbering")
ex = os.path.join(EXT, "example", "outputs", "ascelike-new-example.pdf")
txt = ""
try:
    import pymupdf
    doc = pymupdf.open(ex)
    txt = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    doc.close()
except Exception as e:
    print("ERR", e)
if txt:
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    # uppercase-style headings
    for l in lines:
        if re.fullmatch(r"[A-Z][A-Z ,\-]{4,}", l) and len(l) < 60:
            print("   H:", l)
    print("   has '(1)' or '1.' patterns near headings:")
    for m in re.finditer(r"^\s*(\d+\.?\d*)\s+[A-Z][a-z]", txt, re.M):
        print("      ...", m.group(0).strip()[:60])
    # citation examples
    cits = re.findall(r"\([A-Z][A-Za-z\-]+ (?:et al\.?)?(?:and [A-Z][A-Za-z\-]+ )?\d{4}[a-z]?\)", txt)
    print("   citation examples:", cits[:10])

print()
print("#" * 24, "2. preview guidance: title length / footnote / summary / appendix")
pv = io.open(os.path.join(D, "_asce_template", "asce_preview.txt"), encoding="utf-8").read()
for key in ["Titles should be no longer", "footnote", "Summary", "Acknowledgments", "Notation",
            "KeyWords", "keywords", "in-text", "Supplemental Materials"]:
    hits = list(re.finditer(re.escape(key), pv, re.I))
    print("   [%s] %d hits" % (key, len(hits)))
    if hits:
        i = hits[0].start()
        print("      ", " ".join(pv[max(0, i-120):i+260].split())[:330])

print()
print("#" * 24, "3. main_asce.pdf 'Section' contexts")
t = io.open(os.path.join(PAPER, "_main_asce.txt"), encoding="utf-8", errors="replace").read()
for m in re.finditer(r"Section[ ~\n]*\d[\.\d]*", t):
    i = m.start()
    print("   ...", " ".join(t[max(0, i-100):i+80].split())[-170:])

print()
print("#" * 24, "4. make_github_package rest")
s = io.open(os.path.join(CODE, "make_github_package_20261016.py"), encoding="utf-8").read()
print(s[3000:7000])

print()
print("#" * 24, "5. finish_remaining_tasks.py head")
p = os.path.join(CODE, "finish_remaining_tasks.py")
if os.path.exists(p):
    print(io.open(p, encoding="utf-8").read()[:1500])
else:
    print("missing")
