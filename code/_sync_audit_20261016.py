# -*- coding: utf-8 -*-
"""Audit batch 3: text sync between main_en/main_asce, logs, page counts, zip scripts."""
import io, os, re
import pymupdf

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")
CODE = os.path.join(D, "FCNTM完整代码")

en = io.open(os.path.join(PAPER, "main_en.tex"), encoding="utf-8").read()
asce = io.open(os.path.join(PAPER, "main_asce.tex"), encoding="utf-8").read()

def nw(s):
    return " ".join(s.split())

en_n, asce_n = nw(en), nw(asce)

phrases = [
    # (label, main_en should-now contain)
    ("soften: close on every corpus", "close on every corpus, which frames the two-stream design as a matched alternative"),
    ("soften: upper reference (abstract)", "as the upper reference"),
    ("soften: matched design choice", "matched design choice rather than a source of unconditional coherence gains"),
    ("soften: comparable ranges", "operate within comparable ranges on this task"),
    ("soften: scoped conclusion", "The contribution of this paper is therefore scoped"),
    ("soften: BTM keeps its lead", "BTM keeps its lead under the native"),
    ("soften: BTM keeps a small lead", "BTM keeps a small lead"),
    ("soften: calibration range", "define the current calibration range"),
    ("soften: stability indicator", "present the relative degradation as a stability indicator"),
    ("soften: read together with", "read together with the absolute NLL"),
    ("old: restricted claim", "The claim of this paper is therefore restricted"),
    ("old: not best missing-field prediction", "not the best"),
    ("old: does not replicate", "does not"),
]

print("### phrase sync check (en vs asce)")
for label, ph in phrases:
    e = nw(ph) in en_n
    a = nw(ph) in asce_n
    print("  %-42s en=%s asce=%s" % (label, e, a))

print()
print("### page counts")
for f in ["main_en.pdf", "main_asce.pdf", "main.pdf", "supplementary_en.pdf", "supplementary.pdf"]:
    p = os.path.join(PAPER, f)
    if os.path.exists(p):
        doc = pymupdf.open(p)
        print("  %-22s %d pages" % (f, doc.page_count))
        doc.close()

print()
print("### log tails: main_en.log")
p = os.path.join(PAPER, "main_en.log")
t = io.open(p, encoding="utf-8", errors="replace").read()
lines = [l for l in t.splitlines() if l.strip()]
print("\n".join(lines[-12:]))

print()
print("### log tails: main_asce.log")
p = os.path.join(PAPER, "main_asce.log")
t = io.open(p, encoding="utf-8", errors="replace").read()
lines = [l for l in t.splitlines() if l.strip()]
print("\n".join(lines[-14:]))

print()
print("### make_github_package_20261016.py")
p = os.path.join(CODE, "make_github_package_20261016.py")
print(io.open(p, encoding="utf-8").read()[:3000])

print()
print("### make_fcntm_zips_20261014.py")
p = os.path.join(D, "make_fcntm_zips_20261014.py")
print(io.open(p, encoding="utf-8").read()[:2500])
