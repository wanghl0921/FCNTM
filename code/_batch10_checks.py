# -*- coding: utf-8 -*-
"""Batch 10: footnotes content, title char counts, candidate titles."""
import io, os, re

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")

s = io.open(os.path.join(PAPER, "main_en.tex"), encoding="utf-8").read()

print("### footnotes in main_en.tex")
for m in re.finditer(r"\\footnote\{", s):
    i = m.start()
    # balance braces
    j = i + len("\\footnote{")
    depth = 1
    while depth > 0 and j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    seg = s[i:j]
    print("  at", i, ":", " ".join(seg.split())[:400])
    # context before
    print("    context:", " ".join(s[max(0,i-200):i].split())[-260:])
    print()

print("### title char counts")
base = "Field-Cooperative Neural Topic Modeling of Civil Engineering Literature"
cands = [
    base + ": Read-out Evaluation and Missing-Abstract Inference",
    base + ": Read-out and Missing-Abstract Inference",
    base + ": Missing-Abstract Read-out",
    base + ": Missing-Abstract Inference",
    base + " with Missing-Abstract Read-out",
    base + " for Missing-Abstract Read-out",
    "Field-Cooperative Neural Topic Modeling of Civil Engineering Literature: Missing-Abstract Read-out Evaluation",
    "Field-Cooperative Neural Topic Modeling: Read-out Evaluation and Missing-Abstract Inference for Civil Engineering Literature",
    "Field-Cooperative Neural Topic Modeling with Missing-Abstract Read-out in Civil Engineering Literature",
    "Field-Cooperative Neural Topic Modeling for Civil Engineering Literature: Read-out and Missing-Abstract Inference",
    "Field-Cooperative Neural Topic Modeling of Civil Engineering Literature: Read-out and Missing-Field Inference",
    "Field-Cooperative Neural Topic Modeling of Engineering Literature: Read-out and Missing-Abstract Inference",
]
for c in cands:
    print("  %3d  %s" % (len(c), c))

print()
print("### current title in both files")
for name in ["main_en.tex", "main_asce.tex"]:
    t = io.open(os.path.join(PAPER, name), encoding="utf-8").read()
    m = re.search(r"\\title\{([^}]*)\}", t, re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1).replace("\n", " ")).strip()
        print("  %s: len=%d '%s'" % (name, len(title), title))

print()
print("### section titles in main_asce (for name refs)")
for m in re.finditer(r"\\(section|subsection)\{([^}]*)\}", s):
    print("   ", m.group(1), "|", m.group(2))
