# -*- coding: utf-8 -*-
"""Fix U+0008 control chars written into tex by a buggy repl string.

The corrupted sequence is '$\\x08eta$' which was meant to be '$\\beta$'.
"""
import io, os, sys

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

total = 0
for name in ["main_en.tex", "main_asce.tex", "supplementary_en.tex", "supplementary.tex", "main.tex"]:
    p = os.path.join(PAPER, name)
    if not os.path.exists(p):
        print("skip (missing):", name)
        continue
    s = io.open(p, encoding="utf-8").read()
    n = s.count("\x08")
    if not n:
        print("clean:", name)
        continue
    fixed = s.replace("$\x08eta$", "$\\beta$").replace("\x08", "")
    io.open(p, "w", encoding="utf-8", newline="\n").write(fixed)
    left = fixed.count("\x08")
    total += n
    print("fixed %d occurrence(s) in %s (remaining: %d)" % (n, name, left))

print("TOTAL fixed:", total)
