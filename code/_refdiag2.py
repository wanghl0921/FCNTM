# -*- coding: utf-8 -*-
import io
import re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
tex = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()
bib = tex[tex.index("\\begin{thebibliography}"):]
items = bib.split("\\bibitem{")[1:]

print("== [34], [36], [45] source entries ==")
for i, item in enumerate(items, 1):
    key, _, rest = item.partition("}")
    if i in (34, 36, 45):
        print("##", i, key)
        print(" ".join(rest.split()))
        print()

print("== titles ending with '?.' or '!.' ==")
for i, item in enumerate(items, 1):
    key, _, rest = item.partition("}")
    if re.search(r"[?!]\.\s*$", " ".join(rest.split())):
        print("  ", i, key, "|", " ".join(rest.split())[-80:])

print("== ASCE bbl author names with short particles ==")
bbl = io.open(PAPER + r"\main_asce.bbl", encoding="utf-8").read()
for line in bbl.splitlines():
    if re.match(r"\s*(De|Van|Von|Der|Den|La|Le|Di|Da|Del|Della|El|Al|Bin|Ibn),",
                line):
        print("   suspect:", line.strip()[:90])
