# -*- coding: utf-8 -*-
import io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
s = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()
bib = s[s.index("\\begin{thebibliography}"):]
items = bib.split("\\bibitem{")[1:]
for i, item in enumerate(items, 1):
    key, _, rest = item.partition("}")
    if i in (9, 18, 21, 22, 39, 44, 45):
        print("=" * 12, i, key)
        print(" ".join(rest.split())[:400])
        print()
