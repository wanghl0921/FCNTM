# -*- coding: utf-8 -*-
"""Do captions or text mention colours / line styles?"""

import io
import re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
s = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()

caps = re.findall(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", s, re.S)
pat = re.compile(r"(?i)\b(colou?r|yellow|red|blue|green|orange|grey|gray|dashed|"
                 r"solid|marker|shade|filled|bar|curve)\b")
print("captions:", len(caps))
for i, c in enumerate(caps, 1):
    hits = sorted({h.lower() for h in pat.findall(c)})
    print("%2d %-26s %s" % (i, ",".join(hits) or "-", " ".join(c.split())[:110]))

body = s[:s.index("\\begin{thebibliography}")]
words = re.findall(r"(?i)\b(colou?r|yellow|red|blue|green|orange|grey|gray)\b", body)
print("colour words in body (incl. captions):", len(words),
      sorted({w.lower() for w in words}))
