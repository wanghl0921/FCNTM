# -*- coding: utf-8 -*-
import io
import re

s = io.open(r"C:\Users\Administrator\Downloads\FCNTM完整代码\_docx_build"
            r"\main_en_pre.tex", encoding="utf-8").read()
pat = re.compile(r"\\begin\{table\}(.*?)\\end\{table\}", re.S)
for n, m in enumerate(pat.finditer(s), 1):
    body = m.group(1)
    lab = re.search(r"\\label\{(tab:[^}]*)\}", body)
    cap = re.search(r"\\caption\{([^}]{0,60})", body)
    tabs = body.count("\\begin{tabular}")
    print(n, "| label:", lab.group(1) if lab else None,
          "| tabulars:", tabs,
          "| caption:", (cap.group(1) if cap else None))
