# -*- coding: utf-8 -*-
"""Full survey of main_en.tex / main_asce.tex: bibitem order, citation keys, tables, equations, control chars."""
import io, os, re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

for name in ["main_en.tex", "main_asce.tex"]:
    s = io.open(os.path.join(PAPER, name), encoding="utf-8").read()
    print("=" * 30, name)
    # control chars
    ctrl = sorted(set(ch for ch in s if ord(ch) < 32 and ch not in "\n\r\t"))
    print("control chars:", [(hex(ord(c)), s.count(c)) for c in ctrl])
    # bibitems
    keys = re.findall(r"\\bibitem\{([^}]+)\}", s)
    print("bibitem count:", len(keys))
    if keys:
        print("first 5 keys:", keys[:5])
        print("last 3 keys:", keys[-3:])
    # citation commands used
    cmds = re.findall(r"\\(cite[a-zA-Z]*)\*?(?:\[[^\]]*\])?\{", s)
    from collections import Counter
    print("citation cmds:", Counter(cmds))
    # multi-key citations
    allkeys = []
    for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])?\{([^}]*)\}", s):
        allkeys.append(m.group(1))
    multi = [k for k in allkeys if "," in k]
    print("multi-key cites:", len(multi), "example:", multi[:3])
    # tables: show first structural line of each tabular
    print("-- tables --")
    for i, m in enumerate(re.finditer(r"\\begin\{tabular\}(?:\{[^}]*\})?", s), 1):
        seg = s[m.start():m.start()+220].replace("\n", " ")
        print("T%d: %s" % (i, seg[:200]))
    # equations
    eqs = re.findall(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", s, re.S)
    print("-- equations:", len(eqs))
    for e in eqs[:8]:
        one = " ".join(e.split())[:120]
        print("EQ:", one)
