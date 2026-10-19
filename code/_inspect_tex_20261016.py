# -*- coding: utf-8 -*-
"""Inspect structure of the two tex files for docx conversion planning."""
import io, os, re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

for name in ["main_en.tex", "main_asce.tex"]:
    s = io.open(os.path.join(PAPER, name), encoding="utf-8").read()
    print("=" * 25, name)
    print("graphics:", re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s))
    print("table envs:", len(re.findall(r"\\begin\{table", s)),
          "figure envs:", len(re.findall(r"\\begin\{figure", s)))
    print("multirow:", s.count("\\multirow"), "tabularx:", s.count("tabularx"),
          "tabular*:", s.count("tabular*"), "sidewaystable:", s.count("sidewaystable"))
    secs = re.findall(r"\\(section|subsection)\{([^}]*)\}", s)
    print("sections:", len(secs))
    for k, t in secs:
        print("   ", "##" if k == "section" else "   --", t[:90])
    # footnote / special macros
    for mac in ["\\footnote", "\\thanks", "\\begin{table*}", "\\textsuperscript", "\\citep", "\\citet", "\\citeauthor", "\\citeyear"]:
        n = s.count(mac)
        if n:
            print("macro", mac, ":", n)
    print("bibliography-related:", [l.strip()[:80] for l in s.splitlines() if "bibliographystyle" in l or "\\bibliography{" in l or "\\begin{thebibliography}" in l])
