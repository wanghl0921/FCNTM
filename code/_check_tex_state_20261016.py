# -*- coding: utf-8 -*-
"""Check current state of main_en.tex / main_asce.tex before docx conversion."""
import io, os, re, sys

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

def load(name):
    p = os.path.join(PAPER, name)
    return p, io.open(p, encoding="utf-8").read()

for name in ["main_en.tex", "main_asce.tex"]:
    p, s = load(name)
    print("=" * 20, name)
    print("U+0008 count:", s.count("\x08"))
    print("citep count:", s.count("\\citep"))
    print("cite{ count:", s.count("\\cite{"))
    print("Citet count:", s.count("\\Citet"))
    print("includegraphics count:", len(re.findall(r"\\includegraphics", s)))
    print("input count:", len(re.findall(r"\\input\{", s)))
    print("is there \\bibliography:", "\\bibliography" in s)
    print("is there thebibliography:", "thebibliography" in s)
    print("bibitem count:", len(re.findall(r"\\bibitem", s)))
    print("author lines:")
    for line in s.splitlines():
        if any(k in line for k in ["\\author", "\\affil", "\\email", "\\thanks", "@qq.com", "manqp", "2010254508"]):
            print("   ", line.strip()[:160])
    # cite commands usefrom natbib
    cits = re.findall(r"\\(?:cite|citep|citet|Citet|Citep)\*?(?:\[[^\]]*\])?\{[^}]*\}", s)
    print("total citation commands:", len(cits), "first 3:", cits[:3])
    # equation environments
    print("equation envs:", len(re.findall(r"\\begin\{equation\}", s)),
          "align:", len(re.findall(r"\\begin\{align\}", s)),
          "equation*:", len(re.findall(r"\\begin\{equation\*?\}", s)))
