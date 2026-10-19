# -*- coding: utf-8 -*-
"""Read current main_en.tex state: preamble, abstract, citations, bibliography."""
import io, re

P = r"C:\Users\Administrator\Downloads\FCNTM完整论文\main_en.tex"
s = io.open(P, encoding="utf-8").read()
lines = s.splitlines()

print("== head (1-24) ==")
print("\n".join(lines[:24]))

print()
print("== abstract block ==")
i = s.find("\\begin{abstract}")
j = s.find("\\end{abstract}")
abs_txt = s[i:j + len("\\end{abstract}")]
print(abs_txt)
print("abstract token count:", len(re.findall(r"\S+", s[i:j])))

print()
print("== citation counts ==")
print("\\cite{ :", s.count("\\cite{"), "| \\citep :", s.count("\\citep"), "| \\citet:", s.count("\\citet"))
multi = [k for k in re.findall(r"\\cite\{([^}]*)\}", s) if "," in k]
print("multi-key citations:", len(multi))
for k in multi[:4]:
    print("   ", k)

print()
print("== bibliography head ==")
bi = s.find("\\begin{thebibliography}")
print(s[bi:bi + 1500])

print()
print("== bibliography tail ==")
be = s.find("\\end{thebibliography}")
print(s[be - 700:be + 20])

print()
print("== author block check ==")
for ln, l in enumerate(lines, 1):
    if any(k in l for k in ["\\author", "\\affil", "@qq.com", "manqp", "2010254508", "Harbin", "Corresponding"]):
        print(ln, l.strip()[:160])
