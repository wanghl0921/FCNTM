# -*- coding: utf-8 -*-
import io
import re

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
s = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()
body = s[s.index("\\section{Introduction}"):s.index("\\begin{thebibliography}")]
parts = re.split(r"\\section\{([^}]*)\}", body)
print("front matter words:",
      len(re.findall(r"[A-Za-z][A-Za-z'\-]+",
                     s[s.index("begin{abstract}"):s.index("section{Introduction}")])))
for i in range(1, len(parts), 2):
    name, text = parts[i], parts[i + 1]
    text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", text, flags=re.S)
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", text, flags=re.S)
    words = len(re.findall(r"[A-Za-z][A-Za-z'\-]+", text))
    print("%-32s %5d words  | subsections: %d" %
          (name, words, len(re.findall(r"\\subsection\{", text))))
    for m in re.finditer(r"\\subsection\{([^}]*)\}", text):
        seg = text[m.end():m.end() + 4000]
        seg = re.split(r"\\subsection\{", seg)[0]
        w = len(re.findall(r"[A-Za-z][A-Za-z'\-]+", seg))
        print("      %-52s %4d" % (m.group(1)[:50], w))
