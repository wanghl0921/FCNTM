# -*- coding: utf-8 -*-
import io
import re

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"
d = pymupdf.open(PAPER + r"\main_asce.pdf")
marks = {}
for i in range(d.page_count):
    t = d[i].get_text()
    if "REFERENCES" in t and "refs" not in marks:
        marks["refs"] = i + 1
    if "List of Tables" in t and "lot" not in marks:
        marks["lot"] = i + 1
    if re.search(r"TABLE\s*\n?\s*1\.", t) and "tab1" not in marks:
        marks["tab1"] = i + 1
    if "List of Figures" in t and "lof" not in marks:
        marks["lof"] = i + 1
    if re.search(r"Fig\.\s*1\.", t) and "fig1" not in marks:
        marks["fig1"] = i + 1
print("pages", d.page_count, marks)

s = io.open(PAPER + r"\main_asce.tex", encoding="utf-8").read()
body = s[s.index("begin{document}"):]
body = body[:body.index("Data Availability Statement") if
            "Data Availability Statement" in body else len(body)]
body = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", body, flags=re.S)
body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", body, flags=re.S)
body = re.sub(r"\\caption\{[^}]*\}", "", body)
words = len(re.findall(r"[A-Za-z][A-Za-z'\-]+", body))
print("body words:", words)
print("sections:", re.findall(r"\\section\{([^}]*)\}", s))
print("subsections:", len(re.findall(r"\\subsection\{", s)))
