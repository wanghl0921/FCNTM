# -*- coding: utf-8 -*-
import io
import re
import zipfile

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

print("== 1. docx image check (ASCE) ==")
z = zipfile.ZipFile(PAPER + r"\FC-NTM_manuscript_ASCE.docx")
xml = z.read("word/document.xml").decode("utf-8")
print("drawings:", xml.count("<w:drawing>"))
print("blip refs:", re.findall(r'r:embed="(rId\d+)"', xml)[:8])
rels = z.read("word/_rels/document.xml.rels").decode("utf-8")
print("media rels:",
      re.findall(r'Target="(media/[^"]+)"', rels))

print()
print("== 2. how main_en.tex refers to tables ==")
s = io.open(PAPER + r"\main_en.tex", encoding="utf-8").read()
for m in re.finditer(r"[^.\n]{0,60}Table[ ~\\]{1,2}(?:\\ref\{[^}]*\}|\d+)[^.\n]{0,40}",
                     s):
    print("   ", " ".join(m.group(0).split())[:130])

print()
print("== 3. supplementary_en.tex structure ==")
t = io.open(PAPER + r"\supplementary_en.tex", encoding="utf-8").read()
for m in re.finditer(r"\\(section|subsection)\*?\{([^}]*)\}", t):
    print("   ", m.group(1), m.group(2)[:70])
print("   tables:", len(re.findall(r"\\begin\{table\}", t)),
      "| caption sample:", re.findall(r"\\caption\{([^}]{0,50})", t)[:3])

print()
print("== 4. seed-paired CSV candidates ==")
import glob
for pat in ("*seed*paired*", "*civil*seed*", "*paired*"):
    for f in glob.glob(r"C:\Users\Administrator\Downloads\FCNTM完整代码\outputs\**"
                       "\\" + pat, recursive=True)[:12]:
        print("   ", f)
    print("   ---")
