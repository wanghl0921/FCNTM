# -*- coding: utf-8 -*-
"""Batch 12: Chinese pdf citation style, main_en section numbering, doc_work listing, misc."""
import io, os, re, datetime
import pymupdf

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")

print("#" * 22, "1. Chinese main.pdf citation rendering (scan whole pdf for [n])")
doc = pymupdf.open(os.path.join(PAPER, "main.pdf"))
hits = 0
for pno in range(doc.page_count):
    page = doc[pno]
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            for sp in l["spans"]:
                if re.search(r"\[\d{1,2}(?:[,\-–]\d{1,2})*\]", sp["text"]):
                    print("  p%d size=%.1f y=%.1f %r" % (pno + 1, sp["size"], sp["origin"][1], sp["text"][:80]))
                    hits += 1
    if hits >= 8:
        break
print("  bracket-citation spans found:", hits)
doc.close()

print()
print("#" * 22, "2. main_en.pdf section heading sample (numbered?)")
doc = pymupdf.open(os.path.join(PAPER, "main_en.pdf"))
for pno in range(1, min(6, doc.page_count)):
    t = doc[pno].get_text()
    for pat in [r"^\s*\d+(?:\.\d+)*\s+[A-Z][A-Za-z ]{3,60}$"]:
        for m in re.finditer(pat, t, re.M):
            print("  p%d: %r" % (pno + 1, m.group(0).strip()))
doc.close()

print()
print("#" * 22, "3. doc_work listing")
dw = os.path.join(D, "doc_work")
if os.path.exists(dw):
    for f in sorted(os.listdir(dw)):
        p = os.path.join(dw, f)
        print("  ", datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%m-%d"), os.path.getsize(p) if os.path.isfile(p) else "DIR", f)

print()
print("#" * 22, "4. current time")
print("  now:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print()
print("#" * 22, "5. main_en.tex: equations order + labels; tables order + labels; figures")
s = io.open(os.path.join(PAPER, "main_en.tex"), encoding="utf-8").read()
print("  -- equations in order:")
for i, m in enumerate(re.finditer(r"\\begin\{equation\}(.*?)\\end\{equation\}", s, re.S), 1):
    lab = re.search(r"\\label\{([^}]*)\}", m.group(1))
    print("     %d: %s" % (i, lab.group(1) if lab else "(no label)"))
print("  -- tables in order:")
for i, m in enumerate(re.finditer(r"\\begin\{table\}(.*?)\\end\{table\}", s, re.S), 1):
    lab = re.search(r"\\label\{(tab:[^}]*)\}", m.group(1))
    print("     %d: %s" % (i, lab.group(1) if lab else "(no label)"))
print("  -- figures in order:")
for i, m in enumerate(re.finditer(r"\\begin\{figure\}(.*?)\\end\{figure\}", s, re.S), 1):
    lab = re.search(r"\\label\{(fig:[^}]*)\}", m.group(1))
    print("     %d: %s" % (i, lab.group(1) if lab else "(no label)"))
print("  -- sections/subsections + sec labels:")
idx = 0
secno = 0
subno = 0
for m in re.finditer(r"\\(section|subsection)\{([^}]*)\}(\s*\\label\{(sec:[^}]*)\})?", s):
    if m.group(1) == "section":
        secno += 1
        subno = 0
        num = str(secno)
    else:
        subno += 1
        num = "%d.%d" % (secno, subno)
    if m.group(4):
        print("     %s -> %s %s" % (m.group(4), num, m.group(2)[:50]))
