# -*- coding: utf-8 -*-
"""Batch 4: read ASCE official template docs, check class file match, abstract length, control chars."""
import io, os, re, hashlib
import pymupdf

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")
EXT = os.path.join(D, "_asce_template", "_extract", "ascelike-new-main")

print("#" * 25, "1. README.md of ascelike-new-main")
p = os.path.join(EXT, "README.md")
print(io.open(p, encoding="utf-8").read()[:5000])

print()
print("#" * 25, "2. template.tex (first 100 lines)")
p = os.path.join(EXT, "template.tex")
s = io.open(p, encoding="utf-8").read()
print("\n".join(s.splitlines()[:100]))

print()
print("#" * 25, "3. class file hash compare")
for name in ["ascelike-new.cls", "ascelike-new.bst"]:
    a = os.path.join(PAPER, name)
    b = os.path.join(EXT, name)
    ha = hashlib.md5(open(a, "rb").read()).hexdigest()
    hb = hashlib.md5(open(b, "rb").read()).hexdigest()
    print("  %s paper=%s extract=%s match=%s" % (name, ha[:10], hb[:10], ha == hb))

print()
print("#" * 25, "4. main_asce abstract word count")
t = io.open(os.path.join(PAPER, "main_asce.tex"), encoding="utf-8").read()
m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", t, re.S)
if m:
    body = m.group(1)
    print("words:", len(re.findall(r"\S+", body)))
    print("has math $:", "$" in body)
    print("first 300:", " ".join(body.split())[:300])

print()
print("#" * 25, "5. control char scan")
files = [
    os.path.join(PAPER, "main_en.tex"),
    os.path.join(PAPER, "main_asce.tex"),
    os.path.join(PAPER, "main.tex"),
    os.path.join(PAPER, "supplementary_en.tex"),
    os.path.join(PAPER, "supplementary.tex"),
]
for f in files:
    s2 = io.open(f, encoding="utf-8").read()
    bad = {}
    for ch in s2:
        o = ord(ch)
        if o < 32 and ch != "\n":
            bad[o] = bad.get(o, 0) + 1
    print("  %-22s %s" % (os.path.basename(f), bad if bad else "clean"))

print()
print("#" * 25, "6. Chinese main.pdf citation rendering")
doc = pymupdf.open(os.path.join(PAPER, "main.pdf"))
found = 0
for pno in range(doc.page_count):
    d = doc[pno].get_text("dict")
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            txt = "".join(sp["text"] for sp in l["spans"])
            if re.search(r"\[\d{1,2}\]", txt):
                print("  page", pno + 1, ":", txt[:120])
                for sp in l["spans"]:
                    if "[" in sp["text"] or "]" in sp["text"] or sp["text"].strip().isdigit():
                        print("      size=%.1f y=%.1f %r" % (sp["size"], sp["origin"][1], sp["text"][:80]))
                found += 1
                if found >= 3:
                    break
        if found >= 3:
            break
    if found >= 3:
        break
if not found:
    print("  no [n] pattern found in first scan")
doc.close()
