# -*- coding: utf-8 -*-
"""Deep check: citations in main.pdf (Chinese), main_asce.pdf, current main_en.pdf, delivered main_en.pdf."""
import os, io, zipfile, tempfile
import pymupdf

D = r"C:\Users\Administrator\Downloads"
PAPER = os.path.join(D, "FCNTM完整论文")


def span_scan(pdf, needle_list, maxpages=6):
    doc = pymupdf.open(pdf)
    found = False
    for pno in range(min(maxpages, doc.page_count)):
        page = doc[pno]
        d = page.get_text("dict")
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                line_txt = "".join(s["text"] for s in l["spans"])
                if any(k in line_txt for k in needle_list):
                    print("  page", pno + 1, ":", line_txt[:130])
                    for s in l["spans"]:
                        print("      size=%.1f y0=%.1f font=%s text=%r" % (
                            s["size"], s["origin"][1], s["font"], s["text"][:100]))
                    found = True
                    break
            if found:
                break
        if found:
            break
    if not found:
        print("  [needle not found in first pages]")
    doc.close()


print("### 1. Chinese main.pdf")
span_scan(os.path.join(PAPER, "main.pdf"), ["classical", "LDA", "GSDMM", "主题模型"])

print()
print("### 2. main_asce.pdf (current)")
span_scan(os.path.join(PAPER, "main_asce.pdf"), ["bag-of-words"])

print()
print("### 3. current main_en.pdf")
span_scan(os.path.join(PAPER, "main_en.pdf"), ["bag-of-words"])

print()
print("### 4. delivered main_en.pdf inside FCNTM_github_20261016.zip")
z = zipfile.ZipFile(os.path.join(D, "FCNTM_github_20261016.zip"))
cand = [n for n in z.namelist() if n.endswith("main_en.pdf")]
print("  candidates:", cand)
if cand:
    tmp = os.path.join(tempfile.gettempdir(), "delivered_main_en.pdf")
    with open(tmp, "wb") as fh:
        fh.write(z.read(cand[0]))
    span_scan(tmp, ["bag-of-words"])

print()
print("### 5. main.tex preamble (first 30 non-empty lines)")
s = io.open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
count = 0
for l in s.splitlines():
    if l.strip():
        print("   ", l[:120])
        count += 1
        if count >= 30:
            break
