# -*- coding: utf-8 -*-
"""Check Chinese version main.tex / main.pdf citation style, author block, tone."""
import pymupdf, re, os, io

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

print("### main.pdf citation rendering ###")
doc = pymupdf.open(os.path.join(PAPER, "main.pdf"))
found = False
for pno in range(min(8, doc.page_count)):
    page = doc[pno]
    d = page.get_text("dict")
    for b in d["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            line_txt = "".join(s["text"] for s in l["spans"])
            if any(k in line_txt for k in ["LDA", "GSDMM", "AVITM", "BTM"]):
                print("page", pno + 1, "line:", line_txt[:110])
                for s in l["spans"]:
                    print("    size=%.1f y0=%.1f font=%s text=%r" % (s["size"], s["origin"][1], s["font"], s["text"][:90]))
                found = True
                break
        if found:
            break
    if found:
        break

print()
print("### main.tex citation commands ###")
s = io.open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
for pat in [r"\\cite[a-z]*\*?(\[[^\]]*\])?\{", r"\[[0-9]{1,2}\]", r"上标"]:
    m = re.findall(pat, s)
    print(pat, "->", len(m))
print("author block:", [l.strip() for l in s.splitlines() if "author" in l.lower()][:6])
print("biblio:", [l.strip()[:70] for l in s.splitlines() if "bibli" in l.lower()][:5])

# tone words in Chinese version
print()
print("### negative-ish words in main.tex ###")
for w in ["不足", "局限", "失败", "无法", "不能", "低于", "较差", "较弱", "欠缺", "缺陷", "负面"]:
    n = s.count(w)
    if n:
        print(w, n)
