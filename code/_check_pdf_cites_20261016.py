# -*- coding: utf-8 -*-
"""Check how citations render in main_en.pdf and main_asce.pdf (superscript? brackets?)."""
import pymupdf, re, os

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"

def dump_around(pdf, needle, half=160):
    doc = pymupdf.open(pdf)
    for pno in range(doc.page_count):
        page = doc[pno]
        txt = page.get_text()
        i = txt.find(needle)
        if i < 0:
            continue
        print("PDF:", os.path.basename(pdf), "page", pno + 1)
        # get spans
        d = page.get_text("dict")
        flat = []
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            for l in b["lines"]:
                for s in l["spans"]:
                    flat.append(s)
        # find spans containing distinctive tokens
        for s in flat:
            t = s["text"]
            if any(k in t for k in ["bag-of-words", "2003", "2013]", "[1]", "[2]", "(Blei"]):
                print("  size=%.1f y0=%.1f x0=%.1f font=%s text=%r" % (
                    s["size"], s["origin"][1], s["origin"][0], s["font"], t[:120]))
        return
    print("PDF:", os.path.basename(pdf), "needle not found")

dump_around(os.path.join(PAPER, "main_en.pdf"), "classical bag-of-words")
print("-" * 60)
dump_around(os.path.join(PAPER, "main_asce.pdf"), "classical bag-of-words")
