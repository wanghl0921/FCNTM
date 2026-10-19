# -*- coding: utf-8 -*-
"""Every table/figure of the manuscript must be cited in the running text."""
import re

import pymupdf

PAPER = r"C:\Users\Administrator\Downloads\FCNTM完整论文"


def check(pdf: str, tables: int, figs: int) -> None:
    doc = pymupdf.open(PAPER + "\\" + pdf)
    pages = [doc[i].get_text() for i in range(doc.page_count)]
    cut = next((i for i, t in enumerate(pages) if "REFERENCES" in t),
               len(pages))
    body = "\n".join(pages[:cut])
    cited_t = sorted({int(x) for x in re.findall(r"Table\s+(\d+)", body)})
    cited_f = sorted({int(x) for x in re.findall(r"(?:Figure|Fig\.)\s+(\d+)",
                                                 body)})
    print(pdf, "pages", doc.page_count)
    print("  cited tables:", cited_t, "expected 1..%d" % tables)
    print("  cited figures:", cited_f, "expected 1..%d" % figs)
    print("  missing tables:", [n for n in range(1, tables + 1)
                                if n not in cited_t])
    print("  missing figures:", [n for n in range(1, figs + 1)
                                 if n not in cited_f])
    print("  question marks:", body.count("??"))


check("main_asce.pdf", 9, 6)
check("main_en.pdf", 15, 6)
